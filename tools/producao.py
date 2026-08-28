"""Métricas e guardiões da camada de produção de LLMs (módulo 19).

São as peças pequenas e reutilizáveis do laboratório de engenharia de produção:
orçamento de custo e latência, disjuntor de segurança, contabilidade de tokens e
o resumo de tráfego que uma equipe olha num dash. Tudo determinístico e sem
dependência de modelo — cabe no CPU de qualquer máquina e no CI.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

# Preços por milhão de tokens. Representativos apenas; em produção troque pelos
# valores do seu provedor e versione junto com o manifesto do modelo.
DEFAULT_PRECOS = {
    "entrada": 0.30,
    "saida": 0.60,
}


def _percentil(valores: list[float], p: float) -> float:
    """O percentil que o extrato de tráfego usa (p entre 0 e 1).

    Duplicado de ``serving.percentil`` de propósito: os labs importam as tools por
    nome simples (``from producao import ...``) e as tests pelo pacote (``from
    tools.producao ...``). Como o mesmo módulo precisa funcionar nos dois contextos,
    ele não cruza imports com o pacote irmão — a convenção segue o ``modulos.py``.
    """
    if not valores or not 0 <= p <= 1:
        raise ValueError("percentil requer valores e p entre 0 e 1")
    ordenados = sorted(valores)
    indice = min(len(ordenados) - 1, math.ceil(p * len(ordenados)) - 1)
    return ordenados[max(0, indice)]


def contar_tokens(texto: str) -> int:
    """Conta tokens de forma determinística, SEM tokenizer real.

    Usa uma aproximação simples (1 token ~ grupos de 4 caracteres em texto real,
    mais 1 para o espaço). NUNCA use para cobrar ou para uma conta que decide algo:
    troque pelo tokenizer do modelo. É um brinquedo de escala,
    avisado como brinquedo — a convenção está no README do módulo 19.
    """
    if not texto.strip():
        return 0
    return len(texto) // 4 + sum(1 for c in texto if c == " ")


def calcular_custo(
    tokens_entrada: int,
    tokens_saida: int,
    precos: dict[str, float] | None = None,
) -> float:
    """Custo em moeda, a partir de tokens e dos preços por milhão."""
    if tokens_entrada < 0 or tokens_saida < 0:
        raise ValueError("contagem de tokens não pode ser negativa")
    precos = precos or DEFAULT_PRECOS
    return (
        tokens_entrada * precos["entrada"] / 1_000_000
        + tokens_saida * precos["saida"] / 1_000_000
    )


@dataclass(frozen=True)
class LinhaDeTrafego:
    """Um registro imutável de uma requisição servida (ou recusada)."""

    req_id: int
    ok: bool
    latencia_s: float
    tokens_entrada: int
    tokens_saida: int
    custo: float
    modelo: str
    status_http: int
    motivo: str = ""


def resumir_trafego(linhas: list[LinhaDeTrafego]) -> dict[str, float | int | str]:
    """Transforma a fila de requisições no extrato de um dash.

    Métricas de latência consideram apenas os sucessos (a recusa pelo disjuntor
    não deve arrastar o p95 para cima de quem ficou suspenso — outro registro é que
    conta isso). O motivo mais frequente de status != 200 vira a linha pior da lista.
    """
    if not linhas:
        raise ValueError("não há tráfego para resumir")
    sucessos = [linha for linha in linhas if linha.ok]
    if not sucessos:
        raise ValueError("nenhuma requisição teve sucesso")
    latencias = [linha.latencia_s for linha in sucessos]
    total_saida = sum(linha.tokens_saida for linha in sucessos)
    marcador_inicio = min(linha.req_id for linha in linhas)
    duracao_s = max(linha.latencia_s for linha in sucessos)
    motivo_comum: dict[str, int] = {}
    for linha in linhas:
        if linha.status_http != 200:
            motivo_comum[linha.motivo] = motivo_comum.get(linha.motivo, 0) + 1
    return {
        "requisicoes": len(linhas),
        "sucesso": round(sum(linha.ok for linha in linhas) / len(linhas), 4),
        "latencia_p50_s": round(_percentil(latencias, 0.50), 4),
        "latencia_p95_s": round(_percentil(latencias, 0.95), 4),
        "throughput_tokens_s": round(total_saida / duracao_s, 2),
        "custo_total": round(sum(linha.custo for linha in linhas), 6),
        "motivo_falha_comum": max(motivo_comum, key=motivo_comum.get) if motivo_comum else "",
        "primeiro_req_id": marcador_inicio,
    }


@dataclass
class Disjuntor:
    """Disjuntor de custo/latência estilo circuit breaker.

    Abre quando a taxa de falha numa janela passa do limiar e passa a recusar
    rapidinho (`fast-fail`) — para não gastar dinheiro servindo um modelo quebrado.
    Depois do resfriamento, entra em meio aberto e deixa passar exatamente uma
    requisição-prova; se ela falhar, reabre. Se passar, fecha, limpando a janela.
    """

    limiar_falha: float
    janela_s: float
    resfriamento_s: float
    amostras_minimas: int = 5
    _eventos: list[tuple[float, bool]] = field(default_factory=list)
    _aberto: bool = False
    _meio_aberto: bool = False
    _desde_abertura: float = 0.0
    _prova_feita: bool = False

    def permitir(self, agora: float | None = None) -> bool:
        agora = agora if agora is not None else time.monotonic()
        if self._aberto:
            if (agora - self._desde_abertura) >= self.resfriamento_s:
                self._aberto = False
                self._meio_aberto = True
                # A própria requisição que desperta o circuito é a prova; a
                # seguinte é recusada até o desfecho ser registrado.
                self._prova_feita = True
                return True
            return False
        if self._meio_aberto:
            if self._prova_feita:
                return False
            self._prova_feita = True
            return True
        return True

    def taxa_falha(self, agora: float | None = None) -> float:
        agora = agora if agora is not None else time.monotonic()
        eventos = [ok for instante, ok in self._eventos if (agora - instante) <= self.janela_s]
        if not eventos:
            return 0.0
        return sum(not ok for ok in eventos) / len(eventos)

    def registrar_sucesso(self, agora: float | None = None) -> None:
        agora = agora if agora is not None else time.monotonic()
        self._eventos.append((agora, True))
        self._limpar_janela(agora)
        if self._meio_aberto:
            self._fechar()

    def registrar_falha(self, agora: float | None = None) -> None:
        agora = agora if agora is not None else time.monotonic()
        self._eventos.append((agora, False))
        self._limpar_janela(agora)
        if self._meio_aberto:
            self._aberto = True
            self._meio_aberto = False
            self._desde_abertura = agora
            return
        eventos = [
            instante for instante, ok in self._eventos if (agora - instante) <= self.janela_s
        ]
        if len(eventos) >= self.amostras_minimas and self.taxa_falha(agora) >= self.limiar_falha:
            self._aberto = True
            self._desde_abertura = agora

    def _limpar_janela(self, agora: float) -> None:
        self._eventos = [
            (instante, ok) for instante, ok in self._eventos if (agora - instante) <= self.janela_s
        ]

    def _fechar(self) -> None:
        self._aberto = False
        self._meio_aberto = False
        self._prova_feita = False
        self._eventos.clear()
