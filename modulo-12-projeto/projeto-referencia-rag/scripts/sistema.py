"""Sistema pequeno usado pelo capstone de referência."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJETO = Path(__file__).resolve().parents[1]
RAIZ_REPO = Path(__file__).resolve().parents[3]
if str(RAIZ_REPO) not in sys.path:
    sys.path.insert(0, str(RAIZ_REPO))

from tools.producao import calcular_custo, contar_tokens
from tools.rag import BM25, tokenizar_busca

STOPWORDS_BUSCA = {
    "a",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "é",
    "existe",
    "o",
    "os",
    "para",
    "posso",
    "por",
    "qual",
    "quando",
    "que",
    "um",
    "uma",
}


@dataclass(frozen=True)
class Documento:
    id: str
    titulo: str
    texto: str
    fonte: str
    licenca: str
    atualizado_em: str


@dataclass(frozen=True)
class Resposta:
    status: str
    resposta: str
    citacoes: tuple[str, ...]
    score: float
    tokens_entrada_estimados: int
    tokens_saida_estimados: int
    custo_estimado: float

    def para_dict(self) -> dict:
        return asdict(self)


def carregar_jsonl(caminho: Path) -> list[dict]:
    return [json.loads(linha) for linha in caminho.read_text(encoding="utf-8").splitlines() if linha]


def carregar_documentos() -> list[Documento]:
    return [Documento(**item) for item in carregar_jsonl(PROJETO / "dados" / "corpus.jsonl")]


class AssistentePoliticas:
    def __init__(self, config: dict, documentos: list[Documento] | None = None):
        self.config = config
        self.documentos = documentos or carregar_documentos()
        retrieval = config["retrieval"]
        self.indice = BM25(
            [f"{doc.titulo}. {doc.texto}" for doc in self.documentos],
            k1=retrieval["k1"],
            b=retrieval["b"],
        )

    def responder(self, pergunta: str) -> Resposta:
        retrieval = self.config["retrieval"]
        consulta = " ".join(
            termo for termo in tokenizar_busca(pergunta) if termo not in STOPWORDS_BUSCA
        )
        indices, scores = self.indice.buscar(consulta, k=retrieval["top_k"])
        melhor_score = scores[0] if scores else 0.0

        if not indices or melhor_score < retrieval["score_minimo"]:
            return self._formatar(
                pergunta,
                "abstencao",
                "Não encontrei suporte na base para responder.",
                (),
                melhor_score,
                "",
            )

        documento = self.documentos[indices[0]]
        return self._formatar(
            pergunta,
            "respondido",
            documento.texto,
            (documento.id,),
            melhor_score,
            documento.texto,
        )

    def _formatar(
        self,
        pergunta: str,
        status: str,
        resposta: str,
        citacoes: tuple[str, ...],
        score: float,
        contexto: str,
    ) -> Resposta:
        entrada = contar_tokens(f"{pergunta} {contexto}")
        saida = contar_tokens(resposta)
        custo = calcular_custo(
            entrada,
            saida,
            self.config["serving"]["precos_por_milhao_tokens"],
        )
        return Resposta(status, resposta, citacoes, round(score, 6), entrada, saida, round(custo, 8))


class BaselineTitulo:
    """Baseline deliberadamente simples, mas funcional: overlap com o título."""

    def __init__(self, documentos: list[Documento] | None = None):
        self.documentos = documentos or carregar_documentos()

    def responder(self, pergunta: str) -> Resposta:
        termos = set(tokenizar_busca(pergunta))
        scores = [len(termos & set(tokenizar_busca(doc.titulo))) for doc in self.documentos]
        indice = max(range(len(scores)), key=scores.__getitem__)
        documento = self.documentos[indice]
        return Resposta(
            "respondido",
            documento.texto,
            (documento.id,),
            float(scores[indice]),
            0,
            0,
            0.0,
        )


def carregar_config() -> dict:
    return json.loads((PROJETO / "config.json").read_text(encoding="utf-8"))
