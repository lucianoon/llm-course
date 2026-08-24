"""Cálculo das métricas de carga usadas no laboratório de serving."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AmostraServing:
    latencia_s: float
    tokens_saida: int
    sucesso: bool


def percentil(valores: list[float], p: float) -> float:
    if not valores or not 0 <= p <= 1:
        raise ValueError("percentil requer valores e p entre 0 e 1")
    ordenados = sorted(valores)
    indice = min(len(ordenados) - 1, math.ceil(p * len(ordenados)) - 1)
    return ordenados[max(0, indice)]


def resumir_carga(amostras: list[AmostraServing], duracao_s: float) -> dict[str, float]:
    if not amostras or duracao_s <= 0:
        raise ValueError("carga vazia ou duração inválida")
    latencias = [amostra.latencia_s for amostra in amostras if amostra.sucesso]
    if not latencias:
        raise ValueError("nenhuma requisição teve sucesso")
    tokens = sum(amostra.tokens_saida for amostra in amostras if amostra.sucesso)
    return {
        "sucesso": sum(amostra.sucesso for amostra in amostras) / len(amostras),
        "latencia_p50_s": percentil(latencias, 0.50),
        "latencia_p95_s": percentil(latencias, 0.95),
        "throughput_tokens_s": tokens / duracao_s,
        "requisicoes_s": len(amostras) / duracao_s,
    }
