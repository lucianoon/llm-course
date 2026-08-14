"""Normalização de respostas numéricas compartilhada pelos módulos 9 e 10."""

from __future__ import annotations

import re


def extrair_numero(texto: str) -> str | None:
    """Extrai o último número ou o marcador explícito de resposta final."""
    marcador = re.search(
        r"(?:resposta final|final answer|answer is)[:\s]*\$?\s*([\-0-9.,]+)",
        texto,
        re.IGNORECASE,
    )
    candidato = marcador.group(1) if marcador else None
    if candidato is None:
        numeros = re.findall(r"-?\$?\d[\d,]*\.?\d*", texto)
        if not numeros:
            return None
        candidato = numeros[-1]
    limpo = candidato.replace("$", "").replace(",", "").rstrip(".")
    if limpo.endswith((".0", ".00")):
        limpo = limpo.split(".")[0]
    return limpo or None
