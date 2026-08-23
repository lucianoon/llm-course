"""Leitura e recuperação de arquivos JSONL incrementais."""

from __future__ import annotations

import json
from pathlib import Path


def preparar_jsonl_retomavel(caminho: Path) -> int:
    """Valida JSONL e remove apenas uma última linha truncada.

    Corrupção no meio do arquivo é fatal: descartá-la mudaria a associação entre a
    posição do registro e o trabalho que deve ser retomado.
    """
    if not caminho.exists():
        return 0

    linhas = caminho.read_bytes().splitlines(keepends=True)
    nao_vazias = [indice for indice, linha in enumerate(linhas) if linha.strip()]
    ultima = nao_vazias[-1] if nao_vazias else -1
    validas = 0

    for indice, linha in enumerate(linhas):
        if not linha.strip():
            continue
        try:
            json.loads(linha)
        except (json.JSONDecodeError, UnicodeDecodeError) as erro:
            if indice != ultima:
                raise ValueError(f"JSONL corrompido na linha {indice + 1}: {caminho}") from erro
            caminho.write_bytes(b"".join(linhas[:indice]))
            return validas
        validas += 1
    return validas
