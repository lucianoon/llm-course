"""Registro de reproduções no schema que EVIDENCIAS.md exige.

Uma alegação no curso só passa a **reproduzida** quando existe uma execução preservada,
com commit, ambiente, hardware, seed e métricas. Este helper gera exatamente esse
arquivo em `resultados/<experimento>/<data>-<commit>.json`, com os campos de ambiente
preenchidos automaticamente. O registro vivo de validação está em EVIDENCIAS.md.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _commit_atual(raiz: Path) -> str:
    processo = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=raiz,
        capture_output=True,
        text=True,
        check=False,
    )
    return processo.stdout.strip() if processo.returncode == 0 else "sem-git"


def registrar_reproducao(
    raiz: Path,
    *,
    experimento: str,
    comando: str,
    seed: int = 0,
    modelos: list[dict[str, str]] | None = None,
    dados: list[dict[str, str]] | None = None,
    amostra_n: int = 0,
    metricas: dict[str, Any] | None = None,
    observacoes: str = "",
    python: str | None = None,
    plataforma: str | None = None,
    executado_em: str | None = None,
) -> Path:
    """Grava o registro de reprodução e devolve o caminho do arquivo criado."""
    if not experimento or "/" not in experimento:
        raise ValueError("experimento deve ter a forma 'modulo-NN/nome-do-lab'")
    raiz = Path(raiz)
    agora = datetime.now(UTC)
    registro = {
        "experimento": experimento,
        "commit": _commit_atual(raiz),
        "executado_em": executado_em or agora.isoformat(),
        "comando": comando,
        "python": python or sys.version.split()[0],
        "plataforma": plataforma or f"{platform.system()}-{platform.machine()}",
        "seed": seed,
        "modelos": modelos or [],
        "dados": dados or [],
        "amostra_n": amostra_n,
        "metricas": metricas or {},
        "observacoes": observacoes,
    }
    pasta = raiz / "resultados" / experimento
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / f"{agora.strftime('%Y%m%d')}-{registro['commit'][:8]}.json"
    arquivo.write_text(json.dumps(registro, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return arquivo
