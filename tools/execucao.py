"""Execução consistente de CLIs Python usadas pelos laboratórios MLX."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ResultadoExecucao:
    """Resultado observável de um subprocesso, sem expor ``CompletedProcess``."""

    ok: bool
    saida: str
    segundos: float


def executar_modulo(
    modulo: str,
    *args: str,
    mostrar: int = 0,
    verificar: bool = True,
) -> ResultadoExecucao:
    """Executa ``python -m modulo`` e padroniza log, tempo e tratamento de falhas."""
    inicio = time.perf_counter()
    # Os labs imprimem acentos e setas (→). Sem UTF-8 nas duas pontas, o filho
    # falha ao emitir esses caracteres e a decodificação aqui os corrompe.
    processo = subprocess.run(
        [sys.executable, "-m", modulo, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    segundos = time.perf_counter() - inicio
    saida = processo.stdout
    if processo.stderr:
        saida += "\n--- STDERR ---\n" + processo.stderr

    resumo = " ".join((modulo, *args[:2]))
    print(f"$ {resumo} ... ({segundos:.0f}s, exit {processo.returncode})")
    if mostrar or processo.returncode != 0:
        limite = mostrar if mostrar > 0 else 1200
        print(saida[-limite:])

    if verificar and processo.returncode != 0:
        raise RuntimeError(f"{resumo} falhou com exit {processo.returncode}")
    return ResultadoExecucao(processo.returncode == 0, saida, segundos)
