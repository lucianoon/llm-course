"""Executa de ponta a ponta os labs CPU que não precisam de rede ou acelerador."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.validar_execucao_labs import caminhos_do_perfil

# Lista intencionalmente explícita. Um lab só entra depois de provar que roda com
# TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE e cabe no orçamento do CI.
LABS_OFFLINE = caminhos_do_perfil("ci_offline")

# São offline, mas excederam 60 s no ambiente local. Continuam disponíveis para
# validação agendada/manual sem transformar cada push numa espera de vários minutos.
LABS_OFFLINE_LONGOS = caminhos_do_perfil("offline_longo")


@dataclass(frozen=True)
class ResultadoSmoke:
    caminho: str
    status: str
    duracao_s: float
    detalhe: str = ""


def executar_lab(
    caminho: str,
    *,
    raiz: Path = ROOT,
    python: str = sys.executable,
    timeout_s: float = 180.0,
) -> ResultadoSmoke:
    inicio = time.perf_counter()
    ambiente = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONIOENCODING": "utf-8",
    }
    try:
        processo = subprocess.run(
            [python, caminho],
            cwd=raiz,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=ambiente,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as erro:
        return ResultadoSmoke(
            caminho,
            "timeout",
            round(time.perf_counter() - inicio, 3),
            f"excedeu {erro.timeout}s",
        )
    duracao = round(time.perf_counter() - inicio, 3)
    if processo.returncode != 0:
        detalhe = (processo.stderr or processo.stdout)[-2000:]
        return ResultadoSmoke(caminho, "falhou", duracao, detalhe)
    return ResultadoSmoke(caminho, "passou", duracao)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=180.0, help="limite por lab em segundos")
    parser.add_argument(
        "--incluir-longos",
        action="store_true",
        help="inclui pré-treino, GRPO e MoE, que podem levar vários minutos",
    )
    parser.add_argument("labs", nargs="*", help="subconjunto opcional de caminhos")
    args = parser.parse_args()
    padrao = LABS_OFFLINE + (LABS_OFFLINE_LONGOS if args.incluir_longos else ())
    labs = tuple(args.labs) or padrao
    resultados = [executar_lab(lab, timeout_s=args.timeout) for lab in labs]
    for resultado in resultados:
        print(f"{resultado.status:7} {resultado.duracao_s:8.3f}s  {resultado.caminho}")
        if resultado.detalhe:
            print(resultado.detalhe)
    passaram = sum(resultado.status == "passou" for resultado in resultados)
    duracao = sum(resultado.duracao_s for resultado in resultados)
    print(f"{passaram}/{len(resultados)} labs passaram em {duracao:.1f}s")
    return 0 if passaram == len(resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
