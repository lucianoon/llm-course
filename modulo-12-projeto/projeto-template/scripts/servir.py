"""Serve o modelo com o orçamento e a proteção da camada de produção (módulo 19).

ESQUELETO do contrato do módulo 12. Abre o endpoint, aplica o guardião de custo e o
disjuntor do `tools/producao.py`, e mede p50/p95, throughput, tokens/s e sucesso —
o extrato que o revisor técnico lê antes de perguntar de qualidade.

Troque o corpo por um servidor real (FastAPI + vLLM / MLX) mantendo o desenho.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from preparar import carregar_config


def serve_once(config: dict, prompt: str) -> dict:
    """Uma chamada end-to-end simulada: o desenho que o `FastAPI` implementaria."""
    serv = config["serving"]
    tokens_entrada = len(prompt) // 4
    tokens_saida = serv["max_tokens"]
    # guardião de custo (módulo 19): recusa ANTES de gerar se o teto estourar.
    preco_saida = 0.60
    custo_max = (tokens_entrada * 0.30 + tokens_saida * preco_saida) / 1_000_000
    if custo_max > serv["orcamento_usd_por_requisicao"]:
        return {"resposta": None, "custo_usd": 0.0, "latencia_s": 0.0,
                "status": 403, "motivo": "orcamento_excedido"}
    inicio = time.perf_counter()
    # TODO: substitua pela geração real (vLLM / MLX).
    latencia = time.perf_counter() - inicio
    return {"resposta": "resposta-do-modelo", "custo_usd": round(custo_max, 6),
            "latencia_s": round(latencia, 4), "status": 200, "motivo": ""}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--prompt", default="exemplo de requisição")
    args = parser.parse_args(argv)

    config = carregar_config(Path(args.root) / "config.json")
    print(json.dumps(serve_once(config, args.prompt), ensure_ascii=False, indent=2))
    print("\n(desenho do módulo 19: guardião de custo + disjuntor + extrato p50/p95.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
