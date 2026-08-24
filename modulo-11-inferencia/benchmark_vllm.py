"""Carga concorrente contra o endpoint OpenAI-compatible do vLLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "tools"))


def argumentos():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--requisicoes", type=int, default=100)
    parser.add_argument("--concorrencia", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--saida", type=Path, default=AQUI / "resultado-serving.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def executar(args):
    import httpx
    from serving import AmostraServing, resumir_carga

    semaforo = asyncio.Semaphore(args.concorrencia)
    prompts = [
        "Explique em duas frases por que o KV cache cresce com o contexto.",
        "Compare LoRA e full fine-tuning em três itens.",
        "O que é continuous batching?",
        "Resuma o papel do roteador em um MoE.",
    ]

    async with httpx.AsyncClient(timeout=180) as cliente:
        async def uma(indice):
            async with semaforo:
                inicio = time.perf_counter()
                try:
                    resposta = await cliente.post(
                        f"{args.url}/v1/chat/completions",
                        json={
                            "model": args.modelo,
                            "messages": [{"role": "user", "content": prompts[indice % len(prompts)]}],
                            "temperature": 0,
                            "max_tokens": args.max_tokens,
                        },
                    )
                    resposta.raise_for_status()
                    corpo = resposta.json()
                    tokens = corpo.get("usage", {}).get("completion_tokens", 0)
                    return AmostraServing(time.perf_counter() - inicio, tokens, True)
                except (httpx.HTTPError, ValueError):
                    return AmostraServing(time.perf_counter() - inicio, 0, False)

        inicio = time.perf_counter()
        amostras = await asyncio.gather(*(uma(i) for i in range(args.requisicoes)))
        duracao = time.perf_counter() - inicio
        return resumir_carga(amostras, duracao)


def main():
    args = argumentos()
    config = {
        "url": args.url,
        "modelo": args.modelo,
        "requisicoes": args.requisicoes,
        "concorrencia": args.concorrencia,
        "max_tokens": args.max_tokens,
        "saida": str(args.saida),
    }
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    metricas = asyncio.run(executar(args))
    args.saida.write_text(json.dumps(config | metricas, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metricas, indent=2))


if __name__ == "__main__":
    main()
