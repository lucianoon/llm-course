"""Compara baseline e RAG e regenera resultados.json."""

from __future__ import annotations

import hashlib
import json
import random
import time

from sistema import (
    PROJETO,
    RAIZ_REPO,
    AssistentePoliticas,
    BaselineTitulo,
    carregar_config,
    carregar_jsonl,
)

from tools.reproducao import criar_registro


def acertou(resposta, exemplo: dict) -> bool:
    if not exemplo["respondivel"]:
        return resposta.status == "abstencao"
    return (
        resposta.status == "respondido"
        and exemplo["documento_esperado"] in resposta.citacoes
        and all(termo.casefold() in resposta.resposta.casefold() for termo in exemplo["termos_esperados"])
    )


def intervalo_bootstrap_pareado(
    baseline: list[int],
    final: list[int],
    seed: int,
    amostras: int,
) -> list[float]:
    rng = random.Random(seed)
    n = len(final)
    deltas = []
    for _ in range(amostras):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(final[i] - baseline[i] for i in indices) / n)
    deltas.sort()
    return [round(deltas[int(0.025 * amostras)], 4), round(deltas[int(0.975 * amostras)], 4)]


def main() -> int:
    config = carregar_config()
    exemplos = carregar_jsonl(PROJETO / "dados" / "avaliacao.jsonl")
    baseline = BaselineTitulo()
    final = AssistentePoliticas(config)
    acertos_baseline: list[int] = []
    acertos_final: list[int] = []
    detalhes = []
    latencias_ms = []

    for exemplo in exemplos:
        resposta_baseline = baseline.responder(exemplo["pergunta"])
        inicio = time.perf_counter()
        resposta_final = final.responder(exemplo["pergunta"])
        latencias_ms.append((time.perf_counter() - inicio) * 1000)
        ok_baseline = int(acertou(resposta_baseline, exemplo))
        ok_final = int(acertou(resposta_final, exemplo))
        acertos_baseline.append(ok_baseline)
        acertos_final.append(ok_final)
        detalhes.append(
            {
                "id": exemplo["id"],
                "baseline_ok": bool(ok_baseline),
                "final_ok": bool(ok_final),
                "status_final": resposta_final.status,
                "citacoes": resposta_final.citacoes,
                "score": resposta_final.score,
            }
        )

    n = len(exemplos)
    ordenadas = sorted(latencias_ms)
    p50 = ordenadas[(n - 1) // 2]
    p95 = ordenadas[min(n - 1, int(0.95 * n))]
    bootstrap = config["avaliacao"]["bootstrap_amostras"]
    metricas = {
        "metrica": "acurácia: resposta sustentada e citada, ou abstenção correta",
        "baseline": round(sum(acertos_baseline) / n, 4),
        "modelo_final": round(sum(acertos_final) / n, 4),
        "delta": round((sum(acertos_final) - sum(acertos_baseline)) / n, 4),
        "delta_ic95_bootstrap_pareado": intervalo_bootstrap_pareado(
            acertos_baseline,
            acertos_final,
            config["seed"],
            bootstrap,
        ),
        "bootstrap_amostras": bootstrap,
        "serving_local": {
            "latencia_p50_ms": round(p50, 4),
            "latencia_p95_ms": round(p95, 4),
            "observacao": "processo local sem rede, concorrência ou modelo generativo",
        },
    }
    manifesto_dados = PROJETO / "dataset-manifest.json"
    resultado = criar_registro(
        RAIZ_REPO,
        experimento="modulo-12/capstone-rag-politicas-v1",
        comando="python modulo-12-projeto/projeto-referencia-rag/scripts/avaliar.py",
        seed=config["seed"],
        dados=[
            {
                "id": "politicas-internas-ficticias-v1",
                "revision_ou_sha256": hashlib.sha256(manifesto_dados.read_bytes()).hexdigest(),
            }
        ],
        amostra_n=n,
        metricas=metricas,
        observacoes="Corpus fictício; resultado local não equivale a produção nem reprodução independente.",
    )
    resultado["modelos"] = [
        {"id": "extrativo-bm25-v1", "revision": resultado["commit"]}
    ]
    resultado["detalhes"] = detalhes
    destino = PROJETO / "resultados.json"
    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "n": resultado["amostra_n"],
                **{
                    chave: resultado["metricas"][chave]
                    for chave in (
                        "baseline",
                        "modelo_final",
                        "delta",
                        "delta_ic95_bootstrap_pareado",
                    )
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
