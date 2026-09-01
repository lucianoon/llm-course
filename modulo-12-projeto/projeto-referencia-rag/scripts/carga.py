"""Teste concorrente do servidor HTTP local do capstone."""

from __future__ import annotations

import hashlib
import http.client
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

from servidor_http import criar_servidor
from sistema import PROJETO, RAIZ_REPO, carregar_jsonl

from tools.reproducao import criar_registro
from tools.serving import AmostraServing, resumir_carga

API_KEY_TESTE = "chave-local-de-teste"


def requisitar(
    host: str,
    porta: int,
    pergunta: str,
    api_key: str,
) -> tuple[int, dict, float]:
    inicio = time.perf_counter()
    conexao = http.client.HTTPConnection(host, porta, timeout=5)
    corpo = json.dumps({"pergunta": pergunta}, ensure_ascii=False).encode("utf-8")
    conexao.request(
        "POST",
        "/v1/responder",
        body=corpo,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(corpo)),
            "X-API-Key": api_key,
        },
    )
    resposta = conexao.getresponse()
    payload = json.loads(resposta.read())
    conexao.close()
    return resposta.status, payload, time.perf_counter() - inicio


def main() -> int:
    host = "127.0.0.1"
    servidor = criar_servidor(host, 0, API_KEY_TESTE)
    thread = Thread(target=servidor.serve_forever, daemon=True)
    thread.start()
    porta = servidor.server_port

    try:
        nao_autorizada, _, _ = requisitar(host, porta, "Qual o prazo para férias?", "errada")
        perguntas = [
            item["pergunta"]
            for item in carregar_jsonl(PROJETO / "dados" / "avaliacao.jsonl")
        ] * 3
        inicio_carga = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as executor:
            respostas = list(
                executor.map(
                    lambda pergunta: requisitar(host, porta, pergunta, API_KEY_TESTE),
                    perguntas,
                )
            )
        duracao = time.perf_counter() - inicio_carga
    finally:
        servidor.shutdown()
        servidor.server_close()
        thread.join(timeout=2)

    amostras = [
        AmostraServing(
            latencia_s=latencia,
            tokens_saida=payload.get("tokens_saida_estimados", 0),
            sucesso=status == 200,
        )
        for status, payload, latencia in respostas
    ]
    resumo = resumir_carga(amostras, duracao)
    metricas = {
        "requisicoes": len(respostas),
        "concorrencia": 8,
        "probe_sem_api_key_status": nao_autorizada,
        **{chave: round(valor, 6) for chave, valor in resumo.items()},
    }
    resultado = criar_registro(
        RAIZ_REPO,
        experimento="modulo-12/carga-http-local-v1",
        comando="python modulo-12-projeto/projeto-referencia-rag/scripts/carga.py",
        seed=0,
        dados=[
            {
                "id": "politicas-internas-ficticias-v1",
                "revision_ou_sha256": hashlib.sha256(
                    (PROJETO / "dataset-manifest.json").read_bytes()
                ).hexdigest(),
            }
        ],
        amostra_n=len(respostas),
        metricas=metricas,
        observacoes="Servidor e cliente locais; não representa rede ou modelo generativo.",
    )
    resultado["modelos"] = [
        {"id": "extrativo-bm25-v1", "revision": resultado["commit"]}
    ]
    destino = PROJETO / "carga-resultados.json"
    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
