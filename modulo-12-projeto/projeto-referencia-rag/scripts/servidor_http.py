"""Servidor HTTP local do capstone, somente com a biblioteca padrão."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sistema import AssistentePoliticas, carregar_config

MAX_PAYLOAD_BYTES = 8_192
MAX_PERGUNTA_CARACTERES = 500


def criar_servidor(host: str, porta: int, api_key: str) -> ThreadingHTTPServer:
    if not api_key:
        raise ValueError("api_key não pode ser vazia")
    assistente = AssistentePoliticas(carregar_config())

    class Handler(BaseHTTPRequestHandler):
        server_version = "CapstoneRAG/1.0"

        def do_GET(self) -> None:
            if self.path != "/health":
                self._responder(HTTPStatus.NOT_FOUND, {"erro": "rota inexistente"})
                return
            self._responder(
                HTTPStatus.OK,
                {"status": "ok", "modelo": "extrativo-bm25-v1"},
            )

        def do_POST(self) -> None:
            inicio = time.perf_counter()
            trace_id = self.headers.get("X-Request-ID") or str(uuid.uuid4())
            if self.path != "/v1/responder":
                self._responder(HTTPStatus.NOT_FOUND, {"erro": "rota inexistente"}, trace_id)
                return
            recebida = self.headers.get("X-API-Key", "")
            if not hmac.compare_digest(recebida, api_key):
                self._responder(HTTPStatus.UNAUTHORIZED, {"erro": "não autorizado"}, trace_id)
                self._log(trace_id, HTTPStatus.UNAUTHORIZED, inicio)
                return

            try:
                tamanho = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._responder(HTTPStatus.BAD_REQUEST, {"erro": "Content-Length inválido"}, trace_id)
                return
            if tamanho <= 0 or tamanho > MAX_PAYLOAD_BYTES:
                self._responder(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"erro": f"payload deve ter entre 1 e {MAX_PAYLOAD_BYTES} bytes"},
                    trace_id,
                )
                self._log(trace_id, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, inicio)
                return

            try:
                payload = json.loads(self.rfile.read(tamanho))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._responder(HTTPStatus.BAD_REQUEST, {"erro": "JSON inválido"}, trace_id)
                self._log(trace_id, HTTPStatus.BAD_REQUEST, inicio)
                return

            pergunta = payload.get("pergunta") if isinstance(payload, dict) else None
            if not isinstance(pergunta, str) or not pergunta.strip():
                self._responder(HTTPStatus.UNPROCESSABLE_ENTITY, {"erro": "pergunta obrigatória"}, trace_id)
                self._log(trace_id, HTTPStatus.UNPROCESSABLE_ENTITY, inicio)
                return
            if len(pergunta) > MAX_PERGUNTA_CARACTERES:
                self._responder(HTTPStatus.UNPROCESSABLE_ENTITY, {"erro": "pergunta longa demais"}, trace_id)
                self._log(trace_id, HTTPStatus.UNPROCESSABLE_ENTITY, inicio)
                return

            resposta = assistente.responder(pergunta)
            corpo = {"trace_id": trace_id, **resposta.para_dict()}
            self._responder(HTTPStatus.OK, corpo, trace_id)
            self._log(trace_id, HTTPStatus.OK, inicio, resposta.status)

        def log_message(self, formato: str, *args) -> None:
            # Desativa o log textual da classe base. `_log` emite JSON sem a pergunta.
            return

        def _responder(
            self,
            status: HTTPStatus,
            corpo: dict,
            trace_id: str | None = None,
        ) -> None:
            conteudo = json.dumps(corpo, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(conteudo)))
            if trace_id:
                self.send_header("X-Request-ID", trace_id)
            self.end_headers()
            self.wfile.write(conteudo)

        def _log(
            self,
            trace_id: str,
            status: HTTPStatus,
            inicio: float,
            resultado: str = "",
        ) -> None:
            evento = {
                "trace_id": trace_id,
                "rota": self.path,
                "status_http": int(status),
                "latencia_ms": round((time.perf_counter() - inicio) * 1000, 4),
                "resultado": resultado,
            }
            print(json.dumps(evento, ensure_ascii=False), flush=True)

    servidor = ThreadingHTTPServer((host, porta), Handler)
    servidor.daemon_threads = True
    return servidor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--porta", type=int, default=8080)
    args = parser.parse_args()
    api_key = os.environ.get("CAPSTONE_API_KEY", "")
    if not api_key:
        parser.error("defina CAPSTONE_API_KEY antes de iniciar o servidor")
    servidor = criar_servidor(args.host, args.porta, api_key)
    print(f"servidor em http://{args.host}:{servidor.server_port}", flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
