"""Interface de linha de comando para uma requisição do capstone."""

from __future__ import annotations

import json
import sys

from sistema import AssistentePoliticas, carregar_config


def main(argumentos: list[str] | None = None) -> int:
    argumentos = argumentos if argumentos is not None else sys.argv[1:]
    if not argumentos:
        print('uso: python servir.py "pergunta"', file=sys.stderr)
        return 2
    resposta = AssistentePoliticas(carregar_config()).responder(" ".join(argumentos))
    print(json.dumps(resposta.para_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
