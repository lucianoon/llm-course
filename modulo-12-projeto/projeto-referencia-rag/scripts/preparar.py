"""Valida os dados fictícios e grava um manifesto reproduzível."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sistema import PROJETO, carregar_jsonl


def sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def main() -> int:
    corpus_path = PROJETO / "dados" / "corpus.jsonl"
    avaliacao_path = PROJETO / "dados" / "avaliacao.jsonl"
    corpus = carregar_jsonl(corpus_path)
    avaliacao = carregar_jsonl(avaliacao_path)

    ids = [item["id"] for item in corpus]
    if len(ids) != len(set(ids)):
        raise ValueError("IDs duplicados no corpus")
    if any(item["licenca"] != "CC0-1.0" for item in corpus):
        raise ValueError("todo documento fictício deve declarar CC0-1.0")
    if set(ids) & {item["id"] for item in avaliacao}:
        raise ValueError("IDs do corpus e da avaliação não podem se sobrepor")

    manifesto = {
        "nome": "politicas-internas-ficticias-v1",
        "origem": "conteúdo sintético criado para este capstone",
        "licenca": "CC0-1.0",
        "pii": "nenhuma PII real; o e-mail usa example.org, reservado para exemplos",
        "arquivos": [
            {"caminho": "dados/corpus.jsonl", "sha256": sha256(corpus_path), "linhas": len(corpus)},
            {
                "caminho": "dados/avaliacao.jsonl",
                "sha256": sha256(avaliacao_path),
                "linhas": len(avaliacao),
            },
        ],
        "split": "corpus recuperável e perguntas de avaliação ficam em arquivos separados",
    }
    destino = PROJETO / "dataset-manifest.json"
    destino.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifesto gravado: {destino.relative_to(PROJETO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
