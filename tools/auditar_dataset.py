"""CLI para produzir um manifesto de proveniência e PII antes do treino."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.governanca import criar_manifesto_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arquivos", nargs="+", type=Path)
    parser.add_argument("--nome", required=True)
    parser.add_argument("--origem", required=True)
    parser.add_argument("--licenca", required=True)
    parser.add_argument("--finalidade", required=True)
    parser.add_argument("--saida", type=Path, default=Path("dataset-manifest.json"))
    args = parser.parse_args()
    inexistentes = [str(caminho) for caminho in args.arquivos if not caminho.is_file()]
    if inexistentes:
        parser.error(f"arquivos inexistentes: {inexistentes}")
    destino = criar_manifesto_dataset(
        args.arquivos,
        args.saida,
        nome=args.nome,
        origem=args.origem,
        licenca=args.licenca,
        finalidade=args.finalidade,
    )
    manifesto = json.loads(destino.read_text(encoding="utf-8"))
    total_pii = sum(
        achado["ocorrencias"]
        for arquivo in manifesto["arquivos"]
        for achado in arquivo["pii"]
    )
    print(f"manifesto: {destino} | arquivos={len(args.arquivos)} | ocorrências PII={total_pii}")
    if total_pii:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
