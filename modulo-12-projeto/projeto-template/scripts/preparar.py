"""Prepara os dados do projeto: coleta, dedup, limpeza, splits e manifesto.

Este script é o ESQUELETO do contrato do módulo 12. Ele valida o `config.json`,
lê os dados crus de `dados.origem`, aplica os filtros que o seu problema exigir e
grava os splits com um manifesto de proveniência (módulo 4, `tools/governanca.py`).

Rode com `--dry-run` para ver o plano sem tocar nos dados.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def carregar_config(caminho: Path) -> dict:
    with caminho.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def preparar(config: dict, destino: Path) -> Path:
    """Aplica a pipeline de dados e devolve o caminho do manifesto gerado."""
    exemplo = next(iter(config.get("baselines", [])), "precisa definir baselines")
    # TODO: aqui entra a coleta, dedup, filtros e o split do seu domínio.
    destino.mkdir(parents=True, exist_ok=True)
    manifesto = destino / "dataset-manifest.json"
    manifesto.write_text(
        json.dumps({"origem": config["dados"]["origem"], "split": config["dados"]["splits"]},
                   ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"prévia da pipeline: {len(config['dados']['splits'])} splits | "
          f"baseline de exemplo: {exemplo}")
    print(f"manifesto escrito em {manifesto.name} (substitua pela pipeline real)")
    return manifesto


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = carregar_config(Path(args.root) / "config.json")
    if args.dry_run:
        print("DRY-RUN: nenhum dado será tocado.")
    preparar(config, Path(args.root) / "data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
