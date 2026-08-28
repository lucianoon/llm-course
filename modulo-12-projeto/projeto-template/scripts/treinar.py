"""Treina o modelo conforme a técnica e os hiperparâmetros do `config.json`.

ESQUELETO do contrato do módulo 12. Ele monta o plano de treino a partir de
`config.json`, calcula o número de épocas a partir de passos × batch ÷ exemplos
(módulo 5) e indica onde salvar as curvas e o adapter. O treino em si (LoRA/QLoRA,
SFT, DPO, GRPO) usa as receitas dos módulos 5–10.

Rode com `--dry-run` para imprimir o plano sem treinar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from preparar import carregar_config


def plano_de_treino(config: dict, exemplos: int) -> dict:
    """Calcula o guarda-corpo do treino: épocas a partir de passos × batch."""
    hp = config["modelo"]["hiperparametros"]
    passos = hp.get("passos", 300)
    batch = hp.get("batch", 8)
    # módulo 5: épocas NÃO são um palpite, são iters × batch / exemplos.
    epocas = (passos * batch) / max(1, exemplos)
    return {
        "tecnica": config["modelo"]["tecnica"],
        "passos": passos,
        "batch": batch,
        "epocas_estimadas": round(epocas, 2),
        "modelo_base": config["modelo"]["base"],
        "revision": config["modelo"]["revision"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--exemplos", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = carregar_config(Path(args.root) / "config.json")
    plano = plano_de_treino(config, args.exemplos)
    print("plano de treino:")
    for chave, valor in plano.items():
        print(f"  {chave}: {valor}")
    if args.dry_run:
        print("DRY-RUN: nenhum peso será alterado.")
    else:
        print("→ substitua o corpo deste script pela receita de treino (módulos 5–10).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
