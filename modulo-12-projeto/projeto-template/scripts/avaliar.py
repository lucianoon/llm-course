"""Avalia o modelo contra as baselines e escreve o `resultados.json` do projeto.

ESQUELETO do contrato do módulo 12. Ele executa o conjunto de teste que NÃO foi
usado para otimizar, calcula a métrica do `config.json`, compara com as baselines e
grava `resultados.json` com os campos que o revisor técnico exige (baseline,
modelo_final, n, metrica, intervalo de confiança quando aplicável, commit, revision,
seed, hardware).

Reveja o módulo 14 antes: n pequeno, teste pareado, melhor-de-k infla resultado.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from preparar import carregar_config


def commit_atual(root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def avaliar(root: Path, config: dict) -> Path:
    metricas = config["metrica"]
    # TODO: execute o modelo e as baselines no conjunto de teste e calcule a métrica.
    resultado = {
        "baseline": {metricas["nome"]: "substitua_pelo_valor_baseline"},
        "modelo_final": {metricas["nome"]: "substitua_pelo_valor_medido"},
        "n": 0,
        "metrica": metricas["nome"],
        "intervalo_confianca": "[preencha quando aplicar — módulo 14]",
        "commit": commit_atual(root),
        "revision": config["modelo"]["revision"],
        "seed": 0,
        "hardware": "substitua (fora do CI; é o que torna o número reproduzível)",
        "erros_manuais": "note aqui a leitura manual de 30+ saídas",
    }
    destino = root / "resultados.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"resultados escritos em {destino.name} (preencha as métricas de verdade).")
    return destino


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)

    root = Path(args.root)
    config = carregar_config(root / "config.json")
    avaliar(root, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
