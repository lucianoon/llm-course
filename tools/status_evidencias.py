"""Exibe um resumo dos experimentos com registros preservados."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def registros() -> list[dict]:
    itens = []
    for caminho in sorted((ROOT / "resultados").rglob("*.json")):
        try:
            valor = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        valor["_arquivo"] = str(caminho.relative_to(ROOT))
        itens.append(valor)
    return itens


def main() -> int:
    itens = registros()
    print("# Status das reproduções\n")
    print("| Experimento | Execuções | Último commit | Amostra | Último registro |")
    print("|---|---:|---|---:|---|")
    if not itens:
        print("| — | 0 | — | — | Nenhum registro preservado |")
        return 0

    agrupados: dict[str, list[dict]] = {}
    for item in itens:
        agrupados.setdefault(item.get("experimento", "desconhecido"), []).append(item)
    for experimento, valores in sorted(agrupados.items()):
        ultimo = max(valores, key=lambda item: item.get("executado_em", ""))
        print(
            f"| `{experimento}` | {len(valores)} | `{ultimo.get('commit', '—')[:8]}` | "
            f"{ultimo.get('amostra_n', '—')} | {ultimo.get('executado_em', '—')} |"
        )
    print(f"\nTotal: {len(itens)} registros em {len(agrupados)} experimentos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
