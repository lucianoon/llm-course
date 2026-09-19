"""Valida registros JSON de reprodução preservados em ``resultados/``."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = {
    "experimento",
    "commit",
    "executado_em",
    "comando",
    "python",
    "plataforma",
    "seed",
    "modelos",
    "dados",
    "amostra_n",
    "metricas",
    "observacoes",
}
SHA = re.compile(r"^[0-9a-f]{40}$")


def validate_file(path: Path) -> list[str]:
    errors = []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"{path.relative_to(ROOT)}: JSON inválido: {error}"]
    if not isinstance(value, dict):
        return [f"{path.relative_to(ROOT)}: raiz deve ser um objeto JSON"]
    missing = REQUIRED - value.keys()
    if missing:
        errors.append(f"{path.relative_to(ROOT)}: campos ausentes: {', '.join(sorted(missing))}")
    if not isinstance(value.get("modelos", []), list) or not isinstance(value.get("dados", []), list):
        errors.append(f"{path.relative_to(ROOT)}: modelos e dados devem ser listas")
    if not isinstance(value.get("metricas", {}), dict):
        errors.append(f"{path.relative_to(ROOT)}: metricas deve ser objeto")
    commit = value.get("commit")
    if commit != "sem-git" and not (isinstance(commit, str) and SHA.fullmatch(commit)):
        errors.append(f"{path.relative_to(ROOT)}: commit deve ser SHA completo")
    if not isinstance(value.get("amostra_n"), int) or value["amostra_n"] < 0:
        errors.append(f"{path.relative_to(ROOT)}: amostra_n deve ser inteiro não negativo")
    return errors


def main() -> int:
    arquivos = sorted((ROOT / "resultados").rglob("*.json"))
    errors = [error for arquivo in arquivos for error in validate_file(arquivo)]
    if errors:
        print("FALHA nos resultados:")
        print("\n".join(f"  - {error}" for error in errors))
        return 1
    print(f"OK: {len(arquivos)} registros de reprodução válidos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
