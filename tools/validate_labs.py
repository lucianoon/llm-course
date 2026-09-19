"""Valida contratos baratos dos laboratórios do curso.

O comando não baixa modelos nem datasets. Ele verifica a sintaxe de todos os
labs e executa o modo ``--dry-run`` dos scripts acelerados que o oferecem.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DRY_RUN_SCRIPTS = (
    "modulo-05-sft/lab_cuda.py",
    "modulo-06-lora/lab_adapters.py",
    "modulo-09-rl/lab_cuda.py",
    "modulo-10-distillation/lab_cuda.py",
    "modulo-11-inferencia/lab_moe_cuda.py",
    "modulo-11-inferencia/benchmark_vllm.py",
)


def laboratory_scripts() -> list[Path]:
    """Retorna todos os scripts de laboratório na ordem do currículo."""

    scripts = [ROOT / "00-iniciante-zero" / "lab.py"]
    scripts.extend(sorted(ROOT.glob("modulo-*/lab*.py")))
    return scripts


def validate_syntax(scripts: list[Path]) -> list[str]:
    """Retorna erros de sintaxe sem importar dependências dos laboratórios."""

    errors = []
    for script in scripts:
        try:
            ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        except (OSError, SyntaxError) as error:
            errors.append(f"{script.relative_to(ROOT)}: {error}")
    return errors


def validate_dry_runs() -> list[str]:
    """Executa os contratos ``--dry-run`` e retorna erros legíveis."""

    errors = []
    for relative in DRY_RUN_SCRIPTS:
        script = ROOT / relative
        try:
            result = subprocess.run(
                [sys.executable, str(script), "--dry-run"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"{relative}: {error}")
            continue

        if result.returncode != 0:
            detalhe = result.stderr.strip() or result.stdout.strip()
            errors.append(f"{relative}: saiu com {result.returncode}: {detalhe}")
            continue

        try:
            json.loads(result.stdout)
        except json.JSONDecodeError as error:
            errors.append(f"{relative}: saída não é JSON válido: {error}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--syntax-only",
        action="store_true",
        help="verifica somente a sintaxe e não executa os modos --dry-run",
    )
    args = parser.parse_args()

    scripts = laboratory_scripts()
    errors = validate_syntax(scripts)
    if not args.syntax_only:
        errors.extend(validate_dry_runs())

    if errors:
        print("FALHA na validação dos laboratórios:")
        for error in errors:
            print(f"  - {error}")
        return 1

    dry_runs = 0 if args.syntax_only else len(DRY_RUN_SCRIPTS)
    print(f"OK: {len(scripts)} labs analisados · {dry_runs} dry-runs executados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
