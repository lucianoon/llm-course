"""Verifica links locais e caminhos de scripts citados na documentação."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
COMMAND_PATH = re.compile(
    r"(?:uv run )?(?:python|accelerate launch)\s+([\w./-]+\.py)"
)


def local_target(source: Path, target: str) -> Path | None:
    target = target.strip().strip("<>")
    parsed = urlsplit(target)
    if parsed.scheme or target.startswith("//") or not parsed.path:
        return None
    return (source.parent / unquote(parsed.path)).resolve()


def check_markdown() -> list[str]:
    errors = []
    for source in sorted(ROOT.rglob("*.md")):
        if ".git" in source.parts or ".venv" in source.parts:
            continue
        text = source.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(text):
            path = local_target(source, target)
            if path is not None and not path.exists():
                errors.append(f"{source.relative_to(ROOT)}: link inexistente: {target}")
        for raw_path in COMMAND_PATH.findall(text):
            if "XX" in raw_path or "<" in raw_path or "..." in raw_path:
                continue
            path = (source.parent / raw_path if "/" not in raw_path else ROOT / raw_path).resolve()
            if not path.exists():
                errors.append(f"{source.relative_to(ROOT)}: script inexistente: {raw_path}")
    return errors


def check_module_conventions() -> list[str]:
    """Detecta módulos publicados sem os arquivos mínimos da experiência do curso."""
    errors = []
    for directory in sorted(ROOT.glob("modulo-*")):
        if not directory.is_dir() or directory.name == "modulo-12-projeto":
            continue
        required = ("README.md", "exercicios.md")
        missing = [name for name in required if not (directory / name).exists()]
        labs = sorted(directory.glob("lab*.py"))
        has_lab = bool(labs)
        if missing:
            errors.append(f"{directory.name}: arquivos obrigatórios ausentes: {', '.join(missing)}")
        if not has_lab:
            errors.append(f"{directory.name}: nenhum lab*.py encontrado")
        if directory.name != "modulo-05-sft" and not any(
            path.name in {"lab.py", "lab_cpu.py"} for path in labs
        ):
            errors.append(f"{directory.name}: nenhum lab.py ou lab_cpu.py conceitual encontrado")
    return errors


def main() -> int:
    errors = check_markdown() + check_module_conventions()
    if errors:
        print("FALHA na documentação:")
        print("\n".join(f"  - {error}" for error in errors))
        return 1
    print("OK: links locais e caminhos de scripts verificados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
