"""Contratos estruturais do curso.

Estes testes não executam modelos nem baixam datasets. Eles detectam quebras
de organização que fariam um clone novo perder módulos ou laboratórios.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_RE = re.compile(r"modulo-(\d{2})-")


def _modules() -> list[tuple[int, Path]]:
    modules = []
    for path in ROOT.glob("modulo-*"):
        match = MODULE_RE.match(path.name)
        if match and path.is_dir():
            modules.append((int(match.group(1)), path))
    return sorted(modules)


def test_all_expected_modules_have_teaching_material() -> None:
    modules = _modules()
    assert [number for number, _ in modules] == list(range(1, 20))

    for number, module in modules:
        assert (module / "README.md").is_file(), module
        if number != 12:
            assert (module / "exercicios.md").is_file(), module


def test_every_learning_module_has_a_python_entrypoint() -> None:
    for number, module in _modules():
        if number == 12:
            assert (module / "projeto-template").is_dir(), module
            assert (module / "VALIDACAO.md").is_file(), module
            continue
        scripts = list(module.glob("lab*.py"))
        assert scripts, module


def test_all_laboratory_scripts_parse_without_importing_dependencies() -> None:
    scripts = [ROOT / "00-iniciante-zero" / "lab.py"]
    scripts.extend(path for _, module in _modules() for path in module.glob("lab*.py"))

    # A new lab is welcome; removing an existing entry should not go unnoticed.
    assert len(scripts) >= 35
    for script in scripts:
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
