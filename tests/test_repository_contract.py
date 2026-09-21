"""Contratos estruturais do curso.

Estes testes não executam modelos nem baixam datasets. Eles detectam quebras
de organização que fariam um clone novo perder módulos ou laboratórios.
"""

from __future__ import annotations

# Ruff 0.16.5 oscila a ordenação entre estes módulos com nomes semelhantes.
# O bloco permanece revisado manualmente e a exceção é restrita a este teste.
# ruff: noqa: I001

import ast
import re
from pathlib import Path

from tools.check_docs import check_markdown, check_module_conventions
from tools.modelos import RESOLVIDA_EM_RUNTIME, resolver_revision
from tools.status_evidencias import registros
from tools.validar_resultados import REQUIRED
from tools.validate_labs import DRY_RUN_SCRIPTS, laboratory_scripts


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
    scripts = laboratory_scripts()

    # A new lab is welcome; removing an existing entry should not go unnoticed.
    assert len(scripts) >= 35
    for script in scripts:
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


def test_dry_run_manifest_points_to_existing_scripts() -> None:
    assert len(DRY_RUN_SCRIPTS) == 6
    for relative in DRY_RUN_SCRIPTS:
        script = ROOT / relative
        assert script.is_file(), relative
        assert "--dry-run" in script.read_text(encoding="utf-8")


def test_dependency_security_floors_are_preserved() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires = ["setuptools>=83"]' in pyproject
    assert '"accelerate>=1.15,<2"' in pyproject
    assert 'override-dependencies = ["setuptools>=83"]' in pyproject


def test_curriculum_counts_and_cross_references_are_consistent() -> None:
    plano = (ROOT / "PLANO-MESTRE.md").read_text(encoding="utf-8")
    trilha = (ROOT / "TRILHA-ESSENCIAL.md").read_text(encoding="utf-8")
    fase3 = (ROOT / "FASE-3-MAESTRIA.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert len(list(ROOT.glob("modulo-*/lab_mlx.py"))) == 8
    assert "8 labs MLX" in plano
    assert "19 módulos" in fase3
    assert "19–21" not in trilha
    assert "FASE-3-MAESTRIA.md · etapa 3" in trilha
    assert "O módulo 12 é a exceção" in readme
    assert "substitui o laboratório e os exercícios por um projeto" in readme


def test_sft_hardware_claim_is_explicit() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    mapa = (ROOT / "MAPA-MODULOS.md").read_text(encoding="utf-8")
    certificacao = (ROOT / "TRILHA-CERTIFICACAO-12-SEMANAS.md").read_text(encoding="utf-8")
    assert "| 5 | [Supervised Fine-Tuning]" in readme
    assert "| Mac/GPU |" in mapa
    assert "exige Mac Apple Silicon ou GPU NVIDIA" in certificacao


def test_documentation_has_no_broken_local_references() -> None:
    assert check_markdown() == []


def test_modules_keep_the_minimum_learning_contract() -> None:
    assert check_module_conventions() == []


def test_reproduction_schema_has_required_fields() -> None:
    assert REQUIRED == {
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


def test_model_revision_helper_preserves_explicit_revision() -> None:
    assert RESOLVIDA_EM_RUNTIME == "RESOLVIDA_EM_RUNTIME"
    assert resolver_revision("qualquer/modelo", "abc123") == "abc123"


def test_evidence_status_reads_empty_repository_without_failure() -> None:
    assert registros() == []
