"""Converte os labs em formato percent (lab.py) para notebooks Jupyter (lab.ipynb).

Formato aceito no lab.py:

    # %% [markdown]
    # Texto em markdown, uma linha por linha, cada uma prefixada por "# ".

    # %%
    codigo_python = "aqui"

Uso:
    python tools/build_notebooks.py            # converte todos os modulo-*/lab.py
    python tools/build_notebooks.py modulo-01-fundamentos/lab.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_percent(text: str) -> list[dict]:
    """Quebra o arquivo percent em uma lista de celulas do nbformat."""
    cells: list[dict] = []
    kind = "code"
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        # Remove linhas em branco no inicio e no fim da celula.
        while buffer and not buffer[0].strip():
            buffer.pop(0)
        while buffer and not buffer[-1].strip():
            buffer.pop()
        if not buffer:
            return
        if kind == "markdown":
            # Em celulas markdown cada linha vem comentada com "# ".
            body = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in buffer]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": as_source(body)})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": as_source(buffer),
                }
            )
        buffer.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            continue
        buffer.append(line)
    flush()
    return cells


def as_source(lines: list[str]) -> list[str]:
    """nbformat guarda cada linha com \\n, exceto a ultima."""
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def notebook_bootstrap(caminho_relativo: Path) -> list[str]:
    """Cria um prólogo portátil que dá ao notebook o ``__file__`` do lab-fonte."""
    relativo = caminho_relativo.as_posix()
    return as_source([
        "# Gerado por tools/build_notebooks.py: torna caminhos independentes do cwd do kernel.",
        "from pathlib import Path as _NotebookPath",
        "_notebook_origem = _NotebookPath.cwd().resolve()",
        "_notebook_raiz = next(",
        "    (p for p in (_notebook_origem, *_notebook_origem.parents)",
        "     if (p / 'pyproject.toml').is_file()),",
        "    None,",
        ")",
        "if _notebook_raiz is None:",
        "    raise RuntimeError('abra o notebook dentro do repositório llm-course')",
        f"__file__ = str(_notebook_raiz / {relativo!r})",
        "del _NotebookPath, _notebook_origem, _notebook_raiz",
        "",
    ])


def build(py_path: Path) -> Path:
    cells = parse_percent(py_path.read_text(encoding="utf-8"))
    primeira_codigo = next(cell for cell in cells if cell["cell_type"] == "code")
    primeira_codigo["source"] = notebook_bootstrap(py_path.relative_to(ROOT)) + primeira_codigo["source"]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = py_path.with_suffix(".ipynb")
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    # A partir do módulo 6 há dois labs por módulo: lab_cpu.py (validado, roda em
    # qualquer máquina) e lab_mlx.py (Apple Silicon).
    targets = (
        [Path(a) if Path(a).is_absolute() else ROOT / a for a in sys.argv[1:]]
        or sorted(ROOT.glob("modulo-*/lab*.py"))
    )
    if not targets:
        print("nenhum lab.py encontrado")
        return
    for py_path in targets:
        out = build(py_path)
        n = len(json.loads(out.read_text(encoding="utf-8"))["cells"])
        print(f"{py_path.relative_to(ROOT)} -> {out.name} ({n} celulas)")


if __name__ == "__main__":
    main()
