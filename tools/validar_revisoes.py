"""Valida que downloads de modelos Transformers usam revisões imutáveis."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTO = ROOT / "MODELOS.json"
SHA_GIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ViolacaoRevisao:
    arquivo: Path
    linha: int
    mensagem: str


def carregar_manifesto() -> dict[str, dict[str, str]]:
    return json.loads(MANIFESTO.read_text(encoding="utf-8"))


def _constantes_strings(arvore: ast.AST) -> dict[str, str]:
    constantes: dict[str, str] = {}
    for no in ast.walk(arvore):
        if (
            isinstance(no, (ast.Assign, ast.AnnAssign))
            and isinstance(no.value, ast.Constant)
            and isinstance(no.value.value, str)
        ):
            alvos = no.targets if isinstance(no, ast.Assign) else [no.target]
            for alvo in alvos:
                if isinstance(alvo, ast.Name):
                    constantes[alvo.id] = no.value.value
    return constantes


def _valor_string(no: ast.expr | None, constantes: dict[str, str]) -> str | None:
    if isinstance(no, ast.Constant) and isinstance(no.value, str):
        return no.value
    if isinstance(no, ast.Name):
        return constantes.get(no.id)
    return None


def _dono_chamada(funcao: ast.Attribute) -> str:
    valor = funcao.value
    return valor.id if isinstance(valor, ast.Name) else ""


def validar_arquivo(
    caminho: Path,
    manifesto: dict[str, dict[str, str]],
) -> list[ViolacaoRevisao]:
    arvore = ast.parse(caminho.read_text(encoding="utf-8"), filename=str(caminho))
    constantes = _constantes_strings(arvore)
    violacoes = []
    for no in ast.walk(arvore):
        if not (
            isinstance(no, ast.Call)
            and isinstance(no.func, ast.Attribute)
            and no.func.attr == "from_pretrained"
        ):
            continue
        if _dono_chamada(no.func) == "PeftModel":
            # Adapter pode ser caminho local ou artefato remoto do próprio aluno.
            continue

        revisao = next((kw.value for kw in no.keywords if kw.arg == "revision"), None)
        if revisao is None:
            violacoes.append(
                ViolacaoRevisao(caminho, no.lineno, "from_pretrained sem revision")
            )
            continue

        modelo = _valor_string(no.args[0], constantes) if no.args else None
        revisao_resolvida = _valor_string(revisao, constantes)
        if modelo is not None and "/" in modelo and modelo not in manifesto:
            violacoes.append(
                ViolacaoRevisao(caminho, no.lineno, f"modelo {modelo} ausente de MODELOS.json")
            )
            continue
        if modelo in manifesto and revisao_resolvida is not None:
            esperada = manifesto[modelo]["revision"]
            if revisao_resolvida != esperada:
                violacoes.append(
                    ViolacaoRevisao(
                        caminho,
                        no.lineno,
                        f"revision de {modelo} difere de MODELOS.json",
                    )
                )
    return violacoes


def validar_repositorio(raiz: Path = ROOT) -> list[ViolacaoRevisao]:
    manifesto = carregar_manifesto()
    violacoes = []
    for modelo, metadados in manifesto.items():
        revisao = metadados.get("revision", "")
        if not SHA_GIT.fullmatch(revisao):
            violacoes.append(ViolacaoRevisao(MANIFESTO, 1, f"SHA inválido para {modelo}"))
        if not metadados.get("licenca"):
            violacoes.append(ViolacaoRevisao(MANIFESTO, 1, f"licença ausente para {modelo}"))
        if revisao not in metadados.get("fonte", ""):
            violacoes.append(
                ViolacaoRevisao(MANIFESTO, 1, f"fonte não aponta para a revisão de {modelo}")
            )
    for caminho in raiz.rglob("*.py"):
        relativo = caminho.relative_to(raiz)
        if any(parte.startswith(".") for parte in relativo.parts):
            continue
        violacoes.extend(validar_arquivo(caminho, manifesto))
    return violacoes


def main() -> int:
    violacoes = validar_repositorio()
    for item in violacoes:
        print(f"{item.arquivo.relative_to(ROOT)}:{item.linha}: {item.mensagem}")
    if violacoes:
        print(f"{len(violacoes)} violação(ões) de revisão")
        return 1
    print("Todas as chamadas from_pretrained usam revision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
