"""Valida o manifesto e impede downloads diretos fora do helper auditável."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dados_externos import MANIFESTO, carregar_manifesto_dados

SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT_GITHUB = re.compile(r"raw\.githubusercontent\.com/[^/]+/[^/]+/[0-9a-f]{40}/")


@dataclass(frozen=True)
class ViolacaoDados:
    arquivo: Path
    linha: int
    mensagem: str


def validar_manifesto() -> list[ViolacaoDados]:
    violacoes = []
    manifesto = carregar_manifesto_dados()
    for artefato_id, metadados in manifesto.items():
        if not SHA256.fullmatch(metadados.get("sha256", "")):
            violacoes.append(ViolacaoDados(MANIFESTO, 1, f"SHA-256 inválido: {artefato_id}"))
        if not isinstance(metadados.get("bytes"), int) or metadados["bytes"] <= 0:
            violacoes.append(ViolacaoDados(MANIFESTO, 1, f"tamanho inválido: {artefato_id}"))
        if not metadados.get("licenca"):
            violacoes.append(ViolacaoDados(MANIFESTO, 1, f"licença ausente: {artefato_id}"))
        url = metadados.get("url")
        if url and not url.startswith("https://"):
            violacoes.append(ViolacaoDados(MANIFESTO, 1, f"URL não HTTPS: {artefato_id}"))
        if url and "raw.githubusercontent.com" in url and not COMMIT_GITHUB.search(url):
            violacoes.append(
                ViolacaoDados(MANIFESTO, 1, f"URL GitHub sem commit imutável: {artefato_id}")
            )
        if metadados.get("imutavel") is False and not url:
            violacoes.append(
                ViolacaoDados(MANIFESTO, 1, f"fonte mutável sem URL auditável: {artefato_id}")
            )
    return violacoes


def validar_downloads_diretos(raiz: Path = ROOT) -> list[ViolacaoDados]:
    violacoes = []
    helper = ROOT / "tools" / "dados_externos.py"
    for caminho in raiz.rglob("*.py"):
        relativo = caminho.relative_to(raiz)
        if caminho == helper or any(parte.startswith(".") for parte in relativo.parts):
            continue
        arvore = ast.parse(caminho.read_text(encoding="utf-8"), filename=str(caminho))
        for no in ast.walk(arvore):
            if not isinstance(no, ast.Call) or not isinstance(no.func, ast.Attribute):
                continue
            if no.func.attr == "urlopen":
                violacoes.append(
                    ViolacaoDados(caminho, no.lineno, "urlopen direto; use tools.dados_externos")
                )
    return violacoes


def validar_repositorio() -> list[ViolacaoDados]:
    return [*validar_manifesto(), *validar_downloads_diretos()]


def main() -> int:
    violacoes = validar_repositorio()
    for item in violacoes:
        print(f"{item.arquivo.relative_to(ROOT)}:{item.linha}: {item.mensagem}")
    if violacoes:
        return 1
    print("Datasets externos têm origem, licença, tamanho e checksum declarados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
