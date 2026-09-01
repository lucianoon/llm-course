"""Valida que todo laboratório tem requisitos de execução declarados."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTO = ROOT / "EXECUCAO_LABS.json"


def descobrir_labs(raiz: Path = ROOT) -> set[str]:
    padroes = ("lab.py", "lab_cpu.py", "lab_*.py")
    encontrados: set[str] = set()
    for padrao in padroes:
        for caminho in raiz.glob(f"*/{padrao}"):
            encontrados.add(caminho.relative_to(raiz).as_posix())
    return encontrados


def carregar_manifesto(caminho: Path = MANIFESTO) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def validar_manifesto(raiz: Path = ROOT, caminho: Path = MANIFESTO) -> list[str]:
    dados = carregar_manifesto(caminho)
    erros: list[str] = []
    if dados.get("schema_version") != 1:
        erros.append("schema_version deve ser 1")

    perfis = dados.get("perfis", {})
    entradas = dados.get("labs", [])
    caminhos = [entrada.get("caminho") for entrada in entradas]
    repetidos = sorted({item for item in caminhos if caminhos.count(item) > 1})
    if repetidos:
        erros.append("caminhos duplicados: " + ", ".join(repetidos))

    declarados = {item for item in caminhos if isinstance(item, str)}
    encontrados = descobrir_labs(raiz)
    if faltantes := sorted(encontrados - declarados):
        erros.append("labs sem perfil: " + ", ".join(faltantes))
    if obsoletos := sorted(declarados - encontrados):
        erros.append("entradas sem arquivo: " + ", ".join(obsoletos))

    for entrada in entradas:
        if entrada.get("perfil") not in perfis:
            erros.append(
                f"perfil inválido em {entrada.get('caminho')}: {entrada.get('perfil')}"
            )
    return erros


def caminhos_do_perfil(perfil: str, caminho: Path = MANIFESTO) -> tuple[str, ...]:
    dados = carregar_manifesto(caminho)
    if perfil not in dados["perfis"]:
        raise ValueError(f"perfil desconhecido: {perfil}")
    return tuple(
        entrada["caminho"] for entrada in dados["labs"] if entrada["perfil"] == perfil
    )


def main() -> int:
    erros = validar_manifesto()
    if erros:
        print("Manifesto de execução inválido:")
        for erro in erros:
            print(f"- {erro}")
        return 1
    dados = carregar_manifesto()
    print(f"{len(dados['labs'])} labs têm requisitos de execução declarados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
