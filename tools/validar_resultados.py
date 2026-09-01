"""Valida os artefatos JSON de resultados preservados no repositório."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SHA_GIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

OBRIGATORIOS = {
    "schema_version",
    "experimento",
    "commit",
    "working_tree_dirty",
    "executado_em",
    "comando",
    "python",
    "plataforma",
    "hardware",
    "seed",
    "modelos",
    "dados",
    "amostra_n",
    "metricas",
    "observacoes",
}


@dataclass(frozen=True)
class ViolacaoResultado:
    arquivo: Path
    mensagem: str


def descobrir_resultados(raiz: Path = ROOT) -> list[Path]:
    caminhos = list((raiz / "resultados").glob("**/*.json"))
    caminhos.extend(raiz.glob("modulo-12-projeto/**/resultados.json"))
    caminhos.extend(raiz.glob("modulo-12-projeto/**/carga-resultados.json"))
    return sorted(set(caminhos))


def _validar_lista_referencias(
    arquivo: Path,
    itens: Any,
    campo_revisao: str,
) -> list[ViolacaoResultado]:
    if not isinstance(itens, list):
        return [ViolacaoResultado(arquivo, "referências precisam ser uma lista")]
    violacoes = []
    for indice, item in enumerate(itens):
        if not isinstance(item, dict) or not item.get("id") or not item.get(campo_revisao):
            violacoes.append(
                ViolacaoResultado(
                    arquivo,
                    f"referência {indice} exige id e {campo_revisao}",
                )
            )
    return violacoes


def validar_resultado(arquivo: Path) -> list[ViolacaoResultado]:
    try:
        registro = json.loads(arquivo.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as erro:
        return [ViolacaoResultado(arquivo, f"JSON inválido: {erro}")]
    ausentes = OBRIGATORIOS - set(registro)
    violacoes = [ViolacaoResultado(arquivo, f"campo ausente: {campo}") for campo in sorted(ausentes)]
    if ausentes:
        return violacoes
    if registro["schema_version"] != 1:
        violacoes.append(ViolacaoResultado(arquivo, "schema_version precisa ser 1"))
    if not isinstance(registro["experimento"], str) or "/" not in registro["experimento"]:
        violacoes.append(ViolacaoResultado(arquivo, "experimento precisa conter módulo/nome"))
    if not isinstance(registro["commit"], str) or (
        registro["commit"] != "sem-git" and not SHA_GIT.fullmatch(registro["commit"])
    ):
        violacoes.append(ViolacaoResultado(arquivo, "commit precisa ser SHA Git completo"))
    if not isinstance(registro["working_tree_dirty"], (bool, type(None))):
        violacoes.append(ViolacaoResultado(arquivo, "working_tree_dirty precisa ser boolean/null"))
    try:
        instante = datetime.fromisoformat(registro["executado_em"])
        if instante.tzinfo is None:
            raise ValueError("sem timezone")
    except (TypeError, ValueError):
        violacoes.append(ViolacaoResultado(arquivo, "executado_em precisa ser ISO-8601 com timezone"))
    for campo in ("comando", "python", "plataforma", "hardware"):
        if not isinstance(registro[campo], str) or not registro[campo].strip():
            violacoes.append(ViolacaoResultado(arquivo, f"{campo} não pode ser vazio"))
    if not isinstance(registro["seed"], int):
        violacoes.append(ViolacaoResultado(arquivo, "seed precisa ser inteiro"))
    if not isinstance(registro["amostra_n"], int) or registro["amostra_n"] < 0:
        violacoes.append(ViolacaoResultado(arquivo, "amostra_n precisa ser inteiro não negativo"))
    if not isinstance(registro["metricas"], dict) or not registro["metricas"]:
        violacoes.append(ViolacaoResultado(arquivo, "metricas precisa ser objeto não vazio"))
    if not isinstance(registro["observacoes"], str):
        violacoes.append(ViolacaoResultado(arquivo, "observacoes precisa ser texto"))
    violacoes.extend(_validar_lista_referencias(arquivo, registro["modelos"], "revision"))
    violacoes.extend(
        _validar_lista_referencias(arquivo, registro["dados"], "revision_ou_sha256")
    )
    for dado in registro["dados"] if isinstance(registro["dados"], list) else []:
        revisao = dado.get("revision_ou_sha256", "") if isinstance(dado, dict) else ""
        if len(revisao) == 64 and not SHA256.fullmatch(revisao):
            violacoes.append(ViolacaoResultado(arquivo, "SHA-256 de dados inválido"))
    return violacoes


def validar_repositorio(raiz: Path = ROOT) -> list[ViolacaoResultado]:
    return [item for arquivo in descobrir_resultados(raiz) for item in validar_resultado(arquivo)]


def main() -> int:
    arquivos = descobrir_resultados()
    violacoes = validar_repositorio()
    for item in violacoes:
        print(f"{item.arquivo.relative_to(ROOT)}: {item.mensagem}")
    if violacoes:
        return 1
    sujos = sum(
        bool(json.loads(arquivo.read_text(encoding="utf-8"))["working_tree_dirty"])
        for arquivo in arquivos
    )
    print(
        f"{len(arquivos)} artefato(s) seguem o schema v1; "
        f"{sujos} associado(s) a árvore Git suja."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
