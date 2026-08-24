"""Registro local e reproduzível de experimentos, sem serviço externo obrigatório."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _commit_atual(raiz: Path) -> str | None:
    processo = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=raiz,
        capture_output=True,
        text=True,
        check=False,
    )
    return processo.stdout.strip() or None if processo.returncode == 0 else None


@dataclass(frozen=True)
class MetadadosExecucao:
    experimento: str
    iniciado_em: str
    commit: str | None
    python: str
    plataforma: str
    configuracao: dict[str, Any]


class RegistroExperimento:
    """Persiste configuração e métricas em JSON/JSONL legíveis e versionáveis."""

    def __init__(
        self,
        experimento: str,
        configuracao: dict[str, Any],
        destino: Path,
        raiz_git: Path,
    ):
        if not experimento or any(parte in experimento for parte in ("..", "/", "\\")):
            raise ValueError("experimento deve ser um nome simples e não vazio")
        instante = datetime.now(UTC)
        self.pasta = destino / experimento / instante.strftime("%Y%m%dT%H%M%S%fZ")
        self.pasta.mkdir(parents=True, exist_ok=False)
        self.metricas = self.pasta / "metricas.jsonl"
        metadados = MetadadosExecucao(
            experimento=experimento,
            iniciado_em=instante.isoformat(),
            commit=_commit_atual(raiz_git),
            python=sys.version.split()[0],
            plataforma=f"{platform.system()}-{platform.machine()}",
            configuracao=configuracao,
        )
        (self.pasta / "metadados.json").write_text(
            json.dumps(asdict(metadados), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def registrar(self, passo: int, **metricas: float | str | bool | None) -> None:
        if passo < 0:
            raise ValueError("passo não pode ser negativo")
        registro = {
            "passo": passo,
            "registrado_em": datetime.now(UTC).isoformat(),
            **metricas,
        }
        with self.metricas.open("a", encoding="utf-8") as arquivo:
            arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")

    def concluir(self, **resumo: Any) -> Path:
        caminho = self.pasta / "resultado.json"
        caminho.write_text(
            json.dumps(resumo, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return caminho
