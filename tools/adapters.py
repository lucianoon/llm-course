"""Manifestos e validação de compatibilidade de adapters PEFT/LoRA."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestoAdapter:
    nome: str
    modelo_base: str
    revisao_base: str
    metodo: str
    tarefa: str
    dataset_sha256: str
    metricas: dict[str, float]

    def validar(self) -> None:
        obrigatorios = (
            self.nome,
            self.modelo_base,
            self.revisao_base,
            self.metodo,
            self.tarefa,
            self.dataset_sha256,
        )
        if any(not valor.strip() for valor in obrigatorios):
            raise ValueError("manifesto de adapter possui campo obrigatório vazio")


def salvar_manifesto(manifesto: ManifestoAdapter, destino: Path) -> Path:
    manifesto.validar()
    destino.write_text(
        json.dumps(asdict(manifesto), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destino


def carregar_manifesto(caminho: Path) -> ManifestoAdapter:
    manifesto = ManifestoAdapter(**json.loads(caminho.read_text(encoding="utf-8")))
    manifesto.validar()
    return manifesto


def verificar_compatibilidade(manifesto: ManifestoAdapter, modelo: str, revisao: str) -> None:
    if (manifesto.modelo_base, manifesto.revisao_base) != (modelo, revisao):
        raise ValueError(
            "adapter incompatível: esperado "
            f"{manifesto.modelo_base}@{manifesto.revisao_base}, recebido {modelo}@{revisao}"
        )
