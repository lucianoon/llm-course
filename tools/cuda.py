"""Pré-condições e identificação do ambiente CUDA dos laboratórios."""

from __future__ import annotations

from typing import Any


def exigir_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA indisponível. Use uma GPU NVIDIA e instale `uv sync --extra gpu`."
        )


def descrever_cuda(torch: Any) -> dict[str, str | int]:
    exigir_cuda(torch)
    indice = torch.cuda.current_device()
    propriedades = torch.cuda.get_device_properties(indice)
    return {
        "dispositivo": torch.cuda.get_device_name(indice),
        "capability": f"{propriedades.major}.{propriedades.minor}",
        "vram_bytes": propriedades.total_memory,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda or "desconhecida",
    }
