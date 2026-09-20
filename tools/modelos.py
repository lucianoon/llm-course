"""Helpers para fixar revisões de modelos do Hugging Face."""

from __future__ import annotations

RESOLVIDA_EM_RUNTIME = "RESOLVIDA_EM_RUNTIME"


def resolver_revision(modelo: str, revisao: str = RESOLVIDA_EM_RUNTIME) -> str:
    """Retorna uma revisão fornecida pelo usuário ou resolve a revisão atual uma vez."""

    if revisao != RESOLVIDA_EM_RUNTIME:
        return revisao
    from huggingface_hub import model_info

    resolvida = model_info(modelo).sha
    if not resolvida:
        raise RuntimeError(f"não foi possível resolver a revisão imutável de {modelo}")
    return resolvida
