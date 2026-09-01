"""Registra a decisão de não treinar um modelo para este problema."""

from __future__ import annotations

import json

from sistema import PROJETO, carregar_config


def main() -> int:
    config = carregar_config()
    decisao = {
        "tecnica": "RAG extrativo com BM25",
        "treino": False,
        "justificativa": (
            "A falha é conhecimento externo mutável. Treinar esconderia a proveniência, "
            "aumentaria custo e exigiria novo treino a cada atualização."
        ),
        "alternativas_descartadas": ["SFT", "LoRA", "DPO"],
        "hiperparametros": config["retrieval"],
    }
    destino = PROJETO / "metodo-manifest.json"
    destino.write_text(json.dumps(decisao, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"decisão gravada: {destino.relative_to(PROJETO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
