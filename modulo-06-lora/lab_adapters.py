"""Carregamento, hot-swap e compatibilidade de múltiplos adapters PEFT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "tools"))


def argumentos():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revisao", required=False, default="RESOLVIDA_EM_RUNTIME")
    parser.add_argument("--adapter-a", type=Path, default=AQUI / "adapters-suporte")
    parser.add_argument("--adapter-b", type=Path, default=AQUI / "adapters-segundo-dominio")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = argumentos()
    config = {
        "modelo": args.modelo,
        "revisao": args.revisao,
        "adapters": {"suporte": str(args.adapter_a), "segundo_dominio": str(args.adapter_b)},
    }
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    from adapters import carregar_manifesto, verificar_compatibilidade
    from huggingface_hub import model_info
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revisao = model_info(args.modelo).sha if args.revisao == "RESOLVIDA_EM_RUNTIME" else args.revisao
    base = AutoModelForCausalLM.from_pretrained(args.modelo, revision=revisao, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.modelo, revision=revisao)

    for pasta in (args.adapter_a, args.adapter_b):
        verificar_compatibilidade(
            carregar_manifesto(pasta / "adapter-manifest.json"), args.modelo, revisao
        )
    modelo = PeftModel.from_pretrained(base, args.adapter_a, adapter_name="suporte")
    modelo.load_adapter(args.adapter_b, adapter_name="segundo_dominio")

    prompt = tokenizer("Explique como redefinir uma senha.", return_tensors="pt").to(modelo.device)
    for nome in ("suporte", "segundo_dominio"):
        modelo.set_adapter(nome)
        resposta = modelo.generate(**prompt, max_new_tokens=80, do_sample=False)
        print(f"\n[{nome}] {tokenizer.decode(resposta[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
