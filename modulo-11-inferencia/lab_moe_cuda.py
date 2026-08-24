"""QLoRA de um MoE real e diagnóstico de roteamento em CUDA."""

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
    parser.add_argument("--modelo", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    parser.add_argument("--alvo", choices=("attention", "experts"), default="attention")
    parser.add_argument("--dados", type=Path, default=RAIZ / "modulo-05-sft" / "suporte")
    parser.add_argument("--passos", type=int, default=100)
    parser.add_argument("--saida", type=Path, default=AQUI / "modelo-moe-cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resumo(args):
    alvos = ["q_proj", "k_proj", "v_proj", "o_proj"] if args.alvo == "attention" else [
        "gate_proj", "up_proj", "down_proj"
    ]
    return {
        "modelo": args.modelo,
        "alvo": args.alvo,
        "target_modules": alvos,
        "dados": str(args.dados),
        "passos": args.passos,
        "saida": str(args.saida),
    }


def main():
    args = argumentos()
    especificacao = resumo(args)
    print(json.dumps(especificacao, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    import torch
    from cuda import descrever_cuda
    from datasets import load_dataset
    from experimentos import RegistroExperimento
    from huggingface_hub import model_info
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    gpu = descrever_cuda(torch)
    arquivos = {nome: str(args.dados / f"{nome}.jsonl") for nome in ("train", "valid")}
    if not all(Path(caminho).exists() for caminho in arquivos.values()):
        raise FileNotFoundError("rode `python modulo-05-sft/preparar_dados.py` primeiro")
    dataset = load_dataset("json", data_files=arquivos)
    revisao = model_info(args.modelo).sha
    if not revisao:
        raise RuntimeError("não foi possível resolver a revisão imutável do modelo")

    config_modelo = AutoConfig.from_pretrained(args.modelo, revision=revisao)
    config_modelo.output_router_logits = True
    config_modelo.router_aux_loss_coef = 0.01
    quantizacao = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo,
        revision=revisao,
        config=config_modelo,
        quantization_config=quantizacao,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.modelo, revision=revisao)
    treino = SFTConfig(
        output_dir=str(args.saida),
        max_steps=args.passos,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        bf16=True,
        assistant_only_loss=True,
        max_length=768,
        logging_steps=2,
        save_steps=50,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=modelo,
        args=treino,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=especificacao["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    resultado = trainer.train()
    trainer.save_model()

    # Diagnóstico em dados reais: fração top-1 por expert em todas as camadas.
    contagens = None
    for exemplo in dataset["valid"].select(range(min(20, len(dataset["valid"])))):
        texto = tokenizer.apply_chat_template(exemplo["messages"], tokenize=False)
        entrada = tokenizer(texto, return_tensors="pt", truncation=True, max_length=768).to(
            trainer.model.device
        )
        with torch.no_grad():
            saida = trainer.model(**entrada, output_router_logits=True)
        roteadores = getattr(saida, "router_logits", None)
        if not roteadores:
            raise RuntimeError("o modelo não expôs router_logits; confira a arquitetura escolhida")
        for logits in roteadores:
            camada = torch.bincount(logits.argmax(-1).flatten().cpu(), minlength=logits.shape[-1])
            contagens = camada if contagens is None else contagens + camada
    utilizacao = (contagens / contagens.sum()).tolist()
    print("utilização top-1 por expert:", [f"{valor:.1%}" for valor in utilizacao])

    registro = RegistroExperimento(
        "moe-cuda",
        especificacao | {"revisao": revisao, "gpu": gpu},
        RAIZ / "runs",
        RAIZ,
    )
    registro.registrar(int(resultado.global_step), **resultado.metrics)
    registro.concluir(utilizacao_experts=utilizacao, output_dir=str(args.saida))


if __name__ == "__main__":
    main()
