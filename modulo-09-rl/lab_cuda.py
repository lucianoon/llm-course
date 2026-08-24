"""GRPO portátil em CUDA com TRL e recompensas verificáveis do GSM8K."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "tools"))


def argumentos():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dados", type=Path, default=AQUI / "gsm8k-grpo" / "train.jsonl")
    parser.add_argument("--passos", type=int, default=150)
    parser.add_argument("--geracoes", type=int, default=4)
    parser.add_argument("--saida", type=Path, default=AQUI / "modelo-grpo-cuda")
    parser.add_argument("--report-to", choices=("none", "wandb", "mlflow"), default="none")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extrair_numero(texto: str) -> str | None:
    encontrados = re.findall(r"[-+]?\d+(?:[.,]\d+)?", texto)
    return encontrados[-1].replace(",", "") if encontrados else None


def recompensa_acuracia(completions, answer, **kwargs):
    return [float(extrair_numero(texto) == gabarito) for texto, gabarito in zip(completions, answer)]


def recompensa_formato(completions, **kwargs):
    return [float("<answer>" in texto and "</answer>" in texto) for texto in completions]


def resumo(args):
    return {
        "modelo": args.modelo,
        "dados": str(args.dados),
        "passos": args.passos,
        "geracoes": args.geracoes,
        "saida": str(args.saida),
        "report_to": args.report_to,
    }


def main():
    args = argumentos()
    print(json.dumps(resumo(args), ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    import torch
    from cuda import descrever_cuda
    from datasets import load_dataset
    from experimentos import RegistroExperimento
    from huggingface_hub import model_info
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    gpu = descrever_cuda(torch)
    if not args.dados.exists():
        raise FileNotFoundError("rode `python modulo-09-rl/preparar_dados.py` primeiro")
    revisao = model_info(args.modelo).sha
    if not revisao:
        raise RuntimeError("não foi possível resolver a revisão imutável do modelo")
    dataset = load_dataset("json", data_files=str(args.dados), split="train")
    instrucao = (
        "Resolva passo a passo e termine exatamente com <answer>NUMERO</answer>.\n\n"
    )
    dataset = dataset.map(lambda item: {"prompt": instrucao + item["prompt"]})

    config = GRPOConfig(
        output_dir=str(args.saida),
        max_steps=args.passos,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        num_generations=args.geracoes,
        max_completion_length=384,
        bf16=True,
        beta=0.04,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        report_to=args.report_to,
        log_completions=True,
        model_init_kwargs={"revision": revisao, "dtype": torch.bfloat16},
        seed=42,
    )
    registro = RegistroExperimento(
        "grpo-cuda", resumo(args) | {"revisao": revisao, "gpu": gpu}, RAIZ / "runs", RAIZ
    )
    trainer = GRPOTrainer(
        model=args.modelo,
        args=config,
        reward_funcs=[recompensa_acuracia, recompensa_formato],
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    resultado = trainer.train()
    trainer.save_model()
    registro.registrar(int(resultado.global_step), **resultado.metrics)
    registro.concluir(model_revision=revisao, output_dir=str(args.saida))


if __name__ == "__main__":
    main()
