"""SFT, LoRA ou QLoRA portátil em GPU NVIDIA com TRL/PEFT."""

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
    parser.add_argument("--metodo", choices=("full", "lora", "qlora"), default="lora")
    parser.add_argument("--modelo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dados", type=Path, default=AQUI / "suporte")
    parser.add_argument("--passos", type=int, default=200)
    parser.add_argument("--saida", type=Path, default=AQUI / "modelo-cuda")
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resumo(args):
    return {
        "metodo": args.metodo,
        "modelo": args.modelo,
        "dados": str(args.dados),
        "passos": args.passos,
        "saida": str(args.saida),
        "report_to": args.report_to,
        "assistant_only_loss": True,
    }


def main():
    args = argumentos()
    print(json.dumps(resumo(args), ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    import torch
    from adapters import ManifestoAdapter, salvar_manifesto
    from cuda import descrever_cuda
    from datasets import load_dataset
    from experimentos import RegistroExperimento
    from governanca import sha256_arquivo
    from huggingface_hub import model_info
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    gpu = descrever_cuda(torch)
    revisao = model_info(args.modelo).sha
    if not revisao:
        raise RuntimeError("não foi possível resolver a revisão imutável do modelo")

    arquivos = {nome: str(args.dados / f"{nome}.jsonl") for nome in ("train", "valid")}
    if not all(Path(caminho).exists() for caminho in arquivos.values()):
        raise FileNotFoundError("rode `python modulo-05-sft/preparar_dados.py` primeiro")
    dataset = load_dataset("json", data_files=arquivos)

    quantizacao = None
    if args.metodo == "qlora":
        quantizacao = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo,
        revision=revisao,
        dtype=torch.bfloat16,
        quantization_config=quantizacao,
        device_map="auto" if quantizacao else None,
    )
    peft_config = None
    if args.metodo in {"lora", "qlora"}:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    configuracao = resumo(args) | {"revisao": revisao, "gpu": gpu}
    registro = RegistroExperimento("sft-cuda", configuracao, RAIZ / "runs", RAIZ)
    treino = SFTConfig(
        output_dir=str(args.saida),
        max_steps=args.passos,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5 if args.metodo == "full" else 1e-4,
        bf16=True,
        gradient_checkpointing=True,
        assistant_only_loss=True,
        max_length=1024,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        logging_steps=5,
        report_to=args.report_to,
        seed=42,
    )
    trainer = SFTTrainer(
        model=modelo,
        args=treino,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        peft_config=peft_config,
    )
    resultado = trainer.train()
    trainer.save_model()
    avaliacao = trainer.evaluate()
    if peft_config is not None:
        metricas_adapter = {
            chave: float(valor)
            for chave, valor in avaliacao.items()
            if isinstance(valor, (int, float))
        }
        salvar_manifesto(
            ManifestoAdapter(
                nome=args.saida.name,
                modelo_base=args.modelo,
                revisao_base=revisao,
                metodo=args.metodo,
                tarefa="SFT conversacional",
                dataset_sha256=sha256_arquivo(args.dados / "train.jsonl"),
                metricas=metricas_adapter,
            ),
            args.saida / "adapter-manifest.json",
        )
    registro.registrar(int(resultado.global_step), **resultado.metrics)
    registro.concluir(model_revision=revisao, avaliacao=avaliacao, output_dir=str(args.saida))


if __name__ == "__main__":
    main()
