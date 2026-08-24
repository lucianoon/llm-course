"""Reasoning distillation ponta a ponta em GPU NVIDIA."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
M7 = RAIZ / "modulo-07-reasoning"
sys.path.insert(0, str(RAIZ / "tools"))


def argumentos():
    parser = argparse.ArgumentParser()
    parser.add_argument("--professor", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--aluno", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--problemas", type=int, default=200)
    parser.add_argument("--avaliacao", type=int, default=30)
    parser.add_argument("--passos", type=int, default=200)
    parser.add_argument("--saida", type=Path, default=AQUI / "modelo-destilado-cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extrair_numero(texto: str) -> str | None:
    marcador = re.search(r"(?:Resposta final:|####)\s*([-+]?\d+(?:[.,]\d+)?)", texto)
    if marcador:
        return marcador.group(1).replace(",", "").rstrip(".")
    numeros = re.findall(r"[-+]?\d+(?:[.,]\d+)?", texto)
    return numeros[-1].replace(",", "").rstrip(".") if numeros else None


def resumo(args):
    return {
        "professor": args.professor,
        "aluno": args.aluno,
        "problemas": args.problemas,
        "avaliacao": args.avaliacao,
        "passos": args.passos,
        "saida": str(args.saida),
    }


def main():
    args = argumentos()
    print(json.dumps(resumo(args), ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    import torch
    from cuda import descrever_cuda
    from datasets import Dataset
    from experimentos import RegistroExperimento
    from huggingface_hub import model_info
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    gpu = descrever_cuda(torch)
    treino_path = M7 / "data" / "gsm8k_train.jsonl"
    teste_path = M7 / "data" / "gabarito_teste.jsonl"
    if not treino_path.exists() or not teste_path.exists():
        raise FileNotFoundError("rode `python modulo-07-reasoning/dados.py` primeiro")
    treino = [json.loads(linha) for linha in treino_path.read_text().splitlines()]
    teste = [json.loads(linha) for linha in teste_path.read_text().splitlines()]

    revisao_prof = model_info(args.professor).sha
    revisao_aluno = model_info(args.aluno).sha
    if not revisao_prof or not revisao_aluno:
        raise RuntimeError("não foi possível resolver as revisões imutáveis dos modelos")
    professor = AutoModelForCausalLM.from_pretrained(
        args.professor, revision=revisao_prof, dtype=torch.bfloat16, device_map="auto"
    )
    tok_prof = AutoTokenizer.from_pretrained(args.professor, revision=revisao_prof)

    def gerar(modelo, tokenizer, pergunta, max_tokens=512):
        mensagens = [{"role": "user", "content": pergunta}]
        entrada = tokenizer.apply_chat_template(
            mensagens, add_generation_prompt=True, return_tensors="pt"
        ).to(modelo.device)
        with torch.no_grad():
            saida = modelo.generate(
                entrada,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(saida[0, entrada.shape[1]:], skip_special_tokens=True)

    aprovados = []
    for indice, exemplo in enumerate(treino[: args.problemas], start=1):
        gabarito = extrair_numero(exemplo["answer"])
        resposta = gerar(professor, tok_prof, exemplo["question"])
        if extrair_numero(resposta) == gabarito:
            aprovados.append(
                {"messages": [
                    {"role": "user", "content": exemplo["question"]},
                    {"role": "assistant", "content": resposta},
                ]}
            )
        if indice % 25 == 0:
            print(f"professor: {indice}/{args.problemas} | aprovados={len(aprovados)}")
    del professor
    torch.cuda.empty_cache()
    if len(aprovados) < 20:
        raise RuntimeError("menos de 20 traços aprovados; aumente --problemas ou use outro professor")

    corte = max(1, len(aprovados) // 10)
    dataset_treino = Dataset.from_list(aprovados[corte:])
    dataset_valid = Dataset.from_list(aprovados[:corte])
    aluno = AutoModelForCausalLM.from_pretrained(
        args.aluno, revision=revisao_aluno, dtype=torch.bfloat16
    ).cuda()
    tok_aluno = AutoTokenizer.from_pretrained(args.aluno, revision=revisao_aluno)

    def avaliar(modelo):
        acertos = 0
        for exemplo in teste[: args.avaliacao]:
            resposta = gerar(modelo, tok_aluno, exemplo["question"], max_tokens=384)
            acertos += extrair_numero(resposta) == exemplo["answer"]
        return acertos / args.avaliacao

    antes = avaliar(aluno)
    config = SFTConfig(
        output_dir=str(args.saida),
        max_steps=args.passos,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        bf16=True,
        assistant_only_loss=True,
        max_length=1024,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=aluno,
        args=config,
        train_dataset=dataset_treino,
        eval_dataset=dataset_valid,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    resultado = trainer.train()
    depois = avaliar(trainer.model)
    trainer.save_model()

    registro = RegistroExperimento(
        "distillation-cuda",
        resumo(args) | {
            "revisao_professor": revisao_prof,
            "revisao_aluno": revisao_aluno,
            "gpu": gpu,
            "tracos_aprovados": len(aprovados),
        },
        RAIZ / "runs",
        RAIZ,
    )
    registro.registrar(int(resultado.global_step), acuracia_antes=antes, acuracia_depois=depois)
    registro.concluir(acuracia_antes=antes, acuracia_depois=depois, output_dir=str(args.saida))
    print(f"aluno: {antes:.0%} → {depois:.0%} em n={args.avaliacao}")


if __name__ == "__main__":
    main()
