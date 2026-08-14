"""Datasets do Módulo 7 — GSM8K, baixado direto do GitHub (sem a lib `datasets`).

GSM8K (Cobbe et al., 2021): 8,5k problemas de matemática de ensino fundamental, cada um
com a cadeia de raciocínio escrita por humanos e a resposta final após `####`. É o
benchmark clássico de reasoning — e, importante para nós, cada resposta é VERIFICÁVEL
por comparação exata, o que elimina a necessidade de juiz.

Este script gera dois datasets de SFT no formato do mlx_lm, a partir dos MESMOS problemas:

    gsm8k-cot/     resposta = raciocínio completo + resposta final
    gsm8k-direto/  resposta = APENAS a resposta final

A diferença de acurácia entre os dois modelos treinados é o experimento central do
módulo: o valor dos tokens de raciocínio, isolado de qualquer outra variável.

Uso:
    python dados.py
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path

AQUI = Path(__file__).parent
DATA = AQUI / "data"
BASE_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"

random.seed(42)


def baixar(split: str) -> list[dict]:
    destino = DATA / f"gsm8k_{split}.jsonl"
    if not destino.exists():
        DATA.mkdir(exist_ok=True)
        req = urllib.request.Request(f"{BASE_URL}/{split}.jsonl",
                                     headers={"User-Agent": "Mozilla/5.0"})
        destino.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        print(f"  baixado: {destino.name} ({destino.stat().st_size // 1024} KB)")
    return [json.loads(l) for l in destino.read_text(encoding="utf-8").strip().split("\n")]


def resposta_final(answer: str) -> str:
    """Extrai o número após '####'. É o gabarito verificável."""
    m = re.search(r"####\s*([\-0-9.,]+)", answer)
    assert m, f"sem resposta final: {answer[-80:]}"
    return m.group(1).replace(",", "").rstrip(".")


def limpar_raciocinio(answer: str) -> str:
    """Remove as anotações de calculadora <<...>> e o '#### N' final."""
    corpo = answer.split("####")[0].strip()
    return re.sub(r"<<[^>]*>>", "", corpo)


def montar(exemplos: list[dict], com_raciocinio: bool) -> list[dict]:
    saida = []
    for e in exemplos:
        final = resposta_final(e["answer"])
        if com_raciocinio:
            conteudo = f"{limpar_raciocinio(e['answer'])}\n\nResposta final: {final}"
        else:
            conteudo = f"Resposta final: {final}"
        saida.append({"messages": [
            {"role": "user", "content": e["question"]},
            {"role": "assistant", "content": conteudo},
        ]})
    return saida


def escrever(partes: dict[str, list[dict]], pasta: Path):
    pasta.mkdir(exist_ok=True)
    for nome, dados in partes.items():
        with (pasta / f"{nome}.jsonl").open("w", encoding="utf-8") as f:
            for ex in dados:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {pasta.name}/{nome}.jsonl: {len(dados):,}")


if __name__ == "__main__":
    treino_bruto = baixar("train")
    teste_bruto = baixar("test")
    print(f"GSM8K: {len(treino_bruto):,} treino | {len(teste_bruto):,} teste")

    # Amostra de 1.500 para o treino de LoRA (o suficiente, e rápido no M4).
    random.shuffle(treino_bruto)
    amostra_treino = treino_bruto[:1500]
    amostra_valid = treino_bruto[1500:1650]

    for nome, com_cot in [("gsm8k-cot", True), ("gsm8k-direto", False)]:
        print(f"\n{nome}:")
        escrever({
            "train": montar(amostra_treino, com_cot),
            "valid": montar(amostra_valid, com_cot),
            # o teste é o split OFICIAL do GSM8K — nunca visto, comparável com papers
            "test": montar(teste_bruto[:200], com_cot),
        }, AQUI / nome)

    # E o gabarito puro, para avaliação por comparação exata.
    gabarito = [{"question": e["question"], "answer": resposta_final(e["answer"])}
                for e in teste_bruto[:200]]
    with (DATA / "gabarito_teste.jsonl").open("w", encoding="utf-8") as f:
        for g in gabarito:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"\ngabarito de avaliação: {len(gabarito)} problemas em data/gabarito_teste.jsonl")

    ex = montar([treino_bruto[0]], True)[0]
    print("\nexemplo (com raciocínio):")
    print(json.dumps(ex, ensure_ascii=False, indent=2)[:600])
