"""Gera o dataset de preferências do Módulo 8 por CORRUPÇÃO CONTROLADA.

A técnica da seção 4 do README: para eliminar um comportamento específico, construa o
rejected corrompendo o chosen com exatamente aquele defeito. O par isola o sinal.

O defeito escolhido: BOILERPLATE — os fechos vazios que assistentes adoram
("Hope this helps! Feel free to ask..."). O chosen é a resposta limpa do Alpaca curado
(módulo 5); o rejected é a MESMA resposta com o boilerplate anexado. Nada mais difere.

A métrica de sucesso é objetiva: fração de gerações contendo boilerplate, antes e
depois do DPO. O lab_mlx.py mede.

Formato de saída: {prompt, chosen, rejected} — o que o mlx-lm-lora espera.

Uso:
    python preparar_dados.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

AQUI = Path(__file__).parent
ORIGEM = AQUI.parent / "modulo-05-sft" / "alpaca"
DESTINO = AQUI / "preferencias"

random.seed(42)

BOILERPLATE = [
    " Hope this helps! Feel free to ask if you have any other questions!",
    " I hope this answer was helpful! Let me know if you need anything else! 😊",
    " Thanks for asking! Don't hesitate to reach out with more questions!",
    " Hope that clears things up! I'm always here to help!",
    " Let me know if there's anything else I can assist you with today!",
    " I'm glad I could help! Please don't hesitate to ask more questions! 🙌",
]

# Frases-sonda para a métrica do lab (qualquer uma presente = geração com boilerplate)
SONDAS = ["hope this helps", "feel free to ask", "let me know if", "don't hesitate",
          "glad i could help", "here to help", "anything else"]


def converter(caminho: Path) -> list[dict]:
    pares = []
    for linha in caminho.open(encoding="utf-8"):
        msgs = json.loads(linha)["messages"]
        usuario = next(m["content"] for m in msgs if m["role"] == "user")
        resposta = next(m["content"] for m in msgs if m["role"] == "assistant")
        if len(resposta.split()) < 8:          # respostas curtíssimas viram pares fracos
            continue
        pares.append({
            "prompt": usuario,
            "chosen": resposta,
            # 1 ou 2 fechos, para variar o defeito
            "rejected": resposta + "".join(random.sample(BOILERPLATE,
                                                         random.choice([1, 1, 2]))),
        })
    return pares


if __name__ == "__main__":
    assert (ORIGEM / "train.jsonl").exists(), \
        "rode antes: python ../modulo-05-sft/preparar_dados.py"

    DESTINO.mkdir(exist_ok=True)
    for split in ["train", "valid"]:
        pares = converter(ORIGEM / f"{split}.jsonl")
        with (DESTINO / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for par in pares:
                f.write(json.dumps(par, ensure_ascii=False) + "\n")
        print(f"  preferencias/{split}.jsonl: {len(pares):,} pares")

    with (DESTINO / "sondas.json").open("w", encoding="utf-8") as f:
        json.dump(SONDAS, f)

    exemplo = converter(ORIGEM / "valid.jsonl")[0]
    print("\nexemplo de par:")
    print(f"  prompt  : {exemplo['prompt'][:80]}")
    print(f"  chosen  : ...{exemplo['chosen'][-60:]}")
    print(f"  rejected: ...{exemplo['rejected'][-90:]}")

    # Auditoria obrigatória (README, seção 3): correlação comprimento x preferência.
    pares = converter(ORIGEM / "train.jsonl")
    dif = [len(p["rejected"].split()) - len(p["chosen"].split()) for p in pares]
    print(f"\n⚠️ auditoria de comprimento: rejected tem em média {sum(dif)/len(dif):+.1f} "
          f"palavras a mais que chosen.")
    print("   Neste dataset o REJECTED é sempre o mais longo — o DPO vai associar")
    print("   'longo demais + boilerplate' a ruim. Aceitável aqui (o defeito É o anexo),")
    print("   mas note o confundimento: o exercício B3 o desfaz.")
