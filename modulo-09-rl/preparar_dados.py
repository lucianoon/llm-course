"""Dados do Módulo 9 — GSM8K no formato do GRPO do mlx-lm-lora.

O GRPO não precisa de respostas-modelo nem de pares: só de {prompt, answer}, onde
`answer` é o GABARITO que a função de recompensa usa para verificar. O modelo gera as
próprias tentativas.

Reaproveita o GSM8K baixado no módulo 7.

Também gera `recompensas_r1.py` — as duas funções de recompensa do R1-Zero (acurácia e
formato), no formato de plugin do mlx-lm-lora.

Uso:
    python preparar_dados.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

AQUI = Path(__file__).parent
M7 = AQUI.parent / "modulo-07-reasoning"
DESTINO = AQUI / "gsm8k-grpo"

random.seed(42)


def resposta_final(answer: str) -> str:
    m = re.search(r"####\s*([\-0-9.,]+)", answer)
    assert m, f"sem resposta final: {answer[-80:]}"
    return m.group(1).replace(",", "").rstrip(".")


if __name__ == "__main__":
    origem = M7 / "data" / "gsm8k_train.jsonl"
    if not origem.exists():
        sys.path.insert(0, str(M7))
        import dados as dados_m7
        dados_m7.baixar("train")
        dados_m7.baixar("test")

    treino = [json.loads(l) for l in origem.read_text(encoding="utf-8").strip().split("\n")]
    random.shuffle(treino)

    DESTINO.mkdir(exist_ok=True)
    partes = {"train": treino[:1000], "valid": treino[1000:1080]}
    for nome, exemplos in partes.items():
        with (DESTINO / f"{nome}.jsonl").open("w", encoding="utf-8") as f:
            for e in exemplos:
                f.write(json.dumps({
                    "prompt": e["question"],
                    "answer": resposta_final(e["answer"]),   # só o gabarito!
                }, ensure_ascii=False) + "\n")
        print(f"  gsm8k-grpo/{nome}.jsonl: {len(exemplos):,}")

    print("\nexemplo:")
    print(json.dumps(json.loads((DESTINO / 'train.jsonl').open(encoding='utf-8').readline()),
                     ensure_ascii=False)[:220])
    print("\nNote o que NÃO está aqui: nenhum raciocínio, nenhuma resposta-modelo.")
    print("O modelo vai gerar as tentativas; a recompensa verifica contra `answer`.")
