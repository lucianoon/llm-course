"""As duas recompensas do R1-Zero, como plugin do mlx-lm-lora.

Uso no treino:
    mlx_lm_lora.train ... --train-mode grpo \
        --reward-functions-file ./recompensas_r1.py \
        --reward-functions "recompensa_acuracia,recompensa_formato" \
        --reward-weights "[1.0, 0.5]"

Regras de design (README, seção 6):
- BINÁRIAS e limitadas — sem gradiente para "mais ainda", sem contagens hackeáveis.
- Verificação EXATA — extração + comparação, nenhum juiz neural.
"""

import re

from mlx_lm_lora.reward_functions import register_reward_function


def _extrair_numero(texto: str):
    """A extração robusta do módulo 7 — o último número, normalizado."""
    m = re.search(r"(?:resposta final|final answer|answer is)[:\s]*\$?\s*([\-0-9.,]+)",
                  texto, re.IGNORECASE)
    candidato = m.group(1) if m else None
    if candidato is None:
        numeros = re.findall(r"-?\$?\d[\d,]*\.?\d*", texto)
        if not numeros:
            return None
        candidato = numeros[-1]
    limpo = candidato.replace("$", "").replace(",", "").rstrip(".")
    if limpo.endswith((".0", ".00")):
        limpo = limpo.split(".")[0]
    return limpo or None


@register_reward_function()
def recompensa_acuracia(prompt, completion, reference_answer, **kwargs):
    """1.0 se a resposta final extraída confere com o gabarito. Senão 0.0."""
    extraida = _extrair_numero(completion or "")
    return 1.0 if extraida is not None and extraida == str(reference_answer).strip() else 0.0


@register_reward_function()
def recompensa_formato(prompt, completion, reference_answer, **kwargs):
    """1.0 se a resposta declara explicitamente a resposta final (o 'formato' do R1).

    No R1 real o formato exigido é <think>...</think>; aqui, treinando um modelo
    instruct sem template de thinking, exigimos o marcador 'Final answer:' — mesma
    função: tornar a resposta EXTRAÍVEL, que é o que permite a recompensa de acurácia
    funcionar.
    """
    return 1.0 if re.search(r"(?:resposta final|final answer)\s*:", completion or "",
                            re.IGNORECASE) else 0.0
