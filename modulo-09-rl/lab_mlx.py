# %% [markdown]
# # Módulo 9 — Laboratório B: a receita do R1 no M4
#
# **Só no Mac.** GRPO de verdade: Qwen2.5-1.5B + GSM8K + as duas recompensas do R1-Zero
# (acurácia e formato), via `mlx-lm-lora`.
#
# > ⚠️ Não executado pelo autor. Flags conferidas contra o README do mlx-lm-lora
# > (train-mode grpo, group-size, reward-functions-file). O pacote evolui rápido; se algo
# > mudar, `mlx_lm_lora.train --help` é a fonte da verdade.
#
# > 🔧 **Expectativa de custo:** GRPO gera `group_size` respostas POR EXEMPLO POR PASSO —
# > é ordens de magnitude mais caro que SFT. Num M4 de 16 GB, conte horas, não minutos.
# > O lab usa 150 iterações com grupo 4: o suficiente para ver a recompensa subir, não
# > para reproduzir o R1. Deixe rodando à noite — a vantagem do hardware local.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Baseline: acurácia e taxa de formato antes do RL |
# | 2 | GRPO com as recompensas do R1 |
# | 3 | Depois: as mesmas métricas + leitura das gerações |
# | 4 | Comparação com o caminho SFT (módulo 7) |
#
# Antes: `python preparar_dados.py`  (e `pip install -U mlx-lm-lora`)

# %%
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

assert platform.machine() == "arm64", "este lab requer Apple Silicon; use lab_cpu.py"

AQUI = Path.cwd()
MODELO = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"
DADOS = AQUI / "gsm8k-grpo"
assert (DADOS / "train.jsonl").exists(), "rode antes: python preparar_dados.py"

# O gabarito de avaliação vem do módulo 7 (split de teste oficial).
GABARITO = [json.loads(l) for l in
            (AQUI.parent / "modulo-07-reasoning" / "data" / "gabarito_teste.jsonl")
            .open(encoding="utf-8")]
print(f"{len(GABARITO)} problemas de teste")

sys.path.insert(0, str(AQUI))
from recompensas_r1 import _extrair_numero, recompensa_acuracia, recompensa_formato


def rodar(modulo, *args, mostrar=2000):
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", modulo, *args], capture_output=True, text=True)
    saida = r.stdout if r.returncode == 0 else (r.stdout + "\n--- STDERR ---\n" + r.stderr)
    print(f"$ {modulo} ...  ({time.perf_counter()-t0:.0f}s, exit {r.returncode})")
    if mostrar:
        print(saida[-mostrar:])
    return r.returncode == 0, saida

# %% [markdown]
# ## Lab 1 — Baseline
#
# As duas recompensas, medidas no modelo antes de qualquer RL. A regra de ouro da seção 7
# do README: **a acurácia base não pode ser ~0** — se o grupo nunca acerta, a vantagem é
# sempre zero e o GRPO não tem o que amplificar.

# %%
import mlx.core as mx
from mlx_lm import generate, load


def gerar_resposta(model, tokenizer, pergunta, max_tokens=350, temp=0.0):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": pergunta + "\n\nEnd with 'Final answer: <number>'."}],
        tokenize=False, add_generation_prompt=True)
    try:
        from mlx_lm.sample_utils import make_sampler
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                        sampler=make_sampler(temp=temp), verbose=False)
    except (ImportError, TypeError):
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def avaliar(model, tokenizer, n=60, rotulo=""):
    acuracia = formato = 0
    for i in range(n):
        texto = gerar_resposta(model, tokenizer, GABARITO[i]["question"])
        acuracia += recompensa_acuracia(None, texto, GABARITO[i]["answer"])
        formato += recompensa_formato(None, texto, None)
        if (i + 1) % 20 == 0:
            print(f"  [{rotulo}] {i+1}/{n}: acurácia {acuracia/(i+1):.0%}")
    return {"acuracia": acuracia / n, "formato": formato / n}


model_base, tok = load(MODELO)
antes = avaliar(model_base, tok, rotulo="base")
print(f"\nBASE — acurácia: {antes['acuracia']:.0%} | formato: {antes['formato']:.0%}")
assert antes["acuracia"] > 0.02, (
    "acurácia ~0: o GRPO não terá o que amplificar. Use um modelo maior "
    "ou faça um SFT de cold start primeiro (o caminho do R1 completo).")

del model_base
mx.clear_cache()

# %% [markdown]
# ## Lab 2 — GRPO
#
# As decisões, mapeadas para o README:
#
# - `--group-size 4` — G=4 (o R1 usa 16; 4 é o que a memória e a paciência permitem).
# - recompensas **binárias** via plugin (`recompensas_r1.py`) — a defesa nº 1 contra hack.
# - peso 1.0 na acurácia, 0.5 no formato — o formato é andaime, não o objetivo.
# - `--temperature 0.8` na geração — grupos DIVERSOS; com T=0 as G respostas seriam
#   idênticas e a vantagem, sempre zero.

# %%
rodar("mlx_lm_lora", "train",
      "--model", MODELO, "--train",
      "--train-mode", "grpo",
      "--data", str(DADOS),
      "--adapter-path", str(AQUI / "adapters-grpo"),
      "--reward-functions-file", str(AQUI / "recompensas_r1.py"),
      "--reward-functions", "recompensa_acuracia,recompensa_formato",
      "--reward-weights", "[1.0, 0.5]",
      "--group-size", "4",
      "--temperature", "0.8",
      "--max-completion-length", "350",
      "--iters", "150", "--batch-size", "1",
      "--learning-rate", "1e-5",
      "--max-seq-length", "1024",
      mostrar=3000)

# %% [markdown]
# **No log, procure:** a recompensa média por grupo subindo, e o KL crescendo devagar.
# Recompensa subindo com KL explodindo = a política está fugindo da linguagem para
# satisfazer o verificador — o Lab 5 do lab_cpu em escala real.
#
# ## Lab 3 — Depois

# %%
model_rl, tok_rl = load(MODELO, adapter_path=str(AQUI / "adapters-grpo"))
depois = avaliar(model_rl, tok_rl, rotulo="GRPO")

print(f"\n{'':<10} {'acurácia':>10} {'formato':>9}")
print(f"{'base':<10} {antes['acuracia']:>10.0%} {antes['formato']:>9.0%}")
print(f"{'GRPO':<10} {depois['acuracia']:>10.0%} {depois['formato']:>9.0%}")

# %%
# Leia. Sempre leia.
for i in [2, 7]:
    print(f"\n{'='*70}\nPROBLEMA: {GABARITO[i]['question'][:140]}\n(gabarito: {GABARITO[i]['answer']})\n")
    print(gerar_resposta(model_rl, tok_rl, GABARITO[i]["question"])[:700])

# %% [markdown]
# ## Lab 4 — Os três caminhos para o mesmo lugar
#
# Você agora treinou matemática de TRÊS formas neste curso. Compare (os números do SFT
# vêm do módulo 7, lab_mlx):
#
# | Caminho | Sinal usado | Dados necessários | Custo de treino |
# |---|---|---|---|
# | SFT em traços humanos (mód. 7) | imitação | 1.500 raciocínios escritos | minutos |
# | Distillation de R1 (mód. 7, desafio) | imitação de modelo | traços do professor | minutos |
# | **GRPO (este lab)** | **verificador** | **só perguntas + gabaritos** | **horas** |
#
# A pergunta de engenharia: o GRPO usou MUITO menos informação por exemplo (um bit de
# verificação vs. um raciocínio completo) e MUITO mais compute. Quando ele vence?
#
# 1. Quando não existem traços para imitar (fronteira da capacidade — o caso do R1-Zero).
# 2. Quando os traços existentes têm um teto que o modelo pode superar sozinho.
# 3. Quando o verificador é muito mais barato que a anotação.
#
# Para todo o resto, SFT/distillation chega ao mesmo lugar por uma fração do custo — e é
# por isso que o pipeline do R1 completo usa os dois: RL para DESCOBRIR, SFT/distillation
# para DISSEMINAR. O módulo 10 é a segunda metade dessa frase.

# %%
