# %% [markdown]
# # Módulo 7 — Laboratório B: treinando raciocínio no M4
#
# **Só no Mac.** O experimento central do módulo: dois LoRAs sobre os MESMOS 1.500
# problemas do GSM8K — um treinado com o raciocínio completo, outro só com a resposta
# final — avaliados por comparação exata no split de teste oficial.
#
# > ⚠️ Não executado pelo autor (autoria em Windows sem GPU). Comandos conferidos contra
# > a documentação oficial do mlx-lm. Reporte erros com a saída completa.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Treinar os dois LoRAs |
# | 2 | Avaliação verificável: acurácia por comparação exata |
# | 3 | Custo por resposta: tokens e latência |
# | 4 | Um modelo de reasoning destilado de verdade (R1-Distill) |
#
# Antes: `python dados.py`

# %%
import json
import platform
import re
import sys
import time
from pathlib import Path

assert platform.machine() == "arm64", "este lab requer Apple Silicon; use lab_cpu.py"

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(AQUI.parent / "tools"))
from execucao import executar_modulo

MODELO = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"

for pasta in ["gsm8k-cot", "gsm8k-direto"]:
    assert (AQUI / pasta / "train.jsonl").exists(), "rode antes: python dados.py"

gabarito = [json.loads(l) for l in
            (AQUI / "data" / "gabarito_teste.jsonl").open(encoding="utf-8")]
print(f"{len(gabarito)} problemas de teste (split oficial do GSM8K)")


def rodar(*args, mostrar=1500):
    resultado = executar_modulo("mlx_lm", *args, mostrar=mostrar)
    return resultado.ok, resultado.saida

# %% [markdown]
# ## Lab 1 — Os dois treinos
#
# Idênticos em tudo — modelo, problemas, hiperparâmetros, épocas — exceto no conteúdo
# da resposta. 1.500 exemplos, batch 4, 750 iterações = 2 épocas.
#
# > 🔧 Note o que NÃO fazemos: mascarar os tokens de raciocínio. No SFT de instrução
# > (módulo 5) a loss fica só na resposta; aqui o raciocínio **é** o que se ensina.
# > O `--mask-prompt` mascara apenas o turno do usuário, que é o comportamento certo
# > nos dois casos.

# %%
for nome in ["gsm8k-cot", "gsm8k-direto"]:
    print(f"\n{'='*60}\nTREINO: {nome}\n{'='*60}")
    rodar("lora", "--model", MODELO, "--train",
          "--data", str(AQUI / nome),
          "--adapter-path", str(AQUI / f"adapters-{nome}"),
          "--iters", "750", "--batch-size", "4",
          "--num-layers", "16",
          "--learning-rate", "1e-4",
          "--mask-prompt",
          "--max-seq-length", "768",
          "--steps-per-eval", "100")

# %% [markdown]
# ## Lab 2 — Avaliação verificável
#
# Nada de juiz: gera, extrai o número, compara com o gabarito. A função de extração é a
# mesma do `lab_cpu.py` — testada lá.

# %%
from mlx_lm import generate, load


def extrair_resposta(texto: str):
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


def avaliar(model, tokenizer, n=100, max_tokens=350, rotulo=""):
    acertos, total_tokens, t0 = 0, 0, time.perf_counter()
    for i in range(n):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": gabarito[i]["question"]}],
            tokenize=False, add_generation_prompt=True)
        try:
            from mlx_lm.sample_utils import make_sampler
            texto = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                             sampler=make_sampler(temp=0.0), verbose=False)
        except (ImportError, TypeError):
            texto = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                             verbose=False)
        total_tokens += len(tokenizer.encode(texto))
        if extrair_resposta(texto) == gabarito[i]["answer"]:
            acertos += 1
        if (i + 1) % 20 == 0:
            print(f"  [{rotulo}] {i+1}/{n}: {acertos/(i+1):.0%}")
    dt = time.perf_counter() - t0
    return {"acuracia": acertos / n, "tokens_por_resposta": total_tokens / n,
            "segundos_por_resposta": dt / n}

# %%
N_AVALIACAO = 100
resultados = {}

for nome, adapter in [("base (sem treino)", None),
                      ("LoRA direto", "adapters-gsm8k-direto"),
                      ("LoRA CoT", "adapters-gsm8k-cot")]:
    print(f"\n=== {nome} ===")
    kwargs = {"adapter_path": str(AQUI / adapter)} if adapter else {}
    model, tokenizer = load(MODELO, **kwargs)
    resultados[nome] = avaliar(model, tokenizer, N_AVALIACAO, rotulo=nome)
    del model
    import mlx.core as mx
    mx.clear_cache()

# %%
print(f"{'modelo':<22} {'acurácia':>10} {'tokens/resp':>13} {'s/resp':>8}")
print("-" * 58)
for nome, r in resultados.items():
    print(f"{nome:<22} {r['acuracia']:>10.0%} {r['tokens_por_resposta']:>13.0f} "
          f"{r['segundos_por_resposta']:>8.1f}")

# %% [markdown]
# **As três linhas contam a história inteira do módulo:**
#
# - `LoRA direto` aprendeu o *formato* "Resposta final: N" — mas não tem onde computar.
#   Espere acurácia baixa, possivelmente ABAIXO da base (que ao menos raciocina livremente).
# - `LoRA CoT` aprendeu a escrever os passos antes — os mesmos problemas, o mesmo modelo,
#   e a diferença de acurácia é o valor dos tokens de raciocínio, isolado.
# - A coluna de custo mostra o preço: mais tokens por resposta, mais latência.
#
# Compare com o Lab 2 do `lab_cpu.py`: lá o raciocínio de OURO no contexto deslocava a
# probabilidade; aqui o modelo aprendeu a GERAR o próprio raciocínio. É o mesmo mecanismo,
# nas duas direções.
#
# ## Lab 4 — Um destilado de verdade
#
# O paper do R1 destilou 800k traços em modelos Qwen. O resultado está publicado — e
# cabe na sua máquina. Observe o formato `<think>` em ação.

# %%
R1 = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
model_r1, tok_r1 = load(R1)

problema = gabarito[3]["question"]
prompt = tok_r1.apply_chat_template([{"role": "user", "content": problema}],
                                    tokenize=False, add_generation_prompt=True)
try:
    from mlx_lm.sample_utils import make_sampler
    saida = generate(model_r1, tok_r1, prompt=prompt, max_tokens=1024,
                     sampler=make_sampler(temp=0.6), verbose=False)
except (ImportError, TypeError):
    saida = generate(model_r1, tok_r1, prompt=prompt, max_tokens=1024, verbose=False)

print(f"PROBLEMA: {problema[:150]}\n")
print(saida[:1800])
print(f"\n... [{len(tok_r1.encode(saida))} tokens no total]")
print(f"resposta extraída: {extrair_resposta(saida)} | ouro: {gabarito[3]['answer']}")

# %% [markdown]
# Repare em três coisas na saída:
#
# 1. **O `<think>...</think>`** separando raciocínio de resposta — o formato da seção 3.
# 2. **Auto-verificação espontânea** ("Wait, let me check...") — comportamento que emergiu
#    do RL do R1 original e foi transferido por SFT puro para este 1.5B.
# 3. **O comprimento.** Provavelmente 3–10x os tokens do seu LoRA CoT. É overthinking em
#    ação: o preço da capacidade. Rode o problema 0 (fácil) e veja o exagero.
#
# ---
#
# ## Encerramento
#
# O experimento central do curso até aqui: mesmos dados, mesmo modelo, mesma receita —
# e a única variável é se a resposta contém o raciocínio. A diferença de acurácia é o
# valor computacional dos tokens intermediários.
#
# No módulo 9, o RL vai além do SFT: em vez de imitar raciocínios escritos, o modelo
# descobre os próprios — recompensado apenas pela resposta final verificável. O R1 que
# você acabou de rodar nasceu assim.
#
# Agora o `exercicios.md`.

# %%
