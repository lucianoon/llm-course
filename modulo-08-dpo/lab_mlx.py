# %% [markdown]
# # Módulo 8 — Laboratório B: DPO e ORPO no M4
#
# **Só no Mac.** Usa o pacote comunitário `mlx-lm-lora` (Gökdeniz Gülmez), que estende o
# MLX com treino de preferências: `pip install -U mlx-lm-lora`.
#
# > ⚠️ Não executado pelo autor (autoria em Windows). Comandos conferidos contra o
# > README oficial do mlx-lm-lora. O pacote evolui rápido — se uma flag mudar, rode
# > `mlx_lm_lora.train --help` e reporte.
#
# O experimento: eliminar **boilerplate** ("Hope this helps! 😊") por DPO, com pares
# construídos por corrupção controlada (`preparar_dados.py`). A métrica é objetiva:
# fração de gerações contendo frases-sonda, antes e depois.
#
# | Lab | Assunto |
# |---|---|
# | 1 | A métrica e a baseline |
# | 2 | SFT primeiro (a ordem importa) |
# | 3 | DPO |
# | 4 | Medição final + recompensa implícita |
# | 5 | ORPO: sem referência, metade da memória |
#
# Antes: `python preparar_dados.py`

# %%
import json
import platform
import sys
from pathlib import Path

assert platform.machine() == "arm64", "este lab requer Apple Silicon; use lab_cpu.py"

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(AQUI.parent / "tools"))
from execucao import executar_modulo

MODELO = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"
PREFS = AQUI / "preferencias"
assert (PREFS / "train.jsonl").exists(), "rode antes: python preparar_dados.py"

SONDAS = json.loads((PREFS / "sondas.json").read_text(encoding="utf-8"))
print("frases-sonda:", SONDAS)


def rodar(modulo, *args, mostrar=1800):
    resultado = executar_modulo(modulo, *args, mostrar=mostrar)
    return resultado.ok, resultado.saida

# %% [markdown]
# ## Lab 1 — A métrica, antes de qualquer treino

# %%
from mlx_lm import generate, load


def gerar_texto(model, tokenizer, pergunta, max_tokens=200):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": pergunta}],
                                           tokenize=False, add_generation_prompt=True)
    try:
        from mlx_lm.sample_utils import make_sampler
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                        sampler=make_sampler(temp=0.7), verbose=False)
    except (ImportError, TypeError):
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def taxa_boilerplate(model, tokenizer, perguntas, rotulo=""):
    com_boiler = 0
    for i, p in enumerate(perguntas):
        texto = gerar_texto(model, tokenizer, p).lower()
        if any(s in texto for s in SONDAS):
            com_boiler += 1
    print(f"  [{rotulo}] boilerplate em {com_boiler}/{len(perguntas)} gerações")
    return com_boiler / len(perguntas)


# Perguntas de avaliação: as do valid.jsonl (o modelo nunca treinou nelas).
perguntas_aval = [json.loads(l)["prompt"]
                  for l in (PREFS / "valid.jsonl").open(encoding="utf-8")][:30]

model_base, tok = load(MODELO)
taxa_antes = taxa_boilerplate(model_base, tok, perguntas_aval, "base")

# %% [markdown]
# > Se a taxa da base for ~0%, o experimento morre aqui — não se pode reduzir o que não
# > existe (a lição off-policy do `lab_cpu.py`, aprendida na prática). Nesse caso, faça o
# > SFT do Lab 2 **com os rejected como alvo** por algumas iterações, para instalar o
# > vício antes de curá-lo — é o análogo controlado do que acontece em produção, onde o
# > SFT em dados com boilerplate o instala sem querer.

# %%
import mlx.core as mx

if taxa_antes < 0.1:
    print("base limpa demais — instalando o vício de propósito (SFT nos REJECTED)...")
    vicio = AQUI / "dados-vicio"
    vicio.mkdir(exist_ok=True)
    for split in ["train", "valid"]:
        with (vicio / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for linha in (PREFS / f"{split}.jsonl").open(encoding="utf-8"):
                par = json.loads(linha)
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": par["prompt"]},
                    {"role": "assistant", "content": par["rejected"]},
                ]}, ensure_ascii=False) + "\n")
    rodar("mlx_lm", "lora", "--model", MODELO, "--train",
          "--data", str(vicio), "--adapter-path", str(AQUI / "adapters-vicio"),
          "--iters", "200", "--batch-size", "4", "--num-layers", "8",
          "--learning-rate", "1e-4", "--mask-prompt", "--max-seq-length", "768")
    # funde o vício para servir de ponto de partida único dos labs seguintes
    rodar("mlx_lm", "fuse", "--model", MODELO,
          "--adapter-path", str(AQUI / "adapters-vicio"),
          "--save-path", str(AQUI / "modelo-viciado"))
    PONTO_DE_PARTIDA = str(AQUI / "modelo-viciado")
else:
    PONTO_DE_PARTIDA = MODELO

del model_base
mx.clear_cache()

model_v, tok_v = load(PONTO_DE_PARTIDA)
taxa_partida = taxa_boilerplate(model_v, tok_v, perguntas_aval, "ponto de partida")
del model_v
mx.clear_cache()

# %% [markdown]
# ## Lab 3 — DPO
#
# Nota sobre os hiperparâmetros:
# - `--beta 0.1` — o padrão do README, seção 2.
# - `--learning-rate 5e-6` — **muito menor que o do SFT** (1e-4). DPO mexe na política
#   inteira na direção de uma diferença de log-probs; passos grandes derrubam chosen e
#   rejected juntos (a patologia do lab_cpu, acelerada).
# - sem `--reference-model-path`: o pacote usa o próprio modelo inicial congelado como
#   referência, que é o que queremos.

# %%
rodar("mlx_lm_lora", "train",
      "--model", PONTO_DE_PARTIDA, "--train",
      "--train-mode", "dpo",
      "--data", str(PREFS),
      "--adapter-path", str(AQUI / "adapters-dpo"),
      "--beta", "0.1",
      "--dpo-cpo-loss-type", "sigmoid",
      "--iters", "300", "--batch-size", "2",
      "--learning-rate", "5e-6",
      "--max-seq-length", "768")

# %% [markdown]
# **No log, procure:** a loss partindo de ~0,69 (= ln 2 — o teste de sanidade do
# lab_cpu vale aqui também) e caindo. Se partir de outro valor, o formato dos dados está
# errado. Se cair para ~0 em 20 iterações, o problema é fácil demais (pares muito
# distinguíveis) — o que aqui é esperado: o boilerplate é um sinal grosseiro.

# %%
model_dpo, tok_dpo = load(PONTO_DE_PARTIDA, adapter_path=str(AQUI / "adapters-dpo"))
taxa_depois = taxa_boilerplate(model_dpo, tok_dpo, perguntas_aval, "após DPO")

print(f"\n{'':<20} {'taxa de boilerplate':>20}")
print(f"{'ponto de partida':<20} {taxa_partida:>20.0%}")
print(f"{'após DPO':<20} {taxa_depois:>20.0%}")

# %%
# A qualidade sobreviveu? Gere e LEIA. Métricas não substituem olhos.
for p in perguntas_aval[:3]:
    print(f"\n{'='*70}\nPERGUNTA: {p[:100]}\n")
    print(gerar_texto(model_dpo, tok_dpo, p)[:400])

del model_dpo
mx.clear_cache()

# %% [markdown]
# ## Lab 4 — A recompensa implícita, medida
#
# `r̂(y) = β·[log π(y|x) − log π_ref(y|x)]`. Nos pares de validação, a margem
# `r̂(chosen) − r̂(rejected)` deve ser positiva após o treino — é a métrica interna do DPO.

# %%
from mlx import nn


def logprob_mlx(model, tokenizer, prompt, resposta):
    ids_p = tokenizer.encode(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True))
    ids_r = tokenizer.encode(resposta, add_special_tokens=False) \
        if hasattr(tokenizer, "encode") else tokenizer(resposta)
    seq = mx.array([ids_p + ids_r])
    logits = model(seq)[0]
    lp = nn.log_softmax(logits[len(ids_p) - 1: -1].astype(mx.float32), axis=-1)
    return float(mx.take_along_axis(lp, mx.array(ids_r)[:, None], axis=-1).sum())


pares_val = [json.loads(l) for l in (PREFS / "valid.jsonl").open(encoding="utf-8")][:10]

ref, tok_ref = load(PONTO_DE_PARTIDA)
pol, tok_pol = load(PONTO_DE_PARTIDA, adapter_path=str(AQUI / "adapters-dpo"))

print(f"{'par':>4} {'r̂(chosen)':>12} {'r̂(rejected)':>13} {'margem':>9}")
print("-" * 42)
margens = []
for i, par in enumerate(pares_val):
    rc = 0.1 * (logprob_mlx(pol, tok_pol, par["prompt"], par["chosen"])
                - logprob_mlx(ref, tok_ref, par["prompt"], par["chosen"]))
    rr = 0.1 * (logprob_mlx(pol, tok_pol, par["prompt"], par["rejected"])
                - logprob_mlx(ref, tok_ref, par["prompt"], par["rejected"]))
    margens.append(rc - rr)
    print(f"{i:>4} {rc:>12.3f} {rr:>13.3f} {rc - rr:>9.3f}")
print(f"\nmargem média: {sum(margens)/len(margens):+.3f} | "
      f"acurácia de preferência: {sum(m > 0 for m in margens)}/{len(margens)}")

del ref, pol
mx.clear_cache()

# %% [markdown]
# ## Lab 5 — ORPO: o mesmo efeito, metade da memória
#
# ORPO dispensa o modelo de referência — funde SFT e preferência numa loss só. No M4 de
# 16 GB, isso significa um modelo a menos residente na memória.

# %%
rodar("mlx_lm_lora", "train",
      "--model", PONTO_DE_PARTIDA, "--train",
      "--train-mode", "orpo",
      "--data", str(PREFS),
      "--adapter-path", str(AQUI / "adapters-orpo"),
      "--beta", "0.1",
      "--iters", "300", "--batch-size", "2",
      "--learning-rate", "5e-6",
      "--max-seq-length", "768")

# %%
model_orpo, tok_orpo = load(PONTO_DE_PARTIDA, adapter_path=str(AQUI / "adapters-orpo"))
taxa_orpo = taxa_boilerplate(model_orpo, tok_orpo, perguntas_aval, "após ORPO")

print(f"\n{'método':<20} {'taxa de boilerplate':>20}")
print(f"{'ponto de partida':<20} {taxa_partida:>20.0%}")
print(f"{'DPO':<20} {taxa_depois:>20.0%}")
print(f"{'ORPO':<20} {taxa_orpo:>20.0%}")

# %% [markdown]
# Compare também a memória de pico dos dois treinos no Monitor de Atividade — a
# diferença é o modelo de referência que o ORPO não carrega.
#
# ---
#
# ## Encerramento
#
# O ciclo completo de alinhamento por preferências, no seu Mac:
#
# - pares por corrupção controlada — o defeito isolado cirurgicamente;
# - DPO com a ordem certa (SFT → DPO) e LR 20× menor que o do SFT;
# - métrica objetiva antes/depois + leitura manual (sempre os dois);
# - a recompensa implícita como diagnóstico interno;
# - ORPO como alternativa econômica.
#
# No módulo 9, a última peça: quando existe recompensa **verificável**, nem pares são
# necessários — o modelo gera, um verificador pontua, e o RL faz o resto. É o GRPO do
# DeepSeek-R1.

# %%
