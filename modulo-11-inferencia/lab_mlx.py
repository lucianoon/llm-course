# %% [markdown]
# # Módulo 11 — Laboratório B: inferência de verdade no M4
#
# **Só no Mac.** As três alavancas, medidas no seu hardware: a escada de quantização,
# um MoE real, e decodificação especulativa nativa do mlx_lm.
#
# > ⚠️ Não executado pelo autor. Flags de especulativo (`--draft-model`,
# > `--num-draft-tokens`) confirmadas no repositório do mlx-lm.
#
# | Lab | Assunto |
# |---|---|
# | 1 | A escada de quantização: bits × memória × PPL × velocidade |
# | 2 | MoE real: 14B totais ao preço de 2,7B ativos |
# | 3 | Decodificação especulativa: 7B com draft de 0.5B |
# | 4 | TTFT vs TPOT: as métricas de servir |

# %%
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

assert platform.machine() == "arm64", "este lab requer Apple Silicon; use lab_cpu.py"

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def rodar(modulo, *args, mostrar=0, verificar=True):
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", modulo, *args], capture_output=True, text=True)
    if mostrar or r.returncode != 0:
        print((r.stdout + r.stderr)[-max(mostrar, 1200):])
    if verificar and r.returncode != 0:
        raise RuntimeError(f"{modulo} {' '.join(args[:2])} falhou com exit {r.returncode}")
    return r.returncode == 0, r.stdout + r.stderr, time.perf_counter() - t0


def extrair_metricas(saida: str) -> dict:
    """Parseia o --verbose do mlx_lm.generate: prompt/generation tps e pico de memória."""
    m = {}
    for padrao, chave in [(r"Prompt:.*?([\d.]+) tokens-per-sec", "prefill_tps"),
                          (r"Generation:.*?([\d.]+) tokens-per-sec", "decode_tps"),
                          (r"Peak memory: ([\d.]+) GB", "pico_gb")]:
        achado = re.search(padrao, saida)
        if achado:
            m[chave] = float(achado.group(1))
    return m

PROMPT_TESTE = ("Explique em três parágrafos por que a quantização de modelos de "
                "linguagem reduz o custo de inferência, citando o papel da banda de "
                "memória no decode.")

# %% [markdown]
# ## Lab 1 — A escada de quantização
#
# O mesmo Qwen2.5-1.5B em bf16, 8 bits e 4 bits. Três medições por degrau: memória,
# velocidade de decode e PPL (a qualidade — com a lição do módulo 6: no SEU idioma).

# %%
NIVEIS = {
    "bf16": "mlx-community/Qwen2.5-1.5B-Instruct-bf16",
    "8bit": None,     # convertidos localmente abaixo
    "4bit": None,
}

for bits in ["8", "4"]:
    destino = AQUI / f"qwen15b-{bits}bit"
    if not destino.exists():
        print(f"convertendo para {bits} bits...")
        rodar("mlx_lm", "convert", "--hf-path", "Qwen/Qwen2.5-1.5B-Instruct",
              "-q", "--q-bits", bits, "--q-group-size", "64",
              "--mlx-path", str(destino))
    NIVEIS[f"{bits}bit"] = str(destino)

# %%
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

corpus = AQUI.parent / "modulo-03-treino" / "data" / "corpus.txt"
TEXTO_PT = corpus.read_text(encoding="utf-8")[5000:9000] if corpus.exists() else None


def ppl(model, tokenizer, texto):
    ids = tokenizer.encode(texto)
    logits = model(mx.array([ids]))[0]
    perda = nn.losses.cross_entropy(logits[:-1].astype(mx.float32),
                                    mx.array(ids[1:]), reduction="mean")
    return float(mx.exp(perda))


resultados = {}
for nome, caminho in NIVEIS.items():
    ok, saida, _ = rodar("mlx_lm", "generate", "--model", caminho,
                         "--prompt", PROMPT_TESTE, "--max-tokens", "200", "--verbose")
    met = extrair_metricas(saida)
    if TEXTO_PT:
        m, t = load(caminho)
        met["ppl_pt"] = ppl(m, t, TEXTO_PT)
        del m
        mx.clear_cache()
    resultados[nome] = met

print(f"{'nível':<7} {'pico GB':>9} {'decode tok/s':>13} {'PPL (PT)':>10}")
print("-" * 44)
for nome, met in resultados.items():
    print(f"{nome:<7} {met.get('pico_gb', 0):>9.2f} {met.get('decode_tps', 0):>13.1f} "
          f"{met.get('ppl_pt', float('nan')):>10.2f}")

# %% [markdown]
# **As três colunas são o trade-off inteiro de servir:** cada degrau de bits corta a
# memória (~2× por degrau) e ACELERA o decode (menos bytes para ler por token — a tese
# memory-bound do módulo 1, agora medida), pagando em PPL. Anote onde a PPL portuguesa
# começa a subir de verdade — no módulo 6 o dano de 4 bits foi 4× maior em português
# que em inglês.
#
# ## Lab 2 — Um MoE de verdade
#
# Qwen1.5-MoE-A2.7B: **14,3B parâmetros totais, 2,7B ativos por token** (60 experts,
# top-4 + 4 compartilhados). Em 4 bits, ~8 GB — cabe nos seus 16.
#
# A comparação: um denso de tamanho similar aos ATIVOS (Qwen2.5-3B). Se a teoria da
# seção 2 está certa, os dois devem ter decode parecido — com o MoE carregando 4–5×
# mais parâmetros de capacidade.

# %%
COMPARACAO = {
    "denso 3B (4bit)": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "MoE 14B/2.7B ativos (4bit)": "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit",
}

moe_stats = {}
for nome, modelo_id in COMPARACAO.items():
    ok, saida, _ = rodar("mlx_lm", "generate", "--model", modelo_id,
                         "--prompt", PROMPT_TESTE, "--max-tokens", "200", "--verbose",
                         verificar=False)
    moe_stats[nome] = extrair_metricas(saida)
    if not ok:
        print(f"  ({nome} falhou — verifique se o repo existe na mlx-community)")

print(f"{'modelo':<30} {'pico GB':>9} {'decode tok/s':>13}")
print("-" * 56)
for nome, met in moe_stats.items():
    print(f"{nome:<30} {met.get('pico_gb', 0):>9.2f} {met.get('decode_tps', 0):>13.1f}")

# %% [markdown]
# **Como ler:** o MoE deve gastar ~2,5× a memória do denso 3B (paga TODOS os experts
# residentes) com decode na mesma ordem (só os ativos são lidos por token... com a
# ressalva de que experts diferentes por token quebram parte da localidade de memória —
# o tok/s real fica entre o de um 3B e o de um 14B, mais perto do primeiro).
#
# É a assinatura econômica do MoE: capacidade de 14B, conta de ~3B — quando a memória
# não é o gargalo. No seu M4 de 16 GB, ela é: você paga os 8 GB inteiros para ter os
# 2,7B ativos. MoE brilha onde memória sobra e compute/latência mandam.
#
# ## Lab 3 — Decodificação especulativa nativa
#
# O `mlx_lm.generate` tem o especulativo embutido: `--draft-model`. Alvo: 7B em 4 bits.
# Draft: 0.5B — mesmo tokenizer (obrigatório!), família treinada nos mesmos dados: a
# receita ideal de aceitação alta.

# %%
ALVO_7B = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DRAFT = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

print("=== 7B sozinho ===")
ok, saida, _ = rodar("mlx_lm", "generate", "--model", ALVO_7B,
                     "--prompt", PROMPT_TESTE, "--max-tokens", "250", "--verbose")
sozinho = extrair_metricas(saida)

print("=== 7B + draft 0.5B ===")
ok, saida, _ = rodar("mlx_lm", "generate", "--model", ALVO_7B,
                     "--draft-model", DRAFT, "--num-draft-tokens", "4",
                     "--prompt", PROMPT_TESTE, "--max-tokens", "250", "--verbose")
com_draft = extrair_metricas(saida)

print(f"\n{'configuração':<18} {'decode tok/s':>13} {'pico GB':>9}")
print(f"{'7B sozinho':<18} {sozinho.get('decode_tps', 0):>13.1f} {sozinho.get('pico_gb', 0):>9.2f}")
print(f"{'7B + draft':<18} {com_draft.get('decode_tps', 0):>13.1f} {com_draft.get('pico_gb', 0):>9.2f}")
if sozinho.get("decode_tps") and com_draft.get("decode_tps"):
    print(f"\nspeedup: {com_draft['decode_tps'] / sozinho['decode_tps']:.2f}x "
          f"— com saída de distribuição IDÊNTICA (a prova do lab_cpu)")

# %%
# O especulativo depende do TEXTO: em prosa previsível a aceitação é alta; em conteúdo
# imprevisível, o draft erra e o overhead vence. Medindo os dois regimes:
PROMPTS_REGIME = {
    "código (previsível)": "Write a Python function that implements binary search, with docstring and type hints.",
    "criativo (imprevisível)": "Invente cinco palavras que não existem em português e defina cada uma de forma surreal.",
}
for regime, p in PROMPTS_REGIME.items():
    linha = f"{regime:<26}"
    for draft_flags in [[], ["--draft-model", DRAFT, "--num-draft-tokens", "4"]]:
        ok, saida, _ = rodar("mlx_lm", "generate", "--model", ALVO_7B, *draft_flags,
                             "--prompt", p, "--max-tokens", "200", "--verbose")
        linha += f" {extrair_metricas(saida).get('decode_tps', 0):>8.1f}"
    print(f"{linha}   (sem | com draft)")

# %% [markdown]
# > 🔧 Duas notas de produção:
# > - **Especulativo + MoE não combinam bem** (issue #1132 do mlx-lm): o decode do MoE já
# >   lê poucos parâmetros por token; verificar k tokens de uma vez ativa MAIS experts
# >   (união dos roteamentos), corroendo exatamente a economia que o MoE dava.
# > - O draft ideal é da MESMA família (tokenizer idêntico é requisito duro; distribuição
# >   parecida é o que dá aceitação). Um aluno destilado do próprio alvo — módulo 10 — é
# >   o draft perfeito, e é assim que os provedores fazem.
#
# ## Lab 4 — TTFT e TPOT
#
# As duas métricas de UX, separadas. O `--verbose` já as fornece: o prefill tps determina
# o TTFT; o decode tps é 1/TPOT.

# %%
CONTEXTOS = {
    "prompt curto (~50 tok)": PROMPT_TESTE,
    "prompt longo (~2k tok)": (TEXTO_PT or PROMPT_TESTE * 20)[:6000] +
                              "\n\nResuma o texto acima em um parágrafo.",
}

MODELO_SERVICO = NIVEIS["4bit"]
print(f"{'cenário':<26} {'prefill tok/s':>14} {'TTFT estimado':>14} {'decode tok/s':>13}")
print("-" * 72)
for nome, p in CONTEXTOS.items():
    ok, saida, _ = rodar("mlx_lm", "generate", "--model", MODELO_SERVICO,
                         "--prompt", p, "--max-tokens", "100", "--verbose")
    met = extrair_metricas(saida)
    n_tok_prompt = len(p) // 4          # aproximação
    ttft = n_tok_prompt / met["prefill_tps"] if met.get("prefill_tps") else float("nan")
    print(f"{nome:<26} {met.get('prefill_tps', 0):>14.0f} {ttft:>13.2f}s "
          f"{met.get('decode_tps', 0):>13.1f}")

# %% [markdown]
# **A anatomia do custo, medida:** o TTFT cresce com o prompt (prefill é O(n²) mas
# paraleliza — tok/s de prefill é 10–100× o de decode); o TPOT é ~constante e é quem
# define a experiência de leitura. Em produção com batching, cada usuário a mais no
# batch degrada o TPOT de todos — e o ponto de operação é uma decisão de negócio, não
# de engenharia.
#
# ---
#
# ## Encerramento — e a conta final do curso
#
# Com os números DESTE lab, preencha:
#
# ```
# custo/Mtok do seu M4 ≈ (custo de oportunidade/h) / (decode_tps × 3600) × 10⁶
# ```
#
# Um M4 é "grátis" (você já o pagou), então a conta local vence qualquer API para
# desenvolvimento e volume baixo. Para produção, os mesmos números com uma GPU alugada
# (README, seção 5) dizem se o seu produto se paga.
#
# O módulo 12 fecha o curso: um projeto que atravessa o pipeline inteiro — dados,
# treino, alinhamento e serving — com as decisões justificadas por medição.

# %%
