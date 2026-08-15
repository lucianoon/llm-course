# %% [markdown]
# # Módulo 5 — Laboratório: SFT com MLX no M4
#
# **Roda no Mac (Apple Silicon).** Requer `pip install mlx mlx-lm`.
#
# > ⚠️ Diferente dos módulos 1–4, este lab **não foi executado pelo autor** (o ambiente
# > de autoria é Windows sem GPU). Os comandos e flags foram conferidos contra a
# > documentação oficial do `mlx-lm`, mas a API Python pode variar entre versões. Cada
# > etapa tem uma verificação que falha cedo e com mensagem clara. Se algo quebrar,
# > a mensagem de erro é a informação mais útil.
#
# | Lab | Assunto |
# |---|---|
# | 0 | Verificação do ambiente |
# | 1 | Métrica de avaliação — definida ANTES de treinar |
# | 2 | Baseline: o modelo base com o melhor prompt possível |
# | 3 | Treinar LoRA |
# | 4 | Avaliar: mesma métrica, comparação honesta |
# | 5 | Catastrophic forgetting, medido |
# | 6 | Fundir o adaptador |
# | 7 | Experimento A: Alpaca e capacidade geral |
#
# Antes de tudo: `python preparar_dados.py`

# %%
import json
import re
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODELO = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"
DADOS_SUPORTE = AQUI / "suporte"
ADAPTADORES = AQUI / "adapters-suporte"

# %% [markdown]
# ## Lab 0 — Verificação do ambiente
#
# Falhe agora, não depois de vinte minutos de treino.

# %%
import platform

assert platform.machine() == "arm64", (
    f"Este lab requer Apple Silicon. Detectado: {platform.machine()}. "
    "Nos módulos 1-4 tudo roda em CPU; a partir daqui é MLX/Metal."
)

import mlx.core as mx

a = mx.random.normal((2048, 2048))
mx.eval(a @ a)
print(f"chip        : {platform.processor()} / {platform.machine()}")
print(f"mlx device  : {mx.default_device()}")
print(f"memória ativa: {mx.get_active_memory() / 1e9:.2f} GB")
print(f"pico        : {mx.get_peak_memory() / 1e9:.2f} GB")

assert DADOS_SUPORTE.exists(), "rode antes: python preparar_dados.py"
for split in ["train", "valid", "test"]:
    n = sum(1 for _ in (DADOS_SUPORTE / f"{split}.jsonl").open(encoding="utf-8"))
    print(f"  {split:<6} {n:>4} exemplos")

# %% [markdown]
# ## Lab 1 — A métrica, definida antes do treino
#
# Esta é a etapa que quase todo mundo pula (README, seção 7, passo 2). Sem uma métrica
# definida **antes**, você vai racionalizar qualquer resultado.
#
# O objetivo do experimento B é formato: três seções, nesta ordem, com estes títulos.
# Isso é objetivamente verificável.

# %%
SECOES = ["DIAGNÓSTICO", "SOLUÇÃO", "PREVENÇÃO"]

def aderencia(resposta: str) -> dict:
    """0 a 3 pela presença das seções, mais um bônus se a ordem estiver certa."""
    presentes = [s for s in SECOES if re.search(rf"^{s}\s*:", resposta, re.MULTILINE | re.IGNORECASE)]
    posicoes = [resposta.upper().find(s) for s in SECOES if s in resposta.upper()]
    ordem_ok = posicoes == sorted(posicoes) and len(posicoes) == 3
    return {
        "secoes": len(presentes),
        "ordem_ok": ordem_ok,
        "completo": len(presentes) == 3 and ordem_ok,
        "n_palavras": len(resposta.split()),
    }

# Teste da métrica contra um exemplo real do dataset — a métrica também precisa ser testada.
exemplo = json.loads((DADOS_SUPORTE / "train.jsonl").open(encoding="utf-8").readline())
resposta_ouro = exemplo["messages"][-1]["content"]
print("métrica no exemplo de ouro:", aderencia(resposta_ouro))
assert aderencia(resposta_ouro)["completo"], "a métrica está errada — o ouro deveria passar"

print("\nmétrica em uma resposta genérica:")
print(" ", aderencia("Você pode tentar reiniciar o computador e verificar os cabos."))

# %% [markdown]
# ## Lab 2 — Baseline
#
# A comparação honesta não é contra o modelo base sem instrução nenhuma. É contra o
# modelo base **com o melhor prompt que você consegue escrever**. Se um prompt melhor
# resolve, o fine-tuning não provou nada.

# %%
from mlx_lm import generate, load


def responder(model, tokenizer, pergunta, sistema, max_tokens=300):
    """Wrapper defensivo: a assinatura de `generate` mudou entre versões do mlx-lm."""
    msgs = [{"role": "system", "content": sistema}, {"role": "user", "content": pergunta}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    try:                                    # mlx-lm recente: sampler explícito
        from mlx_lm.sample_utils import make_sampler
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                        sampler=make_sampler(temp=0.0), verbose=False)
    except (ImportError, TypeError):        # versões anteriores
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

modelo_base, tokenizer = load(MODELO)
print(f"modelo carregado | memória: {mx.get_active_memory() / 1e9:.2f} GB")

# %%
testes = [json.loads(l) for l in (DADOS_SUPORTE / "test.jsonl").open(encoding="utf-8")]
perguntas_teste = []
vistas = set()
for ex in testes:
    p = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    if p not in vistas:
        vistas.add(p)
        perguntas_teste.append(p)
print(f"{len(perguntas_teste)} perguntas de teste (problemas NUNCA vistos no treino)\n")
for p in perguntas_teste[:4]:
    print(f"  - {p}")

# %%
SISTEMA_SIMPLES = ("Você é um assistente de suporte técnico. Responda SEMPRE em três seções, "
                   "nesta ordem e com estes títulos exatos: DIAGNÓSTICO, SOLUÇÃO, PREVENÇÃO.")

# O prompt "esforçado": instrução detalhada + exemplo few-shot. Esta é a baseline REAL.
treino = [json.loads(l) for l in (DADOS_SUPORTE / "train.jsonl").open(encoding="utf-8")]
demo = treino[0]
SISTEMA_ESFORCADO = SISTEMA_SIMPLES + (
    "\n\nUse exatamente este formato, sem texto antes ou depois:\n\n"
    f"Exemplo de pergunta: {next(m['content'] for m in demo['messages'] if m['role']=='user')}\n"
    f"Exemplo de resposta:\n{demo['messages'][-1]['content']}"
)

baselines = {}
for nome, sistema in [("prompt simples", SISTEMA_SIMPLES), ("prompt esforçado", SISTEMA_ESFORCADO)]:
    notas = []
    for p in perguntas_teste:
        r = responder(modelo_base, tokenizer, p, sistema)
        notas.append(aderencia(r))
    completos = sum(n["completo"] for n in notas)
    baselines[nome] = {"completo": completos / len(notas),
                       "secoes": sum(n["secoes"] for n in notas) / len(notas),
                       "palavras": sum(n["n_palavras"] for n in notas) / len(notas)}
    print(f"{nome:<18} formato completo: {completos}/{len(notas)} "
          f"({baselines[nome]['completo']:.0%}) | seções médias: {baselines[nome]['secoes']:.2f}")

# %%
print("=== resposta do modelo BASE (prompt esforçado) ===")
print(responder(modelo_base, tokenizer, perguntas_teste[0], SISTEMA_ESFORCADO)[:600])

# %% [markdown]
# ## Lab 3 — Treinar
#
# Cálculo obrigatório antes de rodar (README, seção 3):
#
# ```
# épocas = (iters × batch_size) / número_de_exemplos
# ```

# %%
N_TREINO = sum(1 for _ in (DADOS_SUPORTE / "train.jsonl").open(encoding="utf-8"))
ITERS, BATCH = 300, 4
print(f"exemplos de treino : {N_TREINO}")
print(f"iters × batch      : {ITERS} × {BATCH} = {ITERS * BATCH} exemplos processados")
print(f"épocas             : {ITERS * BATCH / N_TREINO:.1f}")
assert 1 <= ITERS * BATCH / N_TREINO <= 12, "fora da faixa razoável — ajuste iters"

# %%
comando = [
    sys.executable, "-m", "mlx_lm", "lora",
    "--model", MODELO,
    "--train",
    "--data", str(DADOS_SUPORTE),
    "--adapter-path", str(ADAPTADORES),
    "--iters", str(ITERS),
    "--batch-size", str(BATCH),
    "--num-layers", "8",           # das 28 camadas do Qwen2.5-1.5B, só as 8 finais
    "--learning-rate", "1e-4",     # 10x o default: LoRA parte do zero (README, seção 3)
    "--mask-prompt",               # loss só na resposta (módulo 4, seção 9)
    "--steps-per-eval", "50",
    "--max-seq-length", "1024",
]
print(" ".join(comando).replace(sys.executable, "python"), "\n")

resultado = subprocess.run(comando, capture_output=True, text=True, check=False)
print(resultado.stdout[-3000:])
if resultado.returncode != 0:
    print("STDERR:\n", resultado.stderr[-2000:])
    raise RuntimeError(f"treino MLX falhou com exit {resultado.returncode}")

# %% [markdown]
# **Como ler o log.** O `mlx_lm.lora` imprime `Train loss` a cada `steps_per_report` e
# `Val loss` a cada `steps_per_eval`. O que você procura:
#
# - a **loss de validação** descendo junto com a de treino → aprendendo;
# - validação estacionada enquanto treino desce → começou a memorizar, reduza `iters`;
# - nenhuma das duas descendo → learning rate baixo demais ou `num_layers` pequeno demais.
#
# ## Lab 4 — Avaliar

# %%
modelo_sft, tokenizer_sft = load(MODELO, adapter_path=str(ADAPTADORES))
print(f"modelo + adaptador carregado | memória: {mx.get_active_memory() / 1e9:.2f} GB")

notas = [aderencia(responder(modelo_sft, tokenizer_sft, p, SISTEMA_SIMPLES))
         for p in perguntas_teste]
completos = sum(n["completo"] for n in notas)
depois = {"completo": completos / len(notas),
          "secoes": sum(n["secoes"] for n in notas) / len(notas),
          "palavras": sum(n["n_palavras"] for n in notas) / len(notas)}

print(f"\n{'configuração':<28} {'formato completo':>18} {'seções/3':>10} {'palavras':>10}")
print("-" * 70)
for nome, r in baselines.items():
    print(f"base + {nome:<21} {r['completo']:>17.0%} {r['secoes']:>10.2f} {r['palavras']:>10.0f}")
print(f"{'LoRA + prompt simples':<28} {depois['completo']:>17.0%} "
      f"{depois['secoes']:>10.2f} {depois['palavras']:>10.0f}")

# %% [markdown]
# **A comparação que importa** é a última linha contra `base + prompt esforçado`. Se o
# LoRA com prompt curto empata com o prompt longo few-shot, você ganhou: economiza os
# tokens do prompt em toda chamada, para sempre. Se perde, o fine-tuning não se pagou —
# e essa é uma conclusão legítima que muita gente evita registrar.

# %%
print("=== BASE (prompt esforçado) ===")
print(responder(modelo_base, tokenizer, perguntas_teste[1], SISTEMA_ESFORCADO)[:500])
print("\n=== LoRA (prompt simples) ===")
print(responder(modelo_sft, tokenizer_sft, perguntas_teste[1], SISTEMA_SIMPLES)[:500])

# %% [markdown]
# ## Lab 5 — Catastrophic forgetting
#
# Treinar num domínio estreito degrada o resto. Medindo com perplexidade em texto
# **fora** do domínio de fine-tuning.

# %%
from mlx import nn


def perplexidade(model, tokenizer, texto: str) -> float:
    ids = tokenizer.encode(texto)
    entrada = mx.array([ids])
    logits = model(entrada)[0]                       # [seq, vocab]
    perda = nn.losses.cross_entropy(
        logits[:-1].astype(mx.float32), mx.array(ids[1:]), reduction="mean"
    )
    return float(mx.exp(perda))

FORA_DO_DOMINIO = {
    "literatura PT": "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem "
                     "da Central um rapaz aqui do bairro, que eu conheço de vista e de chapéu.",
    "código Python": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "fato geral": "A fotossíntese é o processo pelo qual plantas convertem luz solar, água e "
                  "dióxido de carbono em glicose e oxigênio.",
    "domínio do SFT": "DIAGNÓSTICO: Fila de impressão travada.\n\nSOLUÇÃO: Cancele os documentos.",
}

print(f"{'texto':<18} {'base':>10} {'após SFT':>10} {'variação':>12}")
print("-" * 54)
for nome, texto in FORA_DO_DOMINIO.items():
    p_antes = perplexidade(modelo_base, tokenizer, texto)
    p_depois = perplexidade(modelo_sft, tokenizer_sft, texto)
    print(f"{nome:<18} {p_antes:>10.2f} {p_depois:>10.2f} {(p_depois/p_antes - 1):>+11.1%}")

# %% [markdown]
# **Como interpretar:** a última linha (domínio do SFT) deve **melhorar** muito — é o que
# você treinou. As outras devem piorar pouco. Se a perplexidade em literatura ou código
# subir mais de ~20%, o adaptador está agressivo demais: reduza `iters`, `num_layers` ou
# o learning rate.
#
# Com LoRA, os pesos originais estão congelados — o esquecimento vem só da perturbação de
# baixo posto, e costuma ser modesto. Em full fine-tune este quadro seria bem pior.
#
# ## Lab 6 — Fundir o adaptador
#
# Para servir em produção, funda o adaptador nos pesos base: elimina a indireção e permite
# quantizar ou exportar para GGUF.

# %%
fusao = subprocess.run(
    [sys.executable, "-m", "mlx_lm", "fuse",
     "--model", MODELO,
     "--adapter-path", str(ADAPTADORES),
     "--save-path", str(AQUI / "modelo-suporte-fundido")],
    capture_output=True, text=True, check=False,
)
print(fusao.stdout[-1500:] or fusao.stderr[-1500:])
if fusao.returncode != 0:
    raise RuntimeError(f"fusão MLX falhou com exit {fusao.returncode}")

# %%
# O adaptador é minúsculo comparado ao modelo — a razão de LoRA ser tão prático.
def tamanho(caminho: Path) -> float:
    return sum(f.stat().st_size for f in caminho.rglob("*") if f.is_file()) / 1e6

if ADAPTADORES.exists():
    print(f"adaptador       : {tamanho(ADAPTADORES):>8.1f} MB")
fundido = AQUI / "modelo-suporte-fundido"
if fundido.exists():
    print(f"modelo fundido  : {tamanho(fundido):>8.1f} MB")
    print(f"razão           : {tamanho(fundido) / max(tamanho(ADAPTADORES), 1e-9):>8.0f}x")

# %% [markdown]
# ## Lab 7 — Experimento A: Alpaca
#
# O experimento B mediu **formato**, onde o SFT brilha com poucos exemplos. O A mede
# **capacidade geral de seguir instruções**, com 1.020 exemplos curados — e é bem mais
# difícil de avaliar, porque não existe métrica objetiva de "boa resposta".

# %%
DADOS_ALPACA = AQUI / "alpaca"
ADAPT_ALPACA = AQUI / "adapters-alpaca"

if DADOS_ALPACA.exists():
    n_alpaca = sum(1 for _ in (DADOS_ALPACA / "train.jsonl").open(encoding="utf-8"))
    iters_alpaca = 600
    print(f"exemplos: {n_alpaca} | épocas: {iters_alpaca * 4 / n_alpaca:.1f}")

    r = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora",
         "--model", MODELO, "--train",
         "--data", str(DADOS_ALPACA),
         "--adapter-path", str(ADAPT_ALPACA),
         "--iters", str(iters_alpaca), "--batch-size", "4",
         "--num-layers", "16", "--learning-rate", "1e-4",
         "--mask-prompt", "--max-seq-length", "1024"],
        capture_output=True, text=True, check=False,
    )
    print(r.stdout[-2000:] if r.returncode == 0 else r.stderr[-2000:])
    if r.returncode != 0:
        raise RuntimeError(f"treino Alpaca falhou com exit {r.returncode}")
else:
    print("pasta alpaca/ não encontrada — rode: python preparar_dados.py")

# %%
# Avaliação do experimento A: sem métrica objetiva, resta comparação lado a lado.
# É aqui que o LLM-as-judge (README, seção 5) entra — e é o desafio dos exercícios.
if ADAPT_ALPACA.exists():
    modelo_alpaca, tok_alpaca = load(MODELO, adapter_path=str(ADAPT_ALPACA))
    perguntas = ["Explique o que é uma API REST para alguém não técnico.",
                 "Liste três maneiras de melhorar o desempenho de uma consulta SQL lenta."]
    for p in perguntas:
        print(f"\n{'='*70}\nPERGUNTA: {p}\n")
        print("--- BASE ---")
        print(responder(modelo_base, tokenizer, p, "Você é um assistente útil.", 200)[:400])
        print("\n--- SFT Alpaca ---")
        print(responder(modelo_alpaca, tok_alpaca, p, "Você é um assistente útil.", 200)[:400])

# %% [markdown]
# > ⚠️ Note a dificuldade: o Qwen2.5-1.5B-**Instruct** já passou por SFT e RLHF de altíssima
# > qualidade pela Alibaba, com muito mais dados do que os 1.020 exemplos do Alpaca. É
# > perfeitamente possível — e comum — que o seu fine-tuning o deixe **pior** em capacidade
# > geral, ao especializá-lo numa distribuição mais estreita e mais antiga.
# >
# > Isso não é falha do seu treino; é a razão de o SFT valer a pena para **especializar**
# > (experimento B) e raramente para "melhorar em geral" um modelo instruct moderno. Para
# > sentir o efeito histórico do Alpaca, treine sobre o modelo **base**
# > (`mlx-community/Qwen2.5-1.5B-bf16`, sem `-Instruct`) e compare.
#
# ---
#
# ## Encerramento
#
# Você acabou de:
#
# - definir uma métrica **antes** de treinar e testá-la contra o exemplo de ouro;
# - estabelecer a baseline honesta (modelo base com o melhor prompt);
# - treinar LoRA no seu Mac, de graça, em minutos;
# - comparar na mesma métrica e saber dizer se o fine-tuning se pagou;
# - **medir** catastrophic forgetting em vez de supor;
# - fundir o adaptador para produção.
#
# No módulo 6 vamos abrir o LoRA: o que aquele `rank: 8` significa, por que ele funciona,
# e como treinar modelos de 7B nos seus 16 GB.
#
# Agora o `exercicios.md`.

# %%
