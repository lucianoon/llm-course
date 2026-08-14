# %% [markdown]
# # Módulo 6 — Laboratório B: LoRA e QLoRA no M4
#
# **Só no Mac.** Aplica no seu hardware o que o `lab_cpu.py` verificou numericamente.
#
# > ⚠️ Este lab **não foi executado pelo autor** (ambiente de autoria: Windows sem GPU).
# > Comandos e flags conferidos contra `mlx_lm/LORA.md` e `examples/lora_config.yaml`.
# > As asserções falham cedo e com mensagem clara. Reporte erros com a saída completa.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Quantizar um modelo e medir a economia real |
# | 2 | Medir a degradação — em português |
# | 3 | QLoRA: treinar 3B nos seus 16 GB |
# | 4 | Varredura de rank, com memória e tempo medidos |
# | 5 | Alvos: q,v contra todas as lineares |
# | 6 | DoRA |
# | 7 | O teto da máquina: 7B em 4 bits |

# %%
import platform
import subprocess
import sys
import time
from pathlib import Path

assert platform.machine() == "arm64", "este lab requer Apple Silicon; use lab_cpu.py"

import mlx.core as mx

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DADOS = AQUI.parent / "modulo-05-sft" / "suporte"
assert DADOS.exists(), "rode antes: python ../modulo-05-sft/preparar_dados.py"

BASE_15B = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"
print(f"mlx device: {mx.default_device()}")
print(f"dados     : {DADOS}")

# %%
def rodar(*args, mostrar=2500, verificar=True):
    """Chama a CLI do mlx_lm e devolve (ok, saída)."""
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "mlx_lm", *args], capture_output=True, text=True)
    dt = time.perf_counter() - t0
    saida = r.stdout if r.returncode == 0 else (r.stdout + "\n--- STDERR ---\n" + r.stderr)
    print(f"$ mlx_lm {' '.join(args[:2])} ...  ({dt:.0f}s, exit {r.returncode})")
    print(saida[-mostrar:])
    if verificar and r.returncode != 0:
        raise RuntimeError(f"mlx_lm {' '.join(args[:2])} falhou com exit {r.returncode}")
    return r.returncode == 0, saida

def tamanho_mb(caminho: Path) -> float:
    if not caminho.exists():
        return 0.0
    return sum(f.stat().st_size for f in caminho.rglob("*") if f.is_file()) / 1e6

# %% [markdown]
# ### Configuração por YAML — e por que não por flags
#
# ⚠️ **`rank`, `scale`, `dropout` e `keys` NÃO têm flags de linha de comando.** O
# `build_parser()` do `mlx_lm/lora.py` não os define; eles só existem em `CONFIG_DEFAULTS`
# e só podem ser alterados por um arquivo YAML passado em `--config`.
#
# Escrever `--rank 16` não dá erro de argumento desconhecido em algumas versões — ele é
# simplesmente **ignorado**, e você treina com o default achando que mudou o rank. Por
# isso os labs abaixo geram o YAML explicitamente.

# %%
def escrever_config(caminho: Path, **campos) -> Path:
    """Gera um YAML de configuração do mlx_lm.lora (sem depender de pyyaml)."""
    linhas = []
    for chave, valor in campos.items():
        if isinstance(valor, dict):
            linhas.append(f"{chave}:")
            for k, v in valor.items():
                if isinstance(v, list):
                    itens = ", ".join(f'"{i}"' for i in v)
                    linhas.append(f"  {k}: [{itens}]")
                else:
                    linhas.append(f"  {k}: {v}")
        elif isinstance(valor, bool):
            linhas.append(f"{chave}: {'true' if valor else 'false'}")
        elif isinstance(valor, str):
            linhas.append(f'{chave}: "{valor}"')
        else:
            linhas.append(f"{chave}: {valor}")
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


def treinar(nome: str, **campos):
    """Monta o YAML, roda o treino e devolve (ok, log, segundos)."""
    cfg_path = escrever_config(AQUI / f"config-{nome}.yaml", **campos)
    print(f"\n--- {nome} ---")
    print(cfg_path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    ok, log = rodar("lora", "--config", str(cfg_path), mostrar=600)
    return ok, log, time.perf_counter() - t0


def test_loss(modelo: str, adaptador: Path) -> str:
    ok, log = rodar("lora", "--model", modelo, "--adapter-path", str(adaptador),
                    "--data", str(DADOS), "--test", mostrar=0)
    for linha in log.splitlines():
        if "Test loss" in linha:
            return linha.split("Test loss")[-1].strip().split()[0]
    return "erro"

# %% [markdown]
# ## Lab 1 — Quantizar e medir a economia
#
# O `lab_cpu.py` implementou NF4 do zero e mediu o erro. Aqui a economia é real: os pesos
# ficam **armazenados** em 4 bits, não desquantizados para float32.

# %%
Q4 = AQUI / "qwen15b-4bit"
if not Q4.exists():
    rodar("convert", "--hf-path", "Qwen/Qwen2.5-1.5B-Instruct", "-q",
          "--q-bits", "4", "--q-group-size", "64", "--mlx-path", str(Q4))

# %%
from mlx_lm import load

mx.reset_peak_memory()
modelo_bf16, tok = load(BASE_15B)
mem_bf16 = mx.get_peak_memory() / 1e9
print(f"bf16  — pico de memória ao carregar: {mem_bf16:.2f} GB")

del modelo_bf16
mx.clear_cache()
mx.reset_peak_memory()

modelo_q4, tok_q4 = load(str(Q4))
mem_q4 = mx.get_peak_memory() / 1e9
print(f"4-bit — pico de memória ao carregar: {mem_q4:.2f} GB")
print(f"\nrazão: {mem_bf16 / max(mem_q4, 1e-9):.2f}x   (teórico: 4x)")
print(f"em disco — bf16: ~3.1 GB | 4-bit: {tamanho_mb(Q4)/1000:.2f} GB")

# %% [markdown]
# > A razão real fica abaixo de 4× porque embeddings e `lm_head` costumam ficar em
# > precisão maior, e há overhead de constantes de escala (seção 5 do README: ~0,127
# > bit/peso com double quantization).
#
# ## Lab 2 — A degradação, medida em português
#
# O Lab 8 do `lab_cpu.py` mediu +17,4% de degradação em literatura portuguesa contra
# +4,3% em inglês, no modelo de 0,5B. Confirmando no 1.5B, com a implementação real.

# %%
import mlx.nn as nn

def perplexidade(model, tokenizer, texto: str) -> float:
    ids = tokenizer.encode(texto)
    logits = model(mx.array([ids]))[0]
    perda = nn.losses.cross_entropy(
        logits[:-1].astype(mx.float32), mx.array(ids[1:]), reduction="mean")
    return float(mx.exp(perda))

corpus = (AQUI.parent / "modulo-03-treino" / "data" / "corpus.txt")
TEXTOS = {
    "literatura PT": corpus.read_text(encoding="utf-8")[5000:9000] if corpus.exists() else None,
    "técnico PT": ("A quantização reduz a precisão numérica dos pesos de uma rede neural "
                   "para diminuir o consumo de memória. ") * 12,
    "inglês": ("The transformer architecture relies on self-attention to model dependencies "
               "between tokens regardless of distance. ") * 12,
    "código": ("def quantize(w, bits=4):\n    scale = w.abs().max() / 7\n"
               "    return (w / scale).round().clamp(-8, 7), scale\n\n") * 8,
}
TEXTOS = {k: v for k, v in TEXTOS.items() if v}

modelo_bf16, tok = load(BASE_15B)
print(f"{'texto':<16} {'PPL bf16':>10} {'PPL 4-bit':>11} {'degradação':>12}")
print("-" * 54)
degrad = []
for nome, txt in TEXTOS.items():
    p0 = perplexidade(modelo_bf16, tok, txt)
    p1 = perplexidade(modelo_q4, tok_q4, txt)
    degrad.append(p1 / p0 - 1)
    print(f"{nome:<16} {p0:>10.3f} {p1:>11.3f} {degrad[-1]:>+11.1%}")
print(f"\nmédia: {sum(degrad)/len(degrad):+.1%} | faixa: {min(degrad):+.1%} a {max(degrad):+.1%}")

del modelo_bf16, modelo_q4
mx.clear_cache()

# %% [markdown]
# ## Lab 3 — QLoRA
#
# Como o modelo passado em `--model` é quantizado, o `mlx_lm.lora` usa QLoRA
# automaticamente. Não há flag separada.

# %%
N_TREINO = sum(1 for _ in (DADOS / "train.jsonl").open(encoding="utf-8"))
ITERS, BATCH = 300, 4
print(f"exemplos {N_TREINO} | épocas = {ITERS*BATCH/N_TREINO:.1f}")
assert 1 <= ITERS * BATCH / N_TREINO <= 12

mx.reset_peak_memory()
ok, _ = rodar("lora", "--model", str(Q4), "--train",
              "--data", str(DADOS),
              "--adapter-path", str(AQUI / "adapters-qlora"),
              "--iters", str(ITERS), "--batch-size", str(BATCH),
              "--num-layers", "8", "--learning-rate", "1e-4",
              "--mask-prompt", "--max-seq-length", "1024",
              "--steps-per-eval", "50")
print(f"\npico de memória do processo pai: {mx.get_peak_memory()/1e9:.2f} GB")
print("(o treino roda em subprocesso — use o Monitor de Atividade para o número real)")

# %% [markdown]
# ## Lab 4 — Rank: qualidade, memória e tempo
#
# O `lab_cpu.py` mediu retorno decrescente do rank (r=1 → 7,7% de ganho; r=32 → 10,0%).
# Aqui o custo do outro lado da balança.

# %%
COMUM = dict(model=str(Q4), train=True, data=str(DADOS), iters=200, batch_size=4,
             num_layers=8, learning_rate=1e-4, mask_prompt=True, max_seq_length=1024,
             fine_tune_type="lora")

resultados = {}
for r in [4, 8, 16, 64]:
    destino = AQUI / f"adapters-r{r}"
    ok, log, dt = treinar(
        f"r{r}", **COMUM, adapter_path=str(destino),
        # scale = lora_alpha / r  →  mantemos alpha = 16 constante para isolar o rank
        lora_parameters={"rank": r, "scale": 16.0 / r, "dropout": 0.0,
                         "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
    )
    resultados[r] = {"tempo": dt, "mb": tamanho_mb(destino), "ok": ok}

# %%
print(f"{'rank':>6} {'adaptador':>12} {'tempo':>8} {'test loss':>12}")
print("-" * 42)
for r, d in resultados.items():
    print(f"{r:>6} {d['mb']:>10.1f}MB {d['tempo']:>7.0f}s "
          f"{test_loss(str(Q4), AQUI / f'adapters-r{r}'):>12}")

# %% [markdown]
# ## Lab 5 — Alvos: `q,v` contra todas as lineares
#
# Do módulo 2: a atenção é 12,3% dos parâmetros; o MLP, 87,7%. O default do MLX adapta
# só `q_proj` e `v_proj` — menos de 0,11% do modelo (medido no `lab_cpu.py`, Lab 6).

# %%
CONJUNTOS = {
    "q,v (default)": ["self_attn.q_proj", "self_attn.v_proj"],
    "atenção completa": ["self_attn.q_proj", "self_attn.k_proj",
                         "self_attn.v_proj", "self_attn.o_proj"],
    "todas as lineares": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                          "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
}

print(f"{'alvos':<20} {'adaptador':>12} {'tempo':>8} {'test loss':>12}")
print("-" * 56)
for nome, keys in CONJUNTOS.items():
    destino = AQUI / f"adapters-{nome.split(maxsplit=1)[0]}"
    ok, log, dt = treinar(
        nome.split(maxsplit=1)[0], **COMUM, adapter_path=str(destino),
        lora_parameters={"rank": 8, "scale": 2.0, "dropout": 0.0, "keys": keys},
    )
    print(f"{nome:<20} {tamanho_mb(destino):>10.1f}MB {dt:>7.0f}s "
          f"{test_loss(str(Q4), destino):>12}")

# %% [markdown]
# ## Lab 6 — DoRA
#
# DoRA decompõe a atualização em **magnitude** e **direção**, adaptando cada uma
# separadamente. Costuma superar LoRA em ranks baixos, com custo marginal.

# %%
rodar("lora", "--model", str(Q4), "--train", "--data", str(DADOS),
      "--adapter-path", str(AQUI / "adapters-dora"),
      "--fine-tune-type", "dora",
      "--iters", "200", "--batch-size", "4", "--num-layers", "8",
      "--learning-rate", "1e-4", "--mask-prompt", "--max-seq-length", "1024")

# %% [markdown]
# ## Lab 7 — O teto: 7B em 4 bits
#
# Segundo a tabela do README (seção 1), QLoRA do 7B pede ~4,6 GB de pesos e estados.
# Cabe nos seus ~10 GB úteis — mas ativações e KV cache também consomem.
#
# Se estourar, a ordem de ajuste é: `--batch-size 1` → `--max-seq-length` menor →
# `--grad-checkpoint` → `--num-layers` menor. Use `--grad-accumulation-steps` para
# preservar o batch efetivo.

# %%
MODELO_7B = "mlx-community/Qwen2.5-7B-Instruct-4bit"

ok, _ = rodar("lora", "--model", MODELO_7B, "--train", "--data", str(DADOS),
              "--adapter-path", str(AQUI / "adapters-7b"),
              "--iters", "100",
              "--batch-size", "1",              # o mínimo
              "--grad-accumulation-steps", "4", # batch efetivo 4
              "--num-layers", "4",              # só as 4 camadas finais
              "--grad-checkpoint",              # troca compute por memória
              "--max-seq-length", "512",
              "--learning-rate", "1e-4", "--mask-prompt", verificar=False)

if not ok:
    print("\nSe falhou por memória, tente nesta ordem:")
    print("  1. --max-seq-length 256")
    print("  2. --num-layers 2")
    print("  3. sudo sysctl iogpu.wired_limit_mb=13000   (ver 00-setup-mac.md)")
    print("  4. desista do 7B e use o 3B em 4 bits — não há vergonha nisso")

# %%
# Geração com o adaptador acoplado, para confirmar que o treino pegou.
rodar("generate", "--model", str(Q4),
      "--adapter-path", str(AQUI / "adapters-qlora"),
      "--prompt", "O computador não liga, o que eu faço?",
      "--max-tokens", "250")

# %% [markdown]
# > ⚠️ **Não funda um adaptador QLoRA e requantize.** Você treinou sobre uma base NF4
# > específica; o merge produz pesos em precisão cheia, e requantizá-los gera uma base
# > diferente da que serviu de referência no treino. Sirva com `--adapter-path` (o custo
# > é desprezível) ou estude QA-LoRA.
#
# ---
#
# ## Encerramento
#
# O `lab_cpu.py` provou os algoritmos; este aplicou-os ao seu hardware. Juntos eles
# responderam a pergunta do módulo: **7B cabe em 16 GB** — com base em 4 bits, adaptadores
# de menos de 1% do modelo, batch 1 e gradient checkpointing.
#
# No módulo 7 o assunto muda de eficiência para capacidade: o que acontece quando o
# modelo escreve o próprio raciocínio antes de responder.

# %%
