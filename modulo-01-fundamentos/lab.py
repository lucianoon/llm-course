# %% [markdown]
# # Módulo 1 — Laboratório: Fundamentos de LLMs
#
# Tudo aqui roda em **CPU**. O primeiro download baixa ~1,5 GB de modelos (`gpt2` e
# `Qwen2.5-0.5B-Instruct`) e fica em cache.
#
# Leia o `README.md` do módulo antes. O código segue a mesma ordem da teoria.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Tokenização: português vs inglês, e o custo real |
# | 2 | Anatomia do modelo: parâmetros, VRAM, KV cache |
# | 3 | Forward pass e logits |
# | 4 | Temperatura, na prática |
# | 5 | Implementar sampling do zero |
# | 6 | Loop de geração manual vs `.generate()` |
# | 7 | Perplexidade e o shift de labels |
# | 8 | Base vs instruct e o chat template |
# | 9 | KV cache: o custo de desligá-lo |

# %%
import math
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)
torch.set_grad_enabled(False)  # nada aqui precisa de gradiente

GPT2 = "openai-community/gpt2"
REVISAO_GPT2 = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
REVISAO_QWEN = "7ae557604adf67be50417f59c2c2f167def9a775"

print("torch", torch.__version__, "| device: cpu")

# %% [markdown]
# ## Lab 1 — Tokenização
#
# Primeiro a pergunta que decide custo de API e uso de contexto: **quanto o português
# custa a mais que o inglês?**

# %%
tok_gpt2 = AutoTokenizer.from_pretrained(GPT2, revision=REVISAO_GPT2)
tok_qwen = AutoTokenizer.from_pretrained(QWEN, revision=REVISAO_QWEN)

for name, tk in [("gpt2", tok_gpt2), ("qwen2.5", tok_qwen)]:
    print(f"{name:10} vocabulário = {tk.vocab_size:,}")

# %%
pares = [
    ("The implementation is straightforward", "A implementação é direta"),
    ("Machine learning requires careful evaluation", "Aprendizado de máquina exige avaliação cuidadosa"),
    ("The organization's conclusion was unnecessary", "A conclusão da organização era desnecessária"),
]

print(f"{'texto':52} {'gpt2':>6} {'qwen':>6} {'chars/tok gpt2':>16} {'chars/tok qwen':>16}")
print("-" * 100)
for en, pt in pares:
    for texto in (en, pt):
        n_gpt2 = len(tok_gpt2.encode(texto))
        n_qwen = len(tok_qwen.encode(texto))
        print(
            f"{texto[:50]:52} {n_gpt2:>6} {n_qwen:>6}"
            f" {len(texto) / n_gpt2:>16.2f} {len(texto) / n_qwen:>16.2f}"
        )
    print()

# %% [markdown]
# Repare: o mesmo conteúdo em português consome mais tokens, e a diferença é **muito maior
# no GPT-2** (tokenizer treinado quase só em inglês, vocabulário de 50k) do que no Qwen2.5
# (vocabulário de 151k, corpus multilíngue).
#
# Agora veja *onde* a quebra acontece.

# %%
def mostrar_tokens(tk, texto, nome):
    ids = tk.encode(texto)
    tokens = tk.convert_ids_to_tokens(ids)
    print(f"\n{nome}: {len(ids)} tokens")
    print("  " + " | ".join(t.replace("Ġ", "␣").replace("Ċ", "\\n") for t in tokens))

texto = "A implementação da tokenização é fundamental."
mostrar_tokens(tok_gpt2, texto, "gpt2")
mostrar_tokens(tok_qwen, texto, "qwen2.5")

# %% [markdown]
# Duas coisas a notar na saída acima:
#
# **1. `␣` marca o espaço que pertence ao token seguinte** (byte-level BPE). Por isso
# `"gato"` e `" gato"` são tokens **diferentes** — a armadilha do prompt terminado em espaço.
#
# **2. `Ã§Ã£o` não é bug.** É `ção` visto byte a byte: o BPE opera sobre **bytes UTF-8**,
# e `ç` ocupa dois bytes (`0xC3 0xA7`), que são renderizados como `Ã§` na tabela de
# símbolos do tokenizer. Compare a segmentação de `implementação`:
#
# - GPT-2: `implement` + `a` + `Ã§` + `Ã£o` → **4 pedaços**, e dois deles são meio-caractere
# - Qwen2.5: `implement` + `aÃ§Ã£o` → **2 pedaços**
#
# Todo caractere acentuado do português consome 2 bytes, e um tokenizer que não viu
# português no treino não aprendeu a fundi-los. É essa a origem mecânica do custo extra.

# %%
for variante in ["gato", " gato", "Gato", " Gato"]:
    ids = tok_qwen.encode(variante)
    print(f"{variante!r:10} -> ids={ids}  tokens={tok_qwen.convert_ids_to_tokens(ids)}")

# %% [markdown]
# E os números — a razão pela qual LLMs erram aritmética:

# %%
for numero in ["7", "42", "1234", "12345", "3.14159", "1000000"]:
    a = tok_gpt2.convert_ids_to_tokens(tok_gpt2.encode(numero))
    b = tok_qwen.convert_ids_to_tokens(tok_qwen.encode(numero))
    print(f"{numero:>10}   gpt2={a}   qwen={b}")

# %% [markdown]
# ## Lab 2 — Anatomia do modelo
#
# Carregando o modelo e lendo dele os números que decidem hardware.

# %%
model = AutoModelForCausalLM.from_pretrained(
    QWEN, revision=REVISAO_QWEN, torch_dtype=torch.float32
)
model.eval()
cfg = model.config

print(cfg.__class__.__name__)
for campo in [
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "intermediate_size", "vocab_size",
    "max_position_embeddings", "tie_word_embeddings",
]:
    print(f"  {campo:26} = {getattr(cfg, campo, '—')}")

# %%
total = sum(p.numel() for p in model.parameters())
emb = model.get_input_embeddings().weight.numel()

print(f"parâmetros totais       : {total:>14,}")
print(f"matriz de embeddings    : {emb:>14,}  ({emb / total:.1%} do total)")
print(f"resto (blocos + head)   : {total - emb:>14,}")
print()
print(f"weight tying ativo?     : {cfg.tie_word_embeddings}")
print(f"  lm_head é o mesmo tensor da embedding? "
      f"{model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()}")

# %% [markdown]
# ### Estimativas de memória
#
# As mesmas fórmulas da seção 8 e 10 do README, agora como função. Use-as antes de
# alugar qualquer GPU.

# %%
def estimativas(cfg, n_params, seq_len=8192, batch=1):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    kv_por_token = 2 * cfg.num_hidden_layers * kv_heads * head_dim * 2  # bf16
    return {
        "inferência bf16 (GB)": n_params * 2 / 1e9,
        "inferência 4-bit (GB)": n_params * 0.5 / 1e9,
        "full fine-tune AdamW (GB)": n_params * 16 / 1e9,
        "LoRA sobre base bf16 (GB)": n_params * 2 * 1.05 / 1e9,
        "QLoRA (GB)": n_params * 0.5 * 1.2 / 1e9,
        "KV cache por token (KB)": kv_por_token / 1024,
        f"KV cache {seq_len}tok x{batch} (GB)": kv_por_token * seq_len * batch / 1e9,
    }

for k, v in estimativas(cfg, total).items():
    print(f"  {k:32} {v:>10.2f}")

# %% [markdown]
# Agora rode a mesma conta para um modelo que você **não** tem localmente, só pelo config.
# Isso é o que você faz antes de decidir uma GPU.

# %%
class ConfigFake:
    """Llama-3-8B, valores do config.json oficial."""
    hidden_size = 4096
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8   # GQA: 4x menos KV heads

print("Llama-3-8B (GQA, 8 kv heads):")
for k, v in estimativas(ConfigFake(), 8_030_000_000).items():
    print(f"  {k:32} {v:>10.2f}")

ConfigFake.num_key_value_heads = 32   # e se não houvesse GQA?
print("\nMesmo modelo SEM GQA (32 kv heads):")
for k, v in estimativas(ConfigFake(), 8_030_000_000).items():
    if "KV" in k:
        print(f"  {k:32} {v:>10.2f}")

# %% [markdown]
# 4× mais memória de KV cache. **É por isso que GQA existe** — decisão de arquitetura
# tomada por custo de inferência, não por qualidade.
#
# ## Lab 3 — Forward pass e logits

# %%
prompt = "A capital da França é"
ids = tok_qwen(prompt, return_tensors="pt")

out = model(**ids)
logits = out.logits

print(f"input_ids  : {ids['input_ids'].shape}  -> {ids['input_ids'].tolist()[0]}")
print(f"logits     : {logits.shape}   # [batch, seq, vocab]")
print(f"faixa      : min={logits.min():.2f}  max={logits.max():.2f}  média={logits.mean():.2f}")

# %% [markdown]
# Há um vetor de logits **por posição** — o modelo prevê o próximo token em toda posição
# simultaneamente. Para gerar, só a última interessa.

# %%
ultimo = logits[0, -1]                 # [vocab]
probs = F.softmax(ultimo, dim=-1)
top = torch.topk(probs, 10)

print(f"contexto: {prompt!r}\n")
for p, i in zip(top.values, top.indices):
    print(f"  {p.item():>7.2%}  {tok_qwen.decode([int(i)]).replace(' ', '␣')!r}")

print(f"\nmassa dos 10 primeiros: {top.values.sum():.2%}")
print(f"tokens restantes no vocabulário: {len(probs) - 10:,}")

# %% [markdown]
# ## Lab 4 — Temperatura
#
# A mesma distribuição, reescalada. Repare no colapso e no achatamento.

# %%
for T in [0.1, 0.5, 1.0, 1.5, 3.0]:
    p = F.softmax(ultimo / T, dim=-1)
    top5 = torch.topk(p, 5)
    entropia = -(p * torch.log(p + 1e-12)).sum()
    perplexidade_local = math.exp(entropia)
    linha = "  ".join(f"{v:.1%}" for v in top5.values)
    print(f"T={T:<4} top5=[{linha}]  entropia={entropia:.2f}  n_efetivo={perplexidade_local:>8.1f}")

# %% [markdown]
# `n_efetivo` é entre quantos tokens o modelo está de fato hesitando. Com `T=0.1` ele está
# praticamente decidido; com `T=3.0` está escolhendo entre milhares.
#
# ## Lab 5 — Sampling do zero
#
# Implementando greedy, top-k e top-p sem usar nada da biblioteca. Se você entende estas
# 20 linhas, entende todo o `generate()`.

# %%
def amostrar(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Recebe logits [vocab] e devolve um id de token. Ordem: temperatura -> top_k -> top_p."""
    if temperature == 0:                       # greedy é o caso limite
        return int(logits.argmax())

    logits = logits / temperature

    if top_k > 0:
        corte = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < corte, float("-inf"))

    if top_p < 1.0:
        ordenados, indices = torch.sort(logits, descending=True)
        acumulado = torch.cumsum(F.softmax(ordenados, dim=-1), dim=-1)
        # mantém o núcleo: todos até cruzar p, inclusive o que cruza
        remover = acumulado - F.softmax(ordenados, dim=-1) > top_p
        logits[indices[remover]] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))

# %%
# Quantos tokens sobrevivem a cada estratégia, na MESMA distribuição?
def tamanho_do_nucleo(logits, top_p):
    p = torch.sort(F.softmax(logits, dim=-1), descending=True).values
    return int((torch.cumsum(p, 0) < top_p).sum() + 1)

certo = model(**tok_qwen("A capital da França é", return_tensors="pt")).logits[0, -1]
incerto = model(**tok_qwen("Ontem eu decidi que", return_tensors="pt")).logits[0, -1]

for nome, lg in [("modelo CERTO ", certo), ("modelo INCERTO", incerto)]:
    print(f"{nome}: top_p=0.90 mantém {tamanho_do_nucleo(lg, 0.90):>5} tokens"
          f"   |  top_k=50 mantém sempre 50")

# %% [markdown]
# Aí está a razão de o top-p ter substituído o top-k: o núcleo **se adapta** à confiança
# do modelo. Onde ele está certo, top-k=50 arrasta dezenas de candidatos absurdos junto.
#
# ## Lab 6 — Loop de geração manual

# %%
def gerar(prompt, max_novos=40, **kw):
    ids = tok_qwen(prompt, return_tensors="pt")["input_ids"]
    for _ in range(max_novos):
        logits = model(ids).logits[0, -1]
        proximo = amostrar(logits, **kw)
        if proximo == tok_qwen.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[proximo]])], dim=1)
    return tok_qwen.decode(ids[0])

p = "Uma boa definição de aprendizado de máquina é"

torch.manual_seed(0)
print("GREEDY (T=0):\n ", gerar(p, temperature=0), "\n")
torch.manual_seed(0)
print("T=0.7, top_p=0.9:\n ", gerar(p, temperature=0.7, top_p=0.9), "\n")
torch.manual_seed(0)
print("T=2.0 (caos):\n ", gerar(p, temperature=2.0))

# %% [markdown]
# Compare com a implementação da biblioteca — mesmos parâmetros, mesmo comportamento:

# %%
saida = model.generate(
    **tok_qwen(p, return_tensors="pt"),
    max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.9,
    pad_token_id=tok_qwen.eos_token_id,
)
print(tok_qwen.decode(saida[0]))

# %% [markdown]
# ## Lab 7 — Perplexidade e o shift de labels
#
# Primeiro do jeito certo, na mão. Depois confirmando contra o HuggingFace.

# %%
def perplexidade(texto, modelo, tk):
    ids = tk(texto, return_tensors="pt")["input_ids"]
    logits = modelo(ids).logits

    # a posição t prevê o token t+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return math.exp(loss.item()), loss.item()

textos = {
    "português natural": "O Brasil é um país da América do Sul com uma população de mais de duzentos milhões de habitantes.",
    "inglês natural": "Brazil is a country in South America with a population of more than two hundred million people.",
    "texto embaralhado": "de habitantes população milhões país O um duzentos com Brasil mais é uma de.",
    "aleatório": "xkqp zvbn wrtl mgha jufd cyie",
}

for nome, txt in textos.items():
    ppl, loss = perplexidade(txt, model, tok_qwen)
    print(f"{nome:20} loss={loss:6.3f}   PPL={ppl:>12,.1f}")

# %% [markdown]
# Perplexidade mede o quanto o texto **surpreende** o modelo. Texto natural surpreende
# pouco; texto embaralhado, muito; ruído, absurdamente.
#
# Agora a verificação: o HuggingFace faz o shift internamente quando você passa `labels`.

# %%
txt = textos["português natural"]
ids = tok_qwen(txt, return_tensors="pt")["input_ids"]

loss_hf = model(ids, labels=ids).loss.item()
_, loss_manual = perplexidade(txt, model, tok_qwen)

print(f"loss HF (labels=input_ids) : {loss_hf:.6f}")
print(f"loss manual (shift na mão) : {loss_manual:.6f}")
print(f"iguais? {abs(loss_hf - loss_manual) < 1e-4}")

# %% [markdown]
# ### O erro do shift
#
# Veja o tamanho do estrago quando você esquece de deslocar:

# %%
logits = model(ids).logits
V = logits.size(-1)

errado = F.cross_entropy(logits.view(-1, V), ids.view(-1)).item()
duplo = F.cross_entropy(logits[:, :-2].reshape(-1, V), ids[:, 2:].reshape(-1)).item()

print(f"correto (shift 1)      : loss={loss_manual:6.3f}  PPL={math.exp(loss_manual):>10,.1f}")
print(f"sem shift              : loss={errado:6.3f}  PPL={math.exp(errado):>10,.1f}")
print(f"shift dobrado          : loss={duplo:6.3f}  PPL={math.exp(duplo):>10,.1f}")

# %% [markdown]
# Nenhum dos três dá erro de execução. Um treino com o shift errado **roda até o fim** e
# entrega um modelo inútil. É por isso que se imprime a loss do primeiro batch e se
# compara com `ln(vocab_size)` — o valor esperado de um modelo não treinado:

# %%
print(f"loss esperada de um modelo aleatório: ln({cfg.vocab_size:,}) = {math.log(cfg.vocab_size):.2f}")
print("Se sua loss inicial de fine-tuning começar muito acima disso, algo está errado no pipeline.")

# %% [markdown]
# ## Lab 8 — Base vs instruct e o chat template

# %%
mensagens = [
    {"role": "system", "content": "Você é um assistente objetivo."},
    {"role": "user", "content": "Explique o que é overfitting em uma frase."},
]

formatado = tok_qwen.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
print(repr(formatado))

# %% [markdown]
# **Sempre imprima essa string antes de um treino longo.** Um template errado não gera
# exceção — só degrada o resultado silenciosamente.
#
# A comparação que importa: mesma pergunta, com e sem template.

# %%
def responder(texto_bruto, max_novos=60):
    ids = tok_qwen(texto_bruto, return_tensors="pt")
    saida = model.generate(**ids, max_new_tokens=max_novos, do_sample=False,
                           pad_token_id=tok_qwen.eos_token_id)
    return tok_qwen.decode(saida[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

pergunta = "Explique o que é overfitting em uma frase."

print("=== SEM template (modelo instruct tratado como base) ===")
print(responder(pergunta))
print("\n=== COM template ===")
print(responder(formatado))

# %% [markdown]
# E sem `add_generation_prompt=True` — o modelo não sabe que é a vez dele:

# %%
sem_prompt_geracao = tok_qwen.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=False)
print(repr(sem_prompt_geracao[-80:]))
print("\nresposta:", responder(sem_prompt_geracao))

# %% [markdown]
# ## Lab 9 — KV cache
#
# O cache não muda o resultado; muda o custo. Medindo a diferença:

# %%
entrada = tok_qwen("A história da inteligência artificial começa", return_tensors="pt")

for usar_cache in [True, False]:
    inicio = time.perf_counter()
    model.generate(**entrada, max_new_tokens=60, do_sample=False,
                   use_cache=usar_cache, pad_token_id=tok_qwen.eos_token_id)
    dt = time.perf_counter() - inicio
    print(f"use_cache={usar_cache!s:5}  {dt:6.2f}s  ({60 / dt:5.1f} tok/s)")

# %% [markdown]
# Sem cache, cada novo token reprocessa a sequência inteira do zero — trabalho O(n²)
# desperdiçado. Em contextos longos a diferença passa de uma ordem de grandeza.
#
# ---
#
# ## Encerramento
#
# Você acabou de:
#
# - medir empiricamente o custo do português em tokens;
# - estimar VRAM de treino e KV cache a partir de um `config.json`;
# - reimplementar o `generate()` em 20 linhas;
# - calcular perplexidade com o shift correto e ver os três jeitos de errar;
# - observar o efeito do chat template, a armadilha mais cara do curso.
#
# Agora faça o `exercicios.md` **sem consultar este arquivo**.

# %%
