# %% [markdown]
# # Módulo 3 — Laboratório: treinando um LLM do zero
#
# Roda em **CPU**. O treino do Lab 7 leva alguns minutos.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Corpus e tokenizer BPE treinado do zero |
# | 2 | Packing vs padding: medindo o desperdício |
# | 3 | O modelo (arquitetura do módulo 2, em miniatura) |
# | 4 | AdamW por dentro e o agendamento de learning rate |
# | 5 | Gradient accumulation: prova de equivalência |
# | 6 | fp16 vs bf16: onde cada um quebra |
# | 7 | **Treinar** |
# | 8 | Ler a curva e gerar texto |
# | 9 | Calculadora de custo de pré-treino |
#
# Ao fim do Lab 8 você terá um modelo de linguagem que você treinou, do zero,
# em português.

# %%
import math
import time

import dados
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(1337)

# %% [markdown]
# ## Lab 1 — Corpus e tokenizer
#
# 750 KB de Machado de Assis. Sobre esse corpus treinamos um BPE de 2.048 tokens — o
# mesmo algoritmo do módulo 1, agora do lado de dentro.

# %%
texto = dados.carregar()
print(f"{len(texto):,} caracteres | {len(texto.split()):,} palavras")

# %%
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB = 2048

tk = Tokenizer(models.BPE(unk_token=None))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(
    [texto],
    trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<eos>"], show_progress=False),
)
EOS = tk.token_to_id("<eos>")
print(f"vocabulário: {tk.get_vocab_size()} tokens | <eos> = id {EOS}")

# %%
# O que um BPE de 2k aprende num corpus pequeno — compare com o Qwen (151k) do módulo 1.
for frase in ["Uma noite destas, vindo da cidade", "A implementação da tokenização"]:
    ids = tk.encode(frase).ids
    print(f"\n{frase!r}  ->  {len(ids)} tokens")
    print("  " + " | ".join(tk.decode([i]) for i in ids))

# %% [markdown]
# Note a diferença: no corpus de Machado, `noite` e `cidade` são tokens únicos; palavras
# técnicas que nunca apareceram se despedaçam. **O tokenizer é um retrato do corpus.**

# %%
ids_completos = torch.tensor(tk.encode(texto).ids, dtype=torch.long)
n_treino = int(0.9 * len(ids_completos))
dados_treino, dados_val = ids_completos[:n_treino], ids_completos[n_treino:]

print(f"total   : {len(ids_completos):,} tokens")
print(f"treino  : {len(dados_treino):,}")
print(f"validação: {len(dados_val):,}")
print(f"compressão: {len(texto) / len(ids_completos):.2f} caracteres por token")

# %% [markdown]
# ## Lab 2 — Packing vs padding
#
# Tratando cada parágrafo como um "documento", quanto se desperdiça com padding?

# %%
documentos = [p for p in texto.split("\n\n") if len(p.strip()) > 50]
comprimentos = [len(tk.encode(d).ids) for d in documentos[:2000]]

BLOCO = 128
com_padding = sum(math.ceil(c / BLOCO) * BLOCO for c in comprimentos)
com_packing = sum(c + 1 for c in comprimentos)          # +1 pelo <eos>
uteis = sum(comprimentos)

print(f"documentos              : {len(comprimentos):,}")
print(f"comprimento médio       : {sum(comprimentos) / len(comprimentos):.0f} tokens")
print(f"tokens úteis            : {uteis:,}")
print(f"tokens processados (padding): {com_padding:,}   desperdício = {1 - uteis / com_padding:.1%}")
print(f"tokens processados (packing): {com_packing:,}   desperdício = {1 - uteis / com_packing:.1%}")
print(f"\ncompute economizado pelo packing: {1 - com_packing / com_padding:.1%}")

# %% [markdown]
# Com documentos curtos o desperdício do padding é brutal. Em pré-treino sempre se usa
# packing — o corpus vira **um fluxo contínuo** de tokens, cortado em blocos:

# %%
def pegar_batch(fonte, batch_size, bloco):
    """Amostra posições aleatórias do fluxo. x e y deslocados de 1 — o shift do módulo 1."""
    inicio = torch.randint(len(fonte) - bloco - 1, (batch_size,))
    x = torch.stack([fonte[i: i + bloco] for i in inicio])
    y = torch.stack([fonte[i + 1: i + 1 + bloco] for i in inicio])
    return x, y

x, y = pegar_batch(dados_treino, 4, 8)
print("x[0]:", x[0].tolist())
print("y[0]:", y[0].tolist(), "  <- x deslocado em 1 posição")
print(f"\nx decodificado: {tk.decode(x[0].tolist())!r}")
print(f"y decodificado: {tk.decode(y[0].tolist())!r}")

# %% [markdown]
# ## Lab 3 — O modelo
#
# Arquitetura moderna em miniatura, exatamente as peças do módulo 2: pre-norm com RMSNorm,
# atenção causal multi-head com RoPE, MLP SwiGLU, embeddings amarrados à `lm_head`.

# %%
class Config:
    vocab = VOCAB
    d = 192
    n_camadas = 4
    n_heads = 6
    bloco = 128
    d_ff = 512
    theta = 10_000.0

    @property
    def head_dim(self):
        return self.d // self.n_heads

cfg = Config()

# %%
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.peso = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (self.peso * (x.float() * torch.rsqrt(var + self.eps))).type_as(x)


def rope_cache(bloco, head_dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(bloco).float(), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    m = x.shape[-1] // 2
    return torch.cat([-x[..., m:], x[..., :m]], dim=-1)


class Atencao(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.saida = nn.Linear(cfg.d, cfg.d, bias=False)

    def forward(self, x, cos, sin):
        b, s, d = x.shape
        h, hd = self.cfg.n_heads, self.cfg.head_dim

        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(b, s, h, hd).transpose(1, 2)
        k = k.view(b, s, h, hd).transpose(1, 2)
        v = v.view(b, s, h, hd).transpose(1, 2)

        cos, sin = cos[:s], sin[:s]
        q = q * cos + rotate_half(q) * sin          # RoPE em q e k...
        k = k * cos + rotate_half(k) * sin          # ...nunca em v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # FlashAttention
        return self.saida(y.transpose(1, 2).contiguous().view(b, s, d))


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))"""
    def __init__(self, cfg):
        super().__init__()
        self.portao = nn.Linear(cfg.d, cfg.d_ff, bias=False)
        self.acima = nn.Linear(cfg.d, cfg.d_ff, bias=False)
        self.abaixo = nn.Linear(cfg.d_ff, cfg.d, bias=False)

    def forward(self, x):
        return self.abaixo(F.silu(self.portao(x)) * self.acima(x))


class Bloco(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.d), RMSNorm(cfg.d)
        self.attn, self.mlp = Atencao(cfg), MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)   # pre-norm + residual
        return x + self.mlp(self.norm2(x))


class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocos = nn.ModuleList(Bloco(cfg) for _ in range(cfg.n_camadas))
        self.norm_final = RMSNorm(cfg.d)
        self.cabeca = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self.cabeca.weight = self.emb.weight          # weight tying (módulo 1)

        cos, sin = rope_cache(cfg.bloco, cfg.head_dim, cfg.theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, alvos=None):
        x = self.emb(idx)
        for bloco in self.blocos:
            x = bloco(x, self.cos, self.sin)
        logits = self.cabeca(self.norm_final(x))

        if alvos is None:
            return logits, None
        # Os alvos JÁ vêm deslocados (Lab 2), então não há shift aqui.
        perda = F.cross_entropy(logits.view(-1, logits.size(-1)), alvos.reshape(-1))
        return logits, perda

    @torch.no_grad()
    def gerar(self, idx, n_novos, temperatura=0.8, top_k=40):
        for _ in range(n_novos):
            contexto = idx[:, -self.cfg.bloco:]
            logits, _ = self(contexto)
            logits = logits[:, -1, :] / temperatura
            if top_k:
                corte = torch.topk(logits, top_k).values[:, -1:]
                logits = logits.masked_fill(logits < corte, float("-inf"))
            proximo = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, proximo], dim=1)
        return idx

# %%
modelo = MiniGPT(cfg)
n_params = sum(p.numel() for p in modelo.parameters())
n_emb = modelo.emb.weight.numel()

print(f"parâmetros totais : {n_params:>10,}")
print(f"  embeddings      : {n_emb:>10,}  ({n_emb / n_params:.1%})  — amarrados à lm_head")
print(f"  blocos          : {n_params - n_emb:>10,}")
print(f"\nloss esperada antes do treino: ln({VOCAB}) = {math.log(VOCAB):.3f}")

x, y = pegar_batch(dados_treino, 8, cfg.bloco)
_, perda_inicial = modelo(x, y)
print(f"loss medida no primeiro batch: {perda_inicial.item():.3f}   ← o teste de sanidade do módulo 1")

# %% [markdown]
# ## Lab 4 — AdamW e o agendamento de learning rate
#
# Primeiro o AdamW implementado do zero, comparado ao do PyTorch.

# %%
def adamw_passo(p, g, m, v, t, lr, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)                 # correção de viés
    v_hat = v / (1 - b2 ** t)
    p = p - lr * m_hat / (v_hat.sqrt() + eps) # passo adaptativo
    p = p - lr * wd * p                       # weight decay DESACOPLADO (o "W")
    return p, m, v

p_manual = torch.tensor([1.0, -2.0, 0.5])
m, v = torch.zeros(3), torch.zeros(3)

p_torch = p_manual.clone().requires_grad_(True)
otim = torch.optim.AdamW([p_torch], lr=0.1, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

for t in range(1, 4):
    g = torch.tensor([0.3, -0.1, 0.05]) * t
    p_manual, m, v = adamw_passo(p_manual, g, m, v, t, lr=0.1)
    p_torch.grad = g.clone()
    otim.step()
    otim.zero_grad()

print("manual :", [f"{x:+.6f}" for x in p_manual])
print("pytorch:", [f"{x:+.6f}" for x in p_torch.detach()])
print("iguais? ", torch.allclose(p_manual, p_torch.detach(), atol=1e-6))

# %%
# Os estados do otimizador — a origem dos 16 bytes por parâmetro.
bytes_por_param = {"pesos bf16": 2, "gradientes bf16": 2, "master fp32": 4, "Adam m": 4, "Adam v": 4}
total_bytes = sum(bytes_por_param.values())
for nome, b in bytes_por_param.items():
    print(f"  {nome:<18} {b:>2} bytes/param")
print(f"  {'TOTAL':<18} {total_bytes:>2} bytes/param")
for nome, n in [("nosso MiniGPT", n_params), ("Qwen2.5-0.5B", 494e6), ("Llama-3-8B", 8.03e9)]:
    print(f"    {nome:<16} full fine-tune = {n * total_bytes / 1e9:>8.2f} GB de estados")

# %%
# Agendamento: warmup linear + cosine decay.
def agenda_lr(passo, total, pico=1e-3, warmup=None, minimo_frac=0.1):
    warmup = warmup if warmup is not None else max(1, total // 20)
    if passo < warmup:
        return pico * (passo + 1) / warmup
    progresso = (passo - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * progresso))
    return pico * (minimo_frac + (1 - minimo_frac) * cos)

TOTAL_PASSOS = 400
curva = [agenda_lr(p, TOTAL_PASSOS) for p in range(TOTAL_PASSOS)]

def grafico(valores, altura=12, largura=64, titulo="", log_y=False):
    import math as _m
    vals = [_m.log(v) if log_y and v > 0 else v for v in valores]
    passo = max(1, len(vals) // largura)
    amostra = vals[::passo][:largura]
    lo, hi = min(amostra), max(amostra)
    faixa = (hi - lo) or 1
    print(f"\n{titulo}")
    for linha in range(altura, -1, -1):
        limiar = lo + faixa * linha / altura
        rotulo = (math.exp(limiar) if log_y else limiar)
        print(f"  {rotulo:>8.4f} |" + "".join("█" if v >= limiar else " " for v in amostra))
    print(f"  {'':>8} +" + "-" * len(amostra))

grafico(curva, titulo="learning rate: warmup (5%) + cosine decay até 10% do pico")

# %% [markdown]
# ## Lab 5 — Gradient accumulation
#
# A afirmação: acumular `N` micro-batches equivale a um batch `N×` maior. Provando.

# %%
def gradiente_de(modelo_ref, batches, acumular):
    m = MiniGPT(cfg)
    m.load_state_dict(modelo_ref.state_dict())
    m.zero_grad()
    if acumular:
        for xb, yb in batches:
            _, perda = m(xb, yb)
            (perda / len(batches)).backward()        # ← a divisão obrigatória
    else:
        xg = torch.cat([b[0] for b in batches])
        yg = torch.cat([b[1] for b in batches])
        _, perda = m(xg, yg)
        perda.backward()
    return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])

batches = [pegar_batch(dados_treino, 4, cfg.bloco) for _ in range(4)]
g_acumulado = gradiente_de(modelo, batches, acumular=True)
g_batch_unico = gradiente_de(modelo, batches, acumular=False)

print(f"batch único de 16     : norma = {g_batch_unico.norm():.6f}")
print(f"4 micro-batches de 4  : norma = {g_acumulado.norm():.6f}")
print(f"diferença máxima      : {(g_acumulado - g_batch_unico).abs().max():.3e}")
print(f"equivalentes?           {torch.allclose(g_acumulado, g_batch_unico, atol=1e-5)}")

# %%
# E o erro de esquecer a divisão:
m = MiniGPT(cfg); m.load_state_dict(modelo.state_dict()); m.zero_grad()
for xb, yb in batches:
    _, perda = m(xb, yb)
    perda.backward()                                  # SEM dividir
g_errado = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])

print(f"sem a divisão         : norma = {g_errado.norm():.6f}")
print(f"razão                 : {g_errado.norm() / g_batch_unico.norm():.2f}x")
print("→ equivale a multiplicar o learning rate por 4 sem perceber.")

# %% [markdown]
# ## Lab 6 — fp16 vs bf16
#
# Mesma quantidade de bits, distribuição diferente entre expoente e mantissa.

# %%
print(f"{'valor':>14} {'fp16':>14} {'bf16':>14}  situação")
print("-" * 62)
casos = [
    (70000.0, "overflow em fp16 (máx 65.504)"),
    (1e-8, "underflow em fp16"),
    (1e-30, "underflow em fp16, ok em bf16"),
    (3.14159265, "precisão: bf16 é mais grosseiro"),
    (1.0009765625, "precisão de mantissa"),
]
for valor, nota in casos:
    f16 = torch.tensor(valor, dtype=torch.float16).item()
    b16 = torch.tensor(valor, dtype=torch.bfloat16).item()
    print(f"{valor:>14.8g} {f16:>14.8g} {b16:>14.8g}  {nota}")

print(f"\nfp16: expoente 5 bits, mantissa 10 -> faixa ±{torch.finfo(torch.float16).max:,.0f}")
print(f"bf16: expoente 8 bits, mantissa  7 -> faixa ±{torch.finfo(torch.bfloat16).max:.3g} (igual ao fp32)")

# %% [markdown]
# É por isso que fp16 exige *loss scaling* (multiplicar a loss por ~2¹⁶ para tirar os
# gradientes da zona de underflow) e bf16 não. bf16 troca precisão — que os master weights
# em fp32 recuperam — por faixa dinâmica, que nada recupera.
#
# ## Lab 7 — Treinar
#
# Todos os componentes juntos. Alguns minutos em CPU.

# %%
def grupos_de_parametros(modelo, weight_decay=0.1):
    """Weight decay NÃO se aplica a normas e biases (tensores de 1 dimensão)."""
    com_decay = [p for p in modelo.parameters() if p.dim() >= 2]
    sem_decay = [p for p in modelo.parameters() if p.dim() < 2]
    print(f"  com weight decay: {sum(p.numel() for p in com_decay):,} params em {len(com_decay)} tensores")
    print(f"  sem weight decay: {sum(p.numel() for p in sem_decay):,} params em {len(sem_decay)} tensores")
    return [
        {"params": com_decay, "weight_decay": weight_decay},
        {"params": sem_decay, "weight_decay": 0.0},
    ]

@torch.no_grad()
def avaliar(modelo, fonte, n=20, batch=8):
    modelo.eval()
    perdas = [modelo(*pegar_batch(fonte, batch, cfg.bloco))[1].item() for _ in range(n)]
    modelo.train()
    return sum(perdas) / len(perdas)

# %%
BATCH, MICRO, LR_PICO = 16, 4, 1e-3
ACUMULACAO = BATCH // MICRO

modelo = MiniGPT(cfg)
otim = torch.optim.AdamW(grupos_de_parametros(modelo), lr=LR_PICO, betas=(0.9, 0.95), eps=1e-8)

historico = {"passo": [], "treino": [], "val": [], "lr": [], "norma_grad": []}
inicio = time.perf_counter()

for passo in range(TOTAL_PASSOS):
    lr = agenda_lr(passo, TOTAL_PASSOS, pico=LR_PICO)
    for g in otim.param_groups:
        g["lr"] = lr

    otim.zero_grad(set_to_none=True)
    perda_total = 0.0
    for _ in range(ACUMULACAO):
        xb, yb = pegar_batch(dados_treino, MICRO, cfg.bloco)
        _, perda = modelo(xb, yb)
        (perda / ACUMULACAO).backward()
        perda_total += perda.item() / ACUMULACAO

    norma = torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
    otim.step()

    if passo % 25 == 0 or passo == TOTAL_PASSOS - 1:
        val = avaliar(modelo, dados_val)
        historico["passo"].append(passo)
        historico["treino"].append(perda_total)
        historico["val"].append(val)
        historico["lr"].append(lr)
        historico["norma_grad"].append(float(norma))
        decorrido = time.perf_counter() - inicio
        print(f"passo {passo:>4} | treino {perda_total:.4f} | val {val:.4f} "
              f"| lr {lr:.2e} | |g| {float(norma):5.2f} | {decorrido:5.1f}s")

tokens_vistos = TOTAL_PASSOS * BATCH * cfg.bloco
print(f"\n{time.perf_counter() - inicio:.1f}s | {tokens_vistos:,} tokens "
      f"({tokens_vistos / len(dados_treino):.1f} épocas)")

# %% [markdown]
# ## Lab 8 — Lendo a curva e gerando texto

# %%
grafico(historico["treino"], titulo="loss de treino (eixo y linear)")
print(f"\n  loss inicial : {historico['treino'][0]:.3f}   (ln({VOCAB}) = {math.log(VOCAB):.3f} = modelo aleatório)")
print(f"  loss final   : {historico['treino'][-1]:.3f}")
print(f"  perplexidade : {math.exp(historico['treino'][0]):>8.1f} -> {math.exp(historico['treino'][-1]):.1f}")

# %%
print(f"{'passo':>7} {'treino':>9} {'validação':>11} {'gap':>8} {'|grad|':>8}")
print("-" * 48)
for i in range(len(historico["passo"])):
    gap = historico["val"][i] - historico["treino"][i]
    print(f"{historico['passo'][i]:>7} {historico['treino'][i]:>9.4f} "
          f"{historico['val'][i]:>11.4f} {gap:>+8.4f} {historico['norma_grad'][i]:>8.2f}")

# %% [markdown]
# **Como ler esta tabela.** O `gap` entre validação e treino é o termômetro de
# overfitting: enquanto ele fica pequeno e estável, o modelo generaliza; quando a
# validação sobe e o treino continua caindo, o modelo passou a memorizar. Com 750 KB de
# corpus e várias épocas, é esperado ver o gap crescer — pré-treinos reais fazem
# aproximadamente **uma** época sobre trilhões de tokens, e nunca chegam nesse regime.
#
# A norma do gradiente mostra por que o clipping importa: ela é grande no começo e
# assenta conforme o treino avança.

# %%
contexto = torch.tensor([tk.encode("Uma noite destas").ids])
saida = modelo.gerar(contexto, n_novos=200, temperatura=0.8, top_k=40)
print(tk.decode(saida[0].tolist()))

# %%
# Comparando com um modelo NÃO treinado, para calibrar o que o treino conseguiu:
print("=== modelo aleatório (sem treino) ===")
print(tk.decode(MiniGPT(cfg).gerar(contexto, 80, temperatura=0.8, top_k=40)[0].tolist()))
print("\n=== modelo treinado ===")
print(tk.decode(modelo.gerar(contexto, 80, temperatura=0.8, top_k=40)[0].tolist()))

# %% [markdown]
# Com ~2M de parâmetros e 750 KB de corpus, não espere coerência. Espere **morfologia
# portuguesa plausível, pontuação no lugar e concordância local** — o que já é enorme
# para um modelo que começou em ruído há três minutos, e é exatamente a fase 1 e 2 da
# curva descrita no README. As fases 3 e 4 exigem as outras 14 ordens de grandeza.
#
# ## Lab 9 — Calculadora de custo

# %%
# TFLOP/s em bf16/fp16, tensor cores, DENSOS.
# ⚠️ Fichas técnicas de GPU costumam anunciar o número COM sparsity 2:4, que é o dobro e
# não se aplica a treino de LLM. A H100 SXM aparece como "989 TFLOPS bf16" no material da
# NVIDIA; o valor denso é 495. Usar o número errado subestima o custo pela metade.
GPUS = {
    "T4 (Colab grátis)": 65,
    "L4": 121,
    "A100 40GB": 312,
    "H100 SXM": 495,
}

def orcamento(n_params, n_tokens, gpu="A100 40GB", mfu=0.40, preco_hora=1.50):
    flops = 6 * n_params * n_tokens
    flops_s = GPUS[gpu] * 1e12 * mfu
    segundos = flops / flops_s
    horas = segundos / 3600
    return {"PFLOPs": flops / 1e15, "GPU-horas": horas, "dias (1 GPU)": horas / 24,
            "US$": horas * preco_hora}

print(f"{'modelo':<22} {'PFLOPs':>12} {'GPU-horas':>13} {'dias':>9} {'US$':>14}")
print("-" * 74)
for nome, n, d in [
    ("nosso MiniGPT", n_params, tokens_vistos),
    ("GPT-2 (124M)", 124e6, 300e9),
    ("Qwen2.5-0.5B", 494e6, 18e12),
    ("Llama-3-8B", 8.03e9, 15e12),
    ("Llama-3-70B", 70e9, 15e12),
]:
    r = orcamento(n, d)
    print(f"{nome:<22} {r['PFLOPs']:>12,.1f} {r['GPU-horas']:>13,.0f} "
          f"{r['dias (1 GPU)']:>9,.0f} {r['US$']:>14,.0f}")

# %% [markdown]
# ### Validação contra um número publicado
#
# A Meta reportou **1,3 milhão de H100-horas** para pré-treinar o Llama-3-8B. Vamos ver
# quão perto a fórmula `6ND` chega:

# %%
for gpu in GPUS:
    r = orcamento(8.03e9, 15e12, gpu=gpu)
    print(f"Llama-3-8B numa {gpu:<20} {r['GPU-horas']:>12,.0f} GPU-horas "
          f"= {r['dias (1 GPU)'] / 365:>6.1f} anos numa única GPU")

estimado = orcamento(8.03e9, 15e12, gpu="H100 SXM")["GPU-horas"]
reportado = 1_300_000
print(f"\nestimado (MFU 40%) : {estimado:>10,.0f} H100-horas")
print(f"reportado pela Meta: {reportado:>10,.0f} H100-horas")
print(f"razão              : {reportado / estimado:>10.2f}x")
print(f"MFU implícito real : {0.40 * estimado / reportado:>10.1%}")

# %% [markdown]
# A estimativa fica dentro de ~30% do número real, e a diferença tem explicação: o MFU
# implícito da Meta foi de aproximadamente 31%, não os 40% que assumimos. A fórmula `6ND`
# é grosseira e perfeitamente suficiente para decidir orçamento — desde que você use os
# TFLOPs **densos** e um MFU honesto.

# %% [markdown]
# ---
#
# ## Encerramento
#
# Você acabou de:
#
# - treinar um tokenizer BPE e um modelo de linguagem **do zero**, em português;
# - medir o desperdício do padding contra o packing;
# - reimplementar o AdamW e conferir contra o PyTorch;
# - **provar** a equivalência do gradient accumulation e medir o estrago de esquecer a divisão;
# - ver onde fp16 quebra e bf16 não;
# - ler uma curva de loss com gap de validação e norma de gradiente;
# - orçar o pré-treino de modelos reais e validar a fórmula contra números publicados.
#
# O modelo que você treinou é ruim porque tem 2M de parâmetros e viu 750 KB de texto.
# Tudo o mais — arquitetura, otimizador, agendamento, precisão — é **idêntico** ao que
# treina modelos de fronteira. A diferença é escala, e a escala custa milhões.
#
# É por isso que o resto do curso trata de **partir de um modelo pronto**.
#
# Agora o `exercicios.md`.

# %%
