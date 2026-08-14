# %% [markdown]
# # Módulo 2 — Laboratório: Transformers, Attention e QKV
#
# Roda em **CPU**. Usa o `Qwen/Qwen2.5-0.5B-Instruct` já baixado no módulo 1.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Scaled dot-product attention do zero |
# | 2 | Por que `√d_k` — medindo a saturação do softmax |
# | 3 | Máscara causal e o vazamento do futuro |
# | 4 | Multi-head: o malabarismo de eixos |
# | 5 | GQA e o `repeat_kv` |
# | 6 | RoPE e a identidade da posição relativa |
# | 7 | Mapas de atenção reais e o attention sink |
# | 8 | **Reconstruir uma camada inteira e validar contra o HuggingFace** |
# | 9 | Onde estão os parâmetros |
#
# O Lab 8 é o objetivo do módulo. Se a sua camada bater com a oficial até `1e-5`,
# não sobrou nada de caixa-preta.

# %%
import math

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)
torch.set_grad_enabled(False)

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"

# A API do transformers mudou na v5: `torch_dtype` virou `dtype`, e `config.rope_theta`
# migrou para `config.rope_parameters["rope_theta"]`. Estes dois helpers mantêm o lab
# funcionando tanto na sua máquina (v5) quanto no Colab (ainda em v4).
V5 = int(transformers.__version__.split(".")[0]) >= 5
DTYPE_KW = {"dtype": torch.float32} if V5 else {"torch_dtype": torch.float32}

def obter_rope_theta(cfg):
    if hasattr(cfg, "rope_parameters"):          # transformers >= 5
        return float(cfg.rope_parameters["rope_theta"])
    return float(cfg.rope_theta)                 # transformers 4.x

# attn_implementation="eager" é necessário para extrair mapas de atenção (Lab 7)
# e garante comparação numérica limpa no Lab 8.
tok = AutoTokenizer.from_pretrained(QWEN)
model = AutoModelForCausalLM.from_pretrained(QWEN, attn_implementation="eager", **DTYPE_KW)
model.eval()
cfg = model.config

D = cfg.hidden_size
H = cfg.num_attention_heads
KVH = cfg.num_key_value_heads
HEAD_DIM = D // H
THETA = obter_rope_theta(cfg)

print(f"transformers   = {transformers.__version__}")
print(f"hidden_size    = {D}")
print(f"query heads    = {H}")
print(f"kv heads       = {KVH}   (GQA fator {H // KVH}x)")
print(f"head_dim       = {HEAD_DIM}")
print(f"camadas        = {cfg.num_hidden_layers}")
print(f"d_ff           = {cfg.intermediate_size}")
print(f"rope_theta     = {THETA:,.0f}")
print(f"rms_norm_eps   = {cfg.rms_norm_eps}")

# %% [markdown]
# ## Lab 1 — Attention do zero
#
# A fórmula inteira em cinco linhas.

# %%
def atencao(Q, K, V, mascara_causal=False):
    """Q,K,V: [..., n, d_k]. Devolve saída e a matriz de atenção."""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)     # [..., n, n]

    if mascara_causal:
        n = scores.size(-1)
        triangulo = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(triangulo, float("-inf"))

    pesos = F.softmax(scores, dim=-1)                     # linhas somam 1
    return pesos @ V, pesos

# %%
n, d_k = 5, 16
Q, K, V = torch.randn(3, n, d_k)

saida, pesos = atencao(Q, K, V)
print(f"Q,K,V: {tuple(Q.shape)}   ->   saída: {tuple(saida.shape)}   atenção: {tuple(pesos.shape)}")
print(f"cada linha da matriz de atenção soma 1? {torch.allclose(pesos.sum(-1), torch.ones(n))}")

# Confere contra o kernel do PyTorch (o mesmo que roda em produção).
oficial = F.scaled_dot_product_attention(Q, K, V)
print(f"bate com F.scaled_dot_product_attention? {torch.allclose(saida, oficial, atol=1e-6)}")

# %% [markdown]
# ## Lab 2 — Por que `√d_k`
#
# A afirmação da teoria: `Var(q·k) = d_k`. Verificando empiricamente, e vendo o efeito
# sobre a entropia do softmax — que é o que de fato importa para o gradiente.

# %%
def entropia(p):
    return float(-(p * torch.log(p + 1e-12)).sum(-1).mean())

print(f"{'d_k':>6} {'std(q·k)':>10} {'√d_k':>8} {'entropia SEM escala':>21} {'entropia COM escala':>21}")
print("-" * 72)
for d in [16, 64, 128, 512, 4096]:
    q, k = torch.randn(4096, d), torch.randn(4096, d)
    std_empirico = (q * k).sum(-1).std()          # deve bater com √d_k

    scores = torch.randn(64, d) @ torch.randn(64, d).T
    sem = entropia(F.softmax(scores, dim=-1))
    com = entropia(F.softmax(scores / math.sqrt(d), dim=-1))

    print(f"{d:>6} {std_empirico:>10.2f} {math.sqrt(d):>8.2f} {sem:>21.4f} {com:>21.4f}")

print("\nentropia máxima possível (uniforme sobre 64) = ln(64) =", round(math.log(64), 4))

# %% [markdown]
# Sem a escala, a entropia despenca para perto de zero conforme `d_k` cresce: a atenção
# vira one-hot. E um softmax saturado tem **gradiente ≈ 0** — as projeções `W_Q` e `W_K`
# param de receber sinal de treino. Com a escala, a entropia se mantém numa faixa útil
# independentemente da dimensão.

# %% [markdown]
# E o efeito no **gradiente**, que é o que de fato impede o aprendizado. Usamos uma loss
# artificial (projeção dos pesos de atenção num alvo aleatório) só para medir a magnitude
# do sinal que chega em `W_Q`:

# %%
torch.set_grad_enabled(True)
for d in [64, 512, 4096]:
    linha = f"d_k={d:<6}"
    for escalar in [False, True]:
        torch.manual_seed(0)
        q = torch.randn(64, d, requires_grad=True)
        k = torch.randn(64, d)
        alvo = torch.randn(64, 64)

        scores = q @ k.T
        if escalar:
            scores = scores / math.sqrt(d)
        (F.softmax(scores, dim=-1) * alvo).sum().backward()

        rotulo = "COM escala" if escalar else "SEM escala"
        linha += f"   {rotulo}: |grad|={q.grad.abs().mean():.2e}"
    print(linha)
torch.set_grad_enabled(False)

# %% [markdown]
# ## Lab 3 — Máscara causal
#
# Prova de que sem máscara o modelo enxerga o futuro.

# %%
n = 6
Q, K, V = torch.randn(3, n, 8)

_, sem_mascara = atencao(Q, K, V, mascara_causal=False)
_, com_mascara = atencao(Q, K, V, mascara_causal=True)

def heatmap(m, titulo, escala=" ░▒▓█"):
    """Heatmap em texto — funciona em qualquer terminal, sem matplotlib."""
    print(f"\n{titulo}")
    m = m / (m.max() + 1e-9)
    print("      " + "".join(f"{j:>3}" for j in range(m.shape[1])))
    for i, linha in enumerate(m):
        celulas = "".join(escala[min(int(v * (len(escala) - 1) + 0.5), len(escala) - 1)] * 3 for v in linha)
        print(f"  {i:>3} {celulas}")

heatmap(sem_mascara, "SEM máscara (posição i atende a tudo, inclusive j>i)")
heatmap(com_mascara, "COM máscara causal (triangular inferior)")

# %%
# O teste que importa: mudar um token FUTURO altera a saída da posição atual?
def saida_pos2(V_alterado, causal):
    s, _ = atencao(Q, K, V_alterado, mascara_causal=causal)
    return s[2]

V_mod = V.clone()
V_mod[5] += 100.0   # mexe só no último token

for causal in [False, True]:
    delta = (saida_pos2(V_mod, causal) - saida_pos2(V, causal)).abs().max()
    print(f"causal={str(causal):<5} mudança na saída da posição 2 ao alterar a posição 5: {delta:.6f}")

# %% [markdown]
# Sem máscara, alterar o token 5 muda a representação do token 2 — o modelo estaria
# prevendo o futuro a partir do futuro. Com máscara, a mudança é exatamente zero.
#
# ## Lab 4 — Multi-head
#
# Não existem `h` matrizes: existe uma projeção `[d, d]` e um `reshape`. É onde mais se erra.

# %%
def multi_head(x, Wq, Wk, Wv, Wo, n_heads):
    b, s, d = x.shape
    hd = d // n_heads

    # [b, s, d] -> [b, s, h, hd] -> [b, h, s, hd]
    q = (x @ Wq).view(b, s, n_heads, hd).transpose(1, 2)
    k = (x @ Wk).view(b, s, n_heads, hd).transpose(1, 2)
    v = (x @ Wv).view(b, s, n_heads, hd).transpose(1, 2)

    saida, pesos = atencao(q, k, v, mascara_causal=True)   # cada cabeça em paralelo

    # [b, h, s, hd] -> [b, s, h, hd] -> [b, s, d]
    saida = saida.transpose(1, 2).contiguous().view(b, s, d)
    return saida @ Wo, pesos

x = torch.randn(1, 7, 64)
Ws = [torch.randn(64, 64) / 8 for _ in range(4)]
saida, pesos = multi_head(x, *Ws, n_heads=8)
print(f"entrada {tuple(x.shape)} -> saída {tuple(saida.shape)}   |   atenção por cabeça {tuple(pesos.shape)}")
print(f"8 cabeças de dimensão {64 // 8} = {64} — a dimensão é DIVIDIDA, não multiplicada")

# %% [markdown]
# ## Lab 5 — GQA
#
# `repeat_kv` replica cada KV head para as queries do seu grupo. A replicação é lógica;
# o **cache** guarda apenas os KV heads reais — é daí que vem a economia.

# %%
def repeat_kv(x, n_rep):
    """[b, kv_heads, s, hd] -> [b, kv_heads*n_rep, s, hd]"""
    b, kvh, s, hd = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, kvh, n_rep, s, hd).reshape(b, kvh * n_rep, s, hd)

k_gqa = torch.randn(1, 2, 5, 64)          # 2 KV heads
k_exp = repeat_kv(k_gqa, n_rep=7)         # 14 query heads
print(f"{tuple(k_gqa.shape)} -> {tuple(k_exp.shape)}")
print(f"heads 0..6 são cópias do kv head 0? {torch.equal(k_exp[:, 0], k_exp[:, 6])}")
print(f"head 7 é o kv head 1?               {torch.equal(k_exp[:, 7], k_gqa[:, 1])}")

# %%
def kv_cache_kb_por_token(n_layers, kv_heads, head_dim, bytes_por_valor=2):
    return 2 * n_layers * kv_heads * head_dim * bytes_por_valor / 1024

gqa = kv_cache_kb_por_token(cfg.num_hidden_layers, KVH, HEAD_DIM)
mha = kv_cache_kb_por_token(cfg.num_hidden_layers, H, HEAD_DIM)

print(f"Qwen2.5-0.5B com GQA ({KVH} kv heads): {gqa:6.1f} KB/token")
print(f"o mesmo modelo com MHA ({H} kv heads): {mha:6.1f} KB/token   ({mha / gqa:.0f}x mais)")
for ctx in [4096, 32768, 131072]:
    print(f"  contexto {ctx:>7,}: GQA {gqa * ctx / 1e6:6.2f} GB   MHA {mha * ctx / 1e6:6.2f} GB")

# %% [markdown]
# ## Lab 6 — RoPE
#
# Implementação (convenção "half rotation", a usada por Llama/Qwen/Mistral) e a
# verificação da propriedade que faz tudo funcionar.

# %%
def rope_cos_sin(seq_len, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)   # [s, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)                        # [s, hd]
    return emb.cos(), emb.sin()

def rotate_half(x):
    metade = x.shape[-1] // 2
    return torch.cat([-x[..., metade:], x[..., :metade]], dim=-1)

def aplicar_rope(x, cos, sin):
    """x: [..., s, head_dim]; cos/sin: [s, head_dim]"""
    return x * cos + rotate_half(x) * sin

# %%
# As frequências: pares no início giram rápido, pares no fim giram devagar.
inv_freq = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
print("frequências de rotação (rad por posição):")
for i in [0, 1, 5, 15, 25, 31]:
    periodo = 2 * math.pi / float(inv_freq[i])
    print(f"  par {i:>2}: θ={float(inv_freq[i]):.3e}   período = {periodo:>15,.0f} posições")

# %% [markdown]
# O par 0 completa uma volta a cada ~6 posições (distingue vizinhos); o último par leva
# milhões de posições (distingue regiões distantes). É um relógio de vários ponteiros — e
# a `base` alta (1.000.000 no Qwen2.5) é o que permite contextos longos.

# %%
# A identidade: ⟨R_m q, R_n k⟩ depende SÓ de (m - n).
cos, sin = rope_cos_sin(200, HEAD_DIM, THETA)
q, k = torch.randn(HEAD_DIM), torch.randn(HEAD_DIM)

print("mesma distância relativa, posições absolutas diferentes:")
for m, n in [(10, 5), (50, 45), (100, 95), (180, 175)]:
    qm = aplicar_rope(q, cos[m], sin[m])
    kn = aplicar_rope(k, cos[n], sin[n])
    print(f"  m={m:>3} n={n:>3} (m-n={m - n})  ->  <R_m q, R_n k> = {float(qm @ kn):+.6f}")

print("\ndistâncias relativas diferentes:")
for m, n in [(100, 99), (100, 95), (100, 80), (100, 20)]:
    qm = aplicar_rope(q, cos[m], sin[m])
    kn = aplicar_rope(k, cos[n], sin[n])
    print(f"  m={m:>3} n={n:>3} (m-n={m - n:>3})  ->  <R_m q, R_n k> = {float(qm @ kn):+.6f}")

# %% [markdown]
# Os valores do primeiro bloco são **idênticos**: posição absoluta entra em `q` e `k`, e a
# atenção enxerga posição **relativa**. Nada na fórmula da atenção precisou mudar.
#
# Confirmando contra os pesos reais do modelo:

# %%
inv_freq_oficial = model.model.rotary_emb.inv_freq
print(f"minha inv_freq bate com a do modelo? "
      f"{torch.allclose(inv_freq, inv_freq_oficial.float(), atol=1e-6)}")

# %% [markdown]
# ## Lab 7 — Mapas de atenção reais e o attention sink

# %%
texto = "O gato preto subiu no telhado porque ele estava com fome."
ids = tok(texto, return_tensors="pt")
saida = model(**ids, output_attentions=True)

atencoes = saida.attentions          # tupla de 24 tensores [1, 14, s, s]
tokens = tok.convert_ids_to_tokens(ids["input_ids"][0])
print(f"{len(atencoes)} camadas x {atencoes[0].shape[1]} cabeças, seq={len(tokens)}")
print("tokens:", [t.replace("Ġ", "␣") for t in tokens])

# %%
def heatmap_tokens(m, tokens, titulo):
    escala = " ░▒▓█"
    print(f"\n{titulo}")
    m = m / (m.max() + 1e-9)
    for i, linha in enumerate(m):
        celulas = "".join(escala[min(int(v * 4 + 0.5), 4)] * 2 for v in linha)
        print(f"  {tokens[i].replace('Ġ', '␣')[:12]:>13} |{celulas}")

for camada, cabeca in [(0, 0), (12, 3), (23, 7)]:
    heatmap_tokens(atencoes[camada][0, cabeca], tokens,
                   f"camada {camada}, cabeça {cabeca}")

# %% [markdown]
# Repare na **primeira coluna acesa** em praticamente todas as linhas. Esse é o
# *attention sink*: o softmax precisa somar 1, e cabeças sem nada a buscar despejam a
# massa no primeiro token. Quantificando:

# %%
print(f"{'camada':>7} {'massa no token 0':>18} {'massa na diagonal':>19}")
print("-" * 46)
for camada in range(0, cfg.num_hidden_layers, 4):
    a = atencoes[camada][0]                       # [heads, s, s]
    sink = a[:, 1:, 0].mean()                     # ignora a linha 0 (trivialmente 1.0)
    diag = a.diagonal(dim1=-2, dim2=-1).mean()
    print(f"{camada:>7} {float(sink):>18.1%} {float(diag):>19.1%}")

# %% [markdown]
# > ⚠️ Mapas de atenção são péssima evidência causal. Uma cabeça pode atender fortemente a
# > um token cujo `value` é quase nulo — atenção mostra para onde se olha, não o que se usa.
#
# ## Lab 8 — Reconstruir uma camada inteira
#
# O objetivo do módulo. Vamos reimplementar a camada 0 do Qwen2.5 usando apenas os pesos
# e as funções que escrevemos acima, e comparar com o forward oficial.
#
# Ordem exata de um `Qwen2DecoderLayer` (pre-norm):
#
# ```
# h = h + Attention(RMSNorm(h))
# h = h + MLP(RMSNorm(h))
# ```

# %%
def rmsnorm(x, peso, eps):
    """Upcast para fp32 é obrigatório — mean(x²) estoura em bf16."""
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (peso * (x.float() * torch.rsqrt(var + eps))).to(x.dtype)

def linear(x, modulo):
    return F.linear(x, modulo.weight, modulo.bias)

def camada_manual(h, layer, cfg):
    b, s, d = h.shape
    hd = d // cfg.num_attention_heads
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads

    # ---------- bloco de atenção ----------
    residual = h
    x = rmsnorm(h, layer.input_layernorm.weight, cfg.rms_norm_eps)

    q = linear(x, layer.self_attn.q_proj).view(b, s, cfg.num_attention_heads, hd).transpose(1, 2)
    k = linear(x, layer.self_attn.k_proj).view(b, s, cfg.num_key_value_heads, hd).transpose(1, 2)
    v = linear(x, layer.self_attn.v_proj).view(b, s, cfg.num_key_value_heads, hd).transpose(1, 2)

    cos, sin = rope_cos_sin(s, hd, THETA)
    q = aplicar_rope(q, cos, sin)          # RoPE em q e k...
    k = aplicar_rope(k, cos, sin)          # ...NUNCA em v

    k = repeat_kv(k, n_rep)                # GQA: 2 kv heads -> 14 query heads
    v = repeat_kv(v, n_rep)

    attn, _ = atencao(q, k, v, mascara_causal=True)
    attn = attn.transpose(1, 2).contiguous().view(b, s, d)
    h = residual + linear(attn, layer.self_attn.o_proj)

    # ---------- bloco MLP (SwiGLU) ----------
    residual = h
    x = rmsnorm(h, layer.post_attention_layernorm.weight, cfg.rms_norm_eps)
    portao = F.silu(linear(x, layer.mlp.gate_proj))
    h = residual + linear(portao * linear(x, layer.mlp.up_proj), layer.mlp.down_proj)

    return h

# %%
ids = tok("A atenção é tudo de que você precisa.", return_tensors="pt")
oficial = model(**ids, output_hidden_states=True).hidden_states

entrada = oficial[0]            # embeddings, antes de qualquer camada
esperado = oficial[1]           # saída oficial da camada 0

obtido = camada_manual(entrada, model.model.layers[0], cfg)

erro_max = (obtido - esperado).abs().max()
erro_rel = erro_max / esperado.abs().max()

print(f"forma      : {tuple(obtido.shape)}")
print(f"erro máximo: {erro_max:.3e}")
print(f"erro rel.  : {erro_rel:.3e}")
print(f"allclose(1e-4)? {torch.allclose(obtido, esperado, atol=1e-4)}")

# %%
# E a pilha inteira: 24 camadas, todas manuais.
h = oficial[0]
for i, layer in enumerate(model.model.layers):
    h = camada_manual(h, layer, cfg)

h = rmsnorm(h, model.model.norm.weight, cfg.rms_norm_eps)     # norma final
logits_manual = F.linear(h, model.lm_head.weight)

logits_oficial = model(**ids).logits
print(f"erro máximo nos logits, 24 camadas manuais: {(logits_manual - logits_oficial).abs().max():.3e}")
print(f"mesmo token previsto? "
      f"{int(logits_manual[0, -1].argmax()) == int(logits_oficial[0, -1].argmax())}")
print(f"token: {tok.decode([int(logits_manual[0, -1].argmax())])!r}")

# %% [markdown]
# **Você reconstruiu o modelo.** Em float32 com `attn_implementation="eager"`, o erro
# costuma dar exatamente `0.0`: as operações são as mesmas, na mesma ordem, com os mesmos
# kernels — é literalmente o mesmo cálculo. Em bf16, ou com FlashAttention na GPU, a ordem
# das reduções muda e você veria algo entre `1e-3` e `1e-2` (correto, mas não bit-exact).
#
# ### Um erro zero é suspeito — teste a sua verificação
#
# Se o teste sempre passasse, ele não estaria testando nada. Repetindo a comparação com
# **bugs propositais**, para confirmar que ela detecta problemas reais:

# %%
def camada_com_bug(h, layer, cfg, bug):
    b, s, d = h.shape
    hd = d // cfg.num_attention_heads
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads

    residual = h
    x = rmsnorm(h, layer.input_layernorm.weight, cfg.rms_norm_eps)

    q = linear(x, layer.self_attn.q_proj).view(b, s, cfg.num_attention_heads, hd).transpose(1, 2)
    k = linear(x, layer.self_attn.k_proj).view(b, s, cfg.num_key_value_heads, hd).transpose(1, 2)
    v = linear(x, layer.self_attn.v_proj).view(b, s, cfg.num_key_value_heads, hd).transpose(1, 2)

    cos, sin = rope_cos_sin(s, hd, THETA)
    q = aplicar_rope(q, cos, sin)
    k = aplicar_rope(k, cos, sin)
    if bug == "rope em v":
        v = aplicar_rope(v, cos, sin)

    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)

    causal = bug != "sem máscara causal"
    if bug == "sem √d_k":
        scores = q @ k.transpose(-2, -1)                       # esqueceu de escalar
        if causal:
            tri = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(tri, float("-inf"))
        attn = F.softmax(scores, dim=-1) @ v
    else:
        attn, _ = atencao(q, k, v, mascara_causal=causal)

    attn = attn.transpose(1, 2).contiguous().view(b, s, d)
    h = residual + linear(attn, layer.self_attn.o_proj)

    residual = h
    x = rmsnorm(h, layer.post_attention_layernorm.weight, cfg.rms_norm_eps)
    portao = F.silu(linear(x, layer.mlp.gate_proj))
    saida_mlp = linear(portao * linear(x, layer.mlp.up_proj), layer.mlp.down_proj)

    if bug == "sem residual":
        return saida_mlp
    return residual + saida_mlp

for bug in ["nenhum", "rope em v", "sem máscara causal", "sem √d_k", "sem residual"]:
    obtido_bug = camada_com_bug(entrada, model.model.layers[0], cfg, bug)
    erro = (obtido_bug - esperado).abs().max()
    veredito = "PASSA" if erro < 1e-4 else "FALHA"
    print(f"  bug: {bug:<20} erro máx = {erro:>10.3e}   {veredito}")

# %% [markdown]
# Cada bug produz um erro de ordens de grandeza acima do limiar. A verificação é real —
# e note que **nenhum deles lança exceção**. Um modelo com qualquer um desses defeitos
# treinaria normalmente e entregaria resultados piores, sem nenhum sinal de alerta.
#
# ## Lab 9 — Onde estão os parâmetros

# %%
layer = model.model.layers[0]
grupos = {
    "q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj,
    "v_proj": layer.self_attn.v_proj, "o_proj": layer.self_attn.o_proj,
    "gate_proj": layer.mlp.gate_proj, "up_proj": layer.mlp.up_proj,
    "down_proj": layer.mlp.down_proj,
}

total_bloco = sum(p.numel() for p in layer.parameters())
attn = sum(sum(p.numel() for p in m.parameters()) for n, m in grupos.items() if "proj" in n and n[0] in "qkvo")
mlp = sum(sum(p.numel() for p in m.parameters()) for n, m in grupos.items() if n[0] in "gud")

for nome, m in grupos.items():
    n_p = sum(p.numel() for p in m.parameters())
    forma = tuple(m.weight.shape)
    bias = " +bias" if m.bias is not None else ""
    print(f"  {nome:<10} {str(forma):>14}{bias:<6} {n_p:>12,}  {n_p / total_bloco:>6.1%}")

print(f"\n  {'atenção':<10} {'':>20} {attn:>12,}  {attn / total_bloco:>6.1%}")
print(f"  {'MLP':<10} {'':>20} {mlp:>12,}  {mlp / total_bloco:>6.1%}")
print(f"  {'normas':<10} {'':>20} {total_bloco - attn - mlp:>12,}")
print(f"  {'TOTAL/bloco':<10} {'':>20} {total_bloco:>12,}")

# %%
total = sum(p.numel() for p in model.parameters())
emb = model.get_input_embeddings().weight.numel()
print(f"embeddings              {emb:>12,}  {emb / total:>6.1%}")
print(f"{cfg.num_hidden_layers} blocos               {total_bloco * cfg.num_hidden_layers:>12,}")
print(f"modelo inteiro          {total:>12,}")

# %% [markdown]
# **O MLP domina.** Guarde esse número: no módulo 6, LoRA aplicado só a `q_proj` e `v_proj`
# toca uma fração minúscula do modelo — e é por isso que às vezes não basta. No módulo 11,
# o MoE substitui **apenas o MLP** por especialistas, e agora está claro por quê: é lá que
# estão os parâmetros.
#
# ---
#
# ## Encerramento
#
# Você acabou de:
#
# - implementar attention, multi-head, GQA e RoPE do zero;
# - medir por que `√d_k` existe, em entropia e em gradiente;
# - provar que a máscara causal bloqueia o futuro;
# - verificar numericamente a identidade da posição relativa do RoPE;
# - observar o attention sink em um modelo real;
# - **reconstruir o modelo inteiro** e bater com o oficial.
#
# Agora o `exercicios.md`.

# %%
