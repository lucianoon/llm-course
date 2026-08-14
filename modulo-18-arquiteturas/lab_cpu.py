# %% [markdown]
# # Módulo 18 — Laboratório: a fronteira das arquiteturas (em CPU)
#
# **Roda em CPU (Windows ou Mac), ~8 minutos.** O transformer domina desde 2017, mas não
# é o fim da história. Este lab implementa os desafiantes — SSMs/Mamba, atenção linear,
# MLA — e MEDE o trade-off que cada um faz contra a atenção clássica. Todos rodam em
# miniatura no laptop.
#
# | Lab | Assunto |
# |---|---|
# | 1 | O gargalo que motiva tudo: O(L²) da atenção, medido |
# | 2 | SSM / Mamba: estado fixo, do zero |
# | 3 | Atenção linear: a reordenação que troca L² por L |
# | 4 | MLA (DeepSeek): comprimir o KV cache |
# | 5 | O trade-off recall vs eficiência, medido |
# | 6 | Arquiteturas híbridas: por que ninguém abandonou a atenção |

# %%
import math
import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)

# %% [markdown]
# ## Lab 1 — O gargalo: O(L²)
#
# A atenção (módulo 2) compara cada token com todos os anteriores: `QKᵀ` é uma matriz
# `L×L`. Dobrar o contexto QUADRUPLICA o compute e a memória. Isso é o pecado original
# que todas as arquiteturas deste módulo tentam corrigir.

# %%
def atencao_custo(L, d=64):
    q = torch.randn(L, d)
    k = torch.randn(L, d)
    v = torch.randn(L, d)
    t0 = time.perf_counter()
    scores = q @ k.T / math.sqrt(d)          # [L, L] — o termo quadrático
    mem_matriz = scores.numel() * 4 / 1e6    # MB da matriz de atenção
    a = F.softmax(scores, dim=-1)
    _ = a @ v
    return time.perf_counter() - t0, mem_matriz

print(f"{'L':>8} {'tempo':>10} {'matriz L×L':>12} {'razão vs L/2':>14}")
print("-" * 48)
t_ant = None
for L in [256, 512, 1024, 2048, 4096]:
    t, mem = atencao_custo(L)
    razao = f"{t/t_ant:.1f}x" if t_ant else "—"
    print(f"{L:>8} {t*1000:>8.1f}ms {mem:>10.1f}MB {razao:>14}")
    t_ant = t

print("\n→ dobrar L ~quadruplica tudo. Em L=128k, a matriz de atenção de UMA cabeça")
print("  pesa gigabytes (módulo 2), e o KV cache cresce sem limite (módulo 1).")

# %% [markdown]
# ## Lab 2 — SSM / Mamba: estado de tamanho FIXO
#
# A ideia dos State Space Models: em vez de guardar TODO o passado (KV cache), resumi-lo
# num **estado de tamanho fixo** `h`, atualizado recorrentemente — como uma RNN, mas com
# a matemática que a torna treinável em paralelo e seletiva (Mamba).
#
# ```
# h_t = A·h_{t-1} + B·x_t      (o estado resume o passado)
# y_t = C·h_t                   (a saída lê o estado)
# ```

# %%
def ssm_scan(x, A, B, C):
    """Scan recorrente. x: [batch, L]. Estado h de dimensão d_estado, FIXO em L."""
    batch, L = x.shape
    d_estado = A.shape[-1]
    h = torch.zeros(batch, d_estado)
    saidas = []
    for t in range(L):
        h = A * h + B * x[:, t:t + 1]        # atualização recorrente
        saidas.append((C * h).sum(-1, keepdim=True))
    return torch.cat(saidas, dim=1)

batch, L, D_ESTADO = 2, 16, 8
x = torch.randn(batch, L)
A = torch.rand(D_ESTADO) * 0.9 + 0.05        # |A|<1 para estabilidade
B = torch.rand(D_ESTADO)
C = torch.rand(D_ESTADO)
y = ssm_scan(x, A, B, C)
print(f"SSM: entrada {tuple(x.shape)} -> saída {tuple(y.shape)}, estado de {D_ESTADO} dims")

# %%
# A propriedade que muda tudo: a memória de inferência é CONSTANTE no comprimento.
print(f"\nmemória de 'histórico' por sequência (inferência):")
print(f"{'L':>8} {'attention KV cache':>20} {'SSM estado':>14}")
print("-" * 46)
for L in [1024, 8192, 131072, 1_000_000]:
    kv = L * 128 / 1e6                        # ~128 KB/token (Llama-3-8B, módulo 1)
    ssm = D_ESTADO * 8 * 24 / 1e6             # estado fixo × camadas, em MB
    print(f"{L:>8} {kv:>18.1f} MB {ssm:>11.3f} MB")

print("\n→ O KV cache do transformer cresce LINEARMENTE com o contexto; o estado do SSM")
print("  é CONSTANTE. Em contextos de milhões de tokens, é a diferença entre caber e não.")

# %% [markdown]
# > 🔧 O que o Mamba adiciona ao SSM clássico: os parâmetros A, B, C passam a DEPENDER da
# > entrada (seletividade) — o modelo escolhe o que lembrar e o que esquecer por token.
# > E um "parallel scan" (algoritmo de prefix-sum) treina o recorrente em paralelo na GPU,
# > resolvendo o defeito fatal das RNNs (módulo 2). Aqui usamos o scan sequencial, que dá
# > o mesmo resultado — só é lento de treinar.
#
# ## Lab 3 — Atenção linear: a reordenação mágica
#
# A atenção é `softmax(QKᵀ)V`. O custo O(L²) vem de calcular `QKᵀ` (matriz L×L) ANTES de
# multiplicar por V. Se removermos o softmax e usarmos uma feature `φ`, a
# **associatividade** permite reordenar: `φ(Q)·(φ(K)ᵀ·V)` — e `φ(K)ᵀ·V` é uma matriz
# `d×d`, independente de L!

# %%
def atencao_quadratica(Q, K, V):
    """O(L²): calcula a matriz L×L."""
    return F.softmax(Q @ K.T / math.sqrt(Q.shape[-1]), dim=-1) @ V

def atencao_linear(Q, K, V):
    """O(L): φ(Q)·(φ(K)ᵀ·V). φ = elu+1 (garante positividade)."""
    phi = lambda x: F.elu(x) + 1
    Qp, Kp = phi(Q), phi(K)
    KV = Kp.T @ V                            # [d, d] — NÃO depende de L!
    Z = Qp @ Kp.sum(0, keepdim=True).T       # normalizador
    return (Qp @ KV) / (Z + 1e-6)

print(f"{'L':>8} {'quadrática':>12} {'linear':>10} {'speedup':>9}")
print("-" * 42)
for L in [512, 2048, 8192]:
    Q, K, V = torch.randn(3, L, 64)
    t0 = time.perf_counter(); atencao_quadratica(Q, K, V); tq = time.perf_counter() - t0
    t0 = time.perf_counter(); atencao_linear(Q, K, V); tl = time.perf_counter() - t0
    print(f"{L:>8} {tq*1000:>10.1f}ms {tl*1000:>8.1f}ms {tq/tl:>8.1f}x")

print("\n→ A linear escala O(L); a quadrática, O(L²). O ganho cresce com o contexto.")
print("  O preço: sem o softmax, a capacidade de 'focar' num token específico cai —")
print("  é o trade-off recall vs eficiência que o Lab 5 mede.")

# %% [markdown]
# ## Lab 4 — MLA: comprimir o KV cache sem trocar de arquitetura
#
# O DeepSeek-V3 mantém a atenção (e seu recall) mas ataca só o KV cache: em vez de
# guardar K e V cheios, guarda uma **projeção comprimida** (latente) e a reexpande na
# hora. Menos memória, quase o mesmo recall.

# %%
def kv_cache_mla(n_layers, d, d_latente, L, batch=1):
    """MLA guarda só o vetor latente comprimido por token."""
    return 2 * n_layers * d_latente * L * batch * 2 / 1e9      # GB, bf16

def kv_cache_normal(n_layers, kv_heads, head_dim, L, batch=1):
    return 2 * n_layers * kv_heads * head_dim * L * batch * 2 / 1e9

L = 32768
print(f"KV cache para {L} tokens (modelo tipo Llama-3-8B):")
normal = kv_cache_normal(32, 8, 128, L)          # GQA
mla = kv_cache_mla(32, 4096, 512, L)             # latente de 512 vs 8×128=1024
print(f"  GQA normal (8 kv heads): {normal:.2f} GB")
print(f"  MLA (latente 512):       {mla:.2f} GB")
print(f"  redução: {1 - mla/normal:.0%}")
print("\n→ MLA é a escolha 'conservadora': mantém a atenção que funciona e comprime só")
print("  o que dói (o cache). É por isso que o DeepSeek serve contextos longos barato.")

# %% [markdown]
# ## Lab 5 — O trade-off que decide tudo: recall vs eficiência
#
# A pergunta central da fronteira: as arquiteturas eficientes (SSM, linear) sacrificam
# a capacidade de RECUPERAR um token específico do passado (o que a atenção faz bem). O
# teste clássico: **associative recall** — dado "A→1, B→2, C→3 ... B→?", lembrar o "2".

# %%
def teste_recall(mecanismo, n_pares=8, dim=32, trials=200):
    """Mede: dado pares chave-valor no contexto, o mecanismo recupera o valor certo?"""
    acertos = 0
    for _ in range(trials):
        chaves = torch.randn(n_pares, dim)
        valores = torch.randn(n_pares, dim)
        alvo = torch.randint(n_pares, (1,)).item()
        query = chaves[alvo:alvo + 1]              # pergunta pela chave 'alvo'

        if mecanismo == "atenção":
            recuperado = atencao_quadratica(query, chaves, valores)
        else:  # linear
            recuperado = atencao_linear(query, chaves, valores)

        # acertou se o valor recuperado está mais perto do alvo que de qualquer outro
        dists = ((recuperado - valores) ** 2).sum(-1)
        acertos += (dists.argmin().item() == alvo)
    return acertos / trials

print(f"{'mecanismo':<14} {'recall associativo':>20}")
print("-" * 36)
for mec in ["atenção", "linear"]:
    print(f"{mec:<14} {teste_recall(mec):>19.0%}")

print("\n→ A atenção recupera com precisão (o softmax 'foca' na chave exata);")
print("  a linear borra (sem softmax, a recuperação é uma média ponderada suave).")
print("  Esse é o preço da eficiência — e a razão de os híbridos existirem.")

# %% [markdown]
# ## Lab 6 — Híbridos: o que a indústria de fato faz
#
# Nenhum modelo de fronteira abandonou a atenção. Os que usam SSM (Jamba, os Nemotron
# híbridos) INTERCALAM camadas de atenção e de Mamba: as de Mamba dão eficiência em
# contexto longo; as poucas de atenção preservam o recall preciso onde importa.

# %%
def custo_hibrido(n_layers, frac_atencao, L, d=4096):
    """Custo relativo de um modelo híbrido vs 100% atenção."""
    n_attn = int(n_layers * frac_atencao)
    n_ssm = n_layers - n_attn
    # atenção: O(L²); ssm: O(L). Normalizado por uma camada de atenção.
    custo = n_attn * L + n_ssm * 1            # em unidades de L (o termo dominante)
    custo_full_attn = n_layers * L
    return custo / custo_full_attn

print(f"custo de compute em contexto longo (L grande), vs 100% atenção:")
print(f"{'% de camadas de atenção':>26} {'custo relativo':>16}")
print("-" * 44)
for frac in [1.0, 0.5, 0.25, 0.1, 0.0]:
    c = custo_hibrido(32, frac, L=100000)
    rotulo = {1.0: " (transformer puro)", 0.0: " (Mamba puro)"}.get(frac, "")
    print(f"{frac:>25.0%} {c:>15.1%}{rotulo}")

# %% [markdown]
# **O ponto de equilíbrio da indústria em 2025-26:** um híbrido com ~1 camada de atenção
# a cada 6-8 de Mamba captura quase toda a eficiência do Mamba puro E quase todo o recall
# do transformer. É o "melhor dos dois mundos" que fez SSMs saírem do laboratório para
# produção (Jamba, Nemotron-H, Falcon-Mamba).
#
# > 🔧 A leitura honesta da fronteira: o transformer NÃO foi destronado. Ele foi
# > COMPLEMENTADO. A atenção continua sendo o melhor mecanismo de recall preciso que
# > temos; o que mudou é que agora sabemos usá-la com PARCIMÔNIA, delegando o resto a
# > mecanismos O(L). Multimodalidade segue o mesmo padrão: um encoder de imagem/áudio
# > projeta tokens no mesmo espaço, e a MESMA arquitetura os processa (é por isso que
# > "LLM" virou "modelo de fundação").
#
# ---
#
# ## Encerramento
#
# A fronteira, implementada e medida:
#
# - o O(L²) da atenção, o pecado que tudo tenta corrigir;
# - o SSM/Mamba com estado FIXO — memória constante em contextos de milhões;
# - a atenção linear e a reordenação que troca L² por L;
# - o MLA comprimindo o KV cache sem trocar de arquitetura;
# - o trade-off recall vs eficiência, medido — o preço de cada atalho;
# - os híbridos, a resposta real da indústria.
#
# A lição que fecha a Fase 2: **a arquitetura evolui por pressão de custo** (contexto
# longo, KV cache, O(L²)), exatamente como todo o resto do curso. E o transformer, sete
# anos depois, continua no centro — porque recall preciso ainda é insubstituível.
#
# A Fase 3 começa: parar de aprender o que existe e passar a produzir — reproduzir
# papers, contribuir, pesquisar.

# %%
