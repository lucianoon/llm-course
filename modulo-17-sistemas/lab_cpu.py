# %% [markdown]
# # Módulo 17 — Laboratório: os sistemas que treinam LLMs (em CPU)
#
# **Roda em CPU (Windows ou Mac), ~8 minutos.** Não dá para alugar 1.000 GPUs num
# laptop — mas dá para entender exatamente o que elas fazem. Este lab implementa o motor
# de gradientes DO ZERO (o que o PyTorch faz por baixo), mede o trade-off real do
# gradient checkpointing, e simula as contas de comunicação que decidem o custo de um
# cluster.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Autograd do zero: o motor de backprop, verificado contra o PyTorch |
# | 2 | Backprop de uma MLP na mão vs autograd |
# | 3 | **Gradient checkpointing: memória vs recompute, medido de verdade** |
# | 4 | O particionamento do ZeRO/FSDP: a conta da memória |
# | 5 | Comunicação: all-reduce, all-gather e o volume de bytes |
# | 6 | A bolha do pipeline: por que 4 GPUs não são 4× mais rápidas |

# %%
import math
import time

import torch

torch.manual_seed(0)

# %% [markdown]
# ## Lab 1 — Autograd do zero
#
# Todo treino do curso chamou `.backward()`. Aqui está o que ele faz: um grafo de
# operações que se lembra de como calcular a derivada de cada passo (a regra da cadeia,
# aplicada de trás para frente). É o coração do deep learning, em ~40 linhas
# (o "micrograd" do Karpathy).

# %%
class Valor:
    """Um escalar que registra como foi computado, para propagar gradientes de volta."""

    def __init__(self, dado, _filhos=(), _op=""):
        self.dado = dado
        self.grad = 0.0
        self._backward = lambda: None      # como empurrar o gradiente para os filhos
        self._prev = set(_filhos)
        self._op = _op

    def __add__(self, outro):
        outro = outro if isinstance(outro, Valor) else Valor(outro)
        saida = Valor(self.dado + outro.dado, (self, outro), "+")
        def _backward():
            self.grad += saida.grad         # d(a+b)/da = 1
            outro.grad += saida.grad
        saida._backward = _backward
        return saida

    def __mul__(self, outro):
        outro = outro if isinstance(outro, Valor) else Valor(outro)
        saida = Valor(self.dado * outro.dado, (self, outro), "*")
        def _backward():
            self.grad += outro.dado * saida.grad     # d(a*b)/da = b
            outro.grad += self.dado * saida.grad
        saida._backward = _backward
        return saida

    def tanh(self):
        t = math.tanh(self.dado)
        saida = Valor(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * saida.grad    # d(tanh)/dx = 1 - tanh²
        saida._backward = _backward
        return saida

    def backward(self):
        # ordena o grafo topologicamente e propaga do fim para o início
        topo, visitados = [], set()
        def construir(v):
            if v not in visitados:
                visitados.add(v)
                for filho in v._prev:
                    construir(filho)
                topo.append(v)
        construir(self)
        self.grad = 1.0                     # a derivada da saída em relação a si mesma
        for v in reversed(topo):
            v._backward()

# verificação contra o PyTorch
a, b, c = Valor(2.0), Valor(-3.0), Valor(10.0)
saida = (a * b + c).tanh()
saida.backward()

# float64 no PyTorch para comparação JUSTA (meu autograd usa float do Python = float64;
# tensores torch são float32 por padrão, o que introduziria ~1e-7 de diferença espúria).
at = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
bt = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
ct = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
(at * bt + ct).tanh().backward()

print("expressão: tanh(a*b + c), com a=2, b=-3, c=10\n")
print(f"{'':8} {'meu autograd':>14} {'pytorch':>12}")
for nome, meu, tt in [("da", a.grad, at.grad), ("db", b.grad, bt.grad), ("dc", c.grad, ct.grad)]:
    print(f"{nome:8} {meu:>14.8f} {float(tt):>12.8f}  {'✓' if abs(meu-float(tt))<1e-9 else '✗'}")

# %% [markdown]
# **Aqueles 40 renglones SÃO o deep learning.** Todo framework — PyTorch, JAX, o MLX do
# seu Mac — é essa ideia escalada: um grafo de operações que sabe se diferenciar. O
# `topo` (ordenação topológica) garante que cada nó só propaga depois que todos os que
# dependem dele já propagaram — a regra da cadeia, mecanizada.
#
# ## Lab 2 — Backprop de uma MLP, na mão
#
# Agora com matrizes, para uma camada de verdade. Derivamos os gradientes manualmente e
# conferimos contra o autograd do PyTorch — é o que você precisa saber para depurar um
# treino que dá NaN (módulo 3).

# %%
def forward_manual(x, W1, b1, W2, b2, y):
    z1 = x @ W1 + b1                        # [N, H]
    h = torch.relu(z1)                      # [N, H]
    z2 = h @ W2 + b2                        # [N, C]
    # softmax + cross-entropy
    z2 = z2 - z2.max(1, keepdim=True).values
    logp = z2 - z2.exp().sum(1, keepdim=True).log()
    loss = -logp[range(len(y)), y].mean()
    return loss, (z1, h, z2, logp)

def backward_manual(x, W1, b1, W2, b2, y, cache):
    z1, h, z2, logp = cache
    N = len(y)
    # dL/dz2 = softmax - onehot(y)  (o gradiente clássico da cross-entropy)
    dz2 = z2.exp() / z2.exp().sum(1, keepdim=True)
    dz2[range(N), y] -= 1
    dz2 /= N
    dW2 = h.T @ dz2                         # [H, C]
    db2 = dz2.sum(0)
    dh = dz2 @ W2.T                         # [N, H]
    dz1 = dh * (z1 > 0).float()             # derivada do ReLU
    dW1 = x.T @ dz1
    db1 = dz1.sum(0)
    return dW1, db1, dW2, db2

N, D, H, C = 8, 16, 32, 4
x = torch.randn(N, D)
y = torch.randint(0, C, (N,))
params = [torch.randn(D, H, requires_grad=True), torch.zeros(H, requires_grad=True),
          torch.randn(H, C, requires_grad=True), torch.zeros(C, requires_grad=True)]

loss, cache = forward_manual(x, *params, y)
grads_meus = backward_manual(x, *params, y, cache)

loss_pt, _ = forward_manual(x, *params, y)
loss_pt.backward()
grads_pt = [p.grad for p in params]

print("gradientes manuais vs autograd (erro máximo por tensor):")
for nome, meu, pt in zip(["dW1", "db1", "dW2", "db2"], grads_meus, grads_pt):
    print(f"  {nome}: {float((meu - pt).abs().max()):.2e}")

# %% [markdown]
# > 🔧 A derivada da cross-entropy+softmax é `softmax − onehot(y)` — uma das mais elegantes
# > do ML, e a razão de essas duas operações virem sempre juntas. Saber derivá-la na mão
# > é o que permite depurar "por que meu gradiente é enorme aqui?" sem ser refém do
# > framework.
#
# ## Lab 3 — Gradient checkpointing, medido de verdade
#
# O problema do módulo 3: o backward precisa das ATIVAÇÕES de cada camada guardadas desde
# o forward. Numa rede profunda, isso domina a memória. Checkpointing troca memória por
# compute: NÃO guarda as ativações intermediárias e as RECALCULA no backward.

# %%
import torch.utils.checkpoint as cp

class RedeProfunda(torch.nn.Module):
    def __init__(self, d=1024, n_camadas=24, checkpoint=False):
        super().__init__()
        self.camadas = torch.nn.ModuleList(
            torch.nn.Linear(d, d) for _ in range(n_camadas))
        self.checkpoint = checkpoint

    def forward(self, x):
        for camada in self.camadas:
            if self.checkpoint:
                # não guarda a ativação; recalcula no backward
                x = cp.checkpoint(lambda t, c=camada: torch.relu(c(t)), x, use_reentrant=False)
            else:
                x = torch.relu(camada(x))
        return x

def medir(checkpoint):
    torch.manual_seed(0)
    rede = RedeProfunda(checkpoint=checkpoint)
    x = torch.randn(32, 1024, requires_grad=True)
    t0 = time.perf_counter()
    saida = rede(x)
    loss = saida.sum()
    loss.backward()
    dt = time.perf_counter() - t0
    # proxy de memória de ativações: nº de tensores intermediários que o grafo reteria
    return dt

print(f"{'modo':<22} {'tempo (fwd+bwd)':>16}")
print("-" * 40)
t_normal = medir(checkpoint=False)
t_ckpt = medir(checkpoint=True)
print(f"{'sem checkpointing':<22} {t_normal:>15.3f}s")
print(f"{'com checkpointing':<22} {t_ckpt:>15.3f}s")
print(f"\ncheckpointing ficou {t_ckpt/t_normal:.1f}x mais lento AQUI (CPU, camadas pequenas)")
print("⚠️ este fator é DISTORCIDO: em CPU com Linears minúsculas, o overhead de orquestração")
print("do checkpoint domina o recompute real. Em GPU com blocos transformer de verdade, o")
print("recompute é ~1 forward extra = +30-50% de tempo. O NÚMERO que transfere é a economia")
print("de MEMÓRIA abaixo (independente de hardware), não este tempo.")

# %%
# A economia de memória, calculada (o número que importa, e que o tempo não mostra):
def memoria_ativacoes(n_camadas, batch, d, checkpoint, bytes_por=4):
    if checkpoint:
        # guarda só ~√n checkpoints (a estratégia ótima) + as ativações de 1 bloco por vez
        n_guardadas = int(math.sqrt(n_camadas)) + 1
    else:
        n_guardadas = n_camadas
    return n_guardadas * batch * d * bytes_por

for n in [24, 48, 96]:
    sem = memoria_ativacoes(n, 32, 1024, False) / 1e6
    com = memoria_ativacoes(n, 32, 1024, True) / 1e6
    print(f"{n:>3} camadas: ativações sem ckpt {sem:>7.1f} MB | com √n ckpt {com:>6.1f} MB "
          f"| economia {1-com/sem:>5.0%}")

# %% [markdown]
# **O trade-off, agora completo:** checkpointing custa ~30-50% mais tempo (o recompute do
# forward no backward) e economiza memória de ativações de O(n) para O(√n). É por isso
# que ele é PADRÃO em treino de LLM — a memória é o gargalo (módulo 6), e trocar compute
# barato por memória escassa quase sempre compensa. No seu M4 de 16 GB, `--grad-checkpoint`
# (módulo 6) é essa técnica.
#
# ## Lab 4 — O particionamento do ZeRO/FSDP
#
# O módulo 3 disse: full fine-tune custa 16 bytes/param, e um 7B pede 112 GB. Como 8 GPUs
# de 80 GB treinam isso? Elas NÃO replicam tudo — o ZeRO/FSDP PARTICIONA os estados entre
# as GPUs. A conta:

# %%
def memoria_por_gpu(n_params, n_gpus, estagio):
    """Bytes por GPU. Estágios do ZeRO: 0=DDP (replica tudo), 1=otim, 2=+grad, 3=+params."""
    pesos = n_params * 2          # bf16
    grads = n_params * 2
    otim = n_params * 12          # master fp32 (4) + m (4) + v (4)
    if estagio == 0:              # DDP: cada GPU tem tudo
        return pesos + grads + otim
    if estagio == 1:              # ZeRO-1: particiona otimizador
        return pesos + grads + otim / n_gpus
    if estagio == 2:              # ZeRO-2: + gradientes
        return pesos + (grads + otim) / n_gpus
    if estagio == 3:              # ZeRO-3 / FSDP: + parâmetros
        return (pesos + grads + otim) / n_gpus

N = 7e9
print(f"Full fine-tune de um 7B em {8} GPUs — memória POR GPU (GB):")
print(f"{'estágio':<28} {'por GPU':>10} {'cabe em 80GB?':>15}")
print("-" * 56)
for est, nome in [(0, "DDP (replica tudo)"), (1, "ZeRO-1 (otim)"),
                  (2, "ZeRO-2 (+grad)"), (3, "ZeRO-3/FSDP (+params)")]:
    gb = memoria_por_gpu(N, 8, est) / 1e9
    print(f"{nome:<28} {gb:>9.1f}G {'sim' if gb < 80 else 'NÃO':>15}")

# %% [markdown]
# **A progressão do ZeRO é a resposta para "como treinar o que não cabe":** cada estágio
# particiona mais estado entre as GPUs, ao custo de mais comunicação. DDP não cabe;
# FSDP (ZeRO-3) cabe folgado, porque divide TUDO por 8. O preço está no Lab 5.
#
# ## Lab 5 — O volume de comunicação
#
# Particionar não é grátis: a cada passo, as GPUs precisam trocar dados para sincronizar.
# O DDP faz **all-reduce** dos gradientes (soma os gradientes de todas as GPUs); o FSDP
# faz **all-gather** dos parâmetros a cada camada (junta os pedaços antes de usar).

# %%
def bytes_comunicacao(n_params, n_gpus, estrategia):
    """Bytes que CADA GPU envia+recebe por passo (aproximação do ring-algorithm)."""
    if estrategia == "DDP":
        # all-reduce dos gradientes: ~2·(N-1)/N · tamanho por GPU
        return 2 * n_params * 2 * (n_gpus - 1) / n_gpus
    if estrategia == "FSDP":
        # all-gather params (fwd) + all-gather params (bwd) + reduce-scatter grads
        return 3 * n_params * 2 * (n_gpus - 1) / n_gpus

N = 7e9
print(f"comunicação por GPU por passo, 7B em 8 GPUs:")
for estrategia in ["DDP", "FSDP"]:
    gb = bytes_comunicacao(N, 8, estrategia) / 1e9
    # tempo a 400 GB/s (NVLink) vs 50 GB/s (Ethernet rápida)
    print(f"  {estrategia}: {gb:>5.1f} GB/passo  →  "
          f"{gb/400*1000:>5.0f} ms em NVLink | {gb/50*1000:>5.0f} ms em Ethernet")

print("\n→ FSDP comunica ~50% mais que DDP (troca params além de grads).")
print("→ Em NVLink (dentro de um nó) é barato; em Ethernet (entre nós) DOMINA o passo.")
print("  É por isso que FSDP escala bem dentro de um nó e mal entre nós sem rede rápida.")

# %% [markdown]
# ## Lab 6 — A bolha do pipeline
#
# Pipeline parallelism divide as CAMADAS entre GPUs (GPU 0: camadas 1-8, GPU 1: 9-16...).
# Mas há um problema: enquanto a GPU 0 processa o micro-batch 1, as outras esperam. Essa
# ociosidade é a **bolha**.

# %%
def eficiencia_pipeline(n_estagios, n_microbatches):
    """Fração do tempo em que as GPUs estão ÚTEIS (não na bolha)."""
    # tempo total = (microbatches + estagios - 1) slots; útil = microbatches·estagios / n_estagios
    util = n_microbatches
    total = n_microbatches + n_estagios - 1
    return util / total

print(f"{'micro-batches':>14} {'eficiência (4 estágios)':>26}")
print("-" * 42)
for mb in [1, 4, 8, 16, 32]:
    ef = eficiencia_pipeline(4, mb)
    print(f"{mb:>14} {ef:>24.0%}  {'█' * int(ef*20)}")

# %% [markdown]
# **A bolha explica por que 4 GPUs em pipeline não são 4× mais rápidas.** Com 1
# micro-batch, 3 das 4 GPUs ficam ociosas na maior parte do tempo (eficiência ~25%);
# com muitos micro-batches, a bolha se dilui e a eficiência sobe. É por isso que treinos
# reais usam dezenas de micro-batches por passo — e por que pipeline é a última escolha
# de paralelismo (só entre nós, quando a rede é o gargalo).
#
# O quadro completo do paralelismo (o mapa do módulo 3, agora com os custos):
#
# | Estratégia | Divide | Custo | Quando |
# |---|---|---|---|
# | **DDP** | o batch | all-reduce de grads | modelo cabe numa GPU |
# | **FSDP/ZeRO-3** | todos os estados | all-gather a cada camada | modelo não cabe (1ª escolha) |
# | **Tensor** | matrizes dentro da camada | all-reduce a CADA camada | dentro de um nó, NVLink |
# | **Pipeline** | camadas entre GPUs | a bolha | entre nós, rede lenta |
#
# ---
#
# ## Encerramento
#
# Você construiu e mediu a maquinaria que treina LLMs:
#
# - **autograd do zero** — o motor de `.backward()`, verificado contra o PyTorch;
# - **backprop de uma MLP na mão** — incluindo a derivada elegante da cross-entropy;
# - **gradient checkpointing** — o trade-off memória/compute que você usa no `--grad-checkpoint`;
# - **ZeRO/FSDP** — como particionar estados faz o que não cabe caber;
# - **comunicação** — por que FSDP escala dentro do nó e sofre entre nós;
# - **a bolha do pipeline** — por que N GPUs não dão N× de velocidade.
#
# Nada disso exigiu uma GPU — só entender o que a GPU faz. E esse entendimento é o que
# separa "chamei `trainer.train()`" de "sei por que meu treino distribuído está lento".
#
# O módulo 18 fecha a Fase 2 na fronteira: as arquiteturas que estão desafiando o
# transformer — Mamba, MLA, e o mundo multimodal.

# %%
