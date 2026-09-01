# %% [markdown]
# # Módulo 6 — Laboratório A: LoRA e quantização do zero
#
# **Roda em CPU, no Windows ou no Mac, sem alteração.** Este lab implementa e *verifica
# numericamente* tudo o que o módulo ensina — usando pesos reais do Qwen2.5-0.5B e o
# MiniGPT do módulo 3 para os treinos.
#
# O `lab_mlx.py` traz a receita de produção no seu M4. Este aqui é o que prova que ela
# funciona.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Por que "baixo posto"? SVD e a estrutura da atualização |
# | 2 | LoRA implementado do zero |
# | 3 | Aplicando ao MiniGPT: quantos parâmetros de fato treinam |
# | 4 | **LoRA vs full fine-tune: treino comparado** |
# | 5 | Rank vs qualidade |
# | 6 | LoRA em pesos reais do Qwen2.5-0.5B |
# | 7 | Merge: provando que `W + BA` é equivalente |
# | 8 | **Quantização NF4 do zero, medida em pesos reais** |
# | 9 | O orçamento de memória do QLoRA |

# %%
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(AQUI.parent / "tools"))
import minigpt

torch.manual_seed(0)

# %% [markdown]
# ## Lab 1 — Por que baixo posto funciona
#
# A hipótese do LoRA: a **atualização** que o fine-tuning aplica aos pesos,
# `ΔW = W_final − W_inicial`, tem *posto intrínseco baixo* — mesmo que `W` seja de posto
# cheio. Ou seja, `ΔW` pode ser bem aproximada por `B·A` com `B` e `A` finas.
#
# Isso é testável. Vamos fine-tunar o MiniGPT de verdade e olhar o espectro de `ΔW`.

# %%

corpus_path = AQUI.parent / "modulo-03-treino" / "data" / "corpus.txt"
assert corpus_path.exists(), "rode antes: python ../modulo-03-treino/dados.py"
texto = corpus_path.read_text(encoding="utf-8")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB = 2048
tk = Tokenizer(models.BPE(unk_token=None))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator([texto], trainers.BpeTrainer(vocab_size=VOCAB, show_progress=False))

ids = torch.tensor(tk.encode(texto).ids, dtype=torch.long)
n = int(0.9 * len(ids))
dados_treino, dados_val = ids[:n], ids[n:]
print(f"corpus: {len(ids):,} tokens | treino {len(dados_treino):,} | val {len(dados_val):,}")

# %%
cfg = minigpt.Config(vocab=VOCAB)

print("treinando o modelo 'pré-treinado' (que fará o papel de base)...")
t0 = time.perf_counter()
base, _ = minigpt.treinar(cfg, dados_treino, dados_val, passos=300, batch=16, seed=1337)
print(f"  {time.perf_counter() - t0:.0f}s | val loss {minigpt.avaliar(base, dados_val):.4f}")

# %%
# Agora um fine-tune COMPLETO num subdomínio: só os diálogos (linhas com travessão).
dialogos = "\n".join(l for l in texto.split("\n") if l.strip().startswith(("--", "—")))
ids_dlg = torch.tensor(tk.encode(dialogos).ids, dtype=torch.long)
print(f"subdomínio 'diálogos': {len(ids_dlg):,} tokens ({len(ids_dlg)/len(ids):.1%} do corpus)")

import copy

ajustado = copy.deepcopy(base)
otim = torch.optim.AdamW(minigpt.grupos_de_parametros(ajustado), lr=3e-4, betas=(0.9, 0.95))
for passo in range(150):
    x, y = minigpt.pegar_batch(ids_dlg, 16, cfg.bloco)
    otim.zero_grad(set_to_none=True)
    _, perda = ajustado(x, y)
    perda.backward()
    torch.nn.utils.clip_grad_norm_(ajustado.parameters(), 1.0)
    otim.step()
print(f"fine-tune completo concluído | loss final {perda.item():.4f}")

# %% [markdown]
# ### O espectro de ΔW
#
# Agora a verificação central: quantos valores singulares de `ΔW` carregam a energia?

# %%
def espectro(delta):
    s = torch.linalg.svdvals(delta.float())
    energia = (s ** 2).cumsum(0) / (s ** 2).sum()
    return s, energia

print(f"{'camada':<28} {'posto':>7} {'r p/ 50%':>9} {'r p/ 90%':>9} {'r p/ 99%':>9}")
print("-" * 66)
for nome, p_base in base.named_parameters():
    if p_base.dim() != 2 or "emb" in nome:
        continue
    delta = dict(ajustado.named_parameters())[nome].data - p_base.data
    s, energia = espectro(delta)
    r50 = int((energia < 0.50).sum()) + 1
    r90 = int((energia < 0.90).sum()) + 1
    r99 = int((energia < 0.99).sum()) + 1
    print(f"{nome:<28} {min(delta.shape):>7} {r50:>9} {r90:>9} {r99:>9}")

# %% [markdown]
# **Leia a coluna `r p/ 90%`.** Se ela for muito menor que a coluna `posto`, a hipótese do
# LoRA se sustenta nesse modelo: quase toda a atualização vive num subespaço de dimensão
# pequena, e aproximá-la com `B·A` de posto `r` perde pouco.
#
# > 📐 Note que isso vale para **ΔW**, não para `W`. Os pesos originais são de posto cheio
# > e carregam todo o conhecimento do pré-treino. O que é de baixo posto é a *correção*
# > que o fine-tuning aplica — o que faz sentido: você está mudando comportamento, não
# > reconstruindo o modelo.
#
# ## Lab 2 — LoRA do zero

# %%
class LoRALinear(nn.Module):
    """Envolve uma nn.Linear congelada e soma uma atualização de posto r.

        y = W x + (alpha/r) · B(A x)

    A ~ N(0, σ²) e B = 0  →  no início, BA = 0 e a saída é IDÊNTICA à original.
    """

    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False              # a base congela

        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B fica em ZERO — é o que garante saída idêntica no passo 0.

    def forward(self, x):
        return self.base(x) + self.scaling * F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Funde o adaptador nos pesos: W' = W + (alpha/r)·B·A."""
        fundido = nn.Linear(self.base.in_features, self.base.out_features,
                            bias=self.base.bias is not None)
        fundido.weight.copy_(self.base.weight + self.scaling * (self.lora_B @ self.lora_A))
        if self.base.bias is not None:
            fundido.bias.copy_(self.base.bias)
        return fundido

# %%
# Verificação 1: no passo 0, LoRA é a identidade.
linear = nn.Linear(896, 896)
lora = LoRALinear(linear, r=8, alpha=16)
x = torch.randn(4, 32, 896)

print(f"saída idêntica no início? {torch.allclose(linear(x), lora(x), atol=1e-6)}")
print(f"  (porque lora_B = 0: {bool((lora.lora_B == 0).all())})")

# Verificação 2: depois de mexer em B, deixa de ser identidade.
with torch.no_grad():
    lora.lora_B.normal_(0, 0.02)
print(f"após treinar B, ainda idêntica? {torch.allclose(linear(x), lora(x), atol=1e-6)}")

# Verificação 3: o merge preserva exatamente a função.
fundido = lora.merge()
print(f"merge preserva a saída? {torch.allclose(lora(x), fundido(x), atol=1e-5)}")
print(f"  erro máximo: {(lora(x) - fundido(x)).abs().max():.2e}")

# %% [markdown]
# Essas três verificações são o LoRA inteiro:
#
# 1. **Começa como identidade** — o fine-tuning parte exatamente do modelo base, sem choque.
# 2. **Aprende uma perturbação de baixo posto** — só `A` e `B` recebem gradiente.
# 3. **Funde sem perda** — em produção não há custo de latência.
#
# ## Lab 3 — Quantos parâmetros realmente treinam

# %%
def aplicar_lora(modelo, alvos=("qkv", "saida"), r=8, alpha=16):
    """Substitui as nn.Linear cujo nome contém algum dos alvos."""
    trocados = []
    for nome, modulo in modelo.named_modules():
        for filho_nome, filho in list(modulo.named_children()):
            if isinstance(filho, nn.Linear) and any(a in filho_nome for a in alvos):
                setattr(modulo, filho_nome, LoRALinear(filho, r=r, alpha=alpha))
                trocados.append(f"{nome}.{filho_nome}")
    return trocados

def contar(modelo):
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    return treinaveis, total

modelo_lora = copy.deepcopy(base)
for p in modelo_lora.parameters():
    p.requires_grad = False
trocados = aplicar_lora(modelo_lora, alvos=("qkv", "saida"), r=8)

tr, tot = contar(modelo_lora)
print(f"camadas adaptadas: {len(trocados)}")
print(f"parâmetros treináveis: {tr:>10,}")
print(f"parâmetros totais    : {tot:>10,}")
print(f"fração treinável     : {tr/tot:>10.3%}")

# %%
# A conta que explica a economia, para uma matriz [in, out] com posto r:
print(f"{'matriz':<18} {'full (in×out)':>16} {'LoRA r=8':>12} {'razão':>10}")
print("-" * 60)
for nome, i, o in [("MiniGPT qkv", 192, 576), ("Qwen 0.5B q_proj", 896, 896),
                   ("Llama-3-8B q_proj", 4096, 4096), ("Llama-3-8B down_proj", 14336, 4096)]:
    full = i * o
    lora_p = 8 * (i + o)
    print(f"{nome:<18} {full:>16,} {lora_p:>12,} {full/lora_p:>9.0f}x")

# %% [markdown]
# ## Lab 4 — LoRA vs full fine-tune, medido
#
# Mesmo modelo base, mesmo subdomínio, mesmo número de passos. Muda só o que é treinável.
# Os batches de treino e avaliação também são idênticos entre variantes: cada função usa
# um Generator próprio com seed fixa, sem depender das avaliações executadas antes.

# %%
def treinar_no_subdominio(modelo, passos=150, lr=3e-4, so_treinaveis=True, seed=2024):
    params = [p for p in modelo.parameters() if p.requires_grad] if so_treinaveis else list(modelo.parameters())
    otim = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95))
    generator = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for _ in range(passos):
        x, y = minigpt.pegar_batch(ids_dlg, 16, cfg.bloco, generator=generator)
        otim.zero_grad(set_to_none=True)
        _, perda = modelo(x, y)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        otim.step()
    return perda.item(), time.perf_counter() - t0

@torch.no_grad()
def perda_em(modelo, fonte, n=30, seed=0):
    modelo.eval()
    generator = torch.Generator().manual_seed(seed)
    v = sum(
        modelo(*minigpt.pegar_batch(fonte, 8, cfg.bloco, generator=generator))[1].item()
        for _ in range(n)
    ) / n
    modelo.train()
    return v

alvo_antes = perda_em(base, ids_dlg)
geral_antes = perda_em(base, dados_val)
print(f"MODELO BASE      alvo (diálogos) {alvo_antes:.4f} | geral {geral_antes:.4f}")

# %%
resultados = {}

# (a) full fine-tune
m_full = copy.deepcopy(base)
perda_f, tempo_f = treinar_no_subdominio(m_full, so_treinaveis=False)
resultados["full fine-tune"] = {
    "alvo": perda_em(m_full, ids_dlg), "geral": perda_em(m_full, dados_val),
    "treinaveis": sum(p.numel() for p in m_full.parameters()), "tempo": tempo_f}

# (b) LoRA em vários ranks
for r in [1, 4, 8, 32]:
    m = copy.deepcopy(base)
    for p in m.parameters():
        p.requires_grad = False
    torch.manual_seed(1000 + r)
    aplicar_lora(m, alvos=("qkv", "saida"), r=r, alpha=2 * r)
    perda_l, tempo_l = treinar_no_subdominio(m)
    tr, _ = contar(m)
    resultados[f"LoRA r={r}"] = {
        "alvo": perda_em(m, ids_dlg), "geral": perda_em(m, dados_val),
        "treinaveis": tr, "tempo": tempo_l}

# %%
print(f"{'método':<16} {'treináveis':>12} {'%':>7} {'alvo':>8} {'geral':>8} "
      f"{'esquec.':>9} {'tempo':>7}")
print("-" * 74)
print(f"{'(base)':<16} {'—':>12} {'—':>7} {alvo_antes:>8.4f} {geral_antes:>8.4f} {'—':>9} {'—':>7}")
for nome, r in resultados.items():
    esquec = (r["geral"] - geral_antes) / geral_antes
    print(f"{nome:<16} {r['treinaveis']:>12,} {r['treinaveis']/tot:>6.2%} "
          f"{r['alvo']:>8.4f} {r['geral']:>8.4f} {esquec:>+8.1%} {r['tempo']:>6.0f}s")

# %% [markdown]
# **As três colunas que importam:**
#
# - `alvo` — quanto o modelo melhorou no subdomínio. Menor é melhor.
# - `geral` — quanto ele piorou no corpus completo. É o **catastrophic forgetting** do
#   módulo 5, agora medido diretamente.
# - `esquec.` — a variação percentual da coluna geral. Compare full fine-tune com LoRA.
#
# ## Lab 5 — Rank vs qualidade
#
# Os dados do Lab 4, isolando o efeito do rank.

# %%
print(f"{'rank':>6} {'treináveis':>12} {'loss no alvo':>14} {'ganho vs base':>15}")
print("-" * 52)
for r in [1, 4, 8, 32]:
    d = resultados[f"LoRA r={r}"]
    print(f"{r:>6} {d['treinaveis']:>12,} {d['alvo']:>14.4f} "
          f"{(alvo_antes - d['alvo'])/alvo_antes:>14.1%}")
print(f"{'full':>6} {resultados['full fine-tune']['treinaveis']:>12,} "
      f"{resultados['full fine-tune']['alvo']:>14.4f} "
      f"{(alvo_antes - resultados['full fine-tune']['alvo'])/alvo_antes:>14.1%}")

# %% [markdown]
# ## Lab 6 — LoRA em pesos reais do Qwen2.5-0.5B
#
# O MiniGPT prova o algoritmo. Agora as contas num modelo de verdade.

# %%
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

V5 = int(transformers.__version__.split(".")[0]) >= 5
DTYPE_KW = {"dtype": torch.float32} if V5 else {"torch_dtype": torch.float32}

qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    **DTYPE_KW,
)
tok_qwen = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
)
qcfg = qwen.config
print(f"Qwen2.5-0.5B: {sum(p.numel() for p in qwen.parameters()):,} parâmetros")

# %%
ALVOS = {
    "só q,v (default do MLX)": ["q_proj", "v_proj"],
    "atenção completa": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "todas as lineares (QLoRA)": ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
}

print(f"{'conjunto de alvos':<28} {'r=8':>12} {'r=16':>12} {'r=64':>12} {'% do modelo (r=8)':>19}")
print("-" * 88)
total_qwen = sum(p.numel() for p in qwen.parameters())
for nome, alvos in ALVOS.items():
    linha = f"{nome:<28}"
    for r in [8, 16, 64]:
        n_lora = sum(r * (m.in_features + m.out_features)
                     for nm, m in qwen.named_modules()
                     if isinstance(m, nn.Linear) and any(a in nm for a in alvos))
        linha += f" {n_lora:>12,}"
        if r == 8:
            frac = n_lora / total_qwen
    print(linha + f" {frac:>18.3%}")

# %% [markdown]
# Compare com o módulo 2: a atenção inteira é 12,3% dos parâmetros de um bloco e o MLP
# 87,7%. O default do MLX (`q_proj`, `v_proj`) adapta **menos de 1%** do modelo. O QLoRA
# original recomenda todas as lineares — mais capacidade, mais memória.
#
# ## Lab 7 — Merge, em pesos reais

# %%
camada = qwen.model.layers[0].self_attn.q_proj
print(f"q_proj original: {tuple(camada.weight.shape)}")

lora_q = LoRALinear(camada, r=8, alpha=16)
with torch.no_grad():                       # simula um adaptador treinado
    lora_q.lora_B.normal_(0, 0.01)

entrada = torch.randn(2, 16, camada.in_features)
fundida = lora_q.merge()

print(f"LoRA vs fundido — erro máximo: {(lora_q(entrada) - fundida(entrada)).abs().max():.2e}")
print(f"posto de ΔW = B·A: {torch.linalg.matrix_rank(lora_q.lora_B @ lora_q.lora_A).item()} "
      f"(de {min(camada.weight.shape)} possíveis)")

# %% [markdown]
# ## Lab 8 — Quantização NF4, do zero
#
# O "Q" do QLoRA. NF4 (*4-bit NormalFloat*) usa 16 níveis posicionados nos **quantis de
# uma normal**, não uniformemente — porque pesos de redes neurais são aproximadamente
# gaussianos, e níveis uniformes desperdiçam resolução nas caudas vazias.

# %%
# Os 16 níveis NF4 do paper QLoRA (Dettmers et al., 2023).
NF4 = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
])

def quantizar_nf4(w, bloco=64):
    """Quantiza por blocos: cada bloco é normalizado pelo seu absmax e mapeado a NF4."""
    forma = w.shape
    achatado = w.flatten()
    pad = (-len(achatado)) % bloco
    if pad:
        achatado = torch.cat([achatado, torch.zeros(pad)])
    blocos = achatado.view(-1, bloco)

    absmax = blocos.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    normalizado = blocos / absmax
    indices = (normalizado.unsqueeze(-1) - NF4).abs().argmin(dim=-1)   # 4 bits por peso
    return indices, absmax, forma, pad

def desquantizar_nf4(indices, absmax, forma, pad):
    blocos = NF4[indices] * absmax
    achatado = blocos.flatten()
    if pad:
        achatado = achatado[:-pad]
    return achatado.view(forma)

def quantizar_int4(w, bloco=64):
    """Baseline: quantização LINEAR de 4 bits, para comparação."""
    forma = w.shape
    achatado = w.flatten()
    pad = (-len(achatado)) % bloco
    if pad:
        achatado = torch.cat([achatado, torch.zeros(pad)])
    blocos = achatado.view(-1, bloco)
    absmax = blocos.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    q = ((blocos / absmax) * 7).round().clamp(-8, 7)      # 16 níveis uniformes
    rec = (q / 7) * absmax
    achatado = rec.flatten()
    if pad:
        achatado = achatado[:-pad]
    return achatado.view(forma)

# %%
# Verificação em PESOS REAIS do Qwen.
print(f"{'camada':<34} {'erro NF4':>12} {'erro int4':>12} {'NF4 melhor?':>13}")
print("-" * 74)
melhorias = []
for nome, modulo in list(qwen.named_modules()):
    if isinstance(modulo, nn.Linear) and any(k in nome for k in ["q_proj", "gate_proj", "down_proj"]):
        w = modulo.weight.data
        rec_nf4 = desquantizar_nf4(*quantizar_nf4(w))
        rec_int4 = quantizar_int4(w)
        e_nf4 = (w - rec_nf4).pow(2).mean().sqrt()
        e_int4 = (w - rec_int4).pow(2).mean().sqrt()
        melhorias.append(float(e_int4 / e_nf4))
        print(f"{nome[-34:]:<34} {e_nf4:>12.6f} {e_int4:>12.6f} "
              f"{e_int4/e_nf4:>12.2f}x")
        if len(melhorias) >= 6:
            break
print(f"\nNF4 tem erro {sum(melhorias)/len(melhorias):.2f}x menor que int4 linear, em média")

# %%
# Por que: a distribuição dos pesos é aproximadamente normal.
w = qwen.model.layers[0].mlp.gate_proj.weight.data.flatten()
print(f"pesos de gate_proj: média {w.mean():+.5f} | desvio {w.std():.5f} | "
      f"curtose {float(((w - w.mean())**4).mean() / w.var()**2):.2f} (normal = 3.0)")

quantis = torch.tensor([0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
print(f"\n{'quantil':>9} {'peso real':>12} {'normal teórica':>16}")
for q in quantis:
    real = torch.quantile(w, q)
    teorico = w.mean() + w.std() * math.sqrt(2) * torch.erfinv(2 * q - 1)
    print(f"{float(q):>9.3f} {float(real):>12.5f} {float(teorico):>16.5f}")

# %% [markdown]
# Os quantis reais seguem de perto os da normal — é exatamente a premissa que o NF4
# explora. Níveis uniformes (int4) gastam metade da resolução em regiões onde quase não
# há pesos.
#
# ### O impacto de quantizar no modelo inteiro

# %%
def perplexidade_qwen(modelo, texto):
    ids = tok_qwen(texto, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        return math.exp(modelo(ids, labels=ids).loss.item())

# ⚠️ Uma única frase curta NÃO mede degradação de quantização — a variância entre textos
# é maior que o efeito. Precisa de vários textos, longos, e da média.
TEXTOS_AVALIACAO = {
    "literatura PT": texto[5000:9000],
    "diálogo PT": dialogos[2000:5000],
    "técnico PT": ("A quantização reduz a precisão numérica dos pesos de uma rede neural "
                   "para diminuir o consumo de memória. Em vez de armazenar cada parâmetro "
                   "em 32 bits, utilizam-se 8 ou 4 bits, com uma constante de escala por "
                   "bloco que preserva a faixa dinâmica original. O erro introduzido é "
                   "pequeno quando a distribuição dos pesos é aproximadamente normal. ") * 4,
    "inglês": ("The transformer architecture relies on self-attention to model dependencies "
               "between tokens regardless of their distance in the sequence. Each layer "
               "refines the representation through residual connections. ") * 6,
    "código": ("def quantize(weights, bits=4):\n"
               "    scale = weights.abs().max() / (2 ** (bits - 1) - 1)\n"
               "    return (weights / scale).round().clamp(-8, 7), scale\n\n") * 6,
}

qwen_q = copy.deepcopy(qwen)
n_quantizadas = 0
with torch.no_grad():
    for nome, modulo in qwen_q.named_modules():
        if isinstance(modulo, nn.Linear) and "lm_head" not in nome:
            modulo.weight.copy_(desquantizar_nf4(*quantizar_nf4(modulo.weight.data)))
            n_quantizadas += 1

print(f"camadas quantizadas para 4 bits: {n_quantizadas}\n")
print(f"{'texto':<16} {'tokens':>8} {'PPL fp32':>10} {'PPL NF4':>10} {'degradação':>12}")
print("-" * 60)
degradacoes = []
for nome, txt in TEXTOS_AVALIACAO.items():
    n_tok = len(tok_qwen(txt)["input_ids"])
    p0 = perplexidade_qwen(qwen, txt)
    p1 = perplexidade_qwen(qwen_q, txt)
    degradacoes.append(p1 / p0 - 1)
    print(f"{nome:<16} {n_tok:>8,} {p0:>10.3f} {p1:>10.3f} {degradacoes[-1]:>+11.1%}")

media = sum(degradacoes) / len(degradacoes)
print(f"\ndegradação média: {media:+.1%}")
print(f"faixa           : {min(degradacoes):+.1%} a {max(degradacoes):+.1%}")

# %% [markdown]
# > ⚠️ **Por que a medição precisa de vários textos.** Numa primeira versão deste lab eu
# > medi com uma única frase de 30 tokens e obtive **−13%** — a quantização teria
# > *melhorado* o modelo, o que é impossível. Perplexidade em textos curtos tem variância
# > enorme; o ruído da quantização pode, por acaso, favorecer uma sequência específica.
# >
# > Este é o erro de avaliação mais comum em posts sobre quantização: "quantizei e a
# > perplexidade quase não mudou", medido em um parágrafo. Meça em milhares de tokens,
# > em domínios variados, e reporte a **faixa** além da média.

# %% [markdown]
# > 🔧 Note que quantizamos **e desquantizamos** para float32 — a simulação mede o erro
# > de quantização, não a economia real de memória. Numa implementação de verdade os
# > pesos ficam armazenados em 4 bits e são desquantizados sob demanda, bloco a bloco,
# > durante o forward. É o que o `mlx_lm.convert -q` e o `bitsandbytes` fazem.
#
# ## Lab 9 — O orçamento do QLoRA
#
# Juntando tudo: por que 7B cabe em 16 GB.

# %%
def orcamento(n_params, r=16, frac_alvos=1.0, otimizador_bytes=8):
    """frac_alvos: fração dos parâmetros nas camadas adaptadas."""
    n_lora = n_params * frac_alvos * 0.005 * (r / 8)     # aproximação empírica
    return {
        "full": n_params * 16 / 1e9,
        "lora": (n_params * 2 + n_lora * (2 + otimizador_bytes)) / 1e9,
        "qlora": (n_params * 0.5 + n_lora * (2 + otimizador_bytes)) / 1e9,
        "infer4": n_params * 0.5 / 1e9,
    }

LIMITE = 10.0     # ~10 GB úteis nos 16 GB do Mac (o resto é macOS)
ROTULOS = {"full": "full FT", "lora": "LoRA", "qlora": "QLoRA", "infer4": "inferência"}

print(f"{'modelo':<16} {'full FT':>9} {'LoRA':>9} {'QLoRA':>9} {'infer 4bit':>11}   cabe em 16 GB")
print("-" * 84)
for nome, n in [("Qwen2.5-0.5B", 494e6), ("Qwen2.5-1.5B", 1.54e9),
                ("Qwen2.5-3B", 3.09e9), ("Qwen2.5-7B", 7.62e9), ("Llama-3-8B", 8.03e9)]:
    o = orcamento(n)
    cabe = [ROTULOS[k] for k, v in o.items() if v < LIMITE]
    print(f"{nome:<16} {o['full']:>8.1f}G {o['lora']:>8.1f}G {o['qlora']:>8.1f}G "
          f"{o['infer4']:>10.1f}G   {', '.join(cabe) if cabe else 'NADA'}")

# %% [markdown]
# > ⚠️ Estes são os **pesos e estados do otimizador**. Faltam ativações e KV cache, que
# > dependem de `batch_size` e `max_seq_length` e podem facilmente somar vários GB. Trate
# > a tabela como piso, não como teto — e use `--grad-checkpoint` quando apertar.
#
# ---
#
# ## Encerramento
#
# Tudo neste lab foi **verificado numericamente**:
#
# - o espectro de `ΔW` de um fine-tune real, sustentando a hipótese de baixo posto;
# - LoRA como identidade no passo 0, e o merge preservando a função a `1e-5`;
# - LoRA vs full fine-tune com forgetting medido nos dois;
# - NF4 com erro menor que int4 linear em pesos reais, e a razão (os quantis gaussianos);
# - a degradação de perplexidade ao quantizar o Qwen inteiro para 4 bits.
#
# Agora o `lab_mlx.py` aplica isso ao seu M4, com modelos de verdade.

# %%
