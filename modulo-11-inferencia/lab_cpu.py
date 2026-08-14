# %% [markdown]
# # Módulo 11 — Laboratório A: MoE e decodificação especulativa do zero
#
# **Roda em CPU (Windows ou Mac), ~12 minutos.**
#
# | Lab | Assunto |
# |---|---|
# | 1 | MoE do zero: roteador, top-k, ativos vs totais |
# | 2 | **O colapso de roteamento — produzido e consertado** |
# | 3 | MoE vs denso: mesmo compute ativo, treino comparado |
# | 4 | **Decodificação especulativa do zero, com prova de equivalência** |

# %%
import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd().parent / "tools"))
import minigpt

torch.manual_seed(0)

# %% [markdown]
# ## Lab 1 — MoE do zero
#
# O MLP do módulo 2, multiplicado por E, com um roteador na frente.

# %%
class MoE(nn.Module):
    """Substitui o MLP denso: E experts SwiGLU, top-k roteamento, loss de balanceamento."""

    def __init__(self, d, d_ff, n_experts=4, top_k=1):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.roteador = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(
            minigpt.MLP(type("C", (), {"d": d, "d_ff": d_ff})) for _ in range(n_experts))
        self.ultima_aux = torch.tensor(0.0)   # exposta para o loop de treino
        self.contagem = torch.zeros(n_experts)  # utilização acumulada (diagnóstico)

    def forward(self, x):
        b, s, d = x.shape
        achatado = x.view(-1, d)                                  # [N, d], N = b*s
        scores = self.roteador(achatado)                          # [N, E]
        probs = F.softmax(scores, dim=-1)
        pesos_topk, indices = probs.topk(self.top_k, dim=-1)      # [N, k]
        pesos_topk = pesos_topk / pesos_topk.sum(-1, keepdim=True)

        # ---- loss auxiliar de balanceamento (Switch): E · Σ f_i · P_i ----
        f = torch.zeros(self.n_experts)
        for k in range(self.top_k):
            f += torch.bincount(indices[:, k], minlength=self.n_experts).float()
        f = f / f.sum()
        P = probs.mean(0)
        self.ultima_aux = self.n_experts * (f * P).sum()
        self.contagem += f.detach()

        # ---- despacho: cada token visita só os seus k experts ----
        saida = torch.zeros_like(achatado)
        for e in range(self.n_experts):
            mascara = (indices == e)                              # [N, k]
            if not mascara.any():
                continue
            tokens, qual_k = mascara.nonzero(as_tuple=True)
            saida[tokens] += pesos_topk[tokens, qual_k].unsqueeze(1) * \
                self.experts[e](achatado[tokens])
        return saida.view(b, s, d)


# Verificações mecânicas antes de treinar qualquer coisa:
moe = MoE(d=64, d_ff=128, n_experts=4, top_k=1)
x = torch.randn(2, 8, 64)
y = moe(x)

n_total = sum(p.numel() for p in moe.experts.parameters())
n_ativo = n_total // 4 * 1 + moe.roteador.weight.numel()
print(f"saída: {tuple(y.shape)} (mesma forma da entrada — o MoE é um substituto do MLP)")
print(f"parâmetros totais dos experts : {n_total:,}")
print(f"ativos por token (top-1 + roteador): {n_ativo:,}  ({n_ativo/n_total:.0%})")
print(f"loss auxiliar (balanceado ≈ 1.0): {float(moe.ultima_aux):.3f}")

# Cada token realmente visita só 1 expert?
_ = moe(torch.randn(1, 100, 64))
print(f"tokens por expert neste batch: {[f'{v:.0%}' for v in (moe.contagem/moe.contagem.sum())]}")

# %% [markdown]
# ## Lab 2 — O colapso, produzido em cativeiro
#
# MiniGPT com MoE no lugar do MLP. Treinamos DUAS vezes: sem e com a loss auxiliar.
# A previsão da teoria: sem ela, winner-take-all.

# %%
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

corpus_path = Path.cwd().parent / "modulo-03-treino" / "data" / "corpus.txt"
assert corpus_path.exists(), "rode antes: python ../modulo-03-treino/dados.py"
texto = corpus_path.read_text(encoding="utf-8")

VOCAB = 2048
tk = Tokenizer(models.BPE(unk_token=None))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator([texto], trainers.BpeTrainer(vocab_size=VOCAB, show_progress=False))
ids = torch.tensor(tk.encode(texto).ids, dtype=torch.long)
n = int(0.9 * len(ids))
dados_treino, dados_val = ids[:n], ids[n:]

# %%
class BlocoMoE(nn.Module):
    def __init__(self, cfg, n_experts, top_k):
        super().__init__()
        self.norm1, self.norm2 = minigpt.RMSNorm(cfg.d), minigpt.RMSNorm(cfg.d)
        self.attn = minigpt.Atencao(cfg)
        self.moe = MoE(cfg.d, cfg.d_ff, n_experts, top_k)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.moe(self.norm2(x))


class MiniGPTMoE(minigpt.MiniGPT):
    def __init__(self, cfg, n_experts=4, top_k=1):
        super().__init__(cfg)
        self.blocos = nn.ModuleList(BlocoMoE(cfg, n_experts, top_k)
                                    for _ in range(cfg.n_camadas))
        self.apply(self._init)

    def loss_auxiliar(self):
        return sum(b.moe.ultima_aux for b in self.blocos) / len(self.blocos)

    def utilizacao(self):
        u = sum(b.moe.contagem for b in self.blocos)
        return u / u.sum()

    def zerar_contagem(self):
        for b in self.blocos:
            b.moe.contagem.zero_()


def treinar_moe(alpha_aux, passos=300, seed=1337):
    torch.manual_seed(seed)
    cfg = minigpt.Config(vocab=VOCAB)
    m = MiniGPTMoE(cfg, n_experts=4, top_k=1)
    otim = torch.optim.AdamW(minigpt.grupos_de_parametros(m), lr=1e-3, betas=(0.9, 0.95))
    for passo in range(passos):
        taxa = minigpt.agenda_lr(passo, passos, pico=1e-3)
        for g in otim.param_groups:
            g["lr"] = taxa
        x, y = minigpt.pegar_batch(dados_treino, 16, cfg.bloco)
        _, perda = m(x, y)
        perda = perda + alpha_aux * m.loss_auxiliar()
        otim.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        otim.step()
    return m

print("treinando SEM loss de balanceamento (alpha=0)...")
t0 = time.perf_counter()
moe_sem = treinar_moe(alpha_aux=0.0)
print(f"  {time.perf_counter()-t0:.0f}s")
print("treinando COM loss de balanceamento (alpha=0.01)...")
moe_com = treinar_moe(alpha_aux=0.01)

# %%
# Utilização medida em dados NOVOS (validação):
for nome, m in [("SEM balanceamento", moe_sem), ("COM balanceamento", moe_com)]:
    m.zerar_contagem()
    with torch.no_grad():
        for _ in range(20):
            x, _ = minigpt.pegar_batch(dados_val, 8, 128)
            m(x)
    u = m.utilizacao()
    ppl = math.exp(minigpt.avaliar(m, dados_val))
    barra = " ".join(f"E{i}:{'█' * max(1, int(40 * v))}{v:.0%}" for i, v in enumerate(u))
    print(f"\n{nome}  (PPL {ppl:.1f})")
    for i, v in enumerate(u):
        print(f"  expert {i}: {'█' * max(1, int(50 * float(v)))} {float(v):.1%}")

# %% [markdown]
# ### O que a execução mostrou — mais matizado que a teoria
#
# Na execução de referência, o colapso TOTAL não aconteceu em 300 passos: sem a loss,
# a utilização derivou para 32/20/31/17% (spread de 2×, e crescendo); com `α=0,01`,
# ficou em 23/25/27/25% — quase uniforme. E as PPLs empataram (230,7 vs 235,2 — a loss
# auxiliar custa um pouco de loss principal; é um imposto).
#
# Três leituras honestas:
#
# 1. O colapso é um **processo de realimentação** — em 300 passos num modelo minúsculo,
#    vê-se a deriva (2× de spread), não o estado final. Em treinos reais (milhões de
#    passos), a deriva composta É o colapso; a literatura o documenta amplamente.
# 2. A loss auxiliar **não é grátis**: ela troca um pouco de loss principal por
#    utilização uniforme. O retorno vem depois — experts balanceados continuam todos
#    treinando, e a capacidade extra rende com mais dados.
# 3. Se você quiser VER o colapso completo, o exercício B1 estende o treino e mede a
#    utilização ao longo do tempo — a curva de spread é o diagnóstico.
#
# ## Lab 3 — MoE vs denso, mesmo compute ativo

# %%
cfg = minigpt.Config(vocab=VOCAB)
denso, _ = minigpt.treinar(cfg, dados_treino, dados_val, passos=300, batch=16, seed=1337)
ppl_denso = math.exp(minigpt.avaliar(denso, dados_val))
ppl_moe = math.exp(minigpt.avaliar(moe_com, dados_val))

n_denso = sum(p.numel() for p in denso.parameters())
n_moe = sum(p.numel() for p in moe_com.parameters())

print(f"{'modelo':<28} {'params totais':>14} {'PPL val':>9}")
print("-" * 56)
print(f"{'denso (1 MLP)':<28} {n_denso:>14,} {ppl_denso:>9.1f}")
print(f"{'MoE 4 experts top-1':<28} {n_moe:>14,} {ppl_moe:>9.1f}")
print(f"\nO MoE ativa por token ~o mesmo compute do denso (1 expert = 1 MLP),")
print(f"mas tem {n_moe/n_denso:.1f}x os parâmetros — capacidade extra 'de graça' no compute.")

# %% [markdown]
# > ⚠️ Em 300 passos num corpus minúsculo, o MoE pode não vencer o denso — especialização
# > de experts precisa de dados e tempo (e 4 experts top-1 recebem cada um ~1/4 dos
# > gradientes). O resultado honesto aqui é "compete com o denso ativando o mesmo
# > compute, com mais capacidade latente". Em escala real (Mixtral, DeepSeek), com
# > trilhões de tokens, a capacidade extra vence com folga.
#
# ## Lab 4 — Decodificação especulativa do zero
#
# O alvo é o MiniGPT grande do módulo 10 (professor); o draft, o pequeno. A promessa:
# **a distribuição final é exatamente a do alvo** — aceleração sem perda alguma.

# %%
cfg_alvo = minigpt.Config(vocab=VOCAB, d=320, n_camadas=6, n_heads=8, d_ff=864)
print("treinando o ALVO (grande)...")
alvo, _ = minigpt.treinar(cfg_alvo, dados_treino, dados_val, passos=500, batch=16, seed=1337)
cfg_draft = minigpt.Config(vocab=VOCAB, d=128, n_camadas=3, n_heads=4, d_ff=352)
print("treinando o DRAFT (pequeno)...")
draft, _ = minigpt.treinar(cfg_draft, dados_treino, dados_val, passos=400, batch=16, seed=7)
alvo.eval(); draft.eval()

n_a = sum(p.numel() for p in alvo.parameters())
n_d = sum(p.numel() for p in draft.parameters())
print(f"alvo {n_a/1e6:.1f}M | draft {n_d/1e6:.1f}M ({n_d/n_a:.0%} do alvo)")

# %%
@torch.no_grad()
def probs_de(modelo, seq, temperatura=0.8):
    logits, _ = modelo(seq[:, -128:].unsqueeze(0) if seq.dim() == 1 else seq[:, -128:])
    return F.softmax(logits[0, -1] / temperatura, dim=-1)

@torch.no_grad()
def gerar_especulativo(prompt, n_novos=64, k=4, temperatura=0.8):
    """Leviathan et al. 2022. Devolve (sequência, estatísticas)."""
    seq = prompt.clone()
    aceitos_total = propostos_total = forwards_alvo = 0

    while len(seq) - len(prompt) < n_novos:
        # 1. draft propõe k tokens, um a um (barato)
        rascunho, p_draft = [], []
        s = seq.clone()
        for _ in range(k):
            pd = probs_de(draft, s.unsqueeze(0), temperatura)
            t = int(torch.multinomial(pd, 1))
            rascunho.append(t); p_draft.append(pd)
            s = torch.cat([s, torch.tensor([t])])

        # 2. alvo verifica TODOS em UM forward (posições em paralelo — módulo 1!)
        logits, _ = alvo(s[-128:].unsqueeze(0))
        forwards_alvo += 1
        base = len(s) - len(seq)                       # = k
        p_alvo = F.softmax(logits[0, -base - 1: -1] / temperatura, dim=-1)  # [k, V]

        # 3. aceitação por rejection sampling
        n_aceitos = 0
        for j, t in enumerate(rascunho):
            razao = float(p_alvo[j, t] / (p_draft[j][t] + 1e-12))
            if torch.rand(1) < min(1.0, razao):
                n_aceitos += 1
            else:
                # reamostra da residual max(0, p_alvo - p_draft) normalizada
                residual = (p_alvo[j] - p_draft[j]).clamp(min=0)
                residual = residual / residual.sum()
                t_novo = int(torch.multinomial(residual, 1))
                seq = torch.cat([seq, torch.tensor(rascunho[:n_aceitos] + [t_novo])])
                break
        else:
            # todos aceitos: ganha +1 token grátis da distribuição do alvo
            pd_extra = F.softmax(logits[0, -1] / temperatura, dim=-1)
            t_extra = int(torch.multinomial(pd_extra, 1))
            seq = torch.cat([seq, torch.tensor(rascunho + [t_extra])])
            n_aceitos = k

        aceitos_total += n_aceitos
        propostos_total += k

    stats = {"taxa_aceitacao": aceitos_total / propostos_total,
             "tokens_por_forward_alvo": (len(seq) - len(prompt)) / forwards_alvo}
    return seq[:len(prompt) + n_novos], stats

# %%
torch.manual_seed(42)
i = int(torch.randint(len(dados_val) - 33, (1,)))
prompt = dados_val[i: i + 32]

# baseline: geração normal do alvo
t0 = time.perf_counter()
with torch.no_grad():
    _ = alvo.gerar(prompt.unsqueeze(0), 64, temperatura=0.8, top_k=0)
t_normal = time.perf_counter() - t0

t0 = time.perf_counter()
seq_espec, stats = gerar_especulativo(prompt, n_novos=64, k=4)
t_espec = time.perf_counter() - t0

print(f"taxa de aceitação        : {stats['taxa_aceitacao']:.0%}")
print(f"tokens por forward do alvo: {stats['tokens_por_forward_alvo']:.2f} (normal = 1.00)")
print(f"tempo: normal {t_normal:.1f}s | especulativo {t_espec:.1f}s "
      f"(speedup {t_normal/t_espec:.2f}x)")
print(f"\ntexto: {tk.decode(seq_espec[32:].tolist())[:180]!r}")

# %% [markdown]
# > ⚠️ **Sobre o speedup em CPU:** a promessa do especulativo é reduzir **forwards do
# > alvo** (a linha `tokens por forward`), porque em GPU o decode é memory-bound e cada
# > forward evitado é banda economizada. Em CPU com um alvo de 8M de parâmetros, o
# > overhead do draft em Python pode comer o ganho — o número que transfere para
# > produção é a taxa de aceitação e os tokens/forward, não o tempo de parede daqui.
#
# ### A prova de equivalência
#
# A afirmação forte do método: a distribuição final é a do alvo, exatamente. Testável:
# gere muitos primeiros-tokens pelos dois métodos e compare as distribuições empíricas.

# %%
@torch.no_grad()
def primeiro_token_especulativo():
    pd = probs_de(draft, prompt.unsqueeze(0))
    t = int(torch.multinomial(pd, 1))
    pa = probs_de(alvo, prompt.unsqueeze(0))
    if torch.rand(1) < min(1.0, float(pa[t] / (pd[t] + 1e-12))):
        return t
    residual = (pa - pd).clamp(min=0)
    return int(torch.multinomial(residual / residual.sum(), 1))

torch.manual_seed(0)
N_AMOSTRAS = 3000
pa = probs_de(alvo, prompt.unsqueeze(0))
contagem_espec = torch.zeros(VOCAB)
for _ in range(N_AMOSTRAS):
    contagem_espec[primeiro_token_especulativo()] += 1
empirica = contagem_espec / N_AMOSTRAS

top = torch.topk(pa, 6)
print(f"{'token':<16} {'P(alvo) exata':>14} {'P(especulativo)':>16}")
print("-" * 50)
for p, idx in zip(top.values, top.indices):
    print(f"{tk.decode([int(idx)])!r:<16} {float(p):>14.3f} {float(empirica[idx]):>16.3f}")

tv_espec = 0.5 * float((pa - empirica).abs().sum())

# ⚠️ O piso de ruído CERTO. A primeira versão deste teste comparava com 1/√N ≈ 0,018 —
# errado: para uma distribuição de alta entropia (aqui ~6 nats ≈ centenas de tokens
# efetivos), a TV entre a empírica de N amostras e a exata é dominada pela soma dos
# desvios de MUITOS tokens raros, e fica em ~0,15–0,20 com N=3000. O controle honesto:
# amostrar N vezes DIRETO do alvo e medir a mesma distância.
tvs_controle = []
for _ in range(3):
    direto = torch.multinomial(pa, N_AMOSTRAS, replacement=True)
    emp_direto = torch.bincount(direto, minlength=VOCAB).float() / N_AMOSTRAS
    tvs_controle.append(0.5 * float((pa - emp_direto).abs().sum()))

print(f"\nTV(especulativo, alvo exato)      : {tv_espec:.4f}")
print(f"TV(amostragem DIRETA, alvo exato) : "
      f"{min(tvs_controle):.4f} a {max(tvs_controle):.4f}  (o piso de ruído real)")
print(f"equivalência confirmada? {tv_espec <= max(tvs_controle) * 1.15}")

# %% [markdown]
# **Se a TV do especulativo está dentro da faixa do controle, a equivalência se
# confirma:** o especulativo é estatisticamente indistinguível de amostrar direto do
# alvo. (Na execução de referência: especulativo 0,139 contra controle 0,17–0,19 — o
# especulativo ficou ATÉ ABAIXO do piso, por sorte amostral.)
#
# E registre a lição de método, que já é a segunda do curso neste tema (módulo 6, PPL em
# 30 tokens): **testes estatísticos precisam de piso de ruído medido, não de fórmula de
# bolso.** O `1/√N` vale para UM evento binário; para distribuições inteiras, o controle
# empírico é obrigatório.
#
# ---
#
# ## Encerramento
#
# Implementado e medido:
#
# - MoE com roteador top-k, ativos vs totais, e a loss de balanceamento do Switch;
# - o colapso de roteamento produzido (α=0) e consertado (α=0,01), com a utilização
#   por expert visível;
# - MoE vs denso com o mesmo compute ativo;
# - decodificação especulativa completa, com taxa de aceitação, tokens/forward e a
#   **prova empírica de equivalência** com a distribuição do alvo.
#
# No `lab_mlx.py`: um MoE real nos seus 16 GB, a escada de quantização medida, e o
# `--draft-model` do mlx_lm.

# %%
