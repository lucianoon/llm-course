# %% [markdown]
# # Módulo 9 — Laboratório A: GRPO do zero
#
# **Roda em CPU (Windows ou Mac), ~10 minutos.** Implementa o GRPO completo — grupos,
# vantagem relativa, clipped ratio, KL k3 — e treina o MiniGPT com uma recompensa
# verificável. Depois produz um **reward hack de propósito**, para você nunca esquecer
# a cara de um.
#
# | Lab | Assunto |
# |---|---|
# | 1 | A variância do REINFORCE — e o que a baseline faz com ela |
# | 2 | A vantagem de grupo: mecânica e casos extremos |
# | 3 | **GRPO completo: treinar com recompensa verificável** |
# | 4 | Reward hacking, produzido em cativeiro |
# | 5 | A ablação do KL |

# %%
import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd().parent / "tools"))
import minigpt

torch.manual_seed(0)

# %% [markdown]
# ## Lab 1 — Por que a baseline existe
#
# O estimador REINFORCE é `R·∇log π`. Subtrair uma baseline `b` não muda a média
# (porque `E[∇log π] = 0`), mas muda a variância. Medindo num caso mínimo:
# uma política de 3 ações, recompensas todas positivas.

# %%
logits = torch.tensor([1.0, 0.5, -0.5], requires_grad=True)
recompensas = torch.tensor([1.0, 0.8, 0.6])       # todas positivas!

def gradiente_reinforce(baseline, n_amostras=2000):
    grads = []
    for _ in range(n_amostras):
        probs = F.softmax(logits, dim=-1)
        acao = torch.multinomial(probs, 1)
        logp = torch.log_softmax(logits, dim=-1)[acao]
        g = torch.autograd.grad((recompensas[acao] - baseline) * logp, logits)[0]
        grads.append(g)
    G = torch.stack(grads)
    return G.mean(0), G.var(0).sum()

torch.manual_seed(1)
media_sem, var_sem = gradiente_reinforce(baseline=0.0)
torch.manual_seed(1)
media_com, var_com = gradiente_reinforce(baseline=float(recompensas.mean()))

print(f"{'':<16} {'gradiente médio':>38} {'variância':>11}")
print(f"{'sem baseline':<16} {str([f'{x:+.4f}' for x in media_sem]):>38} {var_sem:>11.4f}")
print(f"{'com baseline':<16} {str([f'{x:+.4f}' for x in media_com]):>38} {var_com:>11.4f}")
print(f"\nmesma direção? {torch.allclose(media_sem, media_com, atol=0.02)}")
print(f"redução de variância: {var_sem / var_com:.1f}x")

# %% [markdown]
# **Mesma média, variância muito menor.** Com recompensas todas positivas e sem baseline,
# toda ação é "reforçada" e a diferenciação afoga no ruído. A baseline converte
# recompensa absoluta em *vantagem* — "melhor ou pior que o esperado". Toda a linhagem
# PPO/GRPO é a história de como estimar essa baseline.
#
# ## Lab 2 — A vantagem de grupo
#
# O GRPO estima a baseline com a média de um GRUPO de gerações do mesmo prompt:
# `A_i = (r_i − média) / desvio`.

# %%
def vantagem_grupo(recompensas, eps=1e-4):
    r = torch.as_tensor(recompensas, dtype=torch.float32)
    return (r - r.mean()) / (r.std(unbiased=False) + eps)

casos = {
    "misto (o caso útil)": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "um único acerto": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "grupo inteiro acerta": [1.0] * 8,
    "grupo inteiro erra": [0.0] * 8,
    "escala 0-100 (mesmo padrão)": [100.0, 0.0, 100.0, 0.0, 0.0, 100.0, 0.0, 0.0],
}
for nome, rs in casos.items():
    a = vantagem_grupo(rs)
    print(f"{nome:<30} A = [{', '.join(f'{x:+.2f}' for x in a[:4])} ...]")

# %% [markdown]
# Três propriedades para guardar:
#
# 1. **Escala não importa** — 0/1 e 0/100 dão vantagens idênticas. A recompensa pode ser
#    qualquer coisa monotônica.
# 2. **Grupo unânime ⇒ vantagem zero** ⇒ gradiente zero. Prompts fáceis demais ou
#    impossíveis não ensinam nada — o treino se concentra sozinho na fronteira da
#    competência (currículo emergente).
# 3. **Um único acerto no grupo recebe vantagem enorme** (+2,65): comportamento raro e
#    correto é amplificado com força — é o mecanismo pelo qual o R1 transformou acertos
#    ocasionais em política confiável.
#
# ## Lab 3 — GRPO completo no MiniGPT
#
# A tarefa verificável: **gerar continuações que abram fala de diálogo** (o corpus de
# Machado marca falas com `--` no início de linha). O verificador é uma busca de
# substring — exato, binário, inhackeável por regra.

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

cfg = minigpt.Config(vocab=VOCAB)
print("treinando a base...")
base, _ = minigpt.treinar(cfg, dados_treino, dados_val, passos=300, batch=16, seed=1337)
ppl_base = math.exp(minigpt.avaliar(base, dados_val))
print(f"PPL base: {ppl_base:.1f}")

# %%
PROMPT_LEN, GER_LEN, G = 32, 24, 8

def recompensa_dialogo(texto_gerado: str) -> float:
    """1.0 se a continuação abre uma fala de diálogo. Binária e limitada — por design."""
    return 1.0 if "\n--" in texto_gerado else 0.0

def prompts_aleatorios(n_prompts, fonte):
    idx = torch.randint(len(fonte) - PROMPT_LEN - 1, (n_prompts,))
    return torch.stack([fonte[i: i + PROMPT_LEN] for i in idx])

@torch.no_grad()
def taxa_de_sucesso(modelo, recompensa, n=48):
    prompts = prompts_aleatorios(n, dados_val)
    saidas = modelo.gerar(prompts, GER_LEN, temperatura=1.0, top_k=40)
    return sum(recompensa(tk.decode(s[PROMPT_LEN:].tolist())) > 0 for s in saidas) / n

torch.manual_seed(3)
taxa_base = taxa_de_sucesso(base, recompensa_dialogo)
print(f"taxa de sucesso da BASE: {taxa_base:.0%}")
print("(precisa estar entre ~5% e ~60%: baixa demais = grupo nunca acerta = vantagem")
print(" sempre zero; alta demais = nada a aprender. É a regra de ouro da seção 7.)")

# %%
def logprobs_por_token(modelo, seqs):
    """log-prob de cada token GERADO. [B, GER_LEN]"""
    logits, _ = modelo(seqs)
    lp = F.log_softmax(logits[:, PROMPT_LEN - 1: -1].float(), dim=-1)
    return lp.gather(2, seqs[:, PROMPT_LEN:].unsqueeze(-1)).squeeze(-1)

def treinar_grpo(recompensa, passos=60, n_prompts=3, lr=1e-4, beta_kl=0.05,
                 clip_eps=0.2, epocas_internas=2, seed=42, log_cada=10):
    torch.manual_seed(seed)
    politica = copy.deepcopy(base)
    referencia = copy.deepcopy(base)
    referencia.eval()
    for p in referencia.parameters():
        p.requires_grad = False
    otim = torch.optim.AdamW(politica.parameters(), lr=lr, betas=(0.9, 0.95))
    historico = []

    for passo in range(passos):
        # ---- 1. gerar G respostas por prompt (on-policy) ----
        prompts = prompts_aleatorios(n_prompts, dados_treino)
        prompts_rep = prompts.repeat_interleave(G, dim=0)            # [n_prompts*G, L]
        with torch.no_grad():
            seqs = politica.gerar(prompts_rep, GER_LEN, temperatura=1.0, top_k=40)
            lp_old = logprobs_por_token(politica, seqs)              # π_old
            lp_ref = logprobs_por_token(referencia, seqs)            # π_ref (p/ KL)

        # ---- 2. recompensas e vantagens por grupo ----
        rs = torch.tensor([recompensa(tk.decode(s[PROMPT_LEN:].tolist())) for s in seqs])
        vantagens = torch.cat([vantagem_grupo(rs[i * G:(i + 1) * G])
                               for i in range(n_prompts)])           # [n_prompts*G]

        # ---- 3. épocas internas com clipped ratio + KL k3 ----
        for _ in range(epocas_internas):
            lp_novo = logprobs_por_token(politica, seqs)
            ratio = torch.exp(lp_novo - lp_old)                      # por token
            A = vantagens.unsqueeze(1)                               # broadcast p/ tokens
            surrogate = torch.min(ratio * A,
                                  ratio.clamp(1 - clip_eps, 1 + clip_eps) * A)
            # KL k3: sempre ≥ 0, não-enviesado
            log_dif = lp_ref - lp_novo
            kl = torch.exp(log_dif) - log_dif - 1
            loss = -(surrogate - beta_kl * kl).mean()

            otim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(politica.parameters(), 1.0)
            otim.step()

        historico.append({"recompensa": float(rs.mean()), "kl": float(kl.detach().mean())})
        if passo % log_cada == 0 or passo == passos - 1:
            print(f"  passo {passo:>3} | recompensa média {rs.mean():.2f} "
                  f"| KL {float(kl.detach().mean()):.4f}")
    return politica, historico

t0 = time.perf_counter()
politica, hist = treinar_grpo(recompensa_dialogo)
print(f"\ntreino GRPO: {time.perf_counter() - t0:.0f}s")

# %%
torch.manual_seed(3)
taxa_grpo = taxa_de_sucesso(politica, recompensa_dialogo)
ppl_grpo = math.exp(minigpt.avaliar(politica, dados_val))

print(f"{'':<12} {'taxa de sucesso':>16} {'PPL validação':>14}")
print(f"{'base':<12} {taxa_base:>16.0%} {ppl_base:>14.1f}")
print(f"{'após GRPO':<12} {taxa_grpo:>16.0%} {ppl_grpo:>14.1f}")

# %%
# Leia as gerações — a regra nº 4 da seção 6 do README.
torch.manual_seed(11)
prompts = prompts_aleatorios(2, dados_val)
for m, nome in [(base, "BASE"), (politica, "GRPO")]:
    torch.manual_seed(11)
    saidas = m.gerar(prompts, GER_LEN, temperatura=1.0, top_k=40)
    print(f"\n=== {nome} ===")
    for s in saidas:
        print(" ", repr(tk.decode(s[PROMPT_LEN:].tolist()))[:110])

# %% [markdown]
# **O que este treino fez, em uma frase:** o modelo já sabia abrir diálogos (a taxa base
# não era zero); o GRPO amplificou esse comportamento raro em comportamento dominante,
# recompensado apenas por um verificador binário — nenhum exemplo-alvo, nenhum par.
#
# É a mecânica do R1 em miniatura: competência ocasional + verificador + RL = competência
# confiável.
#
# ## Lab 4 — Reward hacking, em cativeiro
#
# Agora o experimento que você deve guardar para sempre. Mesma tarefa, uma mudança de
# UMA linha: a recompensa vira **contagem** de `--` em vez de binária. "Mais diálogo =
# melhor", certo?

# %%
def recompensa_hackeavel(texto_gerado: str) -> float:
    return float(texto_gerado.count("--"))          # ⚠️ ilimitada. Erro de design.

politica_hack, hist_hack = treinar_grpo(recompensa_hackeavel, passos=60, log_cada=15)

# %%
print("curva de 'recompensa' do treino hackeado (parece ótima!):")
marcos = [0, 15, 30, 45, 59]
for m in marcos:
    print(f"  passo {m:>3}: recompensa média {hist_hack[m]['recompensa']:.2f}")

torch.manual_seed(11)
saidas = politica_hack.gerar(prompts, GER_LEN, temperatura=1.0, top_k=40)
print("\n=== o que o modelo 'ótimo' de fato gera ===")
for s in saidas:
    print(" ", repr(tk.decode(s[PROMPT_LEN:].tolist()))[:110])

ppl_hack = math.exp(minigpt.avaliar(politica_hack, dados_val))
torca = taxa_de_sucesso(politica_hack, recompensa_dialogo)
print(f"\nPPL: {ppl_base:.0f} -> {ppl_hack:.0f} | "
      f"recompensa média final: {hist_hack[-1]['recompensa']:.1f}")

# %% [markdown]
# **A curva subiu o treino inteiro — e o modelo virou um spammer de travessões.** O RL
# fez exatamente o que a recompensa escrita pedia: maximizar a contagem. A intenção
# ("escreva diálogo") nunca esteve no código.
#
# As defesas, verificadas lado a lado:
#
# - a recompensa **binária** do Lab 3 não tinha gradiente para "mais ainda" — hackeá-la
#   não paga;
# - o **KL** encareceu a fuga da linguagem natural (compare as PPLs);
# - e a defesa definitiva foi esta célula: **ler as gerações**. Na curva, os dois treinos
#   são indistinguíveis de um sucesso.
#
# ## Lab 5 — A ablação do KL

# %%
print("GRPO com recompensa binária, MAS sem KL (beta=0):")
politica_semkl, _ = treinar_grpo(recompensa_dialogo, beta_kl=0.0, passos=60, log_cada=30)

torch.manual_seed(3)
taxa_semkl = taxa_de_sucesso(politica_semkl, recompensa_dialogo)
ppl_semkl = math.exp(minigpt.avaliar(politica_semkl, dados_val))

print(f"\n{'variante':<22} {'sucesso':>9} {'PPL':>9}")
print("-" * 42)
print(f"{'base':<22} {taxa_base:>9.0%} {ppl_base:>9.1f}")
print(f"{'GRPO (β_KL=0.05)':<22} {taxa_grpo:>9.0%} {ppl_grpo:>9.1f}")
print(f"{'GRPO sem KL':<22} {taxa_semkl:>9.0%} {ppl_semkl:>9.1f}")

torch.manual_seed(11)
saidas = politica_semkl.gerar(prompts[:1], GER_LEN, temperatura=1.0, top_k=40)
print("\ngeração sem KL:", repr(tk.decode(saidas[0][PROMPT_LEN:].tolist()))[:110])

# %% [markdown]
# Sem o KL, a política é livre para colapsar na forma mais barata de satisfazer o
# verificador — tipicamente às custas da linguagem (veja a PPL). O KL é o mesmo
# personagem dos módulos anteriores: o β do DPO, o "fique perto da referência" do RLHF.
# Em RL ele não é regularização opcional; é o que mantém o modelo sendo um modelo de
# linguagem.
#
# ---
#
# ## Encerramento
#
# Implementado e medido:
#
# - a baseline reduzindo a variância do REINFORCE em ordens de grandeza, sem viés;
# - a vantagem de grupo: invariância de escala, gradiente zero em grupos unânimes,
#   amplificação máxima do acerto raro;
# - GRPO completo (grupos + clip + KL k3) elevando uma taxa de sucesso verificável;
# - reward hacking produzido, com a curva linda e o modelo destruído;
# - o KL como a diferença entre alinhar e colapsar.
#
# No `lab_mlx.py`: a receita do R1 de verdade — GRPO no Qwen com GSM8K e recompensas de
# acurácia + formato.

# %%
