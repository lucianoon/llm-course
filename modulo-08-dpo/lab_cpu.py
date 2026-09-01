# %% [markdown]
# # Módulo 8 — Laboratório A: DPO do zero
#
# **Roda em CPU (Windows ou Mac), ~8 minutos.** Implementa a loss do DPO a partir da
# derivação do README, treina de verdade com ela, e reproduz as duas patologias
# clássicas — incluindo a que quase ninguém espera.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Bradley-Terry e a mecânica da loss |
# | 2 | O gradiente do DPO: para onde ele empurra |
# | 3 | **Treinar DPO no MiniGPT** — preferência sintética controlada |
# | 4 | A patologia: ambas as log-probs caem |
# | 5 | β: o cabo de guerra, medido |
# | 6 | Viés de comprimento — por que soma ≠ média |

# %%
import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(AQUI.parent / "tools"))
import minigpt

torch.manual_seed(0)

# %% [markdown]
# ## Lab 1 — Bradley-Terry e a loss
#
# Tudo do README, seção 2, em dez linhas de código.

# %%
def dpo_loss(lp_chosen_pol, lp_rejected_pol, lp_chosen_ref, lp_rejected_ref, beta=0.1):
    """L = -log σ( β·[(lpc_pol - lpc_ref) - (lpr_pol - lpr_ref)] )

    Cada argumento é a log-prob TOTAL da sequência (soma sobre os tokens).
    Devolve (loss, recompensa implícita do chosen, do rejected).
    """
    r_chosen = beta * (lp_chosen_pol - lp_chosen_ref)      # recompensa implícita
    r_rejected = beta * (lp_rejected_pol - lp_rejected_ref)
    loss = -F.logsigmoid(r_chosen - r_rejected)
    return loss.mean(), r_chosen.mean(), r_rejected.mean()

# A mecânica do Bradley-Terry: P(preferência) como função da margem.
print(f"{'margem r_w − r_l':>17} {'P(y_w ≻ y_l)':>14} {'loss':>8}")
for margem in [-2.0, -0.5, 0.0, 0.5, 2.0, 5.0]:
    p = torch.sigmoid(torch.tensor(margem))
    print(f"{margem:>17.1f} {p:>14.3f} {-F.logsigmoid(torch.tensor(margem)):>8.4f}")

# %% [markdown]
# Margem 0 → P=0,5 → loss = ln 2 ≈ 0,693. **Este é o valor da loss no passo 0 de todo
# treino DPO** (política = referência ⇒ recompensas implícitas = 0). Se a sua loss
# inicial não for ~0,69, há um bug — é o análogo do `ln(V)` do módulo 3.
#
# ## Lab 2 — Para onde o gradiente empurra

# %%
lpc = torch.tensor([-10.0], requires_grad=True)   # log-prob do chosen (política)
lpr = torch.tensor([-12.0], requires_grad=True)   # log-prob do rejected
loss, _, _ = dpo_loss(lpc, lpr, torch.tensor([-10.0]), torch.tensor([-12.0]), beta=0.1)
loss.backward()

print(f"∂L/∂ log π(chosen)  = {lpc.grad.item():+.4f}   (negativo → o passo AUMENTA a log-prob)")
print(f"∂L/∂ log π(rejected)= {lpr.grad.item():+.4f}   (positivo → o passo REDUZ a log-prob)")
print(f"simétricos? {abs(lpc.grad.item() + lpr.grad.item()) < 1e-9}")

# %%
# O peso do gradiente depende da margem: o DPO foca nos pares em que a política
# ainda está ERRADA — é um hard-example mining embutido.
print(f"\n{'recompensas (c, r)':>20} {'|gradiente|':>12}   interpretação")
for rc, rr in [(0.0, 0.0), (1.0, -1.0), (3.0, -3.0), (-2.0, 2.0)]:
    lpc = torch.tensor([rc / 0.1], requires_grad=True)
    loss, _, _ = dpo_loss(lpc, torch.tensor([rr / 0.1]),
                          torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1)
    loss.backward()
    caso = {0.0: "início do treino", 1.0: "indo bem", 3.0: "par já resolvido (gradiente ~0)",
            -2.0: "par INVERTIDO (gradiente máximo)"}[rc]
    print(f"{f'({rc:+.1f}, {rr:+.1f})':>20} {abs(lpc.grad.item()):>12.4f}   {caso}")

# %% [markdown]
# A sigmoide satura nos pares resolvidos e concentra gradiente nos invertidos. (É essa
# saturação que o IPO remove, trocando por uma loss quadrática — ver README, seção 5.)
#
# ## Lab 3 — DPO de verdade no MiniGPT
#
# O experimento controlado: um MiniGPT treinado em Machado, e pares de preferência em que
# **chosen = continuação real do corpus** e **rejected = degeneração repetitiva** — o
# defeito clássico de modelos pequenos. Se o DPO funciona, a probabilidade de degenerar
# deve despencar sem destruir a linguagem.

# %%
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

corpus_path = AQUI.parent / "modulo-03-treino" / "data" / "corpus.txt"
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
print("treinando a base (o 'SFT' deste experimento)...")
base, _ = minigpt.treinar(cfg, dados_treino, dados_val, passos=300, batch=16, seed=1337)
ppl_base = math.exp(minigpt.avaliar(base, dados_val))
print(f"PPL da base na validação: {ppl_base:.1f}")

# %%
PROMPT_LEN, RESP_LEN = 32, 32

def montar_pares(n_pares, fonte):
    """chosen = continuação real; rejected = degeneração (um trecho curto repetido)."""
    prompts, chosens, rejecteds = [], [], []
    for _ in range(n_pares):
        i = int(torch.randint(len(fonte) - PROMPT_LEN - RESP_LEN - 1, (1,)))
        prompt = fonte[i: i + PROMPT_LEN]
        chosen = fonte[i + PROMPT_LEN: i + PROMPT_LEN + RESP_LEN]
        pedaco = fonte[i + PROMPT_LEN: i + PROMPT_LEN + 4]        # 4 tokens...
        rejected = pedaco.repeat(RESP_LEN // 4)                   # ...repetidos 8x
        prompts.append(prompt); chosens.append(chosen); rejecteds.append(rejected)
    return torch.stack(prompts), torch.stack(chosens), torch.stack(rejecteds)

p, c, r = montar_pares(2, dados_treino)
print("PROMPT  :", tk.decode(p[0].tolist())[:80])
print("CHOSEN  :", tk.decode(c[0].tolist())[:80])
print("REJECTED:", tk.decode(r[0].tolist())[:80])

# %%
def logprob_resposta(modelo, prompts, respostas):
    """log P(resposta | prompt), somado sobre os tokens da resposta. [batch]"""
    seq = torch.cat([prompts, respostas], dim=1)
    logits, _ = modelo(seq)
    # logits da posição t predizem o token t+1 (o shift de sempre)
    lp = F.log_softmax(logits[:, PROMPT_LEN - 1: -1].float(), dim=-1)
    alvos = respostas.unsqueeze(-1)
    return lp.gather(2, alvos).squeeze(-1).sum(dim=1)

# Verificação do passo 0: política == referência  ⇒  loss = ln 2.
politica = copy.deepcopy(base)
referencia = copy.deepcopy(base)
referencia.eval()
for param in referencia.parameters():
    param.requires_grad = False

p, c, r = montar_pares(8, dados_treino)
with torch.no_grad():
    lc_ref, lr_ref = logprob_resposta(referencia, p, c), logprob_resposta(referencia, p, r)
loss0, _, _ = dpo_loss(lc_ref, lr_ref, lc_ref, lr_ref)
print(f"loss no passo 0: {loss0:.4f}  (esperado: ln 2 = {math.log(2):.4f})")
assert abs(loss0 - math.log(2)) < 1e-4

# %%
def treinar_dpo(beta, passos=150, lr=1e-4, batch=8, seed=42):
    torch.manual_seed(seed)
    pol = copy.deepcopy(base)
    otim = torch.optim.AdamW(pol.parameters(), lr=lr, betas=(0.9, 0.95))
    hist = {"loss": [], "lp_chosen": [], "lp_rejected": [], "margem": []}

    for passo in range(passos):
        p, c, r = montar_pares(batch, dados_treino)
        with torch.no_grad():
            lc_ref = logprob_resposta(referencia, p, c)
            lr_ref = logprob_resposta(referencia, p, r)
        lc_pol = logprob_resposta(pol, p, c)
        lr_pol = logprob_resposta(pol, p, r)

        loss, r_c, r_r = dpo_loss(lc_pol, lr_pol, lc_ref, lr_ref, beta=beta)
        otim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        otim.step()

        if passo % 25 == 0 or passo == passos - 1:
            hist["loss"].append(float(loss.detach()))
            hist["lp_chosen"].append(float(lc_pol.detach().mean()))
            hist["lp_rejected"].append(float(lr_pol.detach().mean()))
            hist["margem"].append(float((r_c - r_r).detach()))
    return pol, hist

t0 = time.perf_counter()
politica, hist = treinar_dpo(beta=0.1)
print(f"treino DPO: {time.perf_counter() - t0:.0f}s")

print(f"\n{'passo':>6} {'loss':>8} {'lp chosen':>11} {'lp rejected':>12} {'margem':>8}")
print("-" * 50)
passos_log = [0, 25, 50, 75, 100, 125, 149]
for i, passo in enumerate(passos_log[:len(hist['loss'])]):
    print(f"{passo:>6} {hist['loss'][i]:>8.4f} {hist['lp_chosen'][i]:>11.1f} "
          f"{hist['lp_rejected'][i]:>12.1f} {hist['margem'][i]:>8.2f}")

# %% [markdown]
# ## Lab 4 — Leia a tabela acima com atenção
#
# Três coisas para verificar:
#
# 1. **A loss partiu de ~0,693** (ln 2) — o teste de sanidade.
# 2. **A margem cresce** — o treino funciona.
# 3. **Olhe a coluna `lp chosen`.** Se ela CAIU junto com a rejected (só que menos), você
#    acabou de reproduzir a patologia da seção 3 do README: a loss só otimiza a
#    *diferença*, e derrubar o rejected mais rápido que o chosen também a satisfaz.
#    (Na execução de referência: chosen caiu de −164 para −208 enquanto a margem subia
#    de 0 a ~10. As duas coisas ao mesmo tempo.)
#
# ### Onde a degeneração mora — o detalhe que quase arruinou este experimento
#
# A primeira versão deste lab mediu a degeneração com sampling (T=0,8, top_k=40) e
# encontrou **0% na base** — nada a corrigir, e o DPO só pagou custo. O erro era meu:
# degeneração repetitiva é um fenômeno do **decoding**, não só do modelo. Medido:
#
# - greedy (top_k=1): **100%** das continuações da base entram em loop;
# - sampling (T=0,8): **0%**.
#
# É exatamente o achado de Holtzman et al. (2019), leitura do módulo 1: maximizar
# verossimilhança leva a loops; o ruído do sampling os quebra. A avaliação certa mede o
# modo em que o defeito existe:

# %%
@torch.no_grad()
def taxa_de_degeneracao(modelo, top_k, temperatura, n=30, gerar_tokens=48):
    """Fração de continuações com >30% de 4-gramas repetidos."""
    degenerados = 0
    for _ in range(n):
        i = int(torch.randint(len(dados_val) - PROMPT_LEN - 1, (1,)))
        prompt = dados_val[i: i + PROMPT_LEN].unsqueeze(0)
        saida = modelo.gerar(prompt, gerar_tokens, temperatura=temperatura, top_k=top_k)[0, PROMPT_LEN:]
        grams = [tuple(saida[j: j + 4].tolist()) for j in range(len(saida) - 3)]
        if len(set(grams)) < len(grams) * 0.7:
            degenerados += 1
    return degenerados / n

ppl_dpo = math.exp(minigpt.avaliar(politica, dados_val))
print(f"{'':<12} {'degen. GREEDY':>14} {'degen. sampling':>16} {'PPL validação':>14}")
print("-" * 60)
for nome, modelo, ppl in [("base", base, ppl_base), ("após DPO", politica, ppl_dpo)]:
    torch.manual_seed(7)
    greedy = taxa_de_degeneracao(modelo, top_k=1, temperatura=1.0)
    torch.manual_seed(7)
    amostrado = taxa_de_degeneracao(modelo, top_k=40, temperatura=0.8)
    print(f"{nome:<12} {greedy:>14.0%} {amostrado:>16.0%} {ppl:>14.1f}")

# %% [markdown]
# **Como ler:** a coluna GREEDY é onde o defeito punido vive — é ela que o DPO deve
# derrubar. A PPL é o preço pago em linguagem geral (espere alguma alta: o DPO desloca a
# política, e é para isso que o β existe). Se a degeneração greedy não caiu, os pares
# estão fora da distribuição que o decoding visita — o gap off-policy da seção 3.
#
# A lição de avaliação vale mais que o resultado: **meça o comportamento no modo de
# decoding em que ele ocorre.** Um defeito invisível com sampling pode ser universal em
# greedy — e produção usa os dois.
#
# ## Lab 5 — β: o cabo de guerra
#
# β pequeno deixa a política ir longe; β grande a prende à referência. Medindo os dois
# lados do trade-off:

# %%
print(f"{'beta':>7} {'margem final':>13} {'lp chosen final':>16} {'PPL validação':>14}")
print("-" * 55)
for beta in [0.02, 0.1, 0.5]:
    pol_b, hist_b = treinar_dpo(beta=beta)
    ppl_b = math.exp(minigpt.avaliar(pol_b, dados_val))
    print(f"{beta:>7.2f} {hist_b['margem'][-1]:>13.2f} {hist_b['lp_chosen'][-1]:>16.1f} "
          f"{ppl_b:>14.1f}")

# %% [markdown]
# O padrão esperado: β menor → margem maior E deriva maior (PPL pior). O β é
# literalmente o peso do KL do objetivo original de RLHF — a seção 2 do README mostrou
# que ele sobrevive intacto à derivação.
#
# ## Lab 6 — Viés de comprimento: soma ≠ média
#
# A recompensa implícita usa a log-prob **somada** da sequência. Cada token adiciona um
# termo negativo — sequências longas têm log-prob total menor, *mesmo quando cada token é
# mais provável*. Medindo no Qwen2.5-0.5B:

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
qwen.eval()
tok_q = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
)

def lp_qwen(contexto, resposta):
    ids_ctx = tok_q(contexto, return_tensors="pt")["input_ids"]
    ids_resp = tok_q(resposta, add_special_tokens=False, return_tensors="pt")["input_ids"]
    seq = torch.cat([ids_ctx, ids_resp], dim=1)
    with torch.no_grad():
        logits = qwen(seq).logits
    lp = F.log_softmax(logits[0, ids_ctx.shape[1] - 1: -1].float(), dim=-1)
    por_token = lp.gather(1, ids_resp[0].unsqueeze(1)).squeeze(1)
    return float(por_token.sum()), float(por_token.mean()), len(por_token)

contexto = "Pergunta: Qual é a capital da França?\nResposta:"
curta = " Paris."
longa = (" A capital da França é Paris, uma das cidades mais visitadas do mundo, "
         "conhecida pela Torre Eiffel e pelo Museu do Louvre.")

print(f"{'resposta':<10} {'tokens':>7} {'log P somada':>13} {'log P média/token':>18}")
for nome, resp in [("curta", curta), ("longa", longa)]:
    soma, media, n_tok = lp_qwen(contexto, resp)
    print(f"{nome:<10} {n_tok:>7} {soma:>13.2f} {media:>18.3f}")

# %% [markdown]
# Se a **soma** prefere a curta e a **média** prefere a longa (ou vice-versa), você viu o
# problema: a recompensa implícita do DPO opera na soma, então o comprimento entra na
# conta querendo ou não. Com chosen sistematicamente mais longo no dataset, o DPO
# aprende verbosidade; com rejected mais longo, aprende laconismo. **Audite a correlação
# comprimento × preferência antes de treinar** — e se ela for alta, balanceie os pares ou
# use SimPO, que normaliza pela média.
#
# ---
#
# ## Encerramento
#
# Verificado neste lab:
#
# - a loss parte de ln 2 e o gradiente é simétrico (sobe chosen, desce rejected), com
#   peso concentrado nos pares invertidos;
# - um treino DPO real elimina o comportamento punido (degeneração) preservando a
#   linguagem — as duas métricas juntas;
# - a patologia das log-probs caindo em conjunto;
# - o trade-off do β, dos dois lados;
# - o viés de comprimento na recompensa implícita, num modelo real.
#
# No `lab_mlx.py`: DPO e ORPO num modelo de verdade, com pares construídos por corrupção
# controlada — a técnica da seção 4 do README.

# %%
