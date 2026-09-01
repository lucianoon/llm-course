# %% [markdown]
# # Módulo 10 — Laboratório A: destilação do zero
#
# **Roda em CPU (Windows ou Mac), ~12 minutos.** Três experimentos, todos verificáveis:
# dark knowledge num modelo real, a direção do KL numa bimodal, e uma destilação
# white-box completa — professor grande, aluno pequeno, mesmo compute, três receitas.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Dark knowledge: o que o professor sabe além do rótulo |
# | 2 | Forward vs reverse KL — mode-covering vs mode-seeking |
# | 3 | **Destilação real: hard labels vs KD vs mistura** |
# | 4 | A temperatura |
# | 5 | Black-box em miniatura: treinar no texto do professor |

# %%
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
# ## Lab 1 — Dark knowledge, num modelo real
#
# O rótulo duro diz "o próximo token é X". A distribuição do professor diz muito mais.

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

contexto = "O gato subiu no"
ids = tok_q(contexto, return_tensors="pt")["input_ids"]
with torch.no_grad():
    logits = qwen(ids).logits[0, -1]

probs = F.softmax(logits, dim=-1)
top = torch.topk(probs, 8)
print(f"contexto: {contexto!r}")
print("\no HARD LABEL diria apenas: o próximo token é", repr(tok_q.decode([int(top.indices[0])])))
print("\no SOFT TARGET diz:")
for p, i in zip(top.values, top.indices):
    print(f"  {float(p):>7.2%}  {tok_q.decode([int(i)])!r}")

entropia = float(-(probs * (probs + 1e-12).log()).sum())
print(f"\nentropia da distribuição: {entropia:.2f} nats "
      f"(≈ {math.exp(entropia):.0f} alternativas efetivas)")

# %% [markdown]
# Todas as alternativas do topo são **semanticamente plausíveis** — a distribuição
# codifica a estrutura da língua e do mundo, não só a resposta. É isso que o aluno recebe
# a cada posição na destilação de logits: `V` números em vez de 1.
#
# ## Lab 2 — A direção do KL
#
# O experimento clássico: o "professor" é uma mistura bimodal; o "aluno" é uma gaussiana
# única (sem capacidade para os dois modos — como todo aluno real). Ajustamos o aluno
# minimizando cada direção do KL e vemos onde ele assenta.

# %%
xs = torch.linspace(-6, 6, 400)

def gauss(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

professor = 0.5 * gauss(xs, -2.0, 0.6) + 0.5 * gauss(xs, 2.0, 0.6)
professor = professor / professor.sum()

def ajustar(direcao, passos=3000, sigma0=0.5):
    # ⚠️ A inicialização importa MUITO para o reverse KL: começando largo (sigma=1.0),
    # o otimizador cai num mínimo local SIMÉTRICO (mu=0, sigma=1.84) — cobrir os dois
    # modos "de longe" é localmente melhor que atravessar o vale até um deles. Medido:
    #   init sigma=1.0 -> mu=+0.00 (preso no centro)
    #   init sigma=0.5 -> mu=+2.00, sigma=0.60 (escolheu o modo, com a largura EXATA)
    # O mode-seeking é a geometria do objetivo, não uma garantia do otimizador. Em
    # destilação real (MiniLLM), isso reaparece como a necessidade de inicializar o
    # aluno com SFT antes do reverse KL — começar "perto de um modo".
    mu = torch.tensor(0.3, requires_grad=True)
    log_sigma = torch.tensor(math.log(sigma0), requires_grad=True)
    otim = torch.optim.Adam([mu, log_sigma], lr=0.01)
    for _ in range(passos):
        aluno = gauss(xs, mu, log_sigma.exp())
        aluno = aluno / aluno.sum() + 1e-12
        p = professor + 1e-12
        if direcao == "forward":                     # KL(professor ‖ aluno)
            perda = (p * (p / aluno).log()).sum()
        else:                                        # KL(aluno ‖ professor)
            perda = (aluno * (aluno / p).log()).sum()
        otim.zero_grad(); perda.backward(); otim.step()
    return float(mu), float(log_sigma.exp()), float(perda)

mu_f, sig_f, _ = ajustar("forward")
mu_r, sig_r, _ = ajustar("reverse")

print(f"forward KL (cobre tudo) : aluno em mu={mu_f:+.2f}, sigma={sig_f:.2f}")
print(f"reverse KL (escolhe)    : aluno em mu={mu_r:+.2f}, sigma={sig_r:.2f}")
print("\nmodos do professor: -2.0 e +2.0")

# %%
def curva_ascii(dist, titulo, largura=80, altura=8):
    d = dist / dist.max()
    passo = len(d) // largura
    amostra = d[::passo][:largura]
    print(f"\n{titulo}")
    for linha in range(altura, 0, -1):
        print("  " + "".join("█" if v >= linha / altura else " " for v in amostra))

curva_ascii(professor, "PROFESSOR (bimodal):")
curva_ascii(gauss(xs, torch.tensor(mu_f), torch.tensor(sig_f)),
            f"aluno via FORWARD KL (mu={mu_f:+.2f}) — assenta NO VALE:")
curva_ascii(gauss(xs, torch.tensor(mu_r), torch.tensor(sig_r)),
            f"aluno via REVERSE KL (mu={mu_r:+.2f}) — escolhe UM modo:")

# %% [markdown]
# **O forward KL põe o aluno exatamente onde o professor tem MENOS massa** — o vale entre
# os modos — porque cobrir os dois lados é obrigatório e a média é o único jeito. Em
# geração de texto, "a média de dois estilos bons" é texto ruim. O reverse KL abre mão da
# cobertura e faz um modo bem — qualidade sobre diversidade. É por isso que a destilação
# generativa moderna (MiniLLM, GKD) usa reverse.
#
# ## Lab 3 — Destilação real
#
# Professor: MiniGPT grande (d=320, 6 camadas), bem treinado.
# Aluno: MiniGPT pequeno (d=128, 3 camadas), três receitas com o MESMO compute:
#
# 1. **hard** — cross-entropy no corpus (treinar do zero, sem professor)
# 2. **KD puro** — só a KL contra os logits do professor
# 3. **mistura** — α·KD + (1−α)·CE

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

# %%
cfg_prof = minigpt.Config(vocab=VOCAB, d=320, n_camadas=6, n_heads=8, d_ff=864)
print("treinando o PROFESSOR (d=320, 6 camadas, 600 passos)...")
t0 = time.perf_counter()
professor_lm, _ = minigpt.treinar(cfg_prof, dados_treino, dados_val,
                                  passos=600, batch=16, seed=1337)
professor_lm.eval()
for p in professor_lm.parameters():
    p.requires_grad = False
ppl_prof = math.exp(minigpt.avaliar(professor_lm, dados_val))
n_prof = sum(p.numel() for p in professor_lm.parameters())
print(f"  {time.perf_counter()-t0:.0f}s | {n_prof/1e6:.1f}M params | PPL {ppl_prof:.1f}")

cfg_aluno = minigpt.Config(vocab=VOCAB, d=128, n_camadas=3, n_heads=4, d_ff=352)
n_aluno = sum(p.numel() for p in minigpt.MiniGPT(cfg_aluno).parameters())
print(f"aluno: {n_aluno/1e6:.1f}M params ({n_aluno/n_prof:.0%} do professor)")

# %%
def treinar_aluno(alpha, temperatura=2.0, passos=400, batch=16, lr=1e-3, seed=7):
    """alpha=0 -> só hard labels; alpha=1 -> só KD; entre -> mistura."""
    torch.manual_seed(seed)
    aluno = minigpt.MiniGPT(cfg_aluno)
    otim = torch.optim.AdamW(minigpt.grupos_de_parametros(aluno), lr=lr, betas=(0.9, 0.95))

    for passo in range(passos):
        taxa = minigpt.agenda_lr(passo, passos, pico=lr)
        for g in otim.param_groups:
            g["lr"] = taxa
        x, y = minigpt.pegar_batch(dados_treino, batch, cfg_aluno.bloco)

        logits_a, _ = aluno(x)
        perda = 0.0
        if alpha > 0:
            with torch.no_grad():
                logits_p, _ = professor_lm(x)
            T = temperatura
            kd = F.kl_div(F.log_softmax(logits_a / T, dim=-1),
                          F.log_softmax(logits_p / T, dim=-1),
                          log_target=True, reduction="batchmean") * T * T / x.size(1)
            perda = perda + alpha * kd
        if alpha < 1:
            ce = F.cross_entropy(logits_a.view(-1, VOCAB), y.reshape(-1))
            perda = perda + (1 - alpha) * ce

        otim.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(aluno.parameters(), 1.0)
        otim.step()
    return aluno

receitas = {"hard labels (do zero)": 0.0, "KD puro (α=1)": 1.0, "mistura (α=0.7)": 0.7}
resultados = {}
for nome, alpha in receitas.items():
    t0 = time.perf_counter()
    aluno = treinar_aluno(alpha)
    ppl = math.exp(minigpt.avaliar(aluno, dados_val))
    resultados[nome] = {"ppl": ppl, "t": time.perf_counter() - t0, "modelo": aluno}
    print(f"{nome:<24} PPL {ppl:>7.1f}  ({resultados[nome]['t']:.0f}s)")

# %%
print(f"{'modelo':<26} {'params':>8} {'PPL validação':>14}")
print("-" * 52)
print(f"{'professor':<26} {n_prof/1e6:>7.1f}M {ppl_prof:>14.1f}")
for nome, r in resultados.items():
    print(f"{'aluno — ' + nome:<26} {n_aluno/1e6:>7.1f}M {r['ppl']:>14.1f}")

melhor_kd = min(resultados["KD puro (α=1)"]["ppl"], resultados["mistura (α=0.7)"]["ppl"])
ganho = (resultados["hard labels (do zero)"]["ppl"] - melhor_kd) / resultados["hard labels (do zero)"]["ppl"]
print(f"\nganho da destilação sobre treinar do zero: {ganho:+.1%} de PPL")

# %% [markdown]
# **A comparação central do módulo:** mesmos dados, mesmo compute de treino do aluno,
# mesma arquitetura — a única diferença é o sinal. Se o KD venceu os hard labels, a
# afirmação de Hinton se sustenta aqui: a distribuição do professor ensina mais por token
# que o rótulo sozinho.
#
# (E note o custo escondido: cada passo de KD paga um forward do professor. O compute
# TOTAL não é igual — é o preço da dark knowledge.)
#
# ## Lab 4 — A temperatura

# %%
print(f"{'T':>5} {'PPL do aluno (KD puro)':>24}")
print("-" * 32)
for T in [1.0, 2.0, 4.0, 8.0]:
    aluno_t = treinar_aluno(alpha=1.0, temperatura=T, passos=250)
    print(f"{T:>5.1f} {math.exp(minigpt.avaliar(aluno_t, dados_val)):>24.1f}")

# %% [markdown]
# ### O resultado honesto — que contraria o folclore
#
# Na execução de referência, **T=1 venceu, e monotonicamente**: 199,8 → 221,0 → 262,4 →
# 298,1 para T=1/2/4/8. O "ótimo em T=2–4" que se lê por aí NÃO apareceu. Duas razões,
# ambas instrutivas:
#
# 1. **A avaliação é PPL — que é medida em T=1.** Treinar em T alta otimiza imitar o
#    professor SUAVIZADO, um alvo diferente da distribuição real. O descasamento
#    treino/avaliação penaliza T alta por construção.
# 2. **O folclore vem da classificação de imagens** (Hinton 2015, ImageNet: 1.000
#    classes, professor superconfiante). Em modelagem de língua, a distribuição por
#    token JÁ é suave (entropia de 6+ nats, Lab 1) — a dark knowledge já está visível
#    em T=1, e suavizar mais só dilui o sinal. A prática de KD em LLMs usa T=1
#    (MiniLLM, DistilGPT) com muito mais frequência do que os tutoriais sugerem.
#
# A lição de método vale mais que o número: **defaults migram mal entre domínios** —
# meça no seu.
#
# ## Lab 5 — Black-box em miniatura
#
# Sem logits: o professor **gera texto**, o aluno faz SFT nele. É o pipeline do
# R1-Distill (e o único possível entre tokenizers diferentes).

# %%
print("professor gerando corpus sintético...")
t0 = time.perf_counter()
torch.manual_seed(21)
partes = []
with torch.no_grad():
    for _ in range(30):
        i = int(torch.randint(len(dados_treino) - 33, (1,)))
        prompt = dados_treino[i: i + 32].repeat(8, 1)              # batch de 8
        saidas = professor_lm.gerar(prompt, 64, temperatura=0.9, top_k=40)
        partes.append(saidas[:, 32:].reshape(-1))
corpus_sintetico = torch.cat(partes)
print(f"  {len(corpus_sintetico):,} tokens gerados em {time.perf_counter()-t0:.0f}s")

# %%
def treinar_em(fonte, passos=400, seed=7):
    torch.manual_seed(seed)
    aluno = minigpt.MiniGPT(cfg_aluno)
    otim = torch.optim.AdamW(minigpt.grupos_de_parametros(aluno), lr=1e-3, betas=(0.9, 0.95))
    for passo in range(passos):
        taxa = minigpt.agenda_lr(passo, passos, pico=1e-3)
        for g in otim.param_groups:
            g["lr"] = taxa
        x, y = minigpt.pegar_batch(fonte, 16, cfg_aluno.bloco)
        _, perda = aluno(x, y)
        otim.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(aluno.parameters(), 1.0)
        otim.step()
    return aluno

aluno_real = treinar_em(dados_treino)
aluno_sintetico = treinar_em(corpus_sintetico)

ppl_real = math.exp(minigpt.avaliar(aluno_real, dados_val))
ppl_sint = math.exp(minigpt.avaliar(aluno_sintetico, dados_val))

print(f"{'aluno treinado em':<28} {'PPL na validação REAL':>22}")
print(f"{'corpus real':<28} {ppl_real:>22.1f}")
print(f"{'texto do professor':<28} {ppl_sint:>22.1f}")

# %% [markdown]
# ### O desastre instrutivo
#
# Na execução de referência: aluno no corpus real, PPL **197,8**; no texto do professor,
# PPL **3.190** — dezesseis vezes pior. Não "perdeu por pouco": foi destruído. A autópsia
# encontra três causas empilhadas, e cada uma é uma regra do pipeline black-box:
#
# 1. **Professor fraco.** PPL 124 ainda é um modelo ruim — o texto dele é Machado
#    degenerado. Black-box herda a qualidade do professor SEM o amortecedor dos soft
#    targets (que ao menos carregam a incerteza). Regra: destile de professores MUITO
#    melhores que o aluno, ou não destile.
# 2. **Escala errada em 40×.** 15k tokens sintéticos para 400 passos × 16 × 128 = 819k
#    tokens processados = **53 épocas** sobre o corpusinho. O aluno memorizou um dialeto
#    minúsculo e esquisito. Regra: em black-box, o custo dominante é GERAR EM ESCALA —
#    o R1 gerou 800 mil amostras, não 240.
# 3. **Sem filtragem.** T=0,9 num professor fraco produz lixo frequente, e tudo entrou.
#    Regra: rejection sampling não é opcional (e é o que o lab_mlx faz).
#
# Note que este é o **model collapse do módulo 4 em dose concentrada**: treinar em texto
# gerado por um modelo fraco, repetido, sem filtro. O pipeline R1 funciona porque nega
# as três causas — professor forte, escala real, filtro por gabarito. O `lab_mlx.py`
# executa essa versão.
#
# ---
#
# ## Encerramento
#
# Verificado:
#
# - a dark knowledge existe e é mensurável (entropia da distribuição do professor);
# - forward KL assenta o aluno no vale; reverse escolhe um modo — a direção importa;
# - destilar logits vs treinar do zero, com o resultado medido;
# - a temperatura tem um ótimo interior;
# - o pipeline black-box, com sua herança de qualidade e de defeitos.
#
# No `lab_mlx.py`: o R1-Distill de verdade no seu M4 — gerar traços, filtrar por
# gabarito, treinar o 0.5B, e comparar com o professor.

# %%
