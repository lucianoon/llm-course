# %% [markdown]
# # Módulo 14 — Laboratório: estatística de avaliação, contra o próprio curso
#
# **Roda em CPU (Windows ou Mac), ~8 minutos.** Este lab tem um alvo especial: as
# conclusões DESTE CURSO. A comparação densa vs BM25 do módulo 13 recebe gabarito no
# nível de passagem: o resultado se sustenta com n=25? Um juiz de 0.5B serve para alguma
# coisa? Quanto uma seleção de checkpoint infla resultados?
#
# | Lab | Assunto |
# |---|---|
# | 1 | Acurácia é estimativa: o custo do n pequeno, simulado |
# | 2 | Pareado vs não pareado: o poder, medido |
# | 3 | **Auditoria do módulo 13: métrica válida e comparação pareada** |
# | 4 | **Auditoria de um juiz LLM: acurácia e viés de posição** |
# | 5 | Comparações múltiplas: fabricando melhoras |
# | 6 | Calibração: o Qwen sabe quando não sabe? (ECE) |

# %%
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

torch.manual_seed(0)
AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "tools"))

from rag import PERGUNTAS, IndiceRAG, passagem_relevante

# %% [markdown]
# ## Lab 1 — O custo do n pequeno
#
# Dois modelos com acurácias VERDADEIRAS de 62% e 65% (Δ=3pp — uma melhora típica de
# paper). Avaliamos em conjuntos de vários tamanhos, 10.000 vezes, e contamos: com que
# frequência o ranking observado INVERTE?

# %%
def simular_ranking(acc_a=0.62, acc_b=0.65, ns=(25, 100, 500, 2000), trials=10_000):
    print(f"{'n':>6} {'IC95 típico':>13} {'P(ranking inverte)':>20} {'P(empate)':>11}")
    print("-" * 56)
    for n in ns:
        a = torch.binomial(torch.full((trials,), float(n)), torch.full((trials,), acc_a)) / n
        b = torch.binomial(torch.full((trials,), float(n)), torch.full((trials,), acc_b)) / n
        inverte = float((a > b).float().mean())
        empata = float((a == b).float().mean())
        ic = 1.96 * math.sqrt(0.65 * 0.35 / n)
        print(f"{n:>6} {f'±{ic:.1%}':>13} {inverte:>20.1%} {empata:>11.1%}")

simular_ranking()

# %% [markdown]
# **Leia a linha n=100:** mesmo com B genuinamente 3pp melhor, o ranking observado sai
# INVERTIDO em uma fração alarmante das avaliações. Toda decisão tomada com "rodei 100
# exemplos e A deu 2 pontos a mais" tem essa taxa de erro embutida.
#
# ## Lab 2 — O poder do pareamento
#
# Agora com a estrutura real do problema: perguntas têm DIFICULDADES diferentes, e os
# dois modelos enfrentam AS MESMAS perguntas. A comparação pareada cancela a dificuldade.

# %%
def simular_poder(delta=0.03, n=500, trials=3_000, corr=0.0):
    """corr = fração das perguntas em que os dois modelos compartilham o MESMO sorteio
    (acertam ou erram juntos) — o que modelos reais e similares fazem o tempo todo."""
    detecta_np = detecta_p = 0
    for _ in range(trials):
        dificuldade = torch.rand(n) * 0.6 + 0.2
        base = torch.rand(n)                                # ruído compartilhado
        usa_comum = torch.rand(n) < corr
        ra = torch.where(usa_comum, base, torch.rand(n))
        rb = torch.where(usa_comum, base, torch.rand(n))
        pa = (ra < dificuldade).float()
        pb = (rb < (dificuldade + delta).clamp(0, 1)).float()

        # não pareado: z-test de duas proporções
        ma, mb = pa.mean(), pb.mean()
        p_pool = (ma + mb) / 2
        se = math.sqrt(2 * p_pool * (1 - p_pool) / n)
        if se > 0 and float(mb - ma) / se > 1.96:
            detecta_np += 1

        # pareado: McNemar — só as discordâncias informam
        b_ = int(((pb == 1) & (pa == 0)).sum())
        c_ = int(((pa == 1) & (pb == 0)).sum())
        if b_ + c_ > 0 and (b_ - c_) / math.sqrt(b_ + c_) > 1.96:
            detecta_p += 1
    return detecta_np / trials, detecta_p / trials

print(f"{'correlação':>11} {'não pareado detecta':>21} {'McNemar detecta':>17}   (n=500, Δ=3pp)")
print("-" * 62)
for corr in [0.0, 0.5, 0.8, 0.95]:
    np_, p_ = simular_poder(corr=corr)
    print(f"{corr:>11} {np_:>21.1%} {p_:>17.1%}")

# %% [markdown]
# **A linha que importa é a última:** com correlação 0,95 — o regime real de dois
# checkpoints do mesmo modelo, ou dois sistemas parecidos — o McNemar detecta a
# diferença de 3pp em 88% das vezes; o teste não pareado, em 0,3%. **Trezentas vezes
# mais poder, dos mesmos dados.**
#
# E note a mecânica: o poder do pareamento vem da CORRELAÇÃO entre os modelos, não do
# pareamento em si (na linha corr=0, os dois testes empatam). Modelos reais concordam na
# maioria das perguntas — é exatamente essa concordância que o teste pareado desconta e
# o não pareado paga como ruído.
#
# > 🔬 **Confissão metodológica:** a primeira versão desta simulação não tinha o
# > parâmetro de correlação — os modelos compartilhavam só a dificuldade (corr efetiva
# > ~0,1) — e o ganho do pareamento saiu modesto (6,5% vs 5,7%), desmentindo o
# > "multiplica o poder" do README. O README estava certo para modelos REAIS e a
# > simulação errada para o fenômeno. Foi corrigida — e o episódio é o módulo 14
# > aplicado a si mesmo: a afirmação sobrevive à medição, ou muda a afirmação... ou
# > o experimento estava medindo outra coisa.
#
# ## Lab 3 — Auditoria do módulo 13
#
# Uma versão anterior afirmou densa > BM25 usando o módulo de origem como gabarito.
# Agora reutilizamos o gabarito no nível de passagem do módulo 13: o top-1 só é acerto
# quando contém a evidência da resposta.

# %%
# O próprio módulo 14 fica fora do índice para não revelar as respostas da auditoria.
indice_rag = IndiceRAG(RAIZ, ate_modulo=12)
CHUNKS = indice_rag.chunks
print(f"{len(CHUNKS)} chunks | {len(PERGUNTAS)} perguntas (banco compartilhado)")

hits_bm, hits_dn = [], []
for pergunta, _, _ in PERGUNTAS:
    ordem_bm, _ = indice_rag.bm25.buscar(pergunta, k=1)
    ordem_dn, _ = indice_rag.buscar_densa(pergunta, k=1)
    chunk_bm = CHUNKS[ordem_bm[0]]
    chunk_dn = CHUNKS[ordem_dn[0]]
    hits_bm.append(1 if passagem_relevante(pergunta, chunk_bm.modulo, chunk_bm.texto) else 0)
    hits_dn.append(1 if passagem_relevante(pergunta, chunk_dn.modulo, chunk_dn.texto) else 0)

acc_bm, acc_dn = sum(hits_bm) / len(hits_bm), sum(hits_dn) / len(hits_dn)
print(f"hit@1 — BM25: {acc_bm:.0%} | densa: {acc_dn:.0%} (n={len(PERGUNTAS)})")

# %%
# O teste pareado: McNemar + bootstrap da diferença.
b_ = sum(1 for x, y in zip(hits_dn, hits_bm) if x == 1 and y == 0)   # densa acerta, BM25 erra
c_ = sum(1 for x, y in zip(hits_dn, hits_bm) if x == 0 and y == 1)   # o inverso
print(f"discordâncias: densa✓/BM25✗ = {b_} | densa✗/BM25✓ = {c_} "
      f"(as outras {len(PERGUNTAS) - b_ - c_} perguntas são plateia)")

# McNemar exato: sob H0, b ~ Binomial(b+c, 1/2). p-valor bicaudal:
n_disc = b_ + c_
p_valor = sum(math.comb(n_disc, k) for k in range(max(b_, c_), n_disc + 1)) / 2 ** n_disc * 2
p_valor = min(1.0, p_valor)
print(f"McNemar exato: p = {p_valor:.3f}")

# Bootstrap pareado da diferença:
torch.manual_seed(0)
difs = []
h_dn, h_bm = torch.tensor(hits_dn, dtype=torch.float), torch.tensor(hits_bm, dtype=torch.float)
for _ in range(10_000):
    idx = torch.randint(len(PERGUNTAS), (len(PERGUNTAS),))
    difs.append(float(h_dn[idx].mean() - h_bm[idx].mean()))
difs = torch.tensor(sorted(difs))
lo, hi = difs[int(0.025 * len(difs))], difs[int(0.975 * len(difs))]
print(f"bootstrap pareado: Δ = {acc_dn - acc_bm:+.0%}, IC95 = [{lo:+.0%}, {hi:+.0%}]")
print(f"\nVEREDITO: a diferença {'É' if p_valor < 0.05 else 'NÃO é'} significativa a 5% com n=25.")

# %% [markdown]
# **Este é o resultado mais importante do módulo.** Leia os números produzidos nesta
# execução, não os valores históricos do README. Se o IC cruza zero, a evidência é
# insuficiente para afirmar superioridade geral. E, mesmo quando não cruza, a conclusão
# se limita a este corpus, este gabarito e estes sistemas. n=25 constrói e depura; não
# sustenta generalizações amplas com margens pequenas.
#
# ## Lab 4 — Auditoria de um juiz LLM
#
# O protocolo da seção 3 aplicado ao Qwen2.5-0.5B como juiz: pares com GABARITO
# (resposta certa vs número corrompido), avaliados nas duas ordens.

# %%
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer as AT

V5 = int(transformers.__version__.split(".")[0]) >= 5
DTYPE_KW = {"dtype": torch.float32} if V5 else {"torch_dtype": torch.float32}
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", **DTYPE_KW)
qwen.eval()
tok_q = AT.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

PARES_GABARITO = [
    ("Quanto é 15% de 200?", "30", "45"),
    ("Quantos minutos há em 3 horas?", "180", "150"),
    ("Se um livro custa R$ 40 com 25% de desconto, qual o preço final?", "R$ 30", "R$ 32"),
    ("Quantos lados tem um hexágono?", "6", "8"),
    ("Qual é o dobro de 17?", "34", "27"),
    ("Uma dúzia e meia são quantos ovos?", "18", "16"),
    ("Quanto é 9 × 8?", "72", "63"),
    ("Se hoje é terça, que dia será daqui a 10 dias?", "sexta", "quinta"),
    ("Quantos segundos há em 2 minutos e meio?", "150", "130"),
    ("Qual a metade de 86?", "43", "46"),
    ("Quanto é 100 − 37?", "63", "67"),
    ("Um trem parte às 14h e chega às 17h30. Quanto durou a viagem?", "3h30", "2h30"),
]

@torch.no_grad()
def julgar(pergunta, resp_a, resp_b):
    """Devolve 'A' ou 'B' — comparando a log-prob dos dois tokens de resposta."""
    msgs = [{"role": "user", "content":
             f"Pergunta: {pergunta}\n\nResposta A: {resp_a}\nResposta B: {resp_b}\n\n"
             "Qual resposta está correta? Responda apenas A ou B."}]
    prompt = tok_q.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok_q(prompt, return_tensors="pt")["input_ids"]
    logits = qwen(ids).logits[0, -1]
    id_a = tok_q.encode("A", add_special_tokens=False)[0]
    id_b = tok_q.encode("B", add_special_tokens=False)[0]
    return "A" if logits[id_a] > logits[id_b] else "B"

acertos = consistentes = acerta_consistente = 0
for pergunta, certa, errada in PARES_GABARITO:
    v1 = julgar(pergunta, certa, errada)      # certa na posição A
    v2 = julgar(pergunta, errada, certa)      # certa na posição B
    acerto1, acerto2 = v1 == "A", v2 == "B"
    acertos += acerto1 + acerto2
    # consistente = escolheu o MESMO vencedor nas duas ordens (acerta em ambas
    # ou erra em ambas). Inconsistente = a posição decidiu, não o conteúdo.
    consistente = (acerto1 == acerto2)
    consistentes += consistente
    if consistente and acerto1:
        acerta_consistente += 1

n_j = len(PARES_GABARITO)
print(f"acurácia do juiz (2 ordens × {n_j} pares): {acertos}/{2*n_j} = {acertos/(2*n_j):.0%}")
print(f"consistência entre ordens: {consistentes}/{n_j} = {consistentes/n_j:.0%}")
print(f"acerta NAS DUAS ordens: {acerta_consistente}/{n_j} = {acerta_consistente/n_j:.0%}")
print("\n(um juiz aleatório: 50% de acurácia, 50% de consistência)")

# %% [markdown]
# **A régua:** a distância entre este juiz e a moeda é o máximo de confiança que
# qualquer win rate dele merece. Se a consistência entre ordens está perto de 50%, o
# viés de posição domina — e nenhum resultado desse juiz significa nada sem o protocolo
# das duas ordens. Juízes de 0.5B são termômetros de brinquedo; os de 7B+ melhoram muito,
# e a auditoria continua obrigatória (exercício B3 repete isto no M4 com o 7B).
#
# ## Lab 5 — Fabricando melhoras com comparações múltiplas

# %%
def melhor_de_k(k=20, n=200, acc=0.65, trials=2_000):
    ganhos = []
    for _ in range(trials):
        accs = torch.binomial(torch.full((k,), float(n)), torch.full((k,), acc)) / n
        ganhos.append(float(accs.max() - acc))
    g = torch.tensor(ganhos)
    return float(g.mean()), float(g.quantile(0.95))

print(f"{'checkpoints':>12} {'inflação média':>15} {'p95':>8}   (modelos IDÊNTICOS, n=200)")
print("-" * 58)
for k in [1, 5, 20, 100]:
    m, p95 = melhor_de_k(k=k)
    print(f"{k:>12} {m:>+15.1%} {p95:>+8.1%}")

# %% [markdown]
# **Vinte checkpoints do MESMO modelo, e "o melhor" parece 4–5pp acima da verdade — por
# sorteio.** Toda escolha feita olhando o conjunto de avaliação (checkpoint, prompt,
# seed, temperatura) compra um pedaço dessa inflação. A defesa estrutural: escolher num
# conjunto de desenvolvimento, reportar num teste LACRADO, aberto uma vez.
#
# ## Lab 6 — Calibração: o modelo sabe quando não sabe?
#
# Múltipla escolha com confiança = softmax das log-probs das opções (normalizadas por
# comprimento — módulo 1, desafio). Se o modelo é calibrado, "80% de confiança" acerta
# ~80% das vezes.

# %%
QUESTOES_MC = [
    ("Qual é a capital do Brasil?", ["Brasília", "Rio de Janeiro", "São Paulo", "Salvador"], 0),
    ("Quantos planetas tem o Sistema Solar?", ["8", "9", "7", "10"], 0),
    ("Quem escreveu Dom Casmurro?", ["Machado de Assis", "José de Alencar", "Clarice Lispector", "Jorge Amado"], 0),
    ("Qual o maior oceano?", ["Pacífico", "Atlântico", "Índico", "Ártico"], 0),
    ("Em que ano o homem pisou na Lua?", ["1969", "1959", "1972", "1965"], 0),
    ("Qual elemento tem símbolo Fe?", ["Ferro", "Flúor", "Fósforo", "Frâncio"], 0),
    ("Quantos bits tem um byte?", ["8", "16", "4", "32"], 0),
    ("Qual rio atravessa o Egito?", ["Nilo", "Amazonas", "Tigre", "Eufrates"], 0),
    ("Qual a moeda do Japão?", ["Iene", "Yuan", "Won", "Ringgit"], 0),
    ("Quem pintou a Mona Lisa?", ["Leonardo da Vinci", "Michelangelo", "Rafael", "Botticelli"], 0),
    ("Qual gás as plantas absorvem na fotossíntese?", ["CO2", "O2", "N2", "H2"], 0),
    ("Quantas cordas tem um violão comum?", ["6", "4", "5", "7"], 0),
]

@torch.no_grad()
def confianca_e_acerto(pergunta, opcoes, idx_certa):
    msgs = [{"role": "user", "content": pergunta}]
    prompt = tok_q.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids_ctx = tok_q(prompt, return_tensors="pt")["input_ids"]
    lps = []
    for op in opcoes:
        ids_op = tok_q(" " + op, add_special_tokens=False, return_tensors="pt")["input_ids"]
        logits = qwen(torch.cat([ids_ctx, ids_op], dim=1)).logits
        lp = F.log_softmax(logits[0, ids_ctx.shape[1]-1:-1].float(), dim=-1)
        lps.append(float(lp.gather(1, ids_op[0].unsqueeze(1)).mean()))   # média por token
    probs = F.softmax(torch.tensor(lps), dim=0)
    escolhida = int(probs.argmax())
    return float(probs[escolhida]), escolhida == idx_certa

# embaralha a posição da resposta certa (senão medimos viés de posição, não calibração)
torch.manual_seed(3)
resultados_mc = []
for pergunta, opcoes, certa in QUESTOES_MC:
    perm = torch.randperm(4).tolist()
    ops = [opcoes[i] for i in perm]
    resultados_mc.append(confianca_e_acerto(pergunta, ops, perm.index(certa)))

print(f"{'conf.':>7} {'acertou':>8}")
for conf, ok in sorted(resultados_mc, reverse=True):
    print(f"{conf:>7.0%} {'✓' if ok else '✗':>8}")

acc = sum(ok for _, ok in resultados_mc) / len(resultados_mc)
conf_media = sum(c for c, _ in resultados_mc) / len(resultados_mc)
ece = abs(conf_media - acc)   # com 12 questões, 1 bin honesto; ECE por bins é exercício
print(f"\nacurácia: {acc:.0%} | confiança média: {conf_media:.0%} | "
      f"gap global: {conf_media - acc:+.0%}")
print("gap positivo = superconfiante; negativo = subconfiante.")
print("(12 questões dão UM bin honesto — o ECE completo por faixas é o exercício B5,")
print(" que também explica por que binar 12 pontos seria teatro estatístico.)")

# %% [markdown]
# ---
#
# ## Encerramento
#
# O módulo aplicou o próprio remédio:
#
# - o custo do n pequeno, simulado — e a taxa de rankings invertidos que ninguém reporta;
# - o pareamento multiplicando o poder (McNemar: só as discordâncias falam);
# - **a conclusão do módulo 13 auditada** — e rebaixada à força de afirmação que n=25
#   permite;
# - um juiz auditado contra gabarito, com o viés de posição na mesa;
# - a inflação de melhor-de-k, quantificada;
# - calibração medida — a confiança como informação (ou não).
#
# A regra que fica: **todo número que decide algo carrega um IC; todo número reportado,
# o tamanho da amostra.** É a diferença entre medir e concluir.

# %%
