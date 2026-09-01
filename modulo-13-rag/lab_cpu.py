# %% [markdown]
# # Módulo 13 — Laboratório A: RAG do zero, sobre o próprio curso
#
# **Roda em CPU (Windows ou Mac), ~10 minutos.** A base de conhecimento é o CURSO
# INTEIRO (os READMEs dos módulos 1–12) — você constrói o assistente de estudo que vai
# usar de verdade, e a avaliação é verificável: sabemos em qual módulo cada resposta mora.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Chunking: transformar 12 aulas em pedaços recuperáveis |
# | 2 | BM25 do zero — a baseline de 50 anos que ainda briga |
# | 3 | Busca densa: embeddings multilíngues (e5-small) |
# | 4 | Híbrido: Reciprocal Rank Fusion |
# | 5 | **Avaliação verificável: hit@k e MRR nas 25 perguntas** |
# | 6 | O valor do contexto, medido em probabilidade (a técnica do módulo 7) |
# | 7 | Abstenção: detectar "a resposta não está na base" |

# %%
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

torch.manual_seed(0)
AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "tools"))

from rag import PERGUNTAS, passagem_relevante

# %% [markdown]
# ## Lab 1 — Chunking
#
# A decisão mais subestimada do RAG. Pedaços grandes carregam contexto mas diluem a
# busca; pequenos são precisos mas órfãos. A estratégia aqui: **cortar por seção**
# (títulos `##`) — as fronteiras semânticas que o autor já marcou — e subdividir seções
# longas com sobreposição.

# %%
def extrair_chunks(md_path: Path, alvo_palavras=220, overlap=40):
    """Corta um README por seções ## e subdivide as longas com sobreposição."""
    texto = md_path.read_text(encoding="utf-8")
    modulo = md_path.parent.name
    chunks = []
    # separa por títulos de nível 1-2 mantendo o título junto do corpo
    partes = re.split(r"(?m)^(#{1,2} .+)$", texto)
    titulo_atual = modulo
    for parte in partes:
        if re.match(r"^#{1,2} ", parte or ""):
            titulo_atual = parte.lstrip("# ").strip()
            continue
        palavras = (parte or "").split()
        if len(palavras) < 30:
            continue
        passo = alvo_palavras - overlap
        for i in range(0, len(palavras), passo):
            pedaco = " ".join(palavras[i: i + alvo_palavras])
            if len(pedaco.split()) >= 30:
                chunks.append({"modulo": modulo, "titulo": titulo_atual, "texto": pedaco})
            if i + alvo_palavras >= len(palavras):
                break
    return chunks

CHUNKS = []
for md in sorted(RAIZ.glob("modulo-*/README.md")):
    numero = int(md.parent.name.split("-", 2)[1])
    if numero <= 12:
        CHUNKS.extend(extrair_chunks(md))

tamanhos = [len(c["texto"].split()) for c in CHUNKS]
print(f"{len(CHUNKS)} chunks dos módulos 1–12")
print(f"palavras por chunk: média {sum(tamanhos)/len(tamanhos):.0f} | "
      f"min {min(tamanhos)} | max {max(tamanhos)}")
print(f"\nexemplo: [{CHUNKS[40]['modulo']} / {CHUNKS[40]['titulo'][:40]}]")
print(" ", CHUNKS[40]["texto"][:180], "...")

# %% [markdown]
# ## As 25 perguntas de avaliação
#
# Cada uma tem uma ou mais passagens rotuladas que contêm a evidência da resposta. O
# módulo de origem sozinho não basta: um trecho irrelevante do módulo certo não é hit.
# É a regra de ouro dos módulos 5 e 14: **a métrica vem antes do sistema.**

# %%
# (pergunta, resposta curta, módulos-fonte). O gabarito de passagens fica em
# tools/rag.py para que os módulos 13 e 14 avaliem exatamente os mesmos casos.
print(f"{len(PERGUNTAS)} perguntas | módulos cobertos: "
      f"{len({m for _, _, ms in PERGUNTAS for m in ms})}")

# %% [markdown]
# ## Lab 2 — BM25 do zero
#
# Antes de embeddings, a baseline que dominou a busca por 50 anos: palavras em comum,
# ponderadas por raridade (IDF) e com saturação de frequência. ~25 linhas, zero
# dependências — e você vai ver que ela briga com a busca neural.

# %%
def tokenizar_busca(texto: str) -> list[str]:
    return re.findall(r"[a-záàâãéêíóôõúüç0-9@#\-]+", texto.lower())

class BM25:
    """Okapi BM25. k1 controla a saturação de TF; b, a normalização por comprimento."""

    def __init__(self, docs: list[str], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenizar_busca(d) for d in docs]
        self.N = len(docs)
        self.tam_medio = sum(len(t) for t in self.docs_tokens) / self.N
        df = Counter(tok for t in self.docs_tokens for tok in set(t))
        # IDF: termos raros valem muito; termos onipresentes, quase nada
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}
        self.tf = [Counter(t) for t in self.docs_tokens]

    def buscar(self, consulta: str, k=10):
        q = tokenizar_busca(consulta)
        scores = []
        for i in range(self.N):
            s, tam = 0.0, len(self.docs_tokens[i])
            for termo in q:
                if termo not in self.tf[i]:
                    continue
                f = self.tf[i][termo]
                # saturação: a 10ª ocorrência vale quase nada a mais que a 3ª
                s += self.idf.get(termo, 0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * tam / self.tam_medio))
            scores.append(s)
        ordem = sorted(range(self.N), key=lambda i: -scores[i])[:k]
        return ordem, [scores[i] for i in ordem]

bm25 = BM25([c["texto"] for c in CHUNKS])
ordem, scores = bm25.buscar("quantos bytes por parâmetro custa o AdamW?", k=3)
print("top-3 do BM25 para a pergunta dos 16 bytes:")
for i, s in zip(ordem, scores):
    print(f"  {s:6.2f}  [{CHUNKS[i]['modulo']}] {CHUNKS[i]['texto'][:90]}")

# %% [markdown]
# ## Lab 3 — Busca densa
#
# O bi-encoder: perguntas e documentos viram vetores no MESMO espaço; relevância =
# proximidade. O e5-small (118M, multilíngue) exige os prefixos `query:` / `passage:` —
# ele foi treinado assim, e omiti-los degrada silenciosamente (a armadilha de template
# do módulo 1, versão embeddings).

# %%
from transformers import AutoModel, AutoTokenizer

tok_e5 = AutoTokenizer.from_pretrained(
    "intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
)
e5 = AutoModel.from_pretrained(
    "intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
)
e5.eval()

@torch.no_grad()
def embed(textos: list[str], batch=16) -> torch.Tensor:
    """Mean pooling sobre a última camada + normalização L2 (a receita do e5)."""
    saidas = []
    for i in range(0, len(textos), batch):
        enc = tok_e5(textos[i: i + batch], padding=True, truncation=True,
                     max_length=512, return_tensors="pt")
        h = e5(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1)
        saidas.append(F.normalize(emb, dim=-1))
    return torch.cat(saidas)

t0 = time.perf_counter()
EMB_CHUNKS = embed([f"passage: {c['texto']}" for c in CHUNKS])
print(f"{len(CHUNKS)} chunks embedados em {time.perf_counter()-t0:.0f}s "
      f"-> matriz {tuple(EMB_CHUNKS.shape)}")
print(f"custo do índice: {EMB_CHUNKS.numel() * 4 / 1e6:.1f} MB em fp32 "
      f"(o 'banco vetorial' é literalmente esta matriz)")

def buscar_densa(consulta: str, k=10):
    q = embed([f"query: {consulta}"])
    sims = (q @ EMB_CHUNKS.T)[0]
    ordem = sims.argsort(descending=True)[:k]
    return ordem.tolist(), sims[ordem].tolist()

# %% [markdown]
# ## Lab 4 — Híbrido: Reciprocal Rank Fusion
#
# BM25 acha termos exatos ("--mask-prompt", "NF4"); a densa acha paráfrases ("como
# impedir que o modelo esqueça" → catastrophic forgetting). O RRF combina os dois
# rankings sem precisar calibrar escalas de score — só posições.

# %%
def buscar_hibrida(consulta: str, k=10, c=60):
    """score(d) = Σ 1/(c + rank_d) sobre os dois rankings. c=60 é o padrão da literatura."""
    ordem_bm, _ = bm25.buscar(consulta, k=30)
    ordem_dn, _ = buscar_densa(consulta, k=30)
    pontos = Counter()
    for rank, i in enumerate(ordem_bm):
        pontos[i] += 1 / (c + rank)
    for rank, i in enumerate(ordem_dn):
        pontos[i] += 1 / (c + rank)
    topo = [i for i, _ in pontos.most_common(k)]
    return topo, [pontos[i] for i in topo]

# %% [markdown]
# ## Lab 5 — A avaliação verificável
#
# hit@k: uma passagem que contém a evidência está entre os k primeiros?
# MRR: em que posição, em média, o primeiro acerto aparece (1/rank).

# %%
def avaliar(nome, buscador, k_max=5):
    hits = {1: 0, 3: 0, 5: 0}
    mrr = 0.0
    falhas = []
    for pergunta, _, _ in PERGUNTAS:
        ordem, _ = buscador(pergunta, k=k_max)
        acertos = [
            rank
            for rank, indice in enumerate(ordem)
            if passagem_relevante(
                pergunta,
                CHUNKS[indice]["modulo"],
                CHUNKS[indice]["texto"],
            )
        ]
        primeiro = acertos[0] if acertos else None
        for k in hits:
            if primeiro is not None and primeiro < k:
                hits[k] += 1
        if primeiro is not None:
            mrr += 1 / (primeiro + 1)
        else:
            falhas.append(pergunta)
    n = len(PERGUNTAS)
    return {"nome": nome, "hit@1": hits[1]/n, "hit@3": hits[3]/n,
            "hit@5": hits[5]/n, "MRR": mrr/n, "falhas": falhas}

resultados = [
    avaliar("BM25", bm25.buscar),
    avaliar("densa (e5-small)", buscar_densa),
    avaliar("híbrida (RRF)", buscar_hibrida),
]

print(f"{'sistema':<20} {'hit@1':>7} {'hit@3':>7} {'hit@5':>7} {'MRR':>7}")
print("-" * 52)
for r in resultados:
    print(f"{r['nome']:<20} {r['hit@1']:>7.0%} {r['hit@3']:>7.0%} "
          f"{r['hit@5']:>7.0%} {r['MRR']:>7.2f}")

for r in resultados:
    if r["falhas"]:
        print(f"\nfalhas de {r['nome']} (nem no top-5):")
        for f in r["falhas"][:4]:
            print(f"  - {f}")

# %% [markdown]
# **Como ler:** hit@1 é a métrica de quem manda UM chunk para o modelo; hit@5, de quem
# manda cinco. As falhas listadas são o material de trabalho real — a engenharia de RAG
# é, na prática, olhar as perguntas que a recuperação erra e decidir: chunking melhor?
# consulta reescrita? híbrido? reranker?
#
# ## Lab 6 — O valor do contexto, em probabilidade
#
# A técnica do módulo 7, agora para RAG: quanto o chunk recuperado desloca a
# probabilidade da resposta correta? Três condições — sem contexto, com o chunk que a
# busca REALMENTE devolveu (top-1 híbrido), e com um chunk irrelevante (controle).

# %%
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer as AT

V5 = int(transformers.__version__.split(".")[0]) >= 5
DTYPE_KW = {"dtype": torch.float32} if V5 else {"torch_dtype": torch.float32}
qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    **DTYPE_KW,
)
qwen.eval()
tok_q = AT.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
)

def logprob_resposta(contexto_rag: str | None, pergunta: str, resposta: str) -> float:
    conteudo = (f"Contexto:\n{contexto_rag}\n\nPergunta: {pergunta}" if contexto_rag
                else f"Pergunta: {pergunta}")
    prompt = tok_q.apply_chat_template([{"role": "user", "content": conteudo}],
                                       tokenize=False, add_generation_prompt=True)
    prompt += "Resposta:"
    ids_ctx = tok_q(prompt, return_tensors="pt")["input_ids"]
    ids_resp = tok_q(" " + resposta, add_special_tokens=False, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = qwen(torch.cat([ids_ctx, ids_resp], dim=1)).logits
    lp = F.log_softmax(logits[0, ids_ctx.shape[1] - 1: -1].float(), dim=-1)
    return float(lp.gather(1, ids_resp[0].unsqueeze(1)).sum())

SUBSET = [0, 2, 4, 5, 6, 10, 19, 23]     # perguntas com resposta curta e extraível
torch.manual_seed(0)
chunk_irrelevante = CHUNKS[7]["texto"]    # um chunk qualquer, fixo

print(f"{'pergunta':<44} {'sem ctx':>9} {'ctx recuperado':>14} {'ctx controle':>12}")
print("-" * 78)
ganhos = []
for idx in SUBSET:
    pergunta, resposta, _ = PERGUNTAS[idx]
    topo, _ = buscar_hibrida(pergunta, k=1)
    recuperado = CHUNKS[topo[0]]["texto"]
    sem = logprob_resposta(None, pergunta, resposta)
    com = logprob_resposta(recuperado, pergunta, resposta)
    errado = logprob_resposta(chunk_irrelevante, pergunta, resposta)
    ganhos.append(com - sem)
    print(f"{pergunta[:43]:<44} {sem:>9.2f} {com:>10.2f} {errado:>11.2f}")

media = sum(ganhos) / len(ganhos)
print(f"\nganho médio do contexto recuperado: {media:+.2f} nats "
      f"= resposta {math.exp(media):,.0f}x mais provável")

# %% [markdown]
# **As três colunas contam a história completa do RAG:** o contexto recuperado pode
# deslocar a probabilidade da resposta em ordens de grandeza. Ele não é chamado de
# "certo" por definição: o gabarito de passagens permite verificar isso separadamente.
# A coluna de controle mostra o custo do retrieval ruim: pode ser pior que nenhum
# contexto — o modelo confia no lixo que você entregou.
#
# ## Lab 7 — Abstenção: "não está na base"
#
# Produção precisa detectar perguntas fora da base — melhor responder "não sei" que
# alucinar sobre um chunk irrelevante. O sinal mais simples: o score do top-1.

# %%
FORA_DA_BASE = [
    "Qual a capital da Austrália?",
    "Como fazer um bolo de cenoura?",
    "Quem ganhou a Copa do Mundo de 2022?",
    "Qual o melhor framework de frontend em 2026?",
    "Como investir em renda fixa?",
]

print(f"{'pergunta':<52} {'sim. top-1':>11}")
print("-" * 66)
scores_dentro, scores_fora = [], []
for pergunta, _, _ in PERGUNTAS[:10]:
    _, s = buscar_densa(pergunta, k=1)
    scores_dentro.append(s[0])
    print(f"{pergunta[:50]:<52} {s[0]:>11.3f}")
print()
for pergunta in FORA_DA_BASE:
    _, s = buscar_densa(pergunta, k=1)
    scores_fora.append(s[0])
    print(f"{pergunta[:50]:<52} {s[0]:>11.3f}  ← fora da base")

limiar = (min(scores_dentro) + max(scores_fora)) / 2
separa = min(scores_dentro) > max(scores_fora)
print(f"\ndentro: {min(scores_dentro):.3f}–{max(scores_dentro):.3f} | "
      f"fora: {min(scores_fora):.3f}–{max(scores_fora):.3f}")
print(f"um limiar simples separa os dois grupos? {separa}"
      + (f" (ex.: {limiar:.3f})" if separa else " — os intervalos se sobrepõem"))

# %% [markdown]
# > ⚠️ Cossenos do e5 vivem numa faixa comprimida (~0,75–0,90) — o que importa é a
# > SEPARAÇÃO entre os grupos, não o valor absoluto. Se os intervalos se sobrepõem,
# > o limiar simples não basta: as soluções de produção usam calibração num conjunto
# > rotulado, rerankers (que dão scores mais discriminativos) ou a instrução explícita
# > "responda apenas com base no contexto; se não estiver nele, diga que não sabe" —
# > que o lab_mlx testa.
#
# ---
#
# ## Encerramento
#
# Você construiu e avaliou um sistema de RAG completo — sobre o material que vai
# consultar de verdade:
#
# - chunking por estrutura semântica com sobreposição;
# - BM25 do zero e busca densa multilíngue, comparados na mesma métrica;
# - fusão RRF, e as falhas listadas como material de engenharia;
# - o valor do contexto medido em probabilidade — incluindo o custo do contexto errado;
# - abstenção por score, com a honestidade sobre seus limites.
#
# No `lab_mlx.py`: o assistente de ponta a ponta com geração real (Qwen2.5-1.5B),
# grounding e citação de fontes.

# %%
