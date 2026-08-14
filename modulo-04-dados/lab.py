# %% [markdown]
# # Módulo 4 — Laboratório: curadoria de datasets
#
# Roda em **CPU**. Baixa o Alpaca (22 MB) na primeira execução e reaproveita o corpus
# literário do módulo 3.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Análise exploratória de um dataset de SFT real |
# | 2 | Deduplicação: exata e near-duplicate (MinHash + LSH do zero) |
# | 3 | Filtros heurísticos de qualidade — e o que eles fazem com o português |
# | 4 | Diversidade: medindo o que "variedade" significa |
# | 5 | Contaminação de benchmark por sobreposição de n-gramas |
# | 6 | Formatação para SFT: template, masking e o EOS |
# | 7 | **Experimento controlado: corpus limpo vs poluído** |
# | 8 | Seleção por qualidade: menos dados, melhor modelo |

# %%
import hashlib
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd().parent / "tools"))
sys.path.insert(0, str(Path.cwd()))

import dados
import minigpt

random.seed(42)
torch.manual_seed(42)

# %% [markdown]
# ## Lab 1 — Análise exploratória do Alpaca
#
# Antes de treinar em qualquer dataset, olhe para ele. Sempre.

# %%
alpaca = dados.carregar_alpaca()
print(f"exemplos: {len(alpaca):,}")
print(f"campos  : {list(alpaca[0].keys())}\n")

for i in [0, 7, 100]:
    ex = alpaca[i]
    print(f"--- exemplo {i} ---")
    print(f"instruction: {ex['instruction'][:90]}")
    print(f"input      : {ex['input'][:90]!r}")
    print(f"output     : {ex['output'][:120]}\n")

# %%
comp_inst = [len(e["instruction"].split()) for e in alpaca]
comp_out = [len(e["output"].split()) for e in alpaca]
com_input = sum(1 for e in alpaca if e["input"].strip())

def resumo(valores, nome):
    v = sorted(valores)
    n = len(v)
    print(f"{nome:<14} média {sum(v)/n:>6.1f} | mediana {v[n//2]:>4} | "
          f"p95 {v[int(n*0.95)]:>4} | máx {v[-1]:>5} | vazios {sum(1 for x in v if x == 0):>5}")

resumo(comp_inst, "instrução")
resumo(comp_out, "resposta")
print(f"\ncom campo 'input' preenchido: {com_input:,} ({com_input/len(alpaca):.1%})")

# %%
# Histograma em texto do comprimento das respostas.
faixas = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 10**9)]
print("distribuição do comprimento das respostas (palavras):")
for lo, hi in faixas:
    n = sum(1 for c in comp_out if lo <= c < hi)
    rotulo = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    print(f"  {rotulo:>9} {n:>6,} {'█' * int(60 * n / len(alpaca))}")

# %% [markdown]
# **O que procurar aqui:** respostas de comprimento zero (exemplos quebrados), uma cauda
# muito longa (exemplos que vão dominar a loss), e concentração excessiva numa faixa
# estreita (o modelo vai aprender a sempre responder com aquele tamanho).
#
# Agora a diversidade de **tarefas**, medida pelo verbo inicial da instrução — a métrica
# que o próprio paper do Self-Instruct usa:

# %%
verbos = Counter(e["instruction"].split()[0].lower().strip(".,:") for e in alpaca if e["instruction"].split())
print(f"verbos iniciais distintos: {len(verbos):,}\n")
for verbo, n in verbos.most_common(15):
    print(f"  {verbo:<14} {n:>6,} {n/len(alpaca):>6.1%}")

concentracao = sum(n for _, n in verbos.most_common(10)) / len(alpaca)
print(f"\nos 10 verbos mais comuns cobrem {concentracao:.1%} do dataset")

# %% [markdown]
# ## Lab 2 — Deduplicação
#
# ### Exata

# %%
def hash_doc(texto):
    return hashlib.md5(texto.strip().lower().encode()).hexdigest()

vistos, exatas = set(), 0
for e in alpaca:
    h = hash_doc(e["instruction"] + "\n" + e["input"])
    if h in vistos:
        exatas += 1
    vistos.add(h)

print(f"duplicatas exatas: {exatas:,} ({exatas/len(alpaca):.2%})")
print("→ pouquíssimas. É por isso que a dedup exata é insuficiente: quase nada é idêntico.")

# %% [markdown]
# ### MinHash + LSH, do zero
#
# A propriedade central: `P(minhash(A) == minhash(B)) = J(A,B)`. Verificando primeiro que
# ela é verdadeira, depois usando-a.

# %%
def shingles(texto, n=3):
    """n-gramas de palavras — a representação do documento como conjunto."""
    palavras = re.findall(r"\w+", texto.lower())
    return {" ".join(palavras[i:i + n]) for i in range(max(1, len(palavras) - n + 1))}

def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0

def assinatura(conjunto, k=128, semente=0):
    """k mínimos independentes. Cada 'permutação' é um hash com sal diferente."""
    if not conjunto:
        return [0] * k
    return [
        min(int(hashlib.md5(f"{i}:{s}".encode()).hexdigest()[:12], 16) for s in conjunto)
        for i in range(k)
    ]

# Verificação empírica da propriedade
a = shingles(alpaca[0]["output"])
b = shingles(alpaca[0]["output"] + " Also, drink water regularly and avoid stress.")
sa, sb = assinatura(a), assinatura(b)
estimado = sum(1 for x, y in zip(sa, sb) if x == y) / len(sa)

print(f"Jaccard real     : {jaccard(a, b):.4f}")
print(f"Jaccard estimado : {estimado:.4f}   (por MinHash com k=128)")
print(f"erro             : {abs(jaccard(a, b) - estimado):.4f}   (teoria: ~1/√128 = {1/math.sqrt(128):.3f})")

# %%
def lsh_candidatos(assinaturas, b=16, r=8):
    """Divide cada assinatura em b bandas de r linhas. Colidiu numa banda -> candidato."""
    baldes = {}
    for idx, sig in enumerate(assinaturas):
        for banda in range(b):
            chave = (banda, tuple(sig[banda * r:(banda + 1) * r]))
            baldes.setdefault(chave, []).append(idx)
    pares = set()
    for grupo in baldes.values():
        if len(grupo) > 1:
            for i in range(len(grupo)):
                for j in range(i + 1, len(grupo)):
                    pares.add((grupo[i], grupo[j]))
    return pares

B, R = 16, 8
limiar = (1 / B) ** (1 / R)
print(f"LSH com b={B}, r={R}  ->  limiar aproximado J ≈ {limiar:.3f}")
print("curva S: P(candidato) = 1 - (1 - J^r)^b")
for j in [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]:
    p = 1 - (1 - j ** R) ** B
    print(f"  J={j:.2f} -> P(vira candidato) = {p:>6.1%} {'█' * int(p * 40)}")

# %%
# Rodando sobre uma amostra do Alpaca.
AMOSTRA = 4000
sub = alpaca[:AMOSTRA]
conjuntos = [shingles(e["instruction"] + " " + e["input"]) for e in sub]
assinaturas = [assinatura(c, k=B * R) for c in conjuntos]

pares = lsh_candidatos(assinaturas, B, R)
confirmados = [(i, j, jaccard(conjuntos[i], conjuntos[j])) for i, j in pares]
confirmados = [p for p in confirmados if p[2] >= 0.7]

print(f"amostra              : {AMOSTRA:,} exemplos")
print(f"pares candidatos LSH : {len(pares):,}   (contra {AMOSTRA*(AMOSTRA-1)//2:,} pares possíveis)")
print(f"redução de trabalho  : {1 - len(pares)/(AMOSTRA*(AMOSTRA-1)//2):.4%}")
print(f"near-duplicates (J≥0.7): {len(confirmados):,}\n")

for i, j, s in sorted(confirmados, key=lambda x: -x[2])[:4]:
    print(f"  J={s:.3f}")
    print(f"    [{i}] {sub[i]['instruction'][:80]}")
    print(f"    [{j}] {sub[j]['instruction'][:80]}")

# %% [markdown]
# **O ponto do LSH:** comparar todos os pares seria inviável em escala. O LSH reduz o
# trabalho em ordens de grandeza mantendo praticamente todos os pares realmente similares.
#
# > ⚠️ Repare no que ele encontrou: pares como *"Find the most common noun in this
# > passage"* e *"Describe the style of writing in this passage"*. As **instruções** são
# > diferentes; o que os torna similares é o campo `input` — a mesma passagem longa
# > reaproveitada em vários exemplos. Isso é Jaccard funcionando corretamente e revelando
# > uma decisão sua: se você inclui o `input` no cálculo, está deduplicando por *contexto*;
# > se inclui só a `instruction`, por *tarefa*. As duas escolhas são defensáveis e dão
# > resultados muito diferentes. Decida deliberadamente qual você quer.
#
# ## Lab 3 — Filtros heurísticos
#
# As regras do Gopher. E a armadilha do idioma.

# %%
STOP_EN = {"the", "be", "to", "of", "and", "that", "have", "with"}
STOP_PT = {"de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "não"}

def filtro_gopher(texto, stopwords=STOP_EN, min_palavras=50):
    """Devolve (passou, motivo). Regras de Rae et al. (2021), apêndice A."""
    palavras = texto.split()
    n = len(palavras)
    if n < min_palavras or n > 100_000:
        return False, "número de palavras"

    comp_medio = sum(len(p) for p in palavras) / n
    if not 3 <= comp_medio <= 10:
        return False, "comprimento médio de palavra"

    if (texto.count("#") + texto.count("...")) / n > 0.1:
        return False, "razão símbolo/palavra"

    linhas = [l for l in texto.split("\n") if l.strip()]
    if linhas:
        if sum(1 for l in linhas if l.lstrip().startswith(("•", "-", "*"))) / len(linhas) > 0.9:
            return False, "excesso de bullets"
        if sum(1 for l in linhas if l.rstrip().endswith("...")) / len(linhas) > 0.3:
            return False, "linhas truncadas"

    if sum(1 for p in palavras if any(c.isalpha() for c in p)) / n < 0.8:
        return False, "poucas palavras alfabéticas"

    minusculas = {p.lower().strip(".,;:!?") for p in palavras}
    if len(minusculas & stopwords) < 2:
        return False, "sem stop words"

    return True, "ok"

# %%
corpus = dados.carregar_corpus()
docs_pt = [p for p in corpus.split("\n\n") if len(p.split()) >= 50]
print(f"documentos em português (parágrafos longos de Machado): {len(docs_pt):,}\n")

for nome, stops in [("stop words em INGLÊS", STOP_EN), ("stop words em PORTUGUÊS", STOP_PT)]:
    motivos = Counter()
    for d in docs_pt:
        ok, motivo = filtro_gopher(d, stops)
        motivos[motivo if not ok else "PASSOU"] += 1
    passou = motivos["PASSOU"]
    print(f"{nome}: {passou:,}/{len(docs_pt):,} passaram ({passou/len(docs_pt):.1%})")
    for m, c in motivos.most_common():
        if m != "PASSOU":
            print(f"    rejeitado por {m}: {c:,}")
    print()

# %% [markdown]
# **Machado de Assis reprovado por um filtro de qualidade.** Com a lista de stop words em
# inglês, praticamente todo o corpus é descartado como "não sendo prosa". Esta é a razão
# de corpora multilíngues exigirem filtros recalibrados por idioma — e de tantos modelos
# terem português ruim: o filtro que garantiu a qualidade do inglês jogou fora o resto.
#
# ## Lab 4 — Diversidade

# %%
def distinct_n(textos, n=2, limite=20000):
    """Fração de n-gramas únicos. Alto = variado; baixo = repetitivo."""
    total, unicos = 0, set()
    for t in textos[:limite]:
        palavras = re.findall(r"\w+", t.lower())
        grams = [tuple(palavras[i:i + n]) for i in range(max(0, len(palavras) - n + 1))]
        total += len(grams)
        unicos.update(grams)
    return len(unicos) / total if total else 0

instrucoes = [e["instruction"] for e in alpaca]
respostas = [e["output"] for e in alpaca]

print(f"{'':<16} {'distinct-1':>12} {'distinct-2':>12} {'distinct-3':>12}")
for nome, textos in [("instruções", instrucoes), ("respostas", respostas), ("Machado", docs_pt)]:
    print(f"{nome:<16} " + " ".join(f"{distinct_n(textos, n):>12.4f}" for n in (1, 2, 3)))

# %% [markdown]
# ## Lab 5 — Contaminação de benchmark
#
# O método do GPT-3: se um documento de treino compartilha um n-grama longo com um item
# de teste, marque-o.

# %%
def ngramas(texto, n=13):
    palavras = re.findall(r"\w+", texto.lower())
    return {" ".join(palavras[i:i + n]) for i in range(max(0, len(palavras) - n + 1))}

# Simulando um "benchmark": 200 exemplos que separamos como teste...
random.shuffle(alpaca)
benchmark = alpaca[:200]
treino = alpaca[200:20000]

# ...e um vazamento: 30 itens do benchmark foram parar no treino (com pequenas variações).
vazados = [{**e, "output": e["output"] + " Hope this helps!"} for e in benchmark[:30]]
treino_contaminado = treino + vazados
random.shuffle(treino_contaminado)

indice = set()
for e in treino_contaminado:
    indice |= ngramas(e["instruction"] + " " + e["output"])

detectados = [e for e in benchmark if ngramas(e["instruction"] + " " + e["output"]) & indice]
print(f"itens do benchmark        : {len(benchmark)}")
print(f"itens realmente vazados   : {len(vazados)}")
print(f"itens detectados (13-gram): {len(detectados)}")
print(f"recall da detecção        : {len(detectados)/len(vazados):.1%}")

# %% [markdown]
# > ⚠️ A detecção por n-gramas pega cópias e quase-cópias, mas **não** pega paráfrases ou
# > traduções do benchmark. Contaminação semântica é muito mais difícil de detectar — e é
# > por isso que se deve confiar mais em benchmarks privados e recentes.
#
# ## Lab 6 — Formatação para SFT
#
# Três coisas precisam estar certas: template, masking e EOS.

# %%
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def formatar(exemplo):
    usuario = exemplo["instruction"]
    if exemplo["input"].strip():
        usuario += "\n\n" + exemplo["input"]
    return [{"role": "user", "content": usuario},
            {"role": "assistant", "content": exemplo["output"]}]

msgs = formatar(alpaca[0])
texto_completo = tok.apply_chat_template(msgs, tokenize=False)
print(repr(texto_completo))

# %%
# O masking: loss só nos tokens da resposta.
so_prompt = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
ids_prompt = tok(so_prompt, add_special_tokens=False)["input_ids"]
ids_total = tok(texto_completo, add_special_tokens=False)["input_ids"]

labels = list(ids_total)
for i in range(len(ids_prompt)):
    labels[i] = -100

print(f"tokens totais : {len(ids_total)}")
print(f"tokens do prompt (mascarados com -100): {len(ids_prompt)}")
print(f"tokens que geram loss                 : {sum(1 for l in labels if l != -100)}\n")
for i in [0, 5, len(ids_prompt) - 1, len(ids_prompt), len(ids_prompt) + 3, len(ids_total) - 1]:
    marca = "MASCARADO" if labels[i] == -100 else "treina"
    print(f"  pos {i:>3}  {tok.decode([ids_total[i]])!r:<18} label={labels[i]:>7}  {marca}")

# %%
# O EOS — a armadilha nº 1 do SFT.
print(f"eos_token do Qwen : {tok.eos_token!r} (id {tok.eos_token_id})")
print(f"últimos 5 tokens  : {tok.convert_ids_to_tokens(ids_total[-5:])}")
ocorrencias = [i for i, t in enumerate(ids_total) if t == tok.eos_token_id]
print(f"contém o EOS?       {bool(ocorrencias)}")
print(f"posições do EOS   : {ocorrencias} de {len(ids_total)} tokens\n")

# ⚠️ Dois detalhes que confundem:
# 1. No Qwen, '<|im_end|>' fecha CADA turno (system, user, assistant) — por isso há várias
#    ocorrências. Ele é separador de turno e EOS ao mesmo tempo.
# 2. O EOS não é o ÚLTIMO token: o template fecha com '<|im_end|>\n'. Testar
#    `ids[-1] == eos_token_id` dá falso negativo. O teste correto é a presença.

# %%
# O contraste real: o que acontece quando se monta o prompt à mão.
manual = f"Pergunta: {alpaca[0]['instruction']}\nResposta: {alpaca[0]['output']}"
ids_manual = tok(manual, add_special_tokens=False)["input_ids"]

print("--- montado à mão (o erro comum) ---")
print(f"contém EOS? {tok.eos_token_id in ids_manual}")
print(f"últimos tokens: {tok.convert_ids_to_tokens(ids_manual[-4:])}")
print("\n--- com apply_chat_template ---")
print(f"contém EOS? {tok.eos_token_id in ids_total}")
print("\n→ Um dataset montado à mão treina um modelo que responde certo e NUNCA PARA:")
print("  ele gera a resposta, inventa a próxima pergunta, responde a si mesmo,")
print("  e continua até bater o limite de tokens. Sintoma inconfundível, causa trivial.")

# %% [markdown]
# ## Lab 7 — Experimento controlado: curadoria vale quanto?
#
# Mesmo modelo, mesmo compute, mesmo tokenizer. Muda **só o corpus**:
#
# 1. **limpo** — Machado de Assis
# 2. **poluído** — o mesmo, mais 40% de lixo de web sintético
# 3. **filtrado** — o poluído, passado pelos filtros do Lab 3
#
# Todos avaliados no **mesmo conjunto de validação limpo**.

# %%
def gerar_lixo(n_docs=1200):
    """Lixo de web realista: os padrões que os filtros do Gopher foram feitos para pegar."""
    saida = []
    for i in range(n_docs):
        tipo = i % 5
        if tipo == 0:   # menu de navegação
            saida.append("\n".join("- " + random.choice(
                ["Home", "Sobre", "Contato", "Produtos", "Blog", "Login", "Cadastre-se"]
            ) for _ in range(random.randint(20, 60))))
        elif tipo == 1:  # aviso de cookies repetido
            saida.append(("Este site utiliza cookies para melhorar sua experiência. "
                          "Ao continuar navegando você concorda com nossa política. ") * random.randint(8, 20))
        elif tipo == 2:  # listagem truncada de SEO
            saida.append("\n".join(
                f"{random.choice(['Comprar', 'Melhor', 'Top 10'])} {random.choice(['celular', 'notebook', 'tênis'])} barato..."
                for _ in range(random.randint(25, 70))))
        elif tipo == 3:  # despejo de identificadores
            saida.append(" ".join(hashlib.md5(str(random.random()).encode()).hexdigest()[:12]
                                  for _ in range(random.randint(60, 150))))
        else:            # gibberish
            saida.append(" ".join("".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(2, 14)))
                                  for _ in range(random.randint(60, 140))))
    return saida

lixo = gerar_lixo()
print(f"documentos de lixo gerados: {len(lixo):,} ({sum(len(d) for d in lixo):,} caracteres)")
print(f"\namostra:\n{lixo[2][:200]}")

# %%
docs_limpos = [p for p in corpus.split("\n\n") if len(p.strip()) > 30]
docs_poluidos = docs_limpos + lixo
random.shuffle(docs_poluidos)

docs_filtrados = [d for d in docs_poluidos if filtro_gopher(d, STOP_PT, min_palavras=20)[0]]

# Quanto do lixo o filtro pegou?
set_lixo = set(lixo)
lixo_sobrevivente = sum(1 for d in docs_filtrados if d in set_lixo)
bons_perdidos = len(docs_limpos) - sum(1 for d in docs_filtrados if d not in set_lixo)

print(f"corpus limpo    : {len(docs_limpos):>6,} documentos")
print(f"corpus poluído  : {len(docs_poluidos):>6,} documentos ({len(lixo)/len(docs_poluidos):.1%} é lixo)")
print(f"corpus filtrado : {len(docs_filtrados):>6,} documentos")
print(f"\n  lixo removido pelo filtro : {len(lixo) - lixo_sobrevivente:,}/{len(lixo):,} "
      f"({1 - lixo_sobrevivente/len(lixo):.1%})")
print(f"  bons descartados junto    : {bons_perdidos:,}/{len(docs_limpos):,} "
      f"({bons_perdidos/len(docs_limpos):.1%})  ← o preço do filtro")

# %%
# Tokenizer treinado no corpus LIMPO e usado nos três — comparação justa.
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB = 2048
tk = Tokenizer(models.BPE(unk_token=None))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(["\n\n".join(docs_limpos)],
                       trainers.BpeTrainer(vocab_size=VOCAB, show_progress=False))

def codificar(docs):
    return torch.tensor(tk.encode("\n\n".join(docs)).ids, dtype=torch.long)

# validação SEMPRE limpa: os últimos 10% do corpus original, nunca vistos por ninguém
corte = int(0.9 * len(docs_limpos))
val_limpa = codificar(docs_limpos[corte:])

fontes = {
    "limpo": codificar(docs_limpos[:corte]),
    "poluído": codificar([d for d in docs_poluidos if d not in set(docs_limpos[corte:])]),
    "filtrado": codificar([d for d in docs_filtrados if d not in set(docs_limpos[corte:])]),
}
for nome, t in fontes.items():
    print(f"{nome:<10} {len(t):>9,} tokens de treino")
print(f"{'validação':<10} {len(val_limpa):>9,} tokens (limpa, comum aos três)")

# %%
import time

PASSOS = 200
cfg = minigpt.Config(vocab=VOCAB)
resultados = {}

for nome, fonte in fontes.items():
    inicio = time.perf_counter()
    modelo, hist = minigpt.treinar(cfg, fonte, val_limpa, passos=PASSOS, batch=16, seed=1337)
    resultados[nome] = {"modelo": modelo, "val": hist["val"][-1],
                        "treino": hist["treino"][-1], "tempo": time.perf_counter() - inicio}
    print(f"{nome:<10} loss treino {hist['treino'][-1]:.4f} | "
          f"loss validação LIMPA {hist['val'][-1]:.4f} | {resultados[nome]['tempo']:.0f}s")

# %%
base = resultados["limpo"]["val"]
print(f"{'corpus':<12} {'loss TREINO':>13} {'val (limpa)':>13} {'PPL':>10} {'vs limpo':>11}")
print("-" * 62)
for nome, r in resultados.items():
    print(f"{nome:<12} {r['treino']:>13.4f} {r['val']:>13.4f} {math.exp(r['val']):>10.1f} "
          f"{(r['val'] - base) / base:>+10.1%}")

recuperado = (resultados["poluído"]["val"] - resultados["filtrado"]["val"]) / \
             (resultados["poluído"]["val"] - base + 1e-9)
print(f"\nfração do dano recuperada pela filtragem: {recuperado:.1%}")

# %% [markdown]
# ### ⚠️ A armadilha da loss de treino
#
# Olhe as duas primeiras colunas. O modelo treinado no corpus **poluído** tem a **menor
# loss de treino** de todos — e o **pior desempenho** na validação limpa.
#
# A explicação: lixo de web é trivialmente previsível. Depois de ver "Comprar celular
# barato..." vinte vezes, prever a vigésima primeira é fácil, e a loss desaba. O modelo
# parece estar aprendendo muito quando está apenas modelando repetição.
#
# **Nunca compare modelos pela loss de treino quando os corpora são diferentes.** A loss
# de treino mede o quão previsível é o seu corpus, não o quão bom é o seu modelo. Só a
# avaliação num conjunto fixo e limpo — igual para todos — permite comparação.

# %% [markdown]
# **Leia com atenção o que este experimento mostra.** Os três modelos são idênticos e
# receberam exatamente o mesmo compute. A única variável foi o que estava no corpus. O
# modelo poluído é pior em texto limpo porque gastou capacidade aprendendo a modelar
# menus de navegação e hashes — e capacidade, num modelo de 2M de parâmetros, é escassa.
#
# É a demonstração em miniatura do porquê de o FineWeb-Edu descartar 91% da web e sair
# ganhando.

# %%
contexto = torch.tensor([tk.encode("Uma noite destas").ids])
for nome, r in resultados.items():
    saida = r["modelo"].gerar(contexto, 60, temperatura=0.8, top_k=40)
    print(f"\n=== treinado no corpus {nome} ===")
    print(tk.decode(saida[0].tolist()).replace("\n", " ")[:300])

# %% [markdown]
# ## Lab 8 — Seleção por qualidade
#
# A lição do LIMA: selecionar os melhores exemplos pode superar usar todos. Uma heurística
# simples de pontuação sobre o Alpaca.

# %%
def pontuar(exemplo):
    """Heurística grosseira de qualidade. Em produção use um reward model ou LLM-as-judge."""
    inst, out = exemplo["instruction"], exemplo["output"]
    n_out = len(out.split())
    pontos = 0.0
    pontos += min(n_out / 50, 1.0) * 2          # respostas substanciais
    pontos += min(len(inst.split()) / 15, 1.0)  # instruções específicas
    if n_out < 3:
        pontos -= 3                             # respostas vazias
    if out.strip().endswith((".", "!", "?", "```")):
        pontos += 0.5                           # termina completa
    palavras = out.lower().split()
    if palavras:
        pontos += len(set(palavras)) / len(palavras)   # diversidade lexical interna
    if re.search(r"as an ai|i'm sorry, but|language model", out.lower()):
        pontos -= 2                             # recusas boilerplate
    return pontos

notas = [(pontuar(e), i) for i, e in enumerate(alpaca)]
notas.sort(reverse=True)

print(f"{'percentil':<12} {'nota':>8}  exemplo")
print("-" * 100)
for p in [0, 25, 50, 75, 99]:
    nota, i = notas[int(len(notas) * p / 100)]
    print(f"top {p:>3}%     {nota:>8.2f}  {alpaca[i]['instruction'][:45]:<47} -> {alpaca[i]['output'][:40]}")

piores = [alpaca[i] for _, i in notas[-5:]]
print("\nos 5 piores exemplos do dataset:")
for e in piores:
    print(f"  {e['instruction'][:60]!r} -> {e['output'][:60]!r}")

# %% [markdown]
# Rodar essa seleção e ficar com os 10% melhores é, em muitos casos práticos, melhor do
# que treinar nos 100%. No módulo 5 você vai fazer exatamente isso — e medir.
#
# ---
#
# ## Encerramento
#
# Você acabou de:
#
# - inspecionar um dataset de SFT real e medir sua diversidade de tarefas;
# - implementar MinHash e LSH do zero, e verificar a propriedade que os sustenta;
# - ver Machado de Assis ser reprovado por um filtro de qualidade calibrado em inglês;
# - detectar contaminação de benchmark por sobreposição de n-gramas;
# - formatar um exemplo de SFT com masking correto e confirmar o EOS;
# - **medir** o custo de treinar em dados poluídos e o ganho de filtrá-los;
# - pontuar e ordenar um dataset por qualidade.
#
# Fim dos fundamentos. A partir do módulo 5 você deixa de construir modelos e passa a
# adaptá-los — e vai precisar de GPU.
#
# Agora o `exercicios.md`.

# %%
