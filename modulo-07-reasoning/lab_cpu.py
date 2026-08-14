# %% [markdown]
# # Módulo 7 — Laboratório A: o valor do raciocínio, medido
#
# **Roda em CPU (Windows ou Mac).** Usa o Qwen2.5-0.5B-Instruct e o GSM8K.
# Tempo total: ~15 minutos (a geração em CPU é lenta — ~8 tok/s — e o lab foi
# dimensionado para isso).
#
# | Lab | Assunto | Método |
# |---|---|---|
# | 1 | Extração robusta de resposta | testes de unidade |
# | 2 | **O raciocínio desloca probabilidade** | só forward, sem geração |
# | 3 | CoT vs resposta direta | geração, 10 problemas |
# | 4 | Self-consistency | geração em batch |
# | 5 | O orçamento de tokens | geração truncada |
# | 6 | O raciocínio é usado? (fidelidade causal) | só forward |
#
# Antes: `python dados.py`

# %%
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

torch.manual_seed(0)
AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
V5 = int(transformers.__version__.split(".")[0]) >= 5
DTYPE_KW = {"dtype": torch.float32} if V5 else {"torch_dtype": torch.float32}

MODELO = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODELO)
model = AutoModelForCausalLM.from_pretrained(MODELO, **DTYPE_KW)
model.eval()

gabarito = [json.loads(l) for l in
            (AQUI / "data" / "gabarito_teste.jsonl").open(encoding="utf-8")]
print(f"{len(gabarito)} problemas de teste com resposta verificável")

# %% [markdown]
# ## Lab 1 — Extração robusta
#
# A avaliação inteira depende de extrair o número final da resposta. Uma regex ingênua
# transforma acertos em erros — e você conclui que o modelo é pior do que é.

# %%
def extrair_resposta(texto: str):
    """Extrai o último número da resposta, normalizando moeda, vírgulas e '.00'."""
    # preferência: o que vem depois de "resposta final" / "answer", se existir
    m = re.search(r"(?:resposta final|final answer|answer is)[:\s]*\$?\s*([\-0-9.,]+)",
                  texto, re.IGNORECASE)
    candidato = m.group(1) if m else None
    if candidato is None:
        numeros = re.findall(r"-?\$?\d[\d,]*\.?\d*", texto)
        if not numeros:
            return None
        candidato = numeros[-1]                      # o ÚLTIMO número da resposta
    limpo = candidato.replace("$", "").replace(",", "").rstrip(".")
    if limpo.endswith(".0") or limpo.endswith(".00"):
        limpo = limpo.split(".")[0]
    return limpo or None

# A função de avaliação também precisa de testes:
casos = [
    ("The answer is $108.", "108"),
    ("So we get 48+24 = 72.\n\nResposta final: 72", "72"),
    ("She has 1,250 apples in total.", "1250"),
    ("The total is 35.00 dollars", "35"),
    ("First 10, then 20, finally 15", "15"),
    ("I cannot solve this", None),
]
for texto, esperado in casos:
    obtido = extrair_resposta(texto)
    status = "ok" if obtido == esperado else f"FALHOU (obtido {obtido!r})"
    print(f"  {status:<24} {texto[:45]!r} -> {esperado!r}")
assert all(extrair_resposta(t) == e for t, e in casos), "conserte a extração antes de seguir"

# %% [markdown]
# ## Lab 2 — O raciocínio desloca probabilidade
#
# O experimento mais limpo do módulo, e não gera um único token: medimos a probabilidade
# que o modelo atribui à **resposta correta**, com e sem o raciocínio no contexto.
#
# Se a teoria da seção 1 está certa — raciocínio como memória de trabalho externa — a
# resposta certa deve ficar ordens de magnitude mais provável depois dos passos escritos.

# %%
def logprob_da_resposta(pergunta: str, prefixo_assistente: str, resposta: str) -> float:
    """log P(resposta | pergunta + prefixo), somado sobre os tokens da resposta."""
    msgs = [{"role": "user", "content": pergunta}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    contexto = prompt + prefixo_assistente
    ids_ctx = tok(contexto, return_tensors="pt")["input_ids"]
    ids_resp = tok(resposta, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = torch.cat([ids_ctx, ids_resp], dim=1)

    with torch.no_grad():
        logits = model(ids).logits
    # logits da posição t predizem o token t+1 (o shift do módulo 1)
    lp = F.log_softmax(logits[0, ids_ctx.shape[1] - 1: -1].float(), dim=-1)
    return float(lp.gather(1, ids_resp[0].unsqueeze(1)).sum())

# %%
treino_bruto = [json.loads(l) for l in
                (AQUI / "data" / "gsm8k_test.jsonl").open(encoding="utf-8")]

def raciocinio_ouro(answer: str) -> str:
    return re.sub(r"<<[^>]*>>", "", answer.split("####")[0].strip())

print(f"{'#':>3} {'log P sem raciocínio':>21} {'log P com raciocínio':>21} {'razão':>14}")
print("-" * 66)
ganhos = []
for i in range(8):
    p = treino_bruto[i]
    final = gabarito[i]["answer"]
    alvo = f" {final}"

    sem = logprob_da_resposta(p["question"], "Resposta final:", alvo)
    com = logprob_da_resposta(
        p["question"], raciocinio_ouro(p["answer"]) + "\n\nResposta final:", alvo)
    ganhos.append(com - sem)
    print(f"{i:>3} {sem:>21.2f} {com:>21.2f} {torch.exp(torch.tensor(com - sem)):>13.1f}x")

media = sum(ganhos) / len(ganhos)
print(f"\nganho médio: {media:+.2f} nats = a resposta certa fica "
      f"{torch.exp(torch.tensor(media)):,.0f}x mais provável com o raciocínio no contexto")

# %% [markdown]
# **Isto é a seção 1 do README em números.** O modelo é o mesmo, os pesos são os mesmos —
# a única diferença é que os resultados intermediários estão materializados no contexto,
# onde a atenção os alcança. O raciocínio não é decoração: ele muda a distribuição.
#
# ## Lab 3 — CoT vs resposta direta

# %%
def gerar(pergunta: str, instrucao: str, max_novos: int = 250) -> tuple[str, int]:
    msgs = [{"role": "user", "content": pergunta + "\n\n" + instrucao}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_novos, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    novos = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(novos, skip_special_tokens=True), len(novos)

INSTRUCAO_DIRETA = "Answer with ONLY the final number. No explanation."
INSTRUCAO_COT = "Think step by step, then end with 'Final answer: <number>'."

N = 10
resultados = {"direta": [], "cot": []}
t0 = time.perf_counter()
for i in range(N):
    p, ouro = gabarito[i]["question"], gabarito[i]["answer"]
    for nome, instrucao, max_t in [("direta", INSTRUCAO_DIRETA, 30),
                                   ("cot", INSTRUCAO_COT, 250)]:
        texto, n_tok = gerar(p, instrucao, max_t)
        acertou = extrair_resposta(texto) == ouro
        resultados[nome].append({"acertou": acertou, "tokens": n_tok, "texto": texto})
    print(f"  problema {i}: direta {'✓' if resultados['direta'][-1]['acertou'] else '✗'} "
          f"| cot {'✓' if resultados['cot'][-1]['acertou'] else '✗'}")
print(f"\n{time.perf_counter() - t0:.0f}s")

# %%
print(f"{'modo':<10} {'acurácia':>10} {'tokens médios':>15} {'custo relativo':>16}")
print("-" * 55)
base_tok = sum(r["tokens"] for r in resultados["direta"]) / N
for nome, rs in resultados.items():
    acc = sum(r["acertou"] for r in rs) / N
    toks = sum(r["tokens"] for r in rs) / N
    print(f"{nome:<10} {acc:>10.0%} {toks:>15.0f} {toks / base_tok:>15.1f}x")

# %%
# Um exemplo de cada, para calibrar o olhar:
print("=== resposta DIRETA (problema 0) ===")
print(resultados["direta"][0]["texto"][:200])
print(f"\n=== resposta CoT (problema 0) — acertou: {resultados['cot'][0]['acertou']} ===")
print(resultados["cot"][0]["texto"][:600])

# %% [markdown]
# > ⚠️ **Leia o resultado com o tamanho do modelo em mente.** Este é um 0.5B — a seção 2
# > do README avisa que CoT emergiu com escala e pode até atrapalhar em modelos pequenos,
# > que geram *forma* de raciocínio com passos errados. Qualquer que seja o resultado
# > acima, ele mede este modelo neste benchmark, e o exercício B1 pede a comparação com
# > um modelo maior no M4.
#
# ## Lab 4 — Self-consistency
#
# Amostrar `k` cadeias com temperatura e **votar na resposta extraída**. Geramos em batch
# — e note o `padding_side="left"`, a armadilha do módulo 2.

# %%
def amostrar_k(pergunta: str, k: int = 5, max_novos: int = 250) -> list[str]:
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    msgs = [{"role": "user", "content": pergunta + "\n\n" + INSTRUCAO_COT}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok([prompt] * k, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_novos, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    return [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

from collections import Counter

K = 5
print(f"self-consistency com k={K} em 4 problemas:\n")
for i in range(4):
    p, ouro = gabarito[i]["question"], gabarito[i]["answer"]
    respostas = [extrair_resposta(t) for t in amostrar_k(p, K)]
    votos = Counter(r for r in respostas if r is not None)
    vencedora = votos.most_common(1)[0][0] if votos else None
    greedy_acertou = resultados["cot"][i]["acertou"]
    print(f"  problema {i}: votos {dict(votos)}")
    print(f"    greedy: {'✓' if greedy_acertou else '✗'}  |  "
          f"maioria ({vencedora}): {'✓' if vencedora == ouro else '✗'}  |  ouro: {ouro}")

# %% [markdown]
# A intuição do porquê funciona: erros de raciocínio são **diversos** (cada cadeia erra
# num lugar diferente, produzindo respostas diferentes), enquanto acertos **convergem**
# para o mesmo número. A votação cancela o ruído e preserva o sinal — ao custo de `k×`
# a geração.
#
# > ⚠️ **E quando NÃO funciona — que é o que você provavelmente viu acima.** Na execução
# > de referência (0.5B), os votos saíram espalhados — `{'2': 2, '48': 1, '0.1875': 1,
# > '38': 1}` — e a maioria errou até em um problema em que o greedy havia ACERTADO.
# >
# > A self-consistency pressupõe que a resposta certa é o **modo** da distribuição: o
# > modelo precisa acertar com mais frequência do que repete qualquer erro específico.
# > Num 0.5B no GSM8K (acurácia ~30%), os erros dominam — e votar entre erros diversos
# > só formaliza o ruído. A técnica AMPLIFICA competência existente; não a cria. Nos
# > modelos onde a acurácia base já passa de ~50%, os +10–20 p.p. da literatura aparecem.
# > O exercício B1 mede isso no 1.5B e no 7B.
#
# ## Lab 5 — O orçamento de tokens
#
# A seção 1 previu: problemas de `k` passos precisam de espaço para `k` passos. Cortar
# `max_tokens` não torna o modelo conciso — torna-o errado.

# %%
print(f"{'max_tokens':>11} {'acurácia (6 problemas)':>24} {'truncados':>10}")
print("-" * 50)
for orcamento in [30, 60, 120, 250]:
    acertos, truncados = 0, 0
    for i in range(6):
        texto, n_tok = gerar(gabarito[i]["question"], INSTRUCAO_COT, orcamento)
        if n_tok >= orcamento:
            truncados += 1
        if extrair_resposta(texto) == gabarito[i]["answer"]:
            acertos += 1
    print(f"{orcamento:>11} {acertos}/6{'':>18} {truncados:>9}/6")

# %% [markdown]
# ## Lab 6 — O raciocínio é usado? Fidelidade causal
#
# A seção 5 do README diz que o CoT pode ser infiel — racionalização de uma resposta já
# decidida. Um teste causal barato: **corromper um número intermediário** do raciocínio
# de ouro e medir para onde a probabilidade vai.
#
# Se o modelo *usa* o raciocínio, a resposta consistente com o erro deve ficar mais
# provável que a resposta originalmente correta.

# %%
# Problema 0 do treino: Natalia vende 48 clipes, depois metade disso. 48+24 = 72.
pergunta = ("Natalia sold clips to 48 of her friends in April, and then she sold half "
            "as many clips in May. How many clips did Natalia sell altogether in April and May?")
raciocinio_certo = ("Natalia sold 48/2 = 24 clips in May.\n"
                    "Natalia sold 48+24 = 72 clips altogether in April and May.")
raciocinio_corrompido = ("Natalia sold 48/2 = 30 clips in May.\n"          # 24 -> 30
                         "Natalia sold 48+30 = 78 clips altogether in April and May.")

print(f"{'contexto':<26} {'log P(72)':>12} {'log P(78)':>12} {'mais provável':>15}")
print("-" * 70)
for nome, prefixo in [("sem raciocínio", "Resposta final:"),
                      ("raciocínio CORRETO", raciocinio_certo + "\n\nResposta final:"),
                      ("raciocínio CORROMPIDO", raciocinio_corrompido + "\n\nResposta final:")]:
    lp72 = logprob_da_resposta(pergunta, prefixo, " 72")
    lp78 = logprob_da_resposta(pergunta, prefixo, " 78")
    print(f"{nome:<26} {lp72:>12.2f} {lp78:>12.2f} {'72' if lp72 > lp78 else '78':>15}")

# %% [markdown]
# **Se a última linha aponta para 78**, o modelo está de fato *lendo* o raciocínio — a
# resposta segue a cadeia, mesmo errada. Isso confirma o uso causal **neste caso**, e é
# exatamente o mecanismo que torna o CoT perigoso quando os passos contêm erros: o modelo
# não verifica, ele **continua**.
#
# (A infidelidade do README é o fenômeno complementar: casos em que a resposta NÃO segue
# a cadeia escrita. Os dois coexistem — por isso CoT como auditoria é evidência fraca.)
#
# ---
#
# ## Encerramento
#
# Medido neste lab:
#
# - o raciocínio no contexto torna a resposta correta ordens de magnitude mais provável,
#   sem gerar um token (Lab 2);
# - o trade-off acurácia × custo do CoT num modelo pequeno (Lab 3);
# - self-consistency convertendo diversidade de erros em sinal (Lab 4);
# - truncar o orçamento de tokens destrói a acurácia (Lab 5);
# - o modelo segue o raciocínio — inclusive quando corrompido (Lab 6).
#
# No `lab_mlx.py`: treinar os dois LoRAs (com e sem raciocínio) nos MESMOS problemas e
# medir a diferença — o experimento central do módulo.

# %%
