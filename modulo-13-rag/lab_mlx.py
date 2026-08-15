# %% [markdown]
# # Módulo 13 — Laboratório B: o assistente de ponta a ponta no M4
#
# **Só no Mac.** O `lab_cpu.py` construiu e avaliou a RECUPERAÇÃO; aqui entra a GERAÇÃO
# real (Qwen2.5-1.5B) — com grounding, citação de fontes e a medição de fidelidade.
#
# > ⚠️ Não executado pelo autor. Reusa a infraestrutura do lab_cpu (que roda no Mac
# > também) + o mlx_lm dos módulos anteriores.
#
# | Lab | Assunto |
# |---|---|
# | 1 | O pipeline completo: recuperar → montar prompt → gerar com fontes |
# | 2 | Grounding medido: o modelo respeita "responda só pelo contexto"? |
# | 3 | Lost in the middle: a posição do chunk certo importa? |
#
# O arquivo constrói apenas o índice compartilhado de `tools/rag.py`; não é necessário
# executar o `lab_cpu.py` antes.

# %%
import platform
import sys
from pathlib import Path

assert platform.machine() == "arm64", "este lab requer Apple Silicon"

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "tools"))

from rag import FORA_DA_BASE, PERGUNTAS, IndiceRAG

# Constrói apenas o índice. Importar lab_cpu executaria também as avaliações e
# manteria um Qwen PyTorch residente junto do modelo MLX.
indice_rag = IndiceRAG(RAIZ)
CHUNKS = indice_rag.chunks
buscar_hibrida = indice_rag.buscar_hibrida

# %% [markdown]
# ## Lab 1 — O pipeline completo

# %%
from mlx_lm import generate, load

MODELO = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"
model, tok = load(MODELO)

SISTEMA_RAG = (
    "Você é o assistente de estudo de um curso de customização de LLMs. Responda "
    "APENAS com base no contexto fornecido. Cite a fonte no formato [módulo]. Se a "
    "resposta não estiver no contexto, diga exatamente: 'Não encontrei isso no material.'"
)

def responder_rag(pergunta: str, k: int = 4, max_tokens: int = 300,
                  posicao_do_certo: str | None = None):
    topo, _ = buscar_hibrida(pergunta, k=k)
    blocos = [f"[{CHUNKS[i]['modulo']}] {CHUNKS[i]['texto']}" for i in topo]
    if posicao_do_certo == "fim":
        blocos = blocos[1:] + blocos[:1]      # move o top-1 para o FIM da lista
    contexto = "\n\n---\n\n".join(blocos)
    msgs = [{"role": "system", "content": SISTEMA_RAG},
            {"role": "user", "content": f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    try:
        from mlx_lm.sample_utils import make_sampler
        texto = generate(model, tok, prompt=prompt, max_tokens=max_tokens,
                         sampler=make_sampler(temp=0.0), verbose=False)
    except (ImportError, TypeError):
        texto = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return texto, [CHUNKS[i]["modulo"] for i in topo]

# O assistente, em ação:
for pergunta, _, _ in [PERGUNTAS[0], PERGUNTAS[4], PERGUNTAS[12]]:
    resposta, fontes = responder_rag(pergunta)
    print(f"\n{'='*72}\nP: {pergunta}\nfontes recuperadas: {fontes}\n\nR: {resposta[:400]}")

# %% [markdown]
# ## Lab 2 — Grounding medido
#
# Duas medições objetivas:
# 1. **Perguntas DA base**: a resposta contém a informação correta? (verificação por
#    substring da resposta-curta — a mesma extração honesta do módulo 7)
# 2. **Perguntas FORA da base**: o modelo diz "não encontrei" — ou alucina por cima do
#    contexto irrelevante que a busca inevitavelmente devolve?

# %%
def normalizar(s: str) -> str:
    return s.lower().replace(",", ".").replace("−", "-")

acertos = 0
for pergunta, resposta_curta, _ in PERGUNTAS:
    resposta, _ = responder_rag(pergunta)
    if normalizar(resposta_curta) in normalizar(resposta):
        acertos += 1
print(f"perguntas da base respondidas com a informação correta: "
      f"{acertos}/{len(PERGUNTAS)} ({acertos/len(PERGUNTAS):.0%})")

# %%
absteve = 0
for pergunta in FORA_DA_BASE:
    resposta, fontes = responder_rag(pergunta)
    ok = "não encontrei" in resposta.lower()
    absteve += ok
    print(f"{'✓ absteve' if ok else '✗ RESPONDEU':<14} {pergunta[:44]:<46} "
          f"{'' if ok else '-> ' + resposta[:60]}")
print(f"\nabstenção correta: {absteve}/{len(FORA_DA_BASE)}")

# %% [markdown]
# **As duas taxas juntas são a nota do sistema:** responder o que está na base E recusar
# o que não está. Um sistema que só otimiza a primeira vira gerador de alucinação com
# citações; só a segunda, um atendente inútil. (Compare com a instrução de grounding
# removida — exercício B3 — para medir quanto ela vale.)
#
# ## Lab 3 — Lost in the middle
#
# O chunk certo no INÍCIO do contexto vs no FIM: a taxa de acerto muda?

# %%
for posicao in [None, "fim"]:
    acertos = 0
    for pergunta, resposta_curta, _ in PERGUNTAS:
        resposta, _ = responder_rag(pergunta, k=4, posicao_do_certo=posicao)
        if normalizar(resposta_curta) in normalizar(resposta):
            acertos += 1
    rotulo = "top-1 no INÍCIO (padrão)" if posicao is None else "top-1 no FIM"
    print(f"{rotulo:<28} {acertos}/{len(PERGUNTAS)} ({acertos/len(PERGUNTAS):.0%})")

# %% [markdown]
# Com k=4 chunks o efeito tende a ser pequeno (contexto curto); a literatura o mede em
# contextos de dezenas de documentos. Se quiser vê-lo com força, repita com k=10 e o
# chunk certo enterrado na posição 5 — e lembre a regra prática: **poucos chunks bons,
# o melhor nas bordas.**
#
# ---
#
# ## Encerramento
#
# O assistente está funcional — e é seu: `responder_rag("qualquer dúvida do curso")`.
# Duas extensões naturais (exercícios): reranking com cross-encoder no top-30, e a
# versão agentic em que o modelo decide quando buscar — a ponte para o módulo 15.

# %%
