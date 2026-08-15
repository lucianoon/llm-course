# %% [markdown]
# # Módulo 15 — Laboratório B: agentes capazes no M4
#
# **Só no Mac.** O lab_cpu provou o loop em 0.5B; aqui o roteamento fica bom (7B) e o
# agente ganha uma ferramenta de verdade: **busca no RAG do módulo 13**. É o agentic RAG
# — o modelo decide QUANDO buscar em vez de buscar sempre.
#
# > ⚠️ Não executado pelo autor. O tool calling do mlx_lm passa pelo mesmo
# > apply_chat_template(tools=...); a lógica de loop é idêntica à do lab_cpu.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Roteamento multi-ferramenta: 0.5B vs 7B, medido |
# | 2 | Agentic RAG: o agente que busca no próprio curso |
# | 3 | O agente responde SÓ o que sabe: quando ele decide não buscar |
#
# O índice RAG é construído diretamente por `tools/rag.py`; o módulo 13 não precisa
# ficar carregado na memória.

# %%
import json
import platform
import re
import sys
from pathlib import Path

assert platform.machine() == "arm64", "este lab requer Apple Silicon; use lab_cpu.py"

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "tools"))

import mlx.core as mx
from mlx_lm import generate, load


def gerar_mlx(model, tok, mensagens, ferramentas=None, max_tokens=250):
    prompt = tok.apply_chat_template(mensagens, tools=ferramentas, tokenize=False,
                                     add_generation_prompt=True)
    try:
        from mlx_lm.sample_utils import make_sampler
        return generate(model, tok, prompt=prompt, max_tokens=max_tokens,
                        sampler=make_sampler(temp=0.0), verbose=False)
    except (ImportError, TypeError):
        return generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)

def extrair_tool_call(texto):
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", texto, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None

# %% [markdown]
# ## Lab 1 — Roteamento: o tamanho importa
#
# O lab_cpu mostrou o 0.5B confundindo ferramentas parecidas. Aqui a mesma prova de
# roteamento no 1.5B e no 7B — a curva de capacidade de tool use por tamanho.

# %%
def calculadora(expressao: str) -> str:
    from calculadora import calcular

    try:
        return calcular(expressao)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return "erro"

BASE = {"capital do brasil": "Brasília", "capital da frança": "Paris",
        "maior planeta": "Júpiter", "capital do japão": "Tóquio"}

def consultar_base(pergunta: str) -> str:
    for k, v in BASE.items():
        if k in (pergunta or "").lower():
            return v
    return "não encontrado"

FERRAMENTAS = [
    {"type": "function", "function": {"name": "calculadora",
     "description": "Calcula uma expressão aritmética exata. Use para QUALQUER conta.",
     "parameters": {"type": "object", "properties": {"expressao": {"type": "string"}},
                    "required": ["expressao"]}}},
    {"type": "function", "function": {"name": "consultar_base",
     "description": "Consulta fatos de geografia e astronomia.",
     "parameters": {"type": "object", "properties": {"pergunta": {"type": "string"}},
                    "required": ["pergunta"]}}},
]
CASOS = [("Quanto é 123 + 456?", "calculadora"), ("Qual a capital do Brasil?", "consultar_base"),
         ("Quanto é 50 × 3?", "calculadora"), ("Qual o maior planeta?", "consultar_base"),
         ("Qual a capital da França?", "consultar_base"), ("Quanto é 1000 - 250?", "calculadora")]

for modelo_id in ["mlx-community/Qwen2.5-1.5B-Instruct-bf16",
                  "mlx-community/Qwen2.5-7B-Instruct-4bit"]:
    model, tok = load(modelo_id)
    acertos = 0
    for pergunta, esperada in CASOS:
        chamada = extrair_tool_call(gerar_mlx(model, tok, [{"role": "user", "content": pergunta}], FERRAMENTAS))
        acertos += (chamada is not None and chamada.get("name") == esperada)
    print(f"{modelo_id.split('/')[-1]:<34} roteamento: {acertos}/{len(CASOS)}")
    del model
    mx.clear_cache()

# %% [markdown]
# Espere a curva subir com o tamanho — e compare com o 0.5B do lab_cpu. Tool use é uma
# capacidade emergente com escala (como o CoT do módulo 7): abaixo de certo tamanho, o
# modelo emite JSON válido mas escolhe mal.
#
# ## Lab 2 — Agentic RAG: o agente que busca no curso
#
# A ferramenta agora é a busca do módulo 13. O agente decide QUANDO chamá-la — e para
# perguntas que ele já sabe, pode responder direto. É o meio-termo do módulo 13, seção 6.

# %%
# Reusa apenas a recuperação, sem executar avaliações nem carregar o Qwen PyTorch.
from rag import IndiceRAG

indice_rag = IndiceRAG(RAIZ)
CHUNKS, buscar_hibrida = indice_rag.chunks, indice_rag.buscar_hibrida

def buscar_no_curso(consulta: str) -> str:
    topo, _ = buscar_hibrida(consulta, k=2)
    return "\n---\n".join(f"[{CHUNKS[i]['modulo']}] {CHUNKS[i]['texto'][:300]}" for i in topo)

FERRAMENTA_RAG = [{"type": "function", "function": {"name": "buscar_no_curso",
    "description": "Busca trechos no material do curso de LLMs. Use para perguntas sobre "
                   "o conteúdo técnico do curso (tokenização, LoRA, DPO, RAG, etc.).",
    "parameters": {"type": "object", "properties": {"consulta": {"type": "string"}},
                   "required": ["consulta"]}}}]

model, tok = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

def agente_rag(pergunta, max_passos=4, verbose=True):
    msgs = [{"role": "system", "content":
             "Você é um tutor do curso de LLMs. Para perguntas sobre o conteúdo do curso, "
             "SEMPRE busque antes de responder e cite [módulo]. Para perguntas gerais que "
             "você já sabe, responda direto sem buscar."},
            {"role": "user", "content": pergunta}]
    for passo in range(max_passos):
        saida = gerar_mlx(model, tok, msgs, FERRAMENTA_RAG, max_tokens=350)
        chamada = extrair_tool_call(saida)
        if chamada is None:
            final = re.sub(r"<\|.*?\|>", "", saida).strip()
            if verbose:
                print(f"  RESPOSTA: {final[:250]}")
            return final
        args = chamada.get("arguments", {})
        res = buscar_no_curso(**args)
        if verbose:
            print(f"  BUSCOU: {args.get('consulta','')!r}")
        msgs.append({"role": "assistant", "content": saida.split("<|")[0].strip()})
        msgs.append({"role": "tool", "name": "buscar_no_curso", "content": res})
    return "(max passos)"

for p in ["O que é catastrophic forgetting?",
          "Por que a matriz B do LoRA começa em zeros?",
          "Qual a capital da França?"]:      # esta ele deve responder SEM buscar
    print(f"\nP: {p}")
    agente_rag(p)

# %% [markdown]
# ## Lab 3 — O agente sabe quando NÃO buscar?
#
# A decisão de buscar é ela própria uma capacidade. Medindo: em perguntas DO curso ele
# busca? Em perguntas gerais triviais, ele economiza a busca?

# %%
PERGUNTAS_CURSO = ["O que é NF4?", "Como funciona o GRPO?", "O que é o attention sink?",
                   "Por que bf16 dispensa loss scaling?"]
PERGUNTAS_GERAIS = ["Quanto é 2+2?", "Qual a cor do céu?", "Bom dia, tudo bem?"]

def buscou(pergunta):
    msgs = [{"role": "system", "content":
             "Busque no curso para perguntas técnicas de LLMs; responda direto o resto."},
            {"role": "user", "content": pergunta}]
    saida = gerar_mlx(model, tok, msgs, FERRAMENTA_RAG, max_tokens=80)
    return extrair_tool_call(saida) is not None

buscou_curso = sum(buscou(p) for p in PERGUNTAS_CURSO)
buscou_geral = sum(buscou(p) for p in PERGUNTAS_GERAIS)
print(f"buscou em perguntas DO CURSO:  {buscou_curso}/{len(PERGUNTAS_CURSO)} (queremos alto)")
print(f"buscou em perguntas TRIVIAIS:  {buscou_geral}/{len(PERGUNTAS_GERAIS)} (queremos baixo)")
print("\nA diferença entre as duas taxas é a 'inteligência de roteamento' do agente —")
print("buscar sempre é caro; nunca buscar é o RAG do módulo 13 sem o agente.")

# %% [markdown]
# ---
#
# ## Encerramento
#
# O agente capaz, no seu Mac: roteamento que melhora com escala, e o agentic RAG que
# decide quando consultar o próprio curso. Você agora tem um tutor que busca, cita e
# sabe (às vezes) quando não precisa buscar.
#
# Extensões (exercícios): a ferramenta de execução de código (o agente que roda Python
# para responder), memória entre turnos, e o sandbox de segurança para o Lab 6 do
# lab_cpu levado a sério.
#
# No módulo 16, mudamos de escala de análise: em vez de o que o agente FAZ, o que
# acontece DENTRO do modelo — interpretabilidade mecanicista, e roda em CPU.

# %%
