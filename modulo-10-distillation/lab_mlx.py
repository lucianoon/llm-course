# %% [markdown]
# # Módulo 10 — Laboratório B: o R1-Distill no seu M4
#
# **Só no Mac.** O pipeline black-box completo, de ponta a ponta:
#
# ```
# professor (R1-Distill-Qwen-7B-4bit ou 1.5B)  →  gera traços no GSM8K
#         →  rejection sampling (filtro por gabarito)
#         →  SFT do aluno (Qwen2.5-0.5B)
#         →  avaliação: aluno antes vs depois vs professor
# ```
#
# > ⚠️ Não executado pelo autor. Comandos padrão do mlx_lm (generate/lora), verificados
# > nos módulos anteriores.
#
# > 🔧 Custo: a geração dos traços domina (professor grande, respostas longas). ~300
# > problemas × ~400 tokens ≈ 2–4 h no 7B-4bit, ~40 min no 1.5B. Comece pelo 1.5B; se
# > os resultados animarem, rode o 7B à noite.
#
# Antes: módulo 7 executado (`python ../modulo-07-reasoning/dados.py`).

# %%
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

assert platform.machine() == "arm64", "este lab requer Apple Silicon; use lab_cpu.py"

AQUI = Path.cwd()
M7 = AQUI.parent / "modulo-07-reasoning"

PROFESSOR = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"   # suba p/ 7B se quiser
ALUNO = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"

N_PROBLEMAS = 300      # problemas para o professor tentar
K_TENTATIVAS = 2       # tentativas por problema (rejection sampling melhora com k>1)

# problemas de TREINO do GSM8K que NÃO estão no gabarito de teste
treino_bruto = [json.loads(l) for l in
                (M7 / "data" / "gsm8k_train.jsonl").read_text(encoding="utf-8")
                .strip().split("\n")]
GABARITO_TESTE = [json.loads(l) for l in
                  (M7 / "data" / "gabarito_teste.jsonl").open(encoding="utf-8")]

def resposta_final(answer):
    m = re.search(r"####\s*([\-0-9.,]+)", answer)
    return m.group(1).replace(",", "").rstrip(".")

problemas = [{"question": e["question"], "answer": resposta_final(e["answer"])}
             for e in treino_bruto[2000: 2000 + N_PROBLEMAS]]
print(f"{len(problemas)} problemas para o professor | {len(GABARITO_TESTE)} de teste")

sys.path.insert(0, str(AQUI.parent / "modulo-09-rl"))
from recompensas_r1 import _extrair_numero    # a extração testada do módulo 7

# %% [markdown]
# ## Etapa 1 — O professor gera

# %%
import mlx.core as mx
from mlx_lm import generate, load

def gerar(model, tokenizer, pergunta, max_tokens=600, temp=0.7):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": pergunta}],
                                           tokenize=False, add_generation_prompt=True)
    try:
        from mlx_lm.sample_utils import make_sampler
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                        sampler=make_sampler(temp=temp), verbose=False)
    except (ImportError, TypeError):
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

TRACOS = AQUI / "tracos_professor.jsonl"

if not TRACOS.exists():
    prof, tok_prof = load(PROFESSOR)
    t0 = time.perf_counter()
    with TRACOS.open("w", encoding="utf-8") as f:
        for i, prob in enumerate(problemas):
            for k in range(K_TENTATIVAS):
                traco = gerar(prof, tok_prof, prob["question"],
                              temp=0.7 if k else 0.6)
                f.write(json.dumps({"question": prob["question"],
                                    "answer": prob["answer"],
                                    "traco": traco}, ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                dt = time.perf_counter() - t0
                print(f"  {i+1}/{len(problemas)} ({dt/60:.0f} min, "
                      f"~{dt/(i+1)*(len(problemas)-i-1)/60:.0f} min restantes)")
    del prof
    mx.clear_cache()

tracos = [json.loads(l) for l in TRACOS.open(encoding="utf-8")]
print(f"{len(tracos)} traços gerados")

# %% [markdown]
# ## Etapa 2 — Rejection sampling
#
# O controle de qualidade grátis das tarefas verificáveis: fica só o que ACERTA.
# Mais dois filtros de sanidade do R1 real: comprimento e legibilidade mínima.

# %%
def limpar_traco(texto: str) -> str:
    """Remove o bloco <think> — destilamos o raciocínio LIMPO, não o monólogo bruto.

    Decisão de design com trade-off real: o R1-Distill original treina COM o <think>
    (o aluno herda o formato thinking). Aqui destilamos para um aluno instruct sem
    template de thinking, então usamos só a parte pós-</think> quando existir, com o
    raciocínio resumido que o professor escreve ali. O exercício B4 inverte a escolha.
    """
    if "</think>" in texto:
        pos = texto.split("</think>", 1)[1].strip()
        return pos if len(pos.split()) >= 10 else texto.replace("<think>", "").replace("</think>", "")
    return texto

aprovados, rejeitados = [], {"errado": 0, "sem_numero": 0, "curto": 0, "longo": 0}
for t in tracos:
    limpo = limpar_traco(t["traco"])
    extraida = _extrair_numero(limpo)
    if extraida is None:
        rejeitados["sem_numero"] += 1
    elif extraida != t["answer"]:
        rejeitados["errado"] += 1
    elif len(limpo.split()) < 15:
        rejeitados["curto"] += 1
    elif len(limpo.split()) > 500:
        rejeitados["longo"] += 1
    else:
        aprovados.append({"messages": [
            {"role": "user", "content": t["question"]},
            {"role": "assistant", "content": limpo},
        ]})

print(f"aprovados: {len(aprovados)}/{len(tracos)} ({len(aprovados)/len(tracos):.0%})")
for motivo, n in rejeitados.items():
    print(f"  rejeitado por {motivo}: {n}")

# dedup por pergunta (K_TENTATIVAS pode aprovar duplicatas — fica a primeira)
vistas, unicos = set(), []
for a in aprovados:
    chave = a["messages"][0]["content"]
    if chave not in vistas:
        vistas.add(chave)
        unicos.append(a)
print(f"após dedup por pergunta: {len(unicos)}")

DESTINO = AQUI / "dados-destilados"
DESTINO.mkdir(exist_ok=True)
n_valid = max(10, len(unicos) // 10)
for nome, dados in [("train", unicos[n_valid:]), ("valid", unicos[:n_valid])]:
    with (DESTINO / f"{nome}.jsonl").open("w", encoding="utf-8") as f:
        for ex in dados:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  dados-destilados/{nome}.jsonl: {len(dados)}")

# %% [markdown]
# > ⚠️ Registre a taxa de aprovação. Ela é (a) a acurácia efetiva do professor nesses
# > problemas e (b) a medida do viés de dificuldade: os problemas que TODOS os traços
# > erraram saíram do dataset — o aluno nunca os verá (módulo 7, desafio, item c).
#
# ## Etapa 3 — SFT do aluno

# %%
def rodar(*args, mostrar=1200):
    r = subprocess.run([sys.executable, "-m", "mlx_lm", *args], capture_output=True, text=True)
    print(r.stdout[-mostrar:] if r.returncode == 0 else r.stderr[-mostrar:])
    return r.returncode == 0

n_treino = len(unicos) - n_valid
iters = max(150, n_treino * 3 // 4)          # ~3 épocas
print(f"{n_treino} exemplos, {iters} iters (batch 4) = {iters*4/n_treino:.1f} épocas")

rodar("lora", "--model", ALUNO, "--train",
      "--data", str(DESTINO),
      "--adapter-path", str(AQUI / "adapters-destilado"),
      "--iters", str(iters), "--batch-size", "4",
      "--num-layers", "16", "--learning-rate", "1e-4",
      "--mask-prompt", "--max-seq-length", "1024")

# %% [markdown]
# ## Etapa 4 — A avaliação de três vias

# %%
def avaliar(model, tokenizer, n=60, rotulo="", max_tokens=450):
    acertos = 0
    for i in range(n):
        texto = gerar(model, tokenizer, GABARITO_TESTE[i]["question"],
                      max_tokens=max_tokens, temp=0.0)
        if _extrair_numero(texto) == GABARITO_TESTE[i]["answer"]:
            acertos += 1
        if (i + 1) % 20 == 0:
            print(f"  [{rotulo}] {i+1}/{n}: {acertos/(i+1):.0%}")
    return acertos / n

resultados = {}
for nome, modelo_id, adapter in [
    ("aluno original", ALUNO, None),
    ("aluno destilado", ALUNO, "adapters-destilado"),
    ("professor", PROFESSOR, None),
]:
    kwargs = {"adapter_path": str(AQUI / adapter)} if adapter else {}
    m, t = load(modelo_id, **kwargs)
    resultados[nome] = avaliar(m, t, rotulo=nome,
                               max_tokens=800 if "professor" in nome else 450)
    del m
    mx.clear_cache()

print(f"\n{'modelo':<20} {'acurácia GSM8K':>15}")
print("-" * 38)
for nome, acc in resultados.items():
    print(f"{nome:<20} {acc:>15.0%}")

# %% [markdown]
# **As três linhas contam a história da seção 5 do README:**
#
# - `aluno destilado − aluno original` = o que a destilação transferiu;
# - `professor − aluno destilado` = o que a compressão perdeu (o teto herdado);
# - e o custo total foi: horas de geração do professor + minutos de SFT. Compare com o
#   GRPO do módulo 9 (horas de RL) chegando a um lugar parecido por outro caminho.
#
# A frase final dos dois módulos, agora com os seus números: **RL descobre uma vez;
# distillation dissemina barato.**

# %%
