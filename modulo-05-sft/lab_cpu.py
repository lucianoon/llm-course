# %% [markdown]
# # Módulo 5 — Laboratório: o ciclo de SFT em CPU
#
# **Roda em CPU, em menos de um minuto, sem baixar modelo.** O `lab.py` (Mac) e o
# `lab_cuda.py` (NVIDIA) repetem este ciclo num modelo real. Este arquivo fecha o
# módulo para quem não tem acelerador.
#
# > **Aviso de escala.** O "modelo" é um MiniGPT minúsculo, com vocabulário fechado de
# > brinquedo. O que transfere é o ciclo: baseline → máscara → métrica de formato →
# > esquecimento. O que não transfere é qualquer número desta tabela para um modelo de
# > produção.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Pré-treino num mundo fechado: o modelo sabe os fatos e não sabe o formato |
# | 2 | SFT com máscara e sem máscara, na mesma métrica |
# | 3 | Esquecimento, a mistura que o segura, e o domínio estreito demais |
#
# Seed fixa: `0`. Se uma asserção falhar, o efeito didático não apareceu — rode de
# novo antes de interpretar a tabela.

# %%
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(AQUI.parent / "tools"))
import minigpt

SEED = 0
BLOCO = 24
PASSOS_PRE = 400
PASSOS_SFT = 80
PARES = (
    ("qual a capital da franca", "paris"),
    ("qual a capital do brasil", "brasilia"),
    ("qual a capital do japao", "toquio"),
    ("qual a capital de portugal", "lisboa"),
    ("qual o rio do egito", "nilo"),
    ("qual o rio do brasil", "amazonas"),
    ("qual a cor do ceu", "azul"),
    ("qual a cor da neve", "branca"),
)
PROSA = tuple(f"a resposta de {pergunta} e {resposta}" for pergunta, resposta in PARES)

# %% [markdown]
# ## Lab 1 — Um mundo pequeno o bastante para caber na CPU
#
# O pré-treino vê só prosa: "a resposta de qual a capital da franca e paris".
# O SFT vê outro formato: "usuario ... assistente paris fim".
# A capacidade já está nos pesos; o formato ainda não.

# %%
palavras = ["<pad>", "usuario", "assistente", "fim"]
for pergunta, resposta in PARES:
    palavras.extend(pergunta.split())
    palavras.extend(resposta.split())
for linha in PROSA:
    palavras.extend(linha.split())

vocabulario = []
for palavra in palavras:
    if palavra not in vocabulario:
        vocabulario.append(palavra)
stoi = {palavra: indice for indice, palavra in enumerate(vocabulario)}
itos = {indice: palavra for palavra, indice in stoi.items()}
EOS = stoi["fim"]
PAD = stoi["<pad>"]


def codificar(texto: str) -> list[int]:
    return [stoi[palavra] for palavra in texto.split()]


def exemplo(pergunta: str, resposta: str) -> tuple[list[int], list[int]]:
    prompt = ["usuario", *pergunta.split(), "assistente"]
    completo = prompt + resposta.split() + ["fim"]
    return codificar(" ".join(prompt)), codificar(" ".join(completo))


def lote(pares, mascarar_prompt: bool) -> tuple[torch.Tensor, torch.Tensor]:
    entradas, alvos = [], []
    for pergunta, resposta in pares:
        prompt, completo = exemplo(pergunta, resposta)
        faltam = BLOCO - (len(completo) - 1)
        entrada = completo[:-1] + [PAD] * faltam
        alvo = completo[1:] + [PAD] * faltam
        alvo = [-100 if token == PAD else token for token in alvo]
        if mascarar_prompt:
            # y[i] prevê o token i+1. Alvos ainda dentro do prompt ficam de fora.
            corte = len(prompt) - 1
            alvo = [-100 if indice < corte else token for indice, token in enumerate(alvo)]
        entradas.append(entrada)
        alvos.append(alvo)
    return torch.tensor(entradas), torch.tensor(alvos)


ids_prosa = []
for linha in PROSA:
    codigos = codificar(linha) + [EOS]
    for _ in range(40):
        ids_prosa.extend(codigos)
fonte = torch.tensor(ids_prosa)
cfg = minigpt.Config(
    vocab=len(vocabulario),
    d=64,
    n_camadas=2,
    n_heads=4,
    bloco=BLOCO,
    d_ff=128,
)

torch.manual_seed(SEED)
modelo = minigpt.MiniGPT(cfg)
otimizador = torch.optim.AdamW(minigpt.grupos_de_parametros(modelo), lr=3e-3, betas=(0.9, 0.95))
for _ in range(PASSOS_PRE):
    otimizador.zero_grad(set_to_none=True)
    lote_x, lote_y = minigpt.pegar_batch(fonte, 32, BLOCO)
    _, perda = modelo(lote_x, lote_y)
    perda.backward()
    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
    otimizador.step()

print(f"vocabulario: {len(vocabulario)} tokens | prosa: {len(fonte):,} tokens")
print(f"pre-treino: {PASSOS_PRE} passos | loss final {perda.item():.4f}")
base_state = copy.deepcopy(modelo.state_dict())

# %% [markdown]
# ## Lab 2 — A métrica, antes de comemorar a loss
#
# Acerto exige a resposta certa **e** o token `fim`. Loss no prompt mede se o modelo
# gastou treino copiando a pergunta. Loss na prosa mede o que o pré-treino ainda sabe.

# %%
@torch.no_grad()
def completar(modelo_atual, pergunta: str) -> list[str]:
    prompt, _ = exemplo(pergunta, "paris")
    idx = torch.tensor([prompt])
    gerado = []
    for _ in range(6):
        logits, _ = modelo_atual(idx[:, -BLOCO:])
        proximo = int(logits[0, -1].argmax())
        gerado.append(itos[proximo])
        idx = torch.cat([idx, torch.tensor([[proximo]])], dim=1)
        if proximo == EOS:
            break
    return gerado


@torch.no_grad()
def medir(modelo_atual, pares=PARES) -> dict:
    estava = modelo_atual.training
    modelo_atual.eval()
    previsoes = [(pergunta, resposta, completar(modelo_atual, pergunta)) for pergunta, resposta in pares]
    acertos = [
        gerado == resposta.split() + ["fim"]
        for _, resposta, gerado in previsoes
    ]
    entradas, _ = lote(pares, mascarar_prompt=False)
    logits, _ = modelo_atual(entradas)
    perda_prompt, perda_resposta = [], []
    for indice, (pergunta, resposta) in enumerate(pares):
        prompt, completo = exemplo(pergunta, resposta)
        corte = len(prompt) - 1
        for passo, alvo in enumerate(completo[1:]):
            valor = F.cross_entropy(logits[indice, passo : passo + 1], torch.tensor([alvo])).item()
            (perda_prompt if passo < corte else perda_resposta).append(valor)
    prosa = minigpt.avaliar(modelo_atual, fonte, n=8, batch=8, seed=1)
    modelo_atual.train(estava)
    return {
        "acerto": sum(acertos) / len(acertos),
        "prompt": sum(perda_prompt) / len(perda_prompt),
        "resposta": sum(perda_resposta) / len(perda_resposta),
        "prosa": prosa,
        "previsoes": previsoes,
    }


def sft(modelo_atual, pares, passos, mascarar, lr=3e-3, mistura=0.0, seed=SEED) -> None:
    otim = torch.optim.AdamW(minigpt.grupos_de_parametros(modelo_atual), lr=lr, betas=(0.9, 0.95))
    entradas, alvos = lote(tuple(pares) * 8, mascarar_prompt=mascarar)
    gerador = torch.Generator().manual_seed(seed)
    for _ in range(passos):
        otim.zero_grad(set_to_none=True)
        if mistura and torch.rand((), generator=gerador).item() < mistura:
            lote_x, lote_y = minigpt.pegar_batch(fonte, 16, BLOCO, generator=gerador)
            _, perda_passo = modelo_atual(lote_x, lote_y)
        else:
            indice = torch.randint(len(entradas), (16,), generator=gerador)
            logits, _ = modelo_atual(entradas[indice])
            perda_passo = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                alvos[indice].reshape(-1),
                ignore_index=-100,
            )
        perda_passo.backward()
        torch.nn.utils.clip_grad_norm_(modelo_atual.parameters(), 1.0)
        otim.step()


def restaurar() -> None:
    modelo.load_state_dict(base_state)


medidas = {}
medidas["base"] = medir(modelo)

# %% [markdown]
# ## Lab 3 — Máscara, mistura e domínio estreito
#
# As quatro continuações partem do **mesmo** checkpoint. A única diferença é o dado
# e o que entra na loss.

# %%
restaurar()
sft(modelo, PARES, PASSOS_SFT, mascarar=True, seed=SEED + 1)
medidas["mascara"] = medir(modelo)

restaurar()
sft(modelo, PARES, PASSOS_SFT, mascarar=False, seed=SEED + 2)
medidas["sem_mascara"] = medir(modelo)

restaurar()
sft(modelo, PARES, PASSOS_SFT, mascarar=True, mistura=0.5, seed=SEED + 3)
medidas["mistura"] = medir(modelo)

restaurar()
sft(modelo, PARES[:1], passos=160, mascarar=True, lr=1e-2, seed=SEED + 4)
medidas["estreito"] = medir(modelo)

print()
print(f"{'condicao':<16} {'acerto':>8} {'prompt':>8} {'resposta':>8} {'prosa':>8}")
for nome in ("base", "mascara", "sem_mascara", "mistura", "estreito"):
    item = medidas[nome]
    print(
        f"{nome:<16} {item['acerto']:8.0%} {item['prompt']:8.3f} "
        f"{item['resposta']:8.3f} {item['prosa']:8.3f}"
    )

print("\nbase, primeiras falhas de formato:")
falhas = [item for item in medidas["base"]["previsoes"] if item[2] != item[1].split() + ["fim"]]
for pergunta, resposta, gerado in falhas[:3]:
    print(f"  {pergunta} -> {' '.join(gerado)}  (esperado {resposta} fim)")

print("\ndominio estreito (so 'franca -> paris'), todas as perguntas:")
for pergunta, _, gerado in medidas["estreito"]["previsoes"]:
    print(f"  {pergunta} -> {' '.join(gerado)}")

# %% [markdown]
# ## Leitura obrigatória
#
# - A base acerta pouco o formato: ela continua a prosa em que foi treinada.
# - Com máscara, o acerto do formato sobe e a loss do prompt continua alta: o modelo
#   não gastou gradiente copiando a pergunta.
# - Sem máscara, o formato também sobe, e a loss do prompt desaba. A métrica de
#   formato, sozinha, não vê esse desperdício.
# - A mistura com prosa segura a loss do pré-treino. O SFT só no formato esquece.
# - Um único par, repetido, faz as outras perguntas desabarem na mesma resposta.

# %%
base = medidas["base"]
mascara = medidas["mascara"]
sem_mascara = medidas["sem_mascara"]
mistura = medidas["mistura"]
estreito = medidas["estreito"]
colapso = sum(gerado == ["paris", "fim"] for _, _, gerado in estreito["previsoes"])

assert base["acerto"] <= 0.5, "a base já seguia o formato; o pré-treino vazou o SFT"
assert base["prosa"] < 0.3, "o pré-treino não aprendeu a prosa"
assert mascara["acerto"] >= 0.875, "a máscara não ensinou o formato"
assert mascara["resposta"] < 0.05, "a loss da resposta não caiu"
assert mascara["prompt"] > 2.0, "a loss do prompt caiu mesmo com máscara"
assert sem_mascara["acerto"] >= 0.875, "sem máscara o formato não foi aprendido"
assert sem_mascara["prompt"] < 1.0, "sem máscara a loss do prompt deveria cair"
assert sem_mascara["prompt"] < mascara["prompt"], "a máscara não protegeu o prompt"
assert mistura["acerto"] >= 0.875, "a mistura perdeu o formato"
assert mistura["prosa"] < 0.3, "a mistura não segurou a prosa"
assert mistura["prosa"] < mascara["prosa"], "a mistura não reduziu o esquecimento"
assert estreito["previsoes"][0][2] == ["paris", "fim"], "o par estreito não foi memorizado"
assert colapso >= 5, "o domínio estreito não colapsou as outras respostas"
assert estreito["prosa"] > 5.0, "o domínio estreito não apagou a prosa"

print("\nOK: o ciclo baseline → máscara → métrica → esquecimento apareceu na CPU.")
