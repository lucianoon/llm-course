# %% [markdown]
# # Fase 0 — Laboratório para quem começa do zero
#
# Este laboratório não baixa modelos. Ele começa com Python básico e termina com um
# pequeno preditor de próxima palavra. Execute uma célula por vez e preveja a saída.

# %%
print("Olá, curso de LLMs!")
print("Um programa é uma sequência de instruções executáveis.")

# %% [markdown]
# ## 1. Valores, variáveis e tipos
#
# Uma variável é um nome para um valor. O tipo informa quais operações são válidas.

# %%
nome = "Ada"
idade = 10
estudando = True

print(nome, idade, estudando)
print(type(nome).__name__, type(idade).__name__, type(estudando).__name__)

# %%
proxima_idade = idade + 1
mensagem = nome + " terá " + str(proxima_idade) + " anos."
print(mensagem)

# %% [markdown]
# ## 2. Coleções: vários valores juntos

# %%
notas = [7.0, 8.5, 9.0]
aluno = {"nome": nome, "notas": notas, "ativo": True}

print("primeira nota:", notas[0])
print("nome no dicionário:", aluno["nome"])
print("quantidade de notas:", len(aluno["notas"]))

# %% [markdown]
# ## 3. Decisões e repetições

# %%
media = sum(notas) / len(notas)
if media >= 7:
    situacao = "aprovado"
else:
    situacao = "estudar novamente"

print(f"média={media:.1f} → {situacao}")

# %%
for indice, nota in enumerate(notas, start=1):
    print(f"avaliação {indice}: {nota}")

# %% [markdown]
# ## 4. Funções são contratos
#
# A função abaixo promete receber números e devolver a média. Ela rejeita uma lista vazia
# porque não existe média sem observações.

# %%
def calcular_media(valores: list[float]) -> float:
    """Calcula a média de uma lista não vazia."""
    if not valores:
        raise ValueError("é necessário informar ao menos um valor")
    return sum(valores) / len(valores)


print(calcular_media([10, 8, 9]))

# %% [markdown]
# ## 5. Testes transformam expectativas em código

# %%
assert calcular_media([10, 8, 9]) == 9
assert calcular_media([5]) == 5

try:
    calcular_media([])
except ValueError as erro:
    print("erro esperado:", erro)

print("testes da média passaram")

# %% [markdown]
# Quando um teste falha, leia o traceback de baixo para cima: a última linha diz o tipo do
# erro; as linhas anteriores mostram o caminho percorrido até ele.
#
# ## 6. Registros e JSONL
#
# JSONL guarda um objeto JSON por linha. É comum em datasets porque podemos processar um
# registro de cada vez sem carregar tudo na memória.

# %%
import json
import tempfile
from pathlib import Path

registros = [
    {"pergunta": "Quanto é 2 + 2?", "resposta": "4"},
    {"pergunta": "Capital da França?", "resposta": "Paris"},
]

with tempfile.TemporaryDirectory() as pasta_temporaria:
    caminho = Path(pasta_temporaria) / "exemplos.jsonl"
    with caminho.open("w", encoding="utf-8") as arquivo:
        for registro in registros:
            arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")

    carregados = [json.loads(linha) for linha in caminho.read_text(encoding="utf-8").splitlines()]

assert carregados == registros
print("registros recuperados:", carregados)

# %% [markdown]
# ## 7. Vetores e produto escalar
#
# Um vetor é uma lista ordenada de números. O produto escalar multiplica posições
# correspondentes e soma os resultados. Redes neurais repetem essa operação muitas vezes.

# %%
def produto_escalar(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("os vetores precisam ter o mesmo tamanho")
    return sum(x * y for x, y in zip(a, b))


entrada = [2.0, 3.0]
pesos = [0.5, -1.0]
saida = produto_escalar(entrada, pesos)
print("2×0,5 + 3×(-1) =", saida)
assert saida == -2.0

# %% [markdown]
# ## 8. Tensores fazem a mesma ideia em escala

# %%
import torch

x = torch.tensor([2.0, 3.0])
w = torch.tensor([0.5, -1.0])
print("produto escalar no PyTorch:", torch.dot(x, w).item())

# %% [markdown]
# ## 9. Loss e gradiente
#
# Queremos que `peso × entrada` chegue ao alvo. A loss mede a distância. O gradiente mostra
# como uma pequena mudança no peso altera a loss.

# %%
peso = torch.tensor(1.0, requires_grad=True)
entrada = torch.tensor(3.0)
alvo = torch.tensor(12.0)

previsao = peso * entrada
loss = (previsao - alvo) ** 2
loss.backward()

print("previsão:", previsao.item())
print("loss:", loss.item())
print("gradiente do peso:", peso.grad.item())

# %%
with torch.no_grad():
    peso -= 0.1 * peso.grad

nova_loss = (peso * entrada - alvo) ** 2
print("peso depois de um passo:", peso.item())
print("loss depois de um passo:", nova_loss.item())
assert nova_loss < loss

# %% [markdown]
# ## 10. Um modelo mínimo de próxima palavra
#
# Este modelo não tem rede neural. Ele conta qual palavra apareceu depois de cada palavra
# no treino e escolhe a mais frequente. Mesmo simples, já possui treino, parâmetros,
# inferência e avaliação — o mesmo esqueleto de sistemas maiores.

# %%
from collections import Counter, defaultdict
from itertools import pairwise


def treinar_contagens(frases: list[str]) -> dict[str, Counter]:
    contagens = defaultdict(Counter)
    for frase in frases:
        palavras = frase.lower().split()
        for atual, seguinte in pairwise(palavras):
            contagens[atual][seguinte] += 1
    return dict(contagens)


def prever_proxima(contagens: dict[str, Counter], palavra: str) -> str | None:
    candidatas = contagens.get(palavra.lower())
    return candidatas.most_common(1)[0][0] if candidatas else None


treino = ["o gato dorme", "o gato come", "o cachorro corre", "o gato dorme"]
modelo = treinar_contagens(treino)
print("depois de 'gato':", modelo["gato"])
print("previsão:", prever_proxima(modelo, "gato"))
assert prever_proxima(modelo, "gato") == "dorme"

# %% [markdown]
# ## 11. Treino não é teste
#
# Memorizar exemplos de treino não prova que o modelo funciona em exemplos novos. Sempre
# reserve dados que não participaram do ajuste.

# %%
teste = [("gato", "dorme"), ("cachorro", "corre"), ("peixe", "nada")]
acertos = sum(prever_proxima(modelo, palavra) == esperado for palavra, esperado in teste)
print(f"acurácia: {acertos}/{len(teste)} = {acertos / len(teste):.0%}")

# %% [markdown]
# O erro em `peixe` não é um defeito de Python: o dado nunca apareceu no treino. Antes de
# trocar o algoritmo, pergunte se o sistema recebeu a informação necessária.
#
# ## Checklist final
#
# Explique, sem consultar:
#
# 1. Qual a diferença entre valor, variável e tipo?
# 2. O que uma função promete?
# 3. Por que um teste com `assert` é útil?
# 4. O que produto escalar, loss e gradiente fazem?
# 5. Por que medir no treino não basta?
