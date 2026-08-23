# Fase 0 — Exercícios

Faça sem consultar o `lab.py`. Escreva primeiro sua previsão; só depois abra o gabarito.

## Parte A — Entendimento

### A1. Tipos diferentes

Explique por que `"10" + "5"` produz resultado diferente de `10 + 5`. Como converter a
primeira expressão para obter o resultado numérico esperado?

<details><summary>Gabarito</summary>

Strings são sequências de caracteres, então `+` concatena e produz `"105"`. Inteiros usam
`+` como soma e produzem `15`. A conversão é `int("10") + int("5")`.
</details>

### A2. Contrato de função

Para a função `dividir(total, partes)`, escreva:

1. entradas esperadas;
2. saída;
3. um caso de erro;
4. um teste de sucesso.

<details><summary>Gabarito</summary>

Entradas numéricas, com `partes != 0`; saída `total / partes`; divisão por zero deve gerar
erro; um teste possível é `assert dividir(10, 2) == 5`.
</details>

### A3. Leia o erro

O que há de errado?

```python
idades = [10, 11, 12]
print(idades[3])
```

<details><summary>Gabarito</summary>

Índices começam em zero. A lista possui posições `0`, `1` e `2`; acessar `3` gera
`IndexError`. O último item pode ser acessado com `idades[2]` ou `idades[-1]`.
</details>

### A4. Treino e teste

Um modelo acerta 100% dos exemplos usados para treiná-lo. Podemos afirmar que ele funciona?

<details><summary>Gabarito</summary>

Não. Ele pode apenas ter memorizado. É necessário medir em exemplos não usados no ajuste.
Esse conjunto de teste representa situações novas e estima generalização.
</details>

## Parte B — Código

### B1. Estatísticas pequenas

Implemente uma função que receba uma lista não vazia e devolva um dicionário com mínimo,
máximo e média. Escreva dois testes.

<details><summary>Uma solução</summary>

```python
def resumir(valores):
    if not valores:
        raise ValueError("lista vazia")
    return {
        "minimo": min(valores),
        "maximo": max(valores),
        "media": sum(valores) / len(valores),
    }

assert resumir([1, 2, 3]) == {"minimo": 1, "maximo": 3, "media": 2}
assert resumir([5]) == {"minimo": 5, "maximo": 5, "media": 5}
```
</details>

### B2. Produto escalar

Implemente o produto escalar sem usar PyTorch. Rejeite vetores de tamanhos diferentes.

<details><summary>Uma solução</summary>

```python
def produto_escalar(a, b):
    if len(a) != len(b):
        raise ValueError("tamanhos diferentes")
    return sum(x * y for x, y in zip(a, b))

assert produto_escalar([1, 2], [3, 4]) == 11
```
</details>

### B3. Frequências

Dada uma lista de palavras, devolva as três mais frequentes e suas contagens.

<details><summary>Uma solução</summary>

```python
from collections import Counter

def tres_mais_frequentes(palavras):
    return Counter(palavras).most_common(3)

assert tres_mais_frequentes(["a", "b", "a", "c", "a", "b"])[0] == ("a", 3)
```
</details>

### B4. Avaliação honesta

Modifique o preditor do lab para calcular acurácia em treino e teste separadamente. Explique
por que as duas medidas diferem.

<details><summary>Gabarito esperado</summary>

A acurácia de treino tende a ser maior porque as contagens foram construídas com aqueles
exemplos. No teste aparecem transições novas ou menos frequentes. A diferença entre as duas
é um primeiro sinal de memorização e falta de generalização.
</details>

## Projeto de saída — classificador por regras

Construa um programa que:

1. leia uma lista de mensagens;
2. classifique cada uma como `urgente` ou `normal` usando regras explícitas;
3. compare a previsão com um gabarito;
4. mostre a acurácia e os erros;
5. tenha pelo menos três testes;
6. explique no README quando as regras falham.

O objetivo não é obter inteligência artificial. É praticar o ciclo profissional completo:
contrato → implementação → teste → medição → explicação.
