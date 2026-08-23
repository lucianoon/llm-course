# Fase 0 — Comece aqui se você nunca programou

Esta fase não presume Python, matemática de ensino superior, terminal ou machine learning.
Ela ensina apenas o necessário para você entender e modificar os laboratórios do curso.

> A meta não é decorar Python. É aprender a transformar um problema em dados, passos,
> verificações e uma explicação que outra pessoa consiga reproduzir.

## O que significa sair do zero

Ao concluir esta fase, você conseguirá:

- abrir um terminal, entrar em uma pasta e executar um programa;
- ler variáveis, listas, dicionários, condições, laços e funções em Python;
- interpretar uma mensagem de erro sem entrar em pânico;
- usar `assert` e testes pequenos para verificar comportamento;
- entender vetor, produto escalar, matriz, tensor e gradiente pela operação que realizam;
- ler e escrever dados JSONL;
- explicar o que é um modelo que prevê o próximo item;
- usar Git para registrar uma mudança pequena e intencional.

Isso ainda não torna alguém profissional. Torna possível iniciar a prática que forma um
profissional. O restante do caminho está em [`TRILHA-ESSENCIAL.md`](../TRILHA-ESSENCIAL.md).

---

## Os sete princípios fundamentais

### 1. O computador é literal

Um computador não entende intenção. Ele executa instruções específicas sobre valores
específicos. Quando o programa faz algo inesperado, existe uma diferença concreta entre
o que foi pedido e o que imaginávamos ter pedido.

### 2. Todo programa transforma dados

Entrada → transformação → saída. Um chatbot parece sofisticado, mas continua seguindo
essa estrutura: recebe tokens, transforma números em várias camadas e devolve números que
viram texto.

### 3. Tipos são promessas

`42`, `"42"` e `[42]` parecem relacionados, mas são valores de tipos diferentes. Saber o
tipo de um valor responde quais operações fazem sentido e quais erros esperar.

### 4. Funções escondem detalhes atrás de um contrato

Uma boa função recebe poucos valores, faz uma coisa compreensível e devolve um resultado.
Você não precisa lembrar sua implementação para usá-la — apenas sua promessa.

### 5. Medir vence adivinhar

Antes de mudar algo, defina o resultado esperado. Depois execute e compare. Esse ciclo é
o coração de testes, ciência e engenharia de machine learning.

### 6. Modelos aprendem ajustando números

Um modelo possui parâmetros numéricos. A *loss* mede o erro; o gradiente indica como uma
pequena alteração em cada parâmetro mudaria esse erro; o otimizador aplica a alteração.

### 7. Trabalho profissional é reproduzível

Não basta funcionar uma vez no seu computador. Outra pessoa precisa conseguir instalar,
executar, testar e compreender a decisão. Git, ambientes, testes e documentação fazem
parte do produto.

---

## Roteiro de estudo

| Etapa | Pergunta que você precisa responder | Evidência de aprendizado |
|---|---|---|
| Terminal | Onde estou e qual arquivo estou executando? | Executa o lab pela raiz |
| Python | Como valores viram decisões e repetições? | Resolve exercícios sem copiar |
| Funções | Qual é a entrada, a saída e o erro possível? | Escreve função com teste |
| Dados | Como representar registros reais? | Lê e grava JSONL |
| Matemática | O que vetor e matriz calculam? | Implementa produto escalar |
| Aprendizado | Como um parâmetro reduz um erro? | Interpreta um gradiente |
| Modelos | Como prever o próximo item? | Constrói baseline por contagem |
| Engenharia | Outra pessoa reproduz o resultado? | Commit pequeno e README claro |

## Primeiros comandos

Abra o terminal dentro da pasta do repositório. Os comandos abaixo funcionam no macOS,
Linux e PowerShell:

```bash
pwd                              # mostra a pasta atual
ls                               # lista os arquivos
cd 00-iniciante-zero             # entra na pasta desta fase
python lab.py                    # executa o laboratório inteiro
cd ..                            # volta para a raiz do curso
python tools/build_notebooks.py  # gera os notebooks
```

Não digite o comentário depois de `#`; ele apenas explica o comando. Se `python` não for
encontrado no Mac ou Linux, tente `python3`. Se uma pasta não for encontrada, use `pwd` e
`ls` para descobrir onde está antes de repetir o comando.

Depois de alterar um exercício, use Git para enxergar e registrar somente aquela mudança:

```bash
git status
git diff
git add 00-iniciante-zero/
git diff --staged
git commit -m "feat: conclui projeto da fase zero"
```

`status` mostra o estado, `diff` mostra as linhas alteradas, `add` escolhe o que entrará no
commit e `commit` cria um ponto recuperável no histórico. Leia o `diff --staged` antes de
confirmar: trabalho profissional começa sabendo exatamente o que está sendo entregue.

## Como estudar esta fase

1. Leia uma seção deste README.
2. Abra `lab.py` ou `lab.ipynb`.
3. Antes de cada célula, escreva o que acredita que acontecerá.
4. Execute e compare com sua previsão.
5. Faça `exercicios.md` sem consultar o gabarito.
6. Anote cada erro no diário de erros.

Não avance por velocidade. Avance quando conseguir explicar o código em voz alta.

## Vocabulário mínimo

| Termo | Significado operacional |
|---|---|
| variável | nome que aponta para um valor |
| tipo | conjunto de operações válidas para um valor |
| função | transformação reutilizável com entrada e saída |
| exceção | sinal de que o programa não pode cumprir o contrato |
| teste | exemplo executável do comportamento esperado |
| vetor | lista ordenada de números |
| matriz | tabela de números que transforma vetores |
| tensor | generalização de números, vetores e matrizes |
| parâmetro | número ajustável de um modelo |
| loss | medida numérica do erro |
| gradiente | direção local de mudança da loss |
| token | unidade numérica de texto usada pelo modelo |

## Checklist de saída

Sem consultar o material, você consegue:

- [ ] explicar entrada → transformação → saída com um exemplo;
- [ ] diferenciar string, número, lista e dicionário;
- [ ] escrever e chamar uma função;
- [ ] interpretar a última linha de um traceback;
- [ ] criar ao menos um teste com `assert`;
- [ ] calcular um produto escalar pequeno à mão;
- [ ] explicar loss e gradiente sem fórmulas;
- [ ] descrever por que separamos treino e teste;
- [ ] executar `git status`, `git diff` e `git commit` sabendo o que cada um faz.

Se marcou menos de 7 itens, repita o lab e os exercícios antes do módulo 1.
