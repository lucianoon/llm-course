# Módulo 1 — Exercícios

Faça sem consultar o `lab.py`. O gabarito está no fim de cada exercício, escondido — abra só depois de tentar.

---

## Parte A — Conceituais

### A1. O custo do português

Você vai processar 50.000 documentos de suporte técnico em português, com média de 1.200 caracteres cada, usando um modelo cujo tokenizer rende ~3,2 caracteres/token em português. O provedor cobra US$ 0,50 por milhão de tokens de entrada.

a) Qual o custo total?
b) Um colega estimou usando a regra "1 token ≈ 0,75 palavras" e chegou a um número menor. Por que a regra falhou?

<details><summary>Gabarito</summary>

a) 1.200 / 3,2 = 375 tokens/documento × 50.000 = 18,75M tokens × US$ 0,50/M = **US$ 9,38**.

b) A regra é derivada de corpora em inglês, onde o tokenizer foi treinado. Em português, palavras acentuadas e morfologia mais rica quebram em mais subpalavras — a razão real fica perto de 1 palavra ≈ 1,5–2 tokens. A estimativa subestima em 30–60%. A única forma correta é rodar o tokenizer do modelo alvo sobre uma amostra real.
</details>

---

### A2. O prompt com espaço

Um sistema em produção monta o prompt assim:

```python
prompt = f"Classifique o sentimento do texto abaixo.\n\nTexto: {texto}\n\nResposta: "
```

A qualidade está pior do que o esperado. Aponte o defeito e corrija.

<details><summary>Gabarito</summary>

O prompt termina em espaço. Em BPE byte-level, o espaço pertence ao **início** do token seguinte: o modelo aprendeu `" Positivo"` como um token único, mas agora recebeu o espaço isolado e precisa emitir `"Positivo"` sem espaço — uma continuação rara no treino, fora da distribuição.

Correção: `"...\n\nResposta:"` — sem espaço final. O modelo emite `" Positivo"` naturalmente.
</details>

---

### A3. Perplexidade e comparação

Você avalia dois modelos no mesmo corpus em português:

- Modelo A (vocabulário 32k): PPL = 14,2
- Modelo B (vocabulário 151k): PPL = 19,8

Um relatório conclui que A é melhor. A conclusão é válida?

<details><summary>Gabarito</summary>

Não. Perplexidade é *por token*, e os dois modelos segmentam o mesmo texto em quantidades diferentes de tokens. B, com vocabulário maior, usa menos tokens para o mesmo texto — cada previsão carrega mais informação e é feita sobre um conjunto de candidatos quase 5× maior, então uma PPL maior pode representar desempenho melhor.

Para comparar de forma justa, normalize pela quantidade de texto e não de tokens: *bits per byte* = `(loss_total_em_nats / ln(2)) / número_de_bytes`. Ou compare em tarefas finais (accuracy em benchmark), que é o que de fato importa.
</details>

---

### A4. Diagnóstico de customização

Para cada situação, diga qual técnica atacar **primeiro** e justifique em uma frase:

1. O assistente não conhece a política de reembolso da empresa, criada mês passado.
2. O assistente conhece a política (está no prompt), mas escreve respostas de 5 parágrafos quando a empresa quer 2 frases — e o prompt já pede isso explicitamente.
3. O assistente responde no formato certo, mas entre duas respostas corretas escolhe consistentemente a mais fria e menos empática.
4. O assistente gera código Python; existe uma suíte de testes que diz se o código passa.
5. Tudo funciona, mas a conta de inferência é de US$ 12.000/mês e precisa cair para US$ 3.000.

<details><summary>Gabarito</summary>

1. **RAG.** Problema de conhecimento, e conhecimento volátil — fine-tuning ficaria desatualizado no mês seguinte e é caro de refazer.
2. **SFT / LoRA.** Problema de comportamento: a informação está disponível e o prompt já instrui; o modelo é que não obedece o formato de forma consistente. Poucas centenas de exemplos bem formatados resolvem.
3. **DPO.** As duas saídas são "corretas"; o que falta é preferência entre alternativas. É exatamente o sinal que preferências capturam e que SFT não captura (SFT só sabe imitar um alvo, não ranquear).
4. **RL (GRPO/PPO).** Existe recompensa verificável e automática — o caso em que RL brilha e não precisa de anotação humana.
5. **Quantização primeiro, depois distillation.** Quantizar para 4-bit é barato e imediato; destilar para um modelo menor dá a economia maior, mas custa um ciclo de treino.
</details>

---

### A5. A loss que não cai

Você inicia um fine-tuning de um modelo com vocabulário de 128.256 tokens. A loss do primeiro batch é 14,7 e depois estaciona em torno de 11.

O que isso indica?

<details><summary>Gabarito</summary>

`ln(128.256) ≈ 11,76` é a loss esperada de um modelo **completamente aleatório**. Um modelo pré-treinado deveria começar bem abaixo disso (tipicamente 1–3 em dados do mesmo domínio).

Começar em 14,7 e estacionar em ~11,8 significa que o modelo está efetivamente prevendo ao acaso. Causas prováveis, em ordem de frequência: shift de labels errado (ou aplicado duas vezes); labels todas mascaradas com `-100`; tokenizer diferente do modelo; pesos não carregados de fato (inicialização aleatória).

O diagnóstico de 30 segundos: imprima a loss do primeiro batch e compare com `ln(vocab_size)`.
</details>

---

## Parte B — Práticas

Resolva no Python, no mesmo ambiente do lab.

### B1. Tokenizer comparado

Escreva uma função `custo_relativo(textos_pt, textos_en, tokenizer)` que devolva a razão média de tokens-por-caractere entre português e inglês. Rode nos tokenizers do `gpt2`, do `Qwen/Qwen2.5-0.5B-Instruct` e de um terceiro à sua escolha (sugestão: `bert-base-multilingual-cased`).

Qual tem o menor custo relativo para o português? Isso significa que é o melhor modelo?

<details><summary>Gabarito</summary>

O Qwen2.5 deve apresentar a menor penalidade (razão perto de 1,1–1,2), o GPT-2 a maior (1,5–2,0).

Não, não significa que é o melhor modelo. Tokenizer eficiente reduz custo e consumo de contexto, o que é real e importante — mas a qualidade do modelo depende do pré-treino, do tamanho e do alinhamento. São eixos independentes.
</details>

---

### B2. Min-p

Implemente `min_p_sampling(logits, min_p=0.05)`: mantenha apenas tokens com `p_i ≥ min_p × p_max`, renormalize e amostre.

Compare, sobre a mesma distribuição, quantos tokens sobrevivem a `top_p=0.9`, `top_k=50` e `min_p=0.05`, em dois contextos: um em que o modelo está confiante e outro em que está incerto.

<details><summary>Gabarito</summary>

```python
def min_p_sampling(logits, min_p=0.05, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    limiar = min_p * probs.max()
    logits = logits.masked_fill(probs < limiar, float("-inf"))
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1))
```

No contexto confiante, min-p mantém pouquíssimos tokens (às vezes 1–3), assim como top-p; top-k mantém 50 independentemente. No contexto incerto, min-p costuma manter mais candidatos que top-p, porque seu critério é relativo ao máximo, não à massa acumulada. É essa propriedade que o torna estável em temperaturas altas.
</details>

---

### B3. Beam search

Implemente beam search com `num_beams=3` para 20 tokens: mantenha os 3 prefixos de maior log-probabilidade acumulada, expandindo cada um a cada passo.

Compare o resultado com greedy no mesmo prompt. A saída do beam search tem log-probabilidade maior? E ela é *melhor*?

<details><summary>Gabarito</summary>

A soma de log-probabilidades do beam search será ≥ à do greedy por construção (greedy é beam search com largura 1).

Mas o texto costuma ser **pior** para geração aberta: mais genérico, mais repetitivo, mais "seguro". É o achado central de Holtzman et al. (2019) — a sequência de máxima verossimilhança não é a mais parecida com texto humano, porque texto humano real tem variação de surpresa que um maximizador elimina. Beam search continua útil onde existe uma resposta correta única e curta: tradução, sumarização extrativa, transcrição.
</details>

---

### B4. Calculadora de VRAM

Escreva `planejar(nome_do_modelo, seq_len, batch)` que baixe apenas o `config.json` do Hub (`transformers.AutoConfig.from_pretrained`) — sem baixar pesos — e devolva as estimativas de inferência bf16, inferência 4-bit, full fine-tune, LoRA, QLoRA e KV cache.

Rode para `Qwen/Qwen2.5-7B-Instruct` e responda: cabe QLoRA numa T4 de 16 GB com `seq_len=2048`?

<details><summary>Gabarito</summary>

```python
from transformers import AutoConfig

def planejar(nome, seq_len=2048, batch=1):
    cfg = AutoConfig.from_pretrained(nome)
    # nº de params: estime ou leia do card; aqui, 7,6e9 para o Qwen2.5-7B
    ...
```

Para o Qwen2.5-7B (28 camadas, 4 KV heads via GQA, head_dim 128): QLoRA fica em torno de 5–6 GB de pesos 4-bit, mais adaptadores e otimizador (~0,5 GB), mais KV cache modesto em 2048 tokens, mais ativações — que dependem do batch e do gradient checkpointing.

Cabe em 16 GB, **desde que** você use gradient checkpointing e batch pequeno (1–2) com acumulação de gradiente. Sem checkpointing, as ativações estouram. Esse é exatamente o cálculo que você vai refazer no módulo 6, agora com o número real na mão.
</details>

---

### B5. O template errado, medido

Pegue 10 perguntas factuais curtas em português. Gere a resposta do `Qwen2.5-0.5B-Instruct` de três formas:

1. sem chat template (prompt cru);
2. com template e `add_generation_prompt=True`;
3. com o template **do GPT-2** (isto é, nenhum) aplicado a um modelo que espera ChatML.

Meça o comprimento médio das respostas e leia qualitativamente. Quantifique o estrago.

<details><summary>Gabarito</summary>

Sem template, o modelo tende a continuar o texto em vez de responder — pode gerar mais perguntas, listas ou repetir o enunciado. As respostas costumam ser mais longas e desfocadas.

Com template, respostas mais curtas, diretas e no papel de assistente.

A lição de engenharia: essa diferença é grande e **silenciosa**. Nenhuma exceção é lançada. Em um pipeline de fine-tuning, o mesmo erro (treinar com um formato, servir com outro) desperdiça o treino inteiro sem nenhum sinal de alerta além da qualidade final ruim.
</details>

---

## Desafio

Implemente a **avaliação de escolha múltipla por log-likelihood**, que é como benchmarks tipo MMLU/ENEM realmente funcionam.

Dada uma pergunta e N alternativas, o método correto **não é** pedir ao modelo que escreva "A", "B" ou "C". É calcular, para cada alternativa, a log-probabilidade da alternativa condicionada à pergunta, normalizada pelo número de tokens da alternativa, e escolher a maior.

```python
def escolher(pergunta, alternativas, modelo, tk):
    ...  # devolve o índice da alternativa de maior log-prob normalizada
```

Perguntas a responder depois de implementar:

1. Por que normalizar pelo número de tokens? O que acontece se você não normalizar?
2. Por que esse método é mais confiável do que pedir a letra?

<details><summary>Gabarito</summary>

1. Sem normalizar, alternativas mais **longas** são sistematicamente penalizadas — cada token adicional soma um log-prob negativo. O ranking passaria a medir comprimento, não plausibilidade. Normalizar pela contagem de tokens dá a log-prob média por token. (Variantes mais sofisticadas normalizam pela probabilidade da alternativa *sozinha*, sem a pergunta, para descontar o viés de frequência da própria frase.)

2. Porque separa a **capacidade** do modelo do seu **domínio de formato**. Um modelo base não foi treinado para emitir "A" como resposta e pode falhar em produzir a letra mesmo sabendo o conteúdo — você mediria formatação, não conhecimento. O método por log-likelihood funciona igualmente em modelos base e instruct, o que é indispensável quando você quer medir se o seu fine-tuning melhorou o modelo ou apenas ensinou-o a formatar.
</details>
