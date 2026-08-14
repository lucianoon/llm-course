# Módulo 2 — Exercícios

Faça sem consultar o `lab.py`. Gabaritos escondidos — abra depois de tentar.

---

## Parte A — Conceituais

### A1. Por que `k` e `v` são separados

Suponha uma arquitetura simplificada em que `k = v` (o mesmo vetor serve de chave e de valor). Que capacidade concreta se perde? Dê um exemplo linguístico.

<details><summary>Gabarito</summary>

Perde-se a capacidade de **buscar por um critério e transportar outra informação**.

Com `k = v`, o único jeito de um token ser atendido é ter conteúdo semelhante ao que a query procura, e o que ele entrega é necessariamente esse mesmo conteúdo.

Exemplo: em *"O gato preto subiu no telhado porque **ele** estava com fome"*, a query de `ele` procura por "substantivo masculino singular, sujeito plausível" — um critério **sintático**. Mas o que precisa ser transportado é a informação **semântica** de `gato` (é um animal, tem fome, é preto). Com `k = v`, o token `gato` teria de ser simultaneamente marcado por sua sintaxe e entregar sua semântica pelo mesmo vetor, competindo pelas mesmas dimensões.

A separação permite que `W_K` aprenda um espaço de *endereçamento* e `W_V` um espaço de *conteúdo*, independentes.
</details>

---

### A2. `√d_k` como temperatura

Mostre que dividir os logits de atenção por `√d_k` é o mesmo mecanismo da temperatura do módulo 1. Se você aumentasse a divisão para `d_k` em vez de `√d_k`, o que aconteceria com o treino?

<details><summary>Gabarito</summary>

`softmax(s/√d_k)` é literalmente `softmax(s/T)` com `T = √d_k`. É uma temperatura fixa, escolhida para que a variância dos logits seja 1 independentemente da dimensão.

Dividindo por `d_k` (temperatura muito maior), a distribuição de atenção ficaria quase **uniforme**: cada posição atenderia a todas as outras igualmente, e a saída seria aproximadamente a média dos values. O modelo perderia a capacidade de selecionar. O gradiente existiria, mas apontaria para um espaço de soluções muito plano — o treino seria lento e o modelo, fraco.

O ponto ótimo é onde a distribuição é *seletiva mas não saturada*, e `√d_k` é a normalização que garante isso por construção estatística, não por ajuste empírico.
</details>

---

### A3. Padding side

Você escreve um script de inferência em batch. Tokeniza 8 prompts de comprimentos diferentes com `padding=True` e chama `generate()`. Metade das saídas vem vazia ou sem sentido. O que aconteceu?

<details><summary>Gabarito</summary>

O padding foi para a **direita** (default do tokenizer). Em geração, o próximo token sai da representação da **última posição** da sequência — que, com padding à direita, é um token `<pad>`.

O modelo continua a partir do padding em vez do fim do prompt real. As posições do RoPE também ficam erradas: o texto real ocupa posições 0..n, mas a geração continua da posição `max_len`, criando um salto.

Correção:
```python
tokenizer.padding_side = "left"
```
antes de tokenizar para geração. Para **treino**, mantenha `"right"` — lá a loss é calculada em todas as posições e o padding é mascarado com `-100`, então não há problema.

Regra: `padding_side="left"` para gerar, `"right"` para treinar.
</details>

---

### A4. Esticando o contexto

O Qwen2.5 usa `base = 1.000.000` e `head_dim = 64`. Você quer estender a janela de 32k para 128k (fator `s = 4`) usando NTK-aware scaling, cuja fórmula é:

```
base' = base · s^(d/(d−2))
```

a) Calcule `base'`.
b) Por que NTK-aware é preferível à interpolação linear pura de posições?
c) Por que nada disso seria possível com posições absolutas aprendidas (GPT-2)?

<details><summary>Gabarito</summary>

a) `base' = 10⁶ × 4^(64/62) = 10⁶ × 4^1,0323 ≈ 10⁶ × 4,19 ≈ **4,19 × 10⁶**`.

b) A interpolação linear comprime **todas** as frequências igualmente, inclusive as altas — que são justamente as que distinguem tokens vizinhos. O modelo perde resolução local: passa a confundir a posição `i` com `i+1`. O NTK-aware distribui a compressão de forma não uniforme, deixando as altas frequências praticamente intactas e comprimindo as baixas, que só precisam distinguir regiões distantes. YaRN refina isso ainda mais, com faixas de tratamento distintas por frequência.

c) Com posições absolutas aprendidas, existe um vetor treinado por posição. As posições 2049 em diante **não existem** — não há vetor, e criar um novo por interpolação não funciona porque esses vetores não têm estrutura contínua aprendida. Em RoPE a posição é uma função analítica (rotação), definida para qualquer `m` real, e a relação entre posições é uma propriedade da fórmula, não algo memorizado. É essa continuidade que permite reescalar.
</details>

---

### A5. Onde colocar o LoRA

Dado que 87,7% dos parâmetros de um bloco estão no MLP e apenas 12,3% na atenção, um colega propõe aplicar LoRA **apenas** ao MLP, "onde está o modelo de verdade". Avalie.

<details><summary>Gabarito</summary>

A proposta não é absurda e reflete uma tendência real — o paper do QLoRA recomenda aplicar a **todas** as camadas lineares, e vários trabalhos mostram que incluir o MLP melhora resultados em tarefas que exigem conhecimento novo.

Mas o raciocínio "onde está o modelo" está incompleto por duas razões:

1. **O que se quer mudar importa mais que onde estão os parâmetros.** Se o objetivo é mudar *estilo, formato ou comportamento de seguir instruções*, as projeções de atenção são altamente eficazes com custo mínimo. Se o objetivo é injetar *conhecimento factual*, o MLP tende a ser mais relevante.
2. **Custo de LoRA não escala com o número de parâmetros congelados, mas com a dimensão das matrizes adaptadas.** Um adaptador de rank `r` sobre `down_proj` (4864×896) tem `r×(4864+896)` parâmetros; sobre `q_proj` (896×896), `r×1792`. Adaptar o MLP é ~3× mais caro por matriz.

Resposta prática (que você vai medir no módulo 6): comece com todas as lineares e rank baixo. É o default do QLoRA e costuma dominar as alternativas.
</details>

---

## Parte B — Práticas

### B1. MQA

Implemente MQA (uma única KV head) modificando o `repeat_kv` do lab. Compare o KV cache das três variantes no Qwen2.5-0.5B com contexto de 32k.

Depois responda: por que o GQA venceu, se o MQA economiza mais?

<details><summary>Gabarito</summary>

MQA no Qwen2.5-0.5B: `2 × 24 × 1 × 64 × 2 = 6 KB/token` — metade do GQA (12 KB), 1/14 do MHA (84 KB). Em 32k: 0,20 GB contra 0,39 GB e 2,75 GB.

O GQA venceu porque a economia adicional do MQA é pequena em termos absolutos (de 0,39 para 0,20 GB) e a perda de qualidade é mensurável: com uma única KV head, todas as 14 query heads são forçadas a buscar no mesmo espaço de chaves, eliminando a especialização que justifica ter várias cabeças. O GQA preserva grupos e captura quase toda a economia. Curva de retorno decrescente clássica.
</details>

---

### B2. ALiBi

Implemente ALiBi: em vez de rotacionar `q` e `k`, some diretamente aos scores uma penalidade linear na distância, com uma inclinação `m_h` diferente por cabeça:

```
scores[i,j] += −m_h · (i − j)
```

Compare, sobre a mesma sequência, o padrão de atenção resultante com o do RoPE. Que viés indutivo o ALiBi impõe que o RoPE não impõe?

<details><summary>Gabarito</summary>

```python
def alibi_bias(n, n_heads):
    inclinacoes = torch.tensor([2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)])
    dist = torch.arange(n)[None, :] - torch.arange(n)[:, None]   # j - i, negativo no passado
    return inclinacoes[:, None, None] * dist[None]
```

ALiBi impõe **decaimento monotônico com a distância**: quanto mais longe, menor a atenção, sempre, por construção. Cabeças com inclinação grande enxergam praticamente só o contexto local; com inclinação pequena, enxergam longe.

RoPE não impõe isso — a atenção pode legitimamente saltar para um token muito distante se o conteúdo justificar (ex.: recuperar uma variável declarada no início de um arquivo). Esse é o motivo de o ALiBi ter perdido espaço apesar de extrapolar bem: o viés que o torna robusto é o mesmo que o limita em tarefas de recuperação a longa distância.
</details>

---

### B3. Position interpolation na prática

Pegue o Qwen2.5-0.5B e calcule a perplexidade (módulo 1) sobre um texto de ~1.500 tokens em três configurações: `base` original, `base` × 4, e posições divididas por 4 (interpolação linear).

O que acontece com a perplexidade **sem** nenhum fine-tuning? Por quê?

<details><summary>Gabarito</summary>

A perplexidade **piora em todas as variantes modificadas**, frequentemente muito.

Isso é o esperado e é o ponto do exercício: mexer na base do RoPE muda as posições relativas que o modelo aprendeu a interpretar. É como trocar as unidades de uma régua depois de treinada. As técnicas de extensão de contexto **sempre** exigem um fine-tuning curto (tipicamente algumas centenas de passos) para o modelo se readaptar à nova escala.

A lição prática: quando você vir "estendemos o contexto para 128k só mudando a base", desconfie. O que se lê nos papers é "mudamos a base **e** fizemos fine-tuning em N tokens de contexto longo".
</details>

---

### B4. Caçando uma induction head

Uma *induction head* completa padrões repetidos: dada a sequência `[A][B] ... [A]`, ela atende a `[B]` e prevê `[B]`. É um dos circuitos mais estudados em interpretabilidade.

Construa um prompt com um padrão repetido explícito (ex.: uma lista de pares `chave: valor` que se repete) e varra as 24 camadas × 14 cabeças procurando a cabeça cuja atenção, na segunda ocorrência de `[A]`, se concentra no token que **seguiu** `[A]` na primeira ocorrência.

<details><summary>Gabarito</summary>

```python
texto = "gato: 7, pato: 3, rato: 9, gato:"
ids = tok(texto, return_tensors="pt")
a = model(**ids, output_attentions=True).attentions
tokens = tok.convert_ids_to_tokens(ids["input_ids"][0])

alvo = ... # índice do token "7" (o que seguiu "gato" na 1ª ocorrência)
for camada in range(len(a)):
    for cabeca in range(a[0].shape[1]):
        peso = a[camada][0, cabeca, -1, alvo]
        if peso > 0.3:
            print(f"camada {camada}, cabeça {cabeca}: {peso:.2%}")
```

Você deve encontrar cabeças fortes em camadas intermediárias/tardias. Em modelos pequenos o sinal é mais ruidoso que em modelos grandes.

Ressalva importante, e é o ponto pedagógico: atenção alta **não prova** que a cabeça causa a previsão. A prova exigiria intervenção — zerar aquela cabeça (*ablation*) e verificar se a previsão muda. Esse é o método padrão em interpretabilidade mecanicista, e um bom exercício adicional.
</details>

---

### B5. O orçamento do FlashAttention

Calcule, para o Llama-3-8B (32 camadas, 32 cabeças, `head_dim` 128) com contexto de 32k e batch 4, em bf16:

a) memória para materializar as matrizes de atenção de **uma** camada;
b) o mesmo, se o FlashAttention não existisse e todas as 32 camadas guardassem suas matrizes para o backward;
c) compare com os 80 GB de uma A100.

<details><summary>Gabarito</summary>

a) `4 (batch) × 32 (heads) × 32.768² × 2 bytes = 4 × 32 × 1,074×10⁹ × 2 ≈ **275 GB**` — para uma única camada.

b) 32 camadas → ~8,8 **terabytes**.

c) Cabe em zero A100s. Nem a matriz de uma camada cabe.

É por isso que FlashAttention não é uma otimização opcional: sem ele (ou equivalente), contexto longo é literalmente impossível, não apenas lento. Note também que a conta cresce com o **quadrado** do contexto — dobrar de 32k para 64k quadruplica esse número.
</details>

---

## Desafio — KV cache incremental

Implemente geração autorregressiva com KV cache manual e prove a equivalência.

1. Escreva `forward_com_cache(token_id, cache)` que processe **um único token** por vez, concatenando K e V ao cache e computando a atenção da nova query contra todo o cache.
2. Gere 20 tokens com essa função.
3. Gere os mesmos 20 tokens reprocessando a sequência inteira a cada passo (sem cache).
4. Verifique que os logits são idênticos.
5. Meça a diferença de tempo e explique a curva.

Perguntas:

a) Por que a máscara causal se torna desnecessária no decode com cache?
b) Como o RoPE precisa ser aplicado nesse regime?

<details><summary>Gabarito</summary>

a) Porque no decode existe **uma única query** (a do token novo, na posição `n`) contra `n+1` keys — todas do passado ou a própria. Não há nenhuma posição futura no cache para mascarar. A máscara só é necessária no **prefill**, onde várias queries são processadas em paralelo.

b) O RoPE precisa ser aplicado com a posição **absoluta correta** do token novo — `cos[n], sin[n]`, não `cos[0], sin[0]`. As keys já no cache foram rotacionadas com suas posições originais no momento em que entraram e **não devem ser rotacionadas de novo**. Esquecer de avançar o índice de posição é um dos bugs mais comuns em implementações caseiras de cache: o modelo gera normalmente por alguns tokens e depois degrada, porque todas as queries passam a se comportar como se estivessem na posição 0.

Sobre a curva de tempo: sem cache, o custo do passo `t` cresce com `t` (reprocessa tudo), então o tempo total é `O(n²)`. Com cache, cada passo é `O(n)` — só a atenção contra o cache cresce — e o total é `O(n²)` também, mas com constante muito menor, porque o MLP e as projeções processam 1 token em vez de `t`. Na prática a diferença é de uma ordem de grandeza já em 60 tokens (você mediu isso no Lab 9 do módulo 1).
</details>
