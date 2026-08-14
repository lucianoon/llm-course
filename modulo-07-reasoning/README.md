# Módulo 7 — Reasoning e dados de raciocínio

> **Pergunta central:** o que muda quando o modelo escreve o próprio raciocínio antes de responder — e por que isso funciona?

Este módulo conecta duas pontas do curso. Do módulo 1: o modelo não planeja a resposta antes de emitir o primeiro token — o pensamento **é** a geração. Daqui para os módulos 9 e 10: reasoning é hoje o principal alvo do RL (recompensas verificáveis) e da distillation (traços de raciocínio são o que se destila).

## Objetivos

1. Explicar por que chain-of-thought funciona, em termos de **compute por token**.
2. Medir o efeito do CoT — em probabilidade e em acurácia — num modelo real.
3. Implementar self-consistency e saber quando ela paga o custo.
4. Preparar dados de raciocínio para SFT e conhecer as decisões de formato.
5. Avaliar com respostas verificáveis: extração robusta, exact match, pass@k.
6. Conhecer os limites: infidelidade do CoT, overthinking, custo.

---

## 1. O argumento do compute

Um transformer tem profundidade fixa. Do módulo 2: cada token gerado passa por exatamente `L` camadas — o modelo executa **a mesma quantidade de computação** para responder "2+2" e para responder um problema de olimpíada.

Isso impõe um teto teórico real: há funções que um circuito de profundidade fixa não computa em um passo. Multiplicação de números grandes, busca em grafo, qualquer coisa que exija composição sequencial de mais operações do que a profundidade permite.

**Chain-of-thought quebra o teto.** Cada token gerado é um passo adicional de computação — e volta como **entrada** para o passo seguinte. O raciocínio escrito funciona como memória de trabalho externa: resultados intermediários são materializados no contexto, onde a atenção pode buscá-los, em vez de precisarem sobreviver implicitamente nas ativações.

```
sem CoT:  pergunta ──[L camadas]──> resposta          compute fixo
com CoT:  pergunta ──[L camadas]──> passo 1 ──[L]──> passo 2 ──[L]──> ... ──> resposta
                                                      compute ∝ tokens gerados
```

Formalmente: Feng et al. (2023) e outros mostraram que transformers com CoT computam classes de funções estritamente maiores do que sem. Não é um truque de prompt — é a diferença entre profundidade fixa e computação iterativa.

### Medido, sem gerar um token

O Lab 2 isola o mecanismo: a probabilidade que o Qwen2.5-0.5B atribui à resposta correta do GSM8K, com e sem o raciocínio de ouro no contexto. Só forward — nenhuma geração, nenhum treino:

| | log P(resposta certa) |
|---|---|
| pergunta → "Resposta final:" | −2,0 a −7,5 |
| pergunta → raciocínio → "Resposta final:" | −0,2 a −2,4 |

**Ganho médio: +3,64 nats — a resposta certa fica 38× mais provável** (faixa: 6× a 311×) só por os passos intermediários estarem materializados no contexto, onde a atenção os alcança. Mesmo modelo, mesmos pesos. O raciocínio não é decoração; ele muda a distribuição.

E o teste causal (Lab 6) fecha o argumento: **corrompendo** um passo do raciocínio (24 → 30), a resposta consistente com o erro passa a dominar — log P(78) vai de −8,3 para **−0,07**, enquanto a resposta correta despenca para −12,2. O modelo *lê* a cadeia e a continua; não a verifica. É o mecanismo que faz o CoT funcionar — e que o torna perigoso quando os passos erram.

> 📐 A consequência prática imediata: **problemas que exigem `k` passos sequenciais precisam de espaço para ~`k` passos de geração**. Cortar o `max_tokens` de um modelo de raciocínio não o torna conciso; torna-o errado. Medido no Lab 5: com orçamento de 30–120 tokens, o 0.5B acerta ~0 de 6 problemas (todas as respostas truncadas no meio da cadeia); com 250, acerta 2 — e ainda trunca metade. O modelo não "resume" o raciocínio quando o espaço aperta; ele é cortado no meio dele.

---

## 2. A escada do reasoning

| Técnica | O que é | Custo | Ganho típico |
|---|---|---|---|
| **Few-shot CoT** (Wei et al., 2022) | Exemplos com raciocínio no prompt | prompt maior | O salto original: GSM8K de 18% → 57% no PaLM 540B |
| **Zero-shot CoT** (Kojima et al., 2022) | "Let's think step by step" | ~nada | Surpreendentemente perto do few-shot |
| **Self-consistency** (Wang et al., 2022) | Amostrar `k` cadeias, votar na resposta | `k×` a geração | +10–20 p.p. em matemática |
| **SFT em traços** (este módulo) | Treinar sobre raciocínios escritos | um treino | O formato vira comportamento default |
| **RL verificável** (módulo 9) | Recompensar cadeias que chegam à resposta certa | muito treino | o1, R1 — o estado da arte |

Duas observações sobre a escada:

**CoT emergiu com escala.** Em modelos pequenos, pedir raciocínio pode *piorar* — o modelo gera texto com formato de raciocínio, mas os passos contêm erros que contaminam a resposta. Medido no Lab 3, Qwen2.5-0.5B em 10 problemas do GSM8K:

| Modo | Acurácia | Tokens/resposta | Custo |
|---|---|---|---|
| Resposta direta | **0%** | 4 | 1× |
| Chain-of-thought | **30%** | 231 | 52,5× |

O 0.5B simplesmente **não resolve GSM8K em um passo** — zero em dez. Com espaço para raciocinar, resolve 3 — a um custo de 52× mais tokens. As duas lições juntas: o CoT é o que torna a tarefa possível, e ele não é grátis.

**Self-consistency só funciona com resposta extraível — e com competência de base.** Você vota em `resposta_final`, não em cadeias. Mas há um pressuposto que a execução do Lab 4 expôs: no 0.5B (acurácia ~30%), os votos saíram *espalhados* — cinco amostras, quatro respostas diferentes — e a maioria errou até onde o greedy tinha acertado. A votação exige que a resposta certa seja o **modo** da distribuição; abaixo de ~50% de acurácia base, votar entre erros diversos só formaliza o ruído. **Self-consistency amplifica competência existente; não a cria.**

---

## 3. Modelos de reasoning — o que mudou em 2024–2025

o1 (OpenAI) e depois **DeepSeek-R1** transformaram CoT de técnica de prompt em **regime de treino**: RL sobre problemas com resposta verificável, onde a recompensa é simplesmente "a resposta final está certa?". O modelo aprende sozinho a raciocinar mais e melhor — incluindo comportamentos não programados como auto-verificação e backtracking ("Wait, let me reconsider...").

O detalhe mais relevante para você está no paper do R1: os **modelos destilados**. A DeepSeek pegou 800k traços de raciocínio gerados pelo R1 e fez **SFT simples** em modelos Qwen e Llama de 1.5B a 70B. Resultado: o Qwen-1.5B destilado supera o GPT-4o em matemática. Sem RL nenhum — só SFT em dados de raciocínio de alta qualidade.

A implicação prática é enorme e é a ponte com o módulo 10: **a maneira mais barata de obter capacidade de raciocínio num modelo pequeno é treiná-lo sobre traços de um modelo grande** — exatamente o pipeline deste módulo, com o GSM8K fazendo o papel dos traços.

### O formato `<think>`

Modelos de reasoning modernos separam o raciocínio da resposta:

```
<think>
O usuário quer X. Primeiro preciso calcular Y...
Espera, isso está errado — recalculando...
</think>
A resposta é 42.
```

Decisões de formato que importam ao preparar dados:

- **Delimitadores consistentes** — o runtime corta o `<think>` antes de mostrar ao usuário; se o delimitador variar, vaza raciocínio para a interface.
- **O raciocínio não é a resposta** — treinar sem separação produz modelos que despejam o processo no usuário.
- **Loss em tudo** — diferente do masking de prompt (módulo 5), aqui os tokens de raciocínio *são* o que se quer ensinar. Mascará-los destruiria o propósito.

---

## 4. Avaliação verificável

A grande vantagem de matemática e código: a resposta é **verificável por comparação exata** (ou por teste que passa). Sem juiz, sem viés, sem custo de API.

O GSM8K anota a resposta final após `####`. O pipeline de avaliação é:

```
gerar → extrair o número final → comparar com o gabarito → acurácia
```

> ⚠️ **A extração é onde as avaliações quebram.** O modelo escreve "a resposta é **$108**", "108 dólares", "108.00", "R$ 108" — e uma regex ingênua marca tudo como erro. Resultado: você mede a *capacidade de formatação*, não o raciocínio. A regra prática: extraia o **último número** da resposta (o raciocínio menciona muitos números; o final tende a ser a resposta), normalize vírgulas, moedas e `.00`, e **inspecione manualmente uma amostra dos "erros"** — se muitos são acertos mal extraídos, conserte a extração antes de tirar qualquer conclusão.

### pass@k

Para código e matemática com amostragem: `pass@k` = probabilidade de ao menos uma entre `k` amostras estar certa. Com `n ≥ k` amostras e `c` corretas, o estimador sem viés é:

```
pass@k = 1 − C(n−c, k) / C(n, k)
```

`pass@1` é a métrica honesta de uso único; `pass@k` com `k` grande mede o que a *self-consistency* ou um verificador conseguiriam resgatar.

---

## 5. Os limites

### Infidelidade do CoT

O raciocínio escrito **não é necessariamente o processo que produziu a resposta** — é texto gerado, plausível, que pode racionalizar uma resposta já determinada por outros caminhos. A evidência (Turpin et al., 2023; trabalhos da Anthropic em 2023–2025): é possível enviesar a resposta de um modelo sem que o viés jamais apareça na cadeia; e modelos às vezes chegam à resposta certa com raciocínio errado, ou vice-versa.

Consequência prática: **CoT como explicação para auditoria é evidência fraca.** CoT como técnica para melhorar acurácia é outra coisa — e essa o Lab mede.

### Overthinking

Modelos de reasoning gastam tokens demais em problemas fáceis — R1 pode gerar 2.000 tokens para "quanto é 15% de 200". Custo real em produção: latência e dinheiro. Mitigações: roteamento por dificuldade (problemas fáceis vão para o modelo sem thinking), orçamentos de token, e treinos recentes que penalizam comprimento desnecessário.

### O custo

Um modelo de reasoning que gera 10× mais tokens custa ~10× mais por resposta (o decode domina, módulo 1). A pergunta de engenharia nunca é "reasoning é melhor?" — é "o ganho de acurácia paga o custo por chamada neste caso de uso?".

---

## 6. Leituras

1. **Wei et al. (2022), "Chain-of-Thought Prompting"** — [arXiv:2201.11903](https://arxiv.org/abs/2201.11903). O paper que abriu a área.
2. **Wang et al. (2022), "Self-Consistency"** — [arXiv:2203.11171](https://arxiv.org/abs/2203.11171).
3. **DeepSeek-AI (2025), "DeepSeek-R1"** — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948). Leia as seções de RL (ponte para o módulo 9) e de distillation (ponte para o 10).
4. **Turpin et al. (2023), "Language Models Don't Always Say What They Think"** — [arXiv:2305.04388](https://arxiv.org/abs/2305.04388). A infidelidade do CoT, medida.
5. **Cobbe et al. (2021), "GSM8K"** — [arXiv:2110.14168](https://arxiv.org/abs/2110.14168). O dataset do lab.

---

## 7. Checklist de saída

- [ ] Por que a profundidade fixa do transformer limita o que ele computa em um passo?
- [ ] Em que sentido os tokens de CoT são "memória de trabalho externa"?
- [ ] Por que cortar `max_tokens` de um modelo de raciocínio o torna errado, não conciso?
- [ ] Em que a self-consistency vota, e por que isso exige resposta extraível?
- [ ] O que o paper do R1 mostrou sobre distillation de raciocínio, e por que isso importa para modelos pequenos?
- [ ] Por que a loss NÃO é mascarada nos tokens de raciocínio, ao contrário do prompt?
- [ ] Cite dois modos de uma avaliação de matemática quebrar na extração da resposta.
- [ ] O CoT escrito é evidência confiável do processo interno do modelo? Cite o contra-exemplo.
- [ ] Quando reasoning NÃO compensa em produção?

Depois: `dados.py`, `lab_cpu.py` (executado e validado), `lab_mlx.py` no M4.
