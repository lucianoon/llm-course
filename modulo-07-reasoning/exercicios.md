# Módulo 7 — Exercícios

💻 = roda em qualquer máquina (`lab_cpu.py` como base) | 🍎 = requer o Mac

---

## Parte A — Conceituais

### A1. O argumento do compute

Um transformer de 28 camadas recebe: *"Multiplique 847 por 293 e responda só o número."*

a) Por que a arquitetura torna essa tarefa difícil **em um passo**, mesmo para um modelo grande?
b) Por que a mesma tarefa com CoT é qualitativamente diferente?
c) A tokenização também conspira contra. Como? (módulo 1)

<details><summary>Gabarito</summary>

a) A multiplicação de números de 3 dígitos exige uma sequência de produtos parciais e somas com carrego — computação **serial**. O forward de um transformer tem profundidade fixa (28 camadas): tudo o que ele computa em um passo precisa caber em 28 transformações. Não há loop, não há "guardar e continuar". Modelos grandes memorizam multiplicações frequentes, mas não computam o algoritmo geral em um passo.

b) Com CoT, cada resultado parcial é **escrito no contexto** ("847 × 3 = 2541; 847 × 90 = 76230...") e volta como entrada — a atenção o recupera no passo seguinte. O compute passa a escalar com os tokens gerados, não com a profundidade. É a diferença entre um circuito combinacional e uma máquina com memória.

c) Do módulo 1: `847` pode virar `[8][4][7]` ou `[84][7]` dependendo do tokenizer — o modelo nem vê os dígitos alinhados por casa decimal. Modelos modernos forçam dígito a dígito justamente para mitigar isso.
</details>

---

### A2. Self-consistency — quando não usar

Self-consistency deu +10–20 p.p. em matemática. Um colega propõe aplicá-la a tudo: atendimento, resumos, geração de e-mails.

a) Por que não funciona para essas tarefas?
b) Para quais das seguintes funcionaria: extração de datas de contratos, classificação de sentimento, escrita de poesia, consultas SQL?
c) Qual o custo de `k=10` em produção?

<details><summary>Gabarito</summary>

a) Self-consistency vota na **resposta extraída**, não na cadeia. Um e-mail não tem resposta extraível — dez amostras dão dez e-mails diferentes, e não há em que votar. (Existe uma variante — universal self-consistency — que pede ao próprio modelo para escolher a resposta mais consistente, mas aí voltamos a um juiz, com os vieses do módulo 5.)

b) **Sim**: extração de datas (resposta curta e comparável), classificação (voto entre rótulos), SQL (normalizando a query ou comparando o *resultado* da execução — ainda melhor). **Não**: poesia.

c) `k×` o custo de geração — 10× os tokens de saída, que dominam o preço (módulo 1: decode é o gargalo). Latência também, a menos que as `k` amostras rodem em paralelo (aí o custo vira memória de batch: KV cache × k).
</details>

---

### A3. O masking invertido

No módulo 5, a regra era: loss só nos tokens da resposta, prompt mascarado com `-100`. Nos dados de raciocínio deste módulo, o raciocínio NÃO é mascarado.

a) Justifique a diferença.
b) E se você mascarasse tudo exceto o "Resposta final: N" — treinando com CoT no contexto mas loss só no número? O que o modelo aprenderia?

<details><summary>Gabarito</summary>

a) O masking existe para não ensinar o modelo a *gerar* o que ele nunca precisará gerar (instruções de usuário). O raciocínio é o oposto: é exatamente o que queremos que ele gere. Mascarar o raciocínio ensinaria o modelo a pular direto para a resposta — reproduzindo o `LoRA direto` do Lab 2, que é o grupo de controle, não o objetivo.

b) Armadilha sutil: na **inferência**, o raciocínio não estará no contexto — o modelo teria que gerá-lo, mas nunca recebeu gradiente para isso. Ele aprenderia `P(resposta | pergunta + raciocínio de ouro)`, uma distribuição que nunca ocorre em produção. Na prática, sairia direto para uma resposta (errada), porque nada o ensinou a produzir os passos. Treino e inferência precisam ver a mesma distribuição.
</details>

---

### A4. A extração que mente

Você avalia dois modelos no GSM8K com a regex `r"\d+$"` (número no fim da string). Modelo A: 61%. Modelo B: 34%. Inspecionando 20 "erros" do modelo B, você encontra 12 respostas assim: *"...so the total is 108 apples."*

a) O que aconteceu?
b) Qual a acurácia real aproximada do modelo B nessa amostra?
c) Que procedimento teria evitado a conclusão errada?

<details><summary>Gabarito</summary>

a) A regex exige que a string **termine** em dígito. O modelo B encerra com "apples." — o número existe, a extração falha, o acerto vira erro. Você mediu aderência a um formato que ninguém pediu.

b) Na amostra de 20 "erros", 12 eram acertos mal extraídos: a acurácia real é bem maior. Extrapolando: se 34% acertou "oficialmente" e ~60% dos erros são falsos, a acurácia real fica na faixa de 60–70% — ou seja, **A e B podem ser equivalentes**.

c) O do Lab 1: testar a extração com casos de unidade antes de usá-la, extrair o *último número* com normalização, e **sempre inspecionar manualmente uma amostra dos erros** antes de reportar qualquer número. Vale para toda avaliação automática, não só matemática.
</details>

---

### A5. Reasoning em produção

Sua empresa tem um assistente que responde dúvidas de cobrança (fáceis, alto volume: 50k/dia) e disputas de contrato (difíceis, 200/dia). Alguém propõe migrar tudo para um modelo de reasoning.

Projete a solução correta e justifique com os conceitos do módulo.

<details><summary>Gabarito</summary>

**Roteamento por dificuldade**, não migração total:

- **Cobrança (50k/dia):** modelo padrão sem thinking. Overthinking custaria caro — 2.000 tokens de raciocínio para "qual o valor da minha fatura?" multiplicados por 50k/dia é dinheiro real e latência que degrada a experiência. Perguntas fáceis não têm passos a externalizar; o compute de um forward basta.
- **Disputas (200/dia):** modelo de reasoning, possivelmente com self-consistency (k=3–5) nos casos de maior valor. Volume baixo torna o custo por chamada irrelevante frente ao custo de errar uma disputa.
- **O roteador:** um classificador leve (ou o próprio modelo padrão com uma instrução de triagem) decide o caminho. Casos em que o modelo padrão expressa incerteza sobem para o de reasoning — *escalation*, o padrão que os provedores usam.

A regra do módulo: reasoning se paga quando `ganho de acurácia × valor do acerto > custo extra de tokens × volume`.
</details>

---

## Parte B — Práticas

### B1. 💻🍎 CoT e escala

O Lab 3 mediu CoT vs direto no 0.5B. Repita no Mac com o Qwen2.5-1.5B e o 7B-4bit (mesmos 10 problemas, mesmas instruções).

Monte a tabela acurácia × tamanho × modo. O ganho do CoT cresce com a escala? Existe cruzamento (modelo pequeno onde CoT piora)?

<details><summary>Gabarito esperado</summary>

O padrão da literatura: o **ganho do CoT cresce com a escala**. No 0.5B, o CoT pode empatar ou até perder (a cadeia contém erros aritméticos que contaminam a resposta); no 1.5B já deve haver ganho claro; no 7B, ganho grande.

O achado de Wei et al. (2022) foi exatamente esse: CoT é uma *habilidade emergente* — a curva de ganho por escala não é suave, e abaixo de certo tamanho o raciocínio escrito é imitação de forma sem substância.
</details>

---

### B2. 💻 Self-consistency: a curva de k

Estenda o Lab 4: para 8 problemas, meça a acurácia da votação com k = 1, 3, 5, 9 (reuse as amostras: gere 9 e vote em prefixos).

a) Onde a curva satura?
b) Compare o custo total de tokens com o ganho.
c) Por que reutilizar as mesmas 9 amostras para todos os k economiza sem viciar o resultado?

<details><summary>Gabarito</summary>

a) Tipicamente satura entre k=5 e k=9 para problemas nessa faixa de dificuldade — o voto majoritário estabiliza.

b) k=9 custa 9× os tokens para tipicamente +10-20 p.p. sobre k=1. O ganho marginal de k=5→9 costuma ser pequeno; k=3–5 é o ponto prático.

c) Porque cada amostra é i.i.d. dada a temperatura — votar nos primeiros `k` de 9 é estatisticamente idêntico a gerar `k` novas. (Rigor extra: para k par, defina o desempate de antemão.) É o mesmo truque do estimador de pass@k: gere `n ≥ k` uma vez, avalie todos os k de interesse.
</details>

---

### B3. 💻 pass@k

Implemente o estimador sem viés do pass@k (README, seção 4). Com as 9 amostras por problema do B2:

a) Calcule pass@1, pass@3, pass@5 para cada problema e a média.
b) Compare pass@5 com a acurácia da self-consistency com k=5. Qual é maior? Por quê?
c) O que cada métrica mede, em termos de produto?

<details><summary>Gabarito</summary>

```python
from math import comb
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1 - comb(n - c, k) / comb(n, k)
```

b) **pass@5 ≥ self-consistency@5**, sempre: pass@k conta sucesso se *qualquer* amostra acerta; a votação exige que a resposta certa seja a *mais frequente*. Uma resposta certa minoritária conta para pass@k e perde a votação.

c) pass@k mede o teto com um **verificador perfeito** (você tem como testar cada resposta — caso de código com testes). Self-consistency mede o que se obtém **sem verificador**, só com concordância. A distância entre as duas é o valor de construir um verificador.
</details>

---

### B4. 💻 Fidelidade em escala

O Lab 6 testou fidelidade causal num único problema. Generalize: para 10 problemas do GSM8K, corrompa o primeiro resultado intermediário do raciocínio de ouro (some 3 ao número) e propague o erro no restante da cadeia.

Meça: em quantos problemas a resposta mais provável segue o raciocínio corrompido?

<details><summary>Gabarito esperado</summary>

Espere maioria clara seguindo a corrupção (o modelo continua a cadeia em vez de verificá-la) — mas provavelmente **não** 10/10. Os casos que resistem são interessantes: ou o modelo "conhece" a resposta por memorização e a mantém apesar da cadeia (infidelidade!), ou a corrupção criou uma inconsistência tão gritante que a distribuição fica difusa.

Os dois desvios são instrutivos: seguir a cadeia é o mecanismo que faz CoT funcionar; ignorá-la é a infidelidade de Turpin et al. O mesmo modelo exibe os dois comportamentos, dependendo do caso — e é por isso que nenhuma auditoria séria confia no raciocínio escrito como registro do processo.
</details>

---

### B5. 🍎 O custo do overthinking

Com o R1-Distill-1.5B do Lab 4 e o seu LoRA CoT:

1. Rode ambos nos 20 primeiros problemas do gabarito.
2. Registre: acurácia, tokens por resposta, e a distribuição de comprimento.
3. Separe os problemas em fáceis (resposta em ≤3 passos no ouro) e difíceis.

O R1-Distill gasta quantos tokens a mais nos fáceis? A acurácia extra justifica?

<details><summary>Gabarito esperado</summary>

Espere o R1-Distill gastando 3–10× mais tokens, com o exagero concentrado nos problemas **fáceis** (500+ tokens para problemas de dois passos — re-verificações, reformulações, "wait, let me double-check").

Nos fáceis, o LoRA CoT provavelmente empata em acurácia com fração do custo. Nos difíceis, o R1-Distill deve abrir vantagem real.

É a assinatura do overthinking e a justificativa do roteamento por dificuldade do A5 — agora com os seus números.
</details>

---

## Desafio — destilação de raciocínio (ponte para o módulo 10)

Reproduza em miniatura o pipeline de distillation do R1:

1. **Professor:** use o R1-Distill-Qwen-1.5B (ou o 7B se couber) para gerar raciocínios para 300 problemas do GSM8K de treino que **não** estão no seu subconjunto de 1.500.
2. **Filtragem por rejeição:** descarte os traços cuja resposta final está errada (você tem o gabarito). Guarde a taxa de aproveitamento.
3. **Aluno:** treine um LoRA no Qwen2.5-1.5B sobre os traços aprovados.
4. **Compare** três modelos no teste: LoRA CoT do lab (traços humanos do GSM8K), LoRA destilado (traços do professor), e a base.

Perguntas:

a) Os traços do professor são mais longos e estilisticamente diferentes dos humanos. Isso ajudou ou atrapalhou?
b) A filtragem por resposta correta garante que os *passos* estão corretos?
c) Que viés a filtragem por rejeição introduz na distribuição de problemas?

<details><summary>Notas</summary>

a) Resultado em aberto — os dois desfechos ocorrem na prática. Traços de modelo costumam ser mais verbosos e uniformes (bom para SFT: consistência), mas herdam os tiques do professor.

b) **Não** — um traço pode errar duas vezes e chegar à resposta certa, ou acertar por caminho espúrio. A filtragem por resposta é um proxy barato. O R1 real usa também filtros de legibilidade e mistura de idiomas. Verificar os passos exigiria um verificador de processo (PRM) — assunto do módulo 9.

c) Os problemas **difíceis são descartados desproporcionalmente** (o professor erra mais neles), então o aluno treina numa distribuição enviesada para o fácil. Em escala, isso limita o teto do aluno ao teto do professor — a razão pela qual distillation transfere, mas raramente supera.
</details>
