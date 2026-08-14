# Módulo 9 — Exercícios

💻 = qualquer máquina | 🍎 = Mac

---

## Parte A — Conceituais

### A1. O truque em uma linha

a) Derive `∇E_{y~π}[R(y)] = E[R·∇log π]` a partir de `∇π = π·∇log π`.
b) Por que o resultado é importante — o que ele permite que a diferenciação direta não permitiria?
c) Onde exatamente essa fórmula aparece na loss do GRPO do lab?

<details><summary>Gabarito</summary>

a) `∇E[R] = ∇Σ_y π(y)R(y) = Σ_y R(y)∇π(y) = Σ_y R(y)π(y)∇log π(y) = E_{y~π}[R(y)∇log π(y)]`.

b) Ele converte o gradiente de uma **expectativa sobre amostras** (não-diferenciável: sampling é discreto, e `R` pode ser um teste que passa/falha) num **valor esperado de coisas computáveis**: gere amostras, pese a log-prob de cada uma pela recompensa. Nenhuma derivada de `R` é necessária — `R` pode ser qualquer caixa-preta.

c) No surrogate: `ratio·A` onde `ratio = exp(lp_novo − lp_old)`. Na primeira época interna, `ratio = 1` e o gradiente de `ratio·A` em relação a θ é exatamente `A·∇log π` — REINFORCE com baseline. O clip e as épocas internas são o refinamento PPO por cima.
</details>

---

### A2. O grupo unânime

No GRPO com G=8, considere três prompts: um em que o modelo acerta 8/8, um 0/8, um 3/8.

a) Calcule as vantagens de cada grupo.
b) Qual a consequência para o gradiente de cada prompt?
c) O R1 foi treinado com problemas de competição. Por que problemas *fáceis demais* seriam desperdício de compute, e *difíceis demais* seriam pior que desperdício?

<details><summary>Gabarito</summary>

a) 8/8: todos `r=1`, média 1, desvio 0 → `A = 0/ε = 0` para todos. 0/8: idem, tudo zero. 3/8: média 0,375, desvio ~0,48 → acertos `A ≈ +1,29`, erros `A ≈ −0,77`.

b) Grupos unânimes: gradiente **zero** (o surrogate é `A·(...)`). Só o prompt 3/8 ensina algo. O compute dos outros dois foi gasto em geração e jogado fora.

c) Fáceis: gradiente zero, compute desperdiçado em gerar G respostas certas. Difíceis demais (0/G sempre): também gradiente zero — mas pior, porque nada no treino fará o modelo passar a acertá-los (RL amplifica, não cria), então continuarão custando para sempre. Por isso os pipelines de RLVR **filtram por taxa de acerto do modelo atual** (ex.: manter prompts com acerto entre 10% e 90%) e re-filtram conforme o modelo melhora — o currículo explícito por cima do emergente.
</details>

---

### A3. Hackeando o seu próprio verificador

Para cada recompensa proposta, descreva o hack que o RL vai encontrar e conserte-a:

1. `r = 1/comprimento_da_resposta` (queremos concisão)
2. `r = nota de 0 a 10 dada por um juiz LLM` (queremos qualidade)
3. `r = 1 se o código Python roda sem exceção` (queremos código correto)
4. `r = similaridade de embedding com a resposta de referência` (queremos acurácia)

<details><summary>Gabarito</summary>

1. **Resposta vazia ou de um token.** Concisão sem restrição de conteúdo colapsa no vazio. Correção: `r = acurácia_binária × bônus_se_dentro_do_orçamento` — a concisão só pontua **condicionada ao acerto**.
2. **Bajulação, comprimento, confiança e formato bonito** — os vieses do juiz (módulo 5) viram o alvo. Pior: com iterações suficientes, o RL encontra até adversariais textuais. Correções: juiz com rubrica fechada e comparação par-a-par, recompensa binária extraída ("passou/não passou na rubrica"), e KL apertado. Onde possível, troque o juiz por regra.
3. **`print("ok")` roda sem exceção.** Rodar ≠ correto. Correção: testes de unidade com casos escondidos, `r = fração_de_testes_que_passam` — e os testes fora do alcance do modelo (o hack real de apagar/hardcodar testes aconteceu em produção).
4. **Palavras-chave da referência despejadas sem estrutura** — embeddings são cegos a ordem e lógica. Correção: extração + comparação exata quando a resposta é verificável; embedding só como *shaping* fraco, nunca como sinal principal.

O padrão das correções: **binarizar, condicionar ao acerto exato, e limitar** — as três defesas do lab.
</details>

---

### A4. PPO vs GRPO no seu hardware

a) Liste o que ocupa memória num passo de PPO e num de GRPO, para um modelo de 1,5B em bf16 com LoRA.
b) O GRPO troca o value model por quê, e qual o custo dessa troca?
c) Por que o GRPO gera com `temperature > 0` obrigatoriamente?

<details><summary>Gabarito</summary>

a) **PPO:** política (3 GB) + referência (3 GB) + reward model (RM neural, ~3 GB) + value model (~3 GB, treinando: +estados de otimizador). ~12 GB+ antes de ativações — não cabe no M4. **GRPO com recompensa por regra:** política (3 GB) + referência (3 GB) + adaptadores. ~6,5 GB — cabe. O RM por regra custa zero; o value model não existe.

b) Troca por **estatística do grupo**: a baseline vem da média de G amostras em vez de um preditor aprendido. O custo: G gerações por prompt por passo (compute de decode, o caro do módulo 1) e uma baseline mais ruidosa que um value model bem treinado — compensada por ser impossível de treinar mal.

c) Com T=0, as G gerações do grupo são **idênticas** → recompensas idênticas → desvio zero → vantagem zero → gradiente zero, para sempre. A diversidade do grupo É o mecanismo de exploração do GRPO.
</details>

---

### A5. A curva que mente

Seu treino de GRPO mostra: recompensa média 0,3 → 0,9 em 200 passos; KL crescendo de 0,01 para 4,5; comprimento médio das gerações caindo de 180 para 15 tokens.

Diagnóstico completo, e o que checar primeiro?

<details><summary>Gabarito</summary>

Quadro clássico de **reward hacking com colapso**: a recompensa "melhorou" 3×, mas o KL explodiu (a política abandonou a linguagem da referência) e as gerações encolheram 12× — o modelo provavelmente encontrou um atalho curto que satisfaz o verificador (formato mínimo extraível, resposta memorizada frequente, ou um bug de extração que aceita qualquer número).

Checar primeiro: **ler 20 gerações** — segundos de trabalho, diagnóstico quase certo. Depois: (1) a função de extração aceita lixo? (2) a recompensa é limitada? (3) o β do KL está alto o suficiente? (4) os 15 tokens contêm a resposta certa de fato (aí não é hack — é o modelo ficando eficiente — mas o KL de 4,5 diz que não é o caso).

A regra do lab: curva de recompensa é condição necessária; leitura de gerações é a suficiente.
</details>

---

## Parte B — Práticas

### B1. 💻 Dr. GRPO

O Dr. GRPO remove a divisão por `std` da vantagem (que dá gradiente maior a grupos quase-unânimes) e a normalização por comprimento. Implemente a variante no `treinar_grpo` do lab (`A_i = r_i − média`, sem desvio) e compare com o GRPO padrão em: taxa de sucesso final, PPL, e estabilidade da curva.

<details><summary>Gabarito esperado</summary>

Com recompensa binária e grupos mistos, a diferença tende a ser pequena; ela aparece nos grupos **quase-unânimes** (7/8), onde o GRPO padrão divide por um desvio minúsculo e produz vantagens gigantes — passos bruscos ocasionais na curva. O Dr. GRPO os suaviza.

O ponto conceitual: a divisão por std faz o algoritmo tratar "acertar 7/8" e "acertar 4/8" com a mesma intensidade de sinal, o que enviesa o treino na direção dos prompts quase-resolvidos. Removê-la deixa o sinal proporcional à surpresa real.
</details>

---

### B2. 💻 A curva de G

Rode o GRPO do lab com G = 2, 4, 8, 16 (mesmo compute total de gerações: ajuste os passos para 240, 120, 60, 30).

Qual G vence com compute fixo? Por que G=2 é especialmente ruim?

<details><summary>Gabarito esperado</summary>

G=2 é quase inútil: com recompensa binária, os únicos grupos informativos são os (1,0) — e neles a vantagem é fixa (+1/−1 após normalização), uma baseline de UMA amostra, ruidosíssima. Grupos maiores estimam a baseline melhor, mas com compute fixo veem menos prompts.

Espere o ótimo em G intermediário (4–8 nesta escala). O trade-off é geral: G grande = melhor baseline por prompt; G pequeno = mais prompts vistos. O R1 usou 16 com compute de sobra.
</details>

---

### B3. 💻 Recompensa densa vs esparsa

A recompensa do lab é 1/0 no fim. Crie uma variante com *shaping*: +0,3 se a geração contém `\n` (chegou a quebrar linha), +0,7 adicionais se contém `\n--`. Compare a velocidade de aprendizado e o resultado final com a binária pura.

<details><summary>Gabarito esperado</summary>

O shaping acelera o início (o modelo aprende primeiro o degrau fácil), mas cria um novo risco: se o degrau parcial é mais fácil de maximizar, parte da política estaciona nele (quebras de linha sem diálogo). Espere convergência mais rápida e possivelmente um teto ligeiramente pior — ou um hack parcial.

A lição: shaping é dívida técnica de recompensa. Útil quando o sinal esparso é raro demais para começar (taxa base ~0), perigoso depois. O R1 evitou shaping por isso — e exigiu, em troca, um modelo base que já acertava às vezes.
</details>

---

### B4. 💻 Entropia: o termômetro do colapso

Adicione ao `treinar_grpo` o registro da **entropia média da política** sobre os tokens gerados. Rode o treino normal e o hackeado (Lab 4) e plote as duas curvas.

<details><summary>Gabarito</summary>

```python
probs = torch.exp(lp_novo)          # aproximação: entropia só no token amostrado é
# enviesada; melhor: recomputar softmax completo nas posições geradas
logits, _ = politica(seqs)
lp_full = F.log_softmax(logits[:, PROMPT_LEN-1:-1].float(), dim=-1)
entropia = -(lp_full.exp() * lp_full).sum(-1).mean()
```

Espere: no treino saudável, a entropia cai moderadamente (a política se especializa); no hackeado, **despenca** — o spam é uma política quase determinística. A entropia é o indicador antecedente do colapso: ela cai *antes* de a degeneração ficar visível nas gerações. Todo pipeline de RL sério a monitora (e muitos adicionam um bônus de entropia à loss para sustentar exploração).
</details>

---

### B5. 🍎 Curriculum por taxa de acerto

Implemente o filtro do A2: antes do treino MLX, rode o modelo base nos 1.000 prompts (G=4, T=0,8), meça a taxa de acerto por prompt e separe três subconjuntos: fácil (>75%), fronteira (10–75%), impossível (0%).

Treine 100 iterações só na fronteira e compare com 100 iterações no dataset completo.

<details><summary>Gabarito esperado</summary>

O subconjunto de fronteira deve dar mais ganho por iteração — todo passo tem gradiente, contra o dataset completo onde parte dos grupos sai unânime (gradiente zero, geração paga).

Meça também o custo do filtro: 1.000 prompts × 4 gerações é caro — mas paga-se uma vez, e o pipeline real refaz o filtro a cada N passos conforme a fronteira se move. É exatamente o que os treinos de RLVR de produção fazem, e o motivo de "seleção de dados para RL" ser hoje uma subárea inteira.
</details>

---

## Desafio — o verificador do seu domínio

O módulo inteiro depende de uma pergunta: **o que, no seu trabalho, é verificável?**

1. Liste 5 tarefas do seu dia (ou do seu produto) e classifique o sinal disponível: verificável por regra / verificável por execução / só preferência / só imitação.
2. Para a melhor candidata verificável, escreva o verificador — de verdade, em Python, com testes.
3. Tente hackeá-lo você mesmo: escreva 3 respostas que pontuam bem e são ruins. Conserte o verificador. Repita.
4. Estime o custo de um treino GRPO: taxa de acerto atual do modelo × G × tokens por geração × iterações.
5. Conclua: RL, distillation de um modelo que já faz, ou SFT? (A resposta honesta para a maioria dos casos de negócio é a segunda ou a terceira — saiba dizer por quê.)

O item 3 é o coração do desafio. Um verificador que você mesmo não consegue hackear em meia hora é raro — e cada hack que você encontra sentado à mesa é um que o RL não vai encontrar por você em produção.
