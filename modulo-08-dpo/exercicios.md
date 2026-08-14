# Módulo 8 — Exercícios

💻 = qualquer máquina | 🍎 = Mac

---

## Parte A — Conceituais

### A1. A derivação, de trás para frente

Sem consultar o README: parta da loss do DPO e reconstrua o caminho até o objetivo de RLHF. Em que passo exato a intratabilidade de `Z(x)` desaparece, e qual propriedade do modelo de Bradley-Terry permite isso?

<details><summary>Gabarito</summary>

Caminho reverso: a loss `−log σ(β·Δ)` é a verossimilhança de Bradley-Terry com `r̂ = β·log(π/π_ref)`. Essa expressão de `r̂` vem de inverter a solução ótima `π* ∝ π_ref·exp(r/β)` do objetivo KL-regularizado.

`Z(x)` desaparece no passo do Bradley-Terry: como `P(y_w ≻ y_l) = σ(r_w − r_l)` depende só da **diferença** de recompensas, e `β·log Z(x)` é idêntico para as duas respostas (depende só de `x`), ele cancela na subtração. A propriedade-chave: **Bradley-Terry é invariante a deslocamentos da recompensa por termos que dependem só do contexto.** Sem isso, o DPO não existiria — `Z(x)` é uma soma sobre todas as respostas possíveis, incomputável.
</details>

---

### A2. Diagnóstico por log

Quatro treinos de DPO, quatro logs. Diagnostique cada um:

1. Loss inicial 0,693; margem cresce; `lp_chosen` estável, `lp_rejected` despenca.
2. Loss inicial 0,693; margem cresce; `lp_chosen` cai de −80 para −310; geração vira lixo.
3. Loss inicial 0,45.
4. Loss cai para 0,001 em 15 iterações; na avaliação final, o comportamento-alvo não mudou.

<details><summary>Gabarito</summary>

1. **O treino ideal.** O rejected é suprimido, o chosen preservado. Raro na prática; aproveite.

2. **A patologia em grau terminal.** Ambos caíram (o chosen, catastroficamente): a política fugiu da distribuição inteira dos dados. Causas típicas: LR alto demais, β baixo demais, ou treino longo demais. O lab mediu exatamente isso — chosen de −164 → −208 já com LR moderado. Reduza LR, suba β, pare mais cedo (a margem estabiliza muito antes do fim).

3. **Bug de dados ou de código.** A loss inicial DEVE ser ln 2 = 0,693, porque política = referência no passo 0. Um valor diferente significa que política e referência já divergem (modelo errado carregado como referência?) ou que o cálculo de log-probs está errado (shift? masking?).

4. **Pares fáceis demais e fora da distribuição.** A loss saturou porque distinguir chosen de rejected era trivial (ex.: rejected com defeito grosseiro que o modelo nunca produziria de qualquer forma). O gradiente foi para zero antes de mover qualquer coisa relevante — o gap off-policy. Solução: rejected amostrados do próprio modelo (online DPO), ou defeitos mais próximos do comportamento real.
</details>

---

### A3. Escolhendo o método

Para cada cenário, escolha entre DPO, KTO, ORPO, SimPO (ou nenhum) e justifique:

1. 40k pares anotados por juiz LLM; GPU farta; modelo já passou por SFT.
2. 25k respostas com 👍/👎 dos usuários do seu produto — sem pares.
3. M4 de 16 GB; modelo de 3B; você quer SFT + preferências no mesmo treino.
4. Seus pares têm chosen 3× mais longo que rejected, e você não pode reanotá-los.
5. As respostas "rejected" contêm erros factuais que o seu modelo nunca cometeria.

<details><summary>Gabarito</summary>

1. **DPO** — o caso de projeto: pares, referência disponível, memória folgada.
2. **KTO** — é exatamente o formato dele: exemplos avulsos rotulados bom/ruim, sem pareamento. Transformar 👍/👎 em pares artificiais (cruzando prompts) degrada o sinal.
3. **ORPO** — funde SFT+preferência sem modelo de referência; metade da memória, um treino só.
4. **SimPO** (normaliza a recompensa pelo comprimento) ou rebalancear os dados. DPO puro aprenderia "curto é ruim" junto com o sinal real.
5. **Nenhum, ainda.** Pares off-policy sobre defeitos inexistentes = gradiente irrelevante e custo de deriva (medido no lab: PPL subiu sem benefício quando o defeito não existia no modo avaliado). Primeiro colete amostras reais do modelo e construa pares dos defeitos que ele DE FATO comete.
</details>

---

### A4. β e o KL

a) Escreva o objetivo original de RLHF e aponte onde β aparece.
b) No lab, β=0,02 deu margem 2,8 e PPL 1326; β=0,5 deu margem 26,4 e PPL 371. As direções parecem invertidas — margem MAIOR com β maior e MENOS dano. O que explica?

<details><summary>Gabarito</summary>

a) `max_π E[r(x,y)] − β·KL(π‖π_ref)` — β é o preço do desvio da referência.

b) A pegadinha: a **margem é medida em unidades de recompensa implícita, que contém o próprio β** — `r̂ = β·Δlogprob`. Com β=0,5, cada nat de deslocamento vale 5× mais margem que com β=0,1. A margem de 26,4 com β=0,5 corresponde a um Δlogprob de ~53 nats; a margem 2,8 com β=0,02 corresponde a ~140 nats — **um deslocamento de política muito maior**, e é por isso que a PPL explodiu (1326 vs 371).

Lição dupla: (1) compare margens entre treinos só depois de dividi-las por β; (2) β pequeno permite deriva grande — como a teoria previa, mas com a armadilha de leitura no meio.
</details>

---

### A5. O reward model escondido

Após um treino de DPO, você tem `π` e `π_ref`. Um colega quer jogar as duas fora e treinar um reward model do zero para rankear respostas candidatas em produção (best-of-n).

Mostre que ele já tem um reward model, como usá-lo, e uma limitação real dessa abordagem.

<details><summary>Gabarito</summary>

O DPO define implicitamente `r̂(x,y) = β·log[π(y|x)/π_ref(y|x)]` — treinado exatamente para reproduzir as preferências do dataset. Para best-of-n: gere n candidatas, compute `r̂` de cada (dois forwards por candidata: política e referência) e escolha a maior. O Lab 4 do lab_mlx faz isso.

Limitações reais:
- **Custo:** dois forwards por candidata (o RM dedicado usa um — e é menor).
- **Extrapolação:** `r̂` é confiável perto da distribuição dos pares de treino; para respostas muito fora dela, o log-ratio mede deriva, não qualidade.
- **Viés de comprimento:** `r̂` é soma sobre tokens — candidatas longas têm variância maior (Lab 6 do lab_cpu).

Para best-of-n leve, o `r̂` implícito resolve; para um ranker de produção de alto volume, um RM dedicado menor costuma vencer em custo.
</details>

---

## Parte B — Práticas

### B1. 💻 IPO vs DPO

Implemente a loss do IPO — `(Δ − 1/(2β))²` em vez de `−log σ(β·Δ)` — no `treinar_dpo` do lab.

Compare com DPO em: margem final, `lp_chosen` final, PPL de validação. O IPO evita a saturação? E a patologia?

<details><summary>Gabarito</summary>

```python
def ipo_loss(lpc_pol, lpr_pol, lpc_ref, lpr_ref, beta=0.1):
    delta = (lpc_pol - lpc_ref) - (lpr_pol - lpr_ref)
    return ((delta - 1 / (2 * beta)) ** 2).mean()
```

O IPO tem um **alvo** de margem (`1/(2β)`) em vez de "quanto maior, melhor": o gradiente não desaparece nos pares resolvidos nem cresce sem limite nos invertidos. Espere margens menores e mais estáveis, `lp_chosen` melhor preservado e PPL final menor — ao custo de menos "força" no sinal. É a escolha certa quando o DPO sobreajusta a preferências ruidosas.
</details>

---

### B2. 💻 Online vs offline

O lab treinou com rejected sintéticos (offline). Implemente a variante **online**: a cada passo, gere o rejected amostrando do próprio modelo atual (T=1,0, top_k=0) em vez de construí-lo.

Compare a degeneração greedy final e a PPL. Qual variante move mais o comportamento real?

<details><summary>Gabarito</summary>

A variante online ataca exatamente o que o modelo **produz agora** — o gradiente é sempre on-policy. Espere: efeito maior no comportamento de geração com menos deriva de PPL, porque a punição incide na distribuição visitada, não numa construção externa.

O custo: gerar a cada passo é caro (é o decode do módulo 1 dentro do loop de treino). É o preço que o RL do módulo 9 paga integralmente — e este exercício é a ponte: online DPO é meio caminho entre DPO e RL.
</details>

---

### B3. 💻 Desfazendo o confundimento de comprimento

O `preparar_dados.py` avisa: o rejected é sempre mais longo (o boilerplate é um anexo). Construa uma versão **controlada**: para cada par, corte o chosen e o rejected para o mesmo número de palavras (removendo do fim do chosen o excedente).

Treine DPO nas duas versões e compare a taxa de boilerplate e o comprimento médio das gerações.

<details><summary>Gabarito</summary>

Com os dados originais, o DPO aprende dois sinais somados: "boilerplate é ruim" + "longo é ruim". A versão controlada isola o primeiro. Espere: taxa de boilerplate caindo nas duas, mas o comprimento médio das gerações encolhendo **só na versão original** — a evidência do sinal espúrio.

É o experimento de ablação mínimo que todo dataset de preferências merece: para cada atributo correlacionado com a preferência, pergunte se você o quer no modelo. Se não, controle-o.
</details>

---

### B4. 🍎 Best-of-n com a recompensa implícita

Use o `r̂` do Lab 4 (lab_mlx) como ranker: para 15 perguntas de validação, gere n=4 candidatas (T=0,9), rankeie por `r̂` e compare a taxa de boilerplate de: (a) primeira candidata, (b) melhor por `r̂`, (c) pior por `r̂`.

<details><summary>Gabarito esperado</summary>

Se o DPO aprendeu o sinal, o ranking por `r̂` deve separar: (b) com taxa perto de zero, (c) concentrando o boilerplate restante. A distância entre (a) e (b) é o ganho do best-of-n — capacidade extra sem treinar mais nada, pagando 4× a geração e 8 forwards de ranking.

Se (b) e (c) não se separam, o `r̂` não generalizou além dos pares de treino — volte ao A5, limitação de extrapolação.
</details>

---

### B5. 🍎 KTO com dados de produção simulados

Simule o cenário do A3.2: pegue 200 respostas geradas pelo modelo (não pares!), rotule automaticamente (👎 se contém sonda de boilerplate, 👍 caso contrário) e treine com `--train-mode kto` (se disponível no mlx-lm-lora) ou construa pares artificiais cruzando prompts como controle.

O KTO com rótulos avulsos chega perto do DPO com pares?

<details><summary>Gabarito esperado</summary>

Espere KTO chegando a boa parte do efeito do DPO — o resultado do paper original (Ethayarajh et al., 2024) é que rótulos avulsos capturam a maior parte do sinal quando o dataset é razoavelmente balanceado.

A comparação honesta exige o mesmo número de *exemplos vistos* (um par DPO = 2 exemplos KTO). E note o que este exercício simula: o pipeline mais barato que existe em produção — logs + um detector automático de defeito + KTO, sem anotação humana nenhuma.
</details>

---

## Desafio — o pipeline de preferências do seu produto

Projete (e execute em miniatura) o ciclo completo de alinhamento para um caso seu:

1. **Escolha um defeito real** de um modelo que você usa (verbosidade, tom, formato quebrado, resposta em inglês quando deveria ser português...).
2. **Detector:** escreva uma função que o detecta automaticamente (regex, comprimento, idioma). Valide o detector à mão em 30 gerações.
3. **Colete on-policy:** gere 300+ respostas do modelo em prompts reais. Meça a taxa base do defeito.
4. **Construa os dados** pela via mais barata que o sinal permitir: pares por corrupção (se o defeito é raro), pares reais bom/ruim (se é frequente), ou rótulos avulsos + KTO.
5. **Audite confundimentos:** comprimento, idioma, formato — correlacionados com a preferência?
6. **Treine, meça, leia.** Taxa do defeito antes/depois, PPL ou métrica de qualidade geral, e leitura manual de 20 gerações.
7. **Relatório honesto:** o alinhamento se pagou? O que mais mudou junto?

O item 7 repete o padrão do módulo 5, e a pergunta "o que mais mudou junto?" é a específica de preferências: DPO nunca move só o que você mira — a auditoria de efeitos colaterais é o que separa alinhamento de dano com boa intenção.
