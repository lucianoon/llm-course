# Módulo 8 — Alinhamento por preferências (DPO)

> **Pergunta central:** como ensinar um modelo a *preferir* uma resposta a outra — quando as duas estão corretas?

O SFT (módulo 5) tem um limite estrutural: ele só sabe **imitar um alvo**. Se duas respostas são gramaticais, factuais e no formato certo, mas uma é seca e a outra é útil, o SFT não tem como codificar essa diferença — ele nunca vê a resposta ruim. Preferências veem as duas.

## Objetivos

1. Explicar o pipeline RLHF clássico e o que o DPO eliminou dele.
2. Derivar a loss do DPO a partir do objetivo do RLHF — a derivação inteira, sem pular passos.
3. Implementar DPO do zero e treinar com ele.
4. Diagnosticar as patologias clássicas: ambas as log-probs caindo, viés de comprimento.
5. Construir um dataset de preferências e saber de onde vêm os pares.
6. Escolher entre DPO, ORPO, KTO e SimPO com critério.

---

## 1. De onde isso veio: RLHF em três estágios

O pipeline do InstructGPT (2022), que produziu o ChatGPT:

```
1. SFT           →  modelo que segue instruções           (módulo 5)
2. Reward model  →  treinar um MODELO para prever qual resposta humanos preferem
3. PPO           →  RL: maximizar a recompensa prevista, sem se afastar do SFT
```

O estágio 2 usa pares: humanos veem duas respostas e escolhem. O **modelo de recompensa** (RM) aprende a atribuir um escalar `r(x, y)` que reproduz essas escolhas. O estágio 3 otimiza:

```
max_π  E[ r(x, y) ]  −  β · KL(π ‖ π_ref)
```

Maximize a recompensa, **mas** fique perto do modelo de referência (o SFT). O termo KL é essencial: sem ele, o modelo acha os pontos cegos do RM — respostas com recompensa alta e qualidade real péssima (*reward hacking*).

O problema: PPO é caro e instável. Quatro modelos na memória ao mesmo tempo (policy, referência, RM, value), hiperparâmetros sensíveis, e uma infraestrutura que poucas equipes dominam.

---

## 2. 📐 A derivação do DPO

O insight de Rafailov et al. (2023): **o problema de RL acima tem solução em forma fechada, e ela permite eliminar o RM e o RL inteiros.**

### Passo 1 — a solução ótima do objetivo KL-regularizado

Para o objetivo `max_π E[r] − β·KL(π‖π_ref)`, a política ótima é conhecida (resultado clássico de inferência variacional):

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)
```

A política ótima é a referência, **reponderada exponencialmente pela recompensa**. `Z(x)` normaliza — e é intratável (soma sobre todas as respostas possíveis).

### Passo 2 — inverter: expressar a recompensa pela política

Tomando log e isolando `r`:

```
r(x,y) = β · log[ π*(y|x) / π_ref(y|x) ]  +  β·log Z(x)
```

A recompensa é o **log-ratio entre a política e a referência**, escalado por β — mais um termo que só depende de `x`.

### Passo 3 — a mágica: Z(x) cancela

O modelo de preferências de **Bradley-Terry** diz que a probabilidade de humanos preferirem `y_w` (chosen) a `y_l` (rejected) é:

```
P(y_w ≻ y_l) = σ( r(x,y_w) − r(x,y_l) )
```

Só a **diferença** de recompensas importa. Substituindo o passo 2, o termo `β·log Z(x)` aparece nos dois lados e **cancela**:

```
P(y_w ≻ y_l) = σ( β·log[π(y_w)/π_ref(y_w)] − β·log[π(y_l)/π_ref(y_l)] )
```

### Passo 4 — a loss

Maximizar a verossimilhança das preferências observadas = minimizar:

```
L_DPO = −E[ log σ( β·(log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)) ) ]
```

**Quatro quantidades, todas computáveis com forwards simples:** log-prob do chosen e do rejected, na política e na referência congelada. Sem RM, sem sampling, sem PPO. O RL virou uma classificação binária.

### A recompensa implícita

O subproduto conceitual mais importante: o DPO define

```
r̂(x,y) = β · log[ π(y|x) / π_ref(y|x) ]
```

**Seu modelo é secretamente um reward model.** O quanto a política aumentou a probabilidade de uma resposta em relação à referência *é* a recompensa que ela atribui. O Lab mede isso diretamente, e a **margem** `r̂(y_w) − r̂(y_l)` é a métrica de progresso do treino.

### O papel do β

`β` controla o cabo de guerra entre maximizar a margem e ficar perto da referência:

- **β pequeno** (0,01–0,05): margens grandes, modelo se afasta muito — risco de degenerar.
- **β grande** (0,3–1,0): quase não sai da referência — treino seguro e fraco.
- **Padrão: 0,1.** Comece aí.

Medido no Lab 5 (MiniGPT, 150 passos):

| β | Margem final | log P(chosen) final | PPL validação |
|---|---|---|---|
| 0,02 | 2,77 | −233,9 | **1.326** |
| 0,10 | 9,68 | −207,7 | 705 |
| 0,50 | 26,37 | −184,1 | **371** |

> ⚠️ **A armadilha de leitura:** a margem *parece* maior com β maior — mas a margem é medida em unidades de recompensa implícita, **que contém o próprio β** (`r̂ = β·Δlogprob`). Dividindo por β: o treino com β=0,02 deslocou a política ~140 nats; o com β=0,5, ~53. O β pequeno permitiu a maior deriva — exatamente o que a teoria prevê, e a PPL de 1.326 é o preço. Nunca compare margens entre treinos com β diferentes sem normalizar.

---

## 3. As patologias — o que olhar no treino

### Ambas as log-probs caem

O fenômeno mais contraintuitivo do DPO, e o lab o reproduz: **a log-prob do chosen também cai** durante o treino. A loss só exige que a *diferença* cresça — e frequentemente o caminho de menor resistência é derrubar o rejected muito mais rápido do que o chosen, com os dois caindo.

Medido (Lab 3, β=0,1):

| Passo | log P(chosen) | log P(rejected) | Margem |
|---|---|---|---|
| 0 | −163,7 | −190,6 | 0,00 |
| 75 | −184,4 | −330,4 | 11,00 |
| 149 | **−207,7** | −328,5 | 9,68 |

O chosen perdeu 44 nats enquanto a margem subia — treino "bem-sucedido" pela loss, com a política fugindo dos dois lados do par. Não é necessariamente um bug (a massa migra para respostas parecidas com o chosen), mas se o chosen despenca, o modelo está abandonando a distribuição dos seus dados. Monitoramento padrão: log-probs de chosen e rejected **separadas**, mais a margem — nunca só a loss.

### O experimento completo do lab, em uma tabela

Preferência sintética: chosen = continuação real de Machado; rejected = degeneração repetitiva. A sutileza que quase arruinou o experimento: a degeneração é um fenômeno do **decoding** — a base degenera em 100% das continuações greedy e 0% das amostradas (Holtzman et al., módulo 1). A avaliação tem que medir o modo onde o defeito vive:

| | Degen. greedy | Degen. sampling | PPL validação |
|---|---|---|---|
| Base | **100%** | 0% | 222,6 |
| Após DPO | **43%** | 7% | 705,1 |

O DPO cortou o comportamento punido em mais da metade — pagando deriva real de PPL (o β=0,1 da tabela acima; com β=0,5 a PPL final é 371). As duas colunas juntas são a avaliação completa: efeito no alvo e custo na capacidade.

### Viés de comprimento

Se os seus pares têm chosen sistematicamente mais longo que rejected (comum: anotadores preferem respostas detalhadas), o DPO aprende **"longo é melhor"** — e o modelo fica verboso. É o viés mais documentado em modelos alinhados. Diagnóstico: correlação entre comprimento e preferência no dataset, **antes** de treinar. Mitigações: balancear comprimentos nos pares, ou SimPO (normaliza por comprimento).

### Distribuição fora da política

Os pares de preferência foram gerados por *algum* modelo (frequentemente outro). Se a política atual nunca geraria aquelas respostas, o gradiente empurra em regiões irrelevantes. Por isso o receituário: **SFT primeiro, DPO depois, sobre respostas parecidas com o que o modelo já produz** — e é também a motivação dos métodos *online* (módulo 9).

---

## 4. De onde vêm os pares

| Fonte | Custo | Qualidade | Nota |
|---|---|---|---|
| Anotação humana (estilo InstructGPT) | Alto | Alta | O padrão-ouro; caro de escalar |
| **LLM-as-judge sobre amostras do próprio modelo** | Baixo | Boa | RLAIF; o mais comum hoje |
| Heurística verificável (testes passam / formato válido) | ~zero | Alta no que mede | Quando existe, use |
| Datasets públicos (UltraFeedback, HH-RLHF) | zero | Variável | Cuidado com o gap de distribuição (seção 3) |
| **Sintético controlado: corromper o chosen** | ~zero | Alta para o alvo | Ensina "não faça X" com precisão cirúrgica |

A última linha é a mais subestimada e é a que os labs usam: se você quer eliminar um comportamento específico (boilerplate, formato quebrado, resposta na língua errada), gere o rejected **corrompendo o chosen com exatamente aquele defeito**. O par isola o sinal; nada mais difere.

---

## 5. A família de variantes

| Método | Diferença | Quando usar |
|---|---|---|
| **DPO** | O baseline derivado acima | Padrão; comece aqui |
| **IPO** | Loss quadrática em vez de log-sigmoid — não satura | Quando o DPO sobreajusta às preferências |
| **KTO** | Não precisa de *pares* — só exemplos bons e ruins avulsos | Quando você tem 👍/👎 soltos (logs de produção!) |
| **ORPO** | Funde SFT + preferência numa loss só, **sem referência** | Metade da memória; bom no M4 |
| **SimPO** | Sem referência + normalização por comprimento | Ataca o viés de comprimento |

> 🔧 No seu M4, a economia do ORPO/SimPO importa: DPO mantém **dois** modelos na memória (política + referência congelada). Para um 1.5B em bf16, são ~6 GB só de pesos antes de qualquer gradiente. ORPO corta isso pela metade.

---

## 6. DPO no pipeline completo

```
pré-treino  →  SFT  →  DPO
 (módulo 3)   (módulo 5)  (este)
```

- **SFT ensina o quê fazer** (formato, comportamento). DPO ensina **o que preferir** entre alternativas plausíveis.
- DPO **não** instala capacidades — ele desloca massa entre respostas que o modelo já produz. Se nenhuma amostra do modelo é boa, não há o que preferir; volte ao SFT.
- A ordem importa: DPO direto no modelo base (sem SFT) funciona mal — as respostas preferidas estão fora da distribuição do modelo.
- Poucos dados bastam: **5k–50k pares** é a faixa típica; o LIMA das preferências também existe.

---

## 7. Leituras

1. **Rafailov et al. (2023), "Direct Preference Optimization"** — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290). A derivação da seção 2 está na seção 4 do paper; compare.
2. **Ouyang et al. (2022), "InstructGPT"** — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155). O pipeline que o DPO simplificou.
3. **Hong et al. (2024), "ORPO"** — [arXiv:2403.07691](https://arxiv.org/abs/2403.07691).
4. **Meng et al. (2024), "SimPO"** — [arXiv:2405.14734](https://arxiv.org/abs/2405.14734). A seção sobre viés de comprimento vale sozinha.
5. **Razin et al. (2024), "Unintentional Unalignment in DPO"** — sobre o fenômeno das log-probs caindo.

---

## 8. Checklist de saída

- [ ] O que o DPO eliminou do pipeline RLHF, e à custa de qual suposição?
- [ ] Derive: por que `Z(x)` cancela no Bradley-Terry?
- [ ] O que é a recompensa implícita, e como monitorá-la no treino?
- [ ] O que β pequeno demais causa? E grande demais?
- [ ] Por que a log-prob do chosen pode CAIR num treino saudável? Quando é sinal de problema?
- [ ] Seu dataset tem chosen 2× mais longo que rejected em média. O que vai acontecer, e como evitar?
- [ ] Você tem 30k 👍/👎 avulsos de produção, sem pares. Qual método?
- [ ] Por que DPO sem SFT prévio funciona mal?
- [ ] Por que ORPO usa metade da memória do DPO?

Depois: `lab_cpu.py` (executado e validado), `preparar_dados.py`, `lab_mlx.py`.
