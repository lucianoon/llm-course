# Módulo 14 — Avaliação como disciplina

> **Pergunta central:** quantas amostras você precisa para afirmar que o modelo A é melhor que o B — e quantas conclusões da área (e deste curso) sobrevivem a essa pergunta?

Todo módulo do curso terminou numa medição. Este módulo pergunta: **quando uma medição vira conhecimento?** A resposta é estatística elementar que quase ninguém na área aplica — e o lab a aplica contra as conclusões do próprio curso, com resultado desconfortável e instrutivo.

É, na minha leitura, a habilidade mais escassa da área: gente que treina modelo há muita; gente que sabe dizer *com que confiança* um modelo é melhor que outro, pouca.

## Objetivos

1. Tratar toda acurácia como o que ela é: uma **estimativa** com intervalo de confiança.
2. Usar bootstrap para qualquer métrica, e testes pareados para comparações — entendendo por que parear multiplica o poder estatístico.
3. Dimensionar conjuntos de avaliação: quantos exemplos para detectar uma diferença de X pontos?
4. Medir os vieses do LLM-as-judge em vez de citá-los.
5. Reconhecer a inflação por comparações múltiplas — a fraude estatística acidental mais comum da área.
6. Medir calibração (o modelo sabe quando não sabe?).

---

## 1. Acurácia é uma estimativa

"O modelo acertou 65% das 100 perguntas" **não** significa que a acurácia do modelo é 65%. Significa que uma moeda com viés desconhecido deu 65 caras em 100 lançamentos. A pergunta certa: que vieses são compatíveis com essa observação?

### 📐 O intervalo de confiança binomial

Erro-padrão de uma proporção: `√(p(1−p)/n)`. Com p≈0,65 e n=100: **±4,8 pontos** por desvio — o IC de 95% vai de ~55% a ~74%. Uma "melhora de 3 pontos" medida em 100 exemplos é indistinguível de ruído.

A tabela que deveria estar colada em todo monitor da área:

| n | IC 95% de uma acurácia ~65% | Menor Δ detectável (não pareado) |
|---|---|---|
| 25 | ±19 pp | ~26 pp |
| 100 | ±9 pp | ~13 pp |
| 500 | ±4 pp | ~6 pp |
| 2.000 | ±2 pp | ~3 pp |
| 10.000 | ±0,9 pp | ~1,3 pp |

Medido no Lab 1 (dois modelos com acurácias VERDADEIRAS de 62% e 65%): com n=25, o ranking observado sai **invertido em 36% das avaliações**; com n=100, em 31,5%; até n=500 erra 15% das vezes. Toda decisão "rodei 100 exemplos e A deu 2 pontos a mais" carrega essa taxa de erro embutida.

**E a auditoria do próprio curso (Lab 3):** o "densa 92% > BM25 84%" do módulo 13, testado com o rigor devido — 3 discordâncias a favor da densa, 1 contra; McNemar exato **p = 0,625**; bootstrap pareado da diferença: **+8pp, IC95 [−8pp, +24pp]**. O IC cruza zero com folga. A conclusão honesta rebaixada: *"a densa venceu nesta amostra; n=25 não distingue os sistemas"*. Nada no módulo 13 estava errado — o que muda é a força da afirmação permitida.

**As outras auditorias do lab, medidas:**
- **O juiz de 0.5B é um selecionador de posição puro:** 50% de acurácia em pares com gabarito e **0% de consistência entre ordens** — ele escolhe a mesma letra sempre, independente do conteúdo. Qualquer win rate dele é ruído com formato de número.
- **Melhor-de-k:** o melhor de 20 checkpoints IDÊNTICOS parece **+6,2pp** melhor que a verdade (p95: +9pp) em n=200. De 100: +8,3pp. Toda seleção olhando o conjunto de avaliação compra um pedaço disso.
- **Calibração do 0.5B:** 33% de acurácia com 68% de confiança média — **+35pp de superconfiança**, incluindo uma resposta com 99% de confiança... errada.

### 📐 Bootstrap — o canivete universal

Para métricas sem fórmula fechada (MRR, win rate com empates, médias truncadas): reamostre o conjunto de avaliação **com reposição** milhares de vezes, recompute a métrica em cada reamostra, e os percentis 2,5/97,5 da distribuição são o IC. Três linhas de código, vale para qualquer métrica, e é o padrão de facto (usado pelo lm-eval-harness e pelos leaderboards sérios).

---

## 2. Comparações: pareie, sempre

O erro estrutural mais comum: avaliar A e B e comparar as acurácias como números soltos. Se A e B respondem **as mesmas perguntas**, a comparação certa é por pergunta — e a diferença de poder é dramática.

A intuição: modelos parecidos acertam e erram JUNTOS na maioria das perguntas — e essa concordância, que o teste não pareado paga como ruído, o pareado desconta. Medido no Lab 2 (Δ real de 3pp, n=500):

| Correlação entre os modelos | Não pareado detecta | McNemar detecta |
|---|---|---|
| 0,0 (independentes) | 15,1% | 17,9% |
| 0,5 | 7,7% | 28,3% |
| 0,8 | 2,1% | 50,6% |
| **0,95 (regime real)** | **0,2%** | **87,7%** |

**Trezentas vezes mais poder, dos mesmos dados** — e a fonte do poder é a correlação, não o pareamento em si (na linha 0,0 os testes empatam). Dois checkpoints do mesmo modelo, duas versões de prompt, dois sistemas de RAG sobre o mesmo índice: tudo isso vive na última linha.

📐 As ferramentas pareadas:

- **Bootstrap pareado:** reamostre *perguntas* (não respostas); em cada reamostra compute `acc_A − acc_B`. IC da diferença que exclui zero = significativo.
- **Teste de McNemar:** só as perguntas em que os modelos **discordam** carregam informação. Se A acerta onde B erra `b` vezes e o inverso `c` vezes, sob a hipótese nula `b ~ Binomial(b+c, ½)`. As perguntas em que ambos acertam ou ambos erram são plateia.

> 🔧 A consequência prática do McNemar é um princípio de design: **conjuntos de avaliação ganham poder pelas perguntas que discriminam** — as que um modelo típico acerta e outro erra. Perguntas fáceis demais ou impossíveis são custo sem informação (o mesmo princípio da vantagem de grupo do GRPO, módulo 9 — grupos unânimes não ensinam nada).

---

## 3. LLM-as-judge com rigor

O juiz neural (módulos 4–5) tem vieses conhecidos. A diferença deste módulo: **medi-los no seu juiz** antes de confiar em qualquer win rate.

O protocolo mínimo de auditoria, que o lab executa:

1. **Pares com gabarito** — monte comparações em que você SABE qual resposta é melhor (correta vs corrompida). A acurácia do juiz nesses pares é o teto de confiança dele.
2. **Viés de posição** — avalie cada par nas duas ordens. A taxa de inconsistência (juiz muda de ideia quando a ordem inverte) é ruído puro; win rates dentro dessa faixa não significam nada.
3. **Viés de comprimento** — nos pares em que a resposta pior é mais longa, o juiz cai quanto?
4. **Concordância com humanos** (quando houver anotação): kappa de Cohen, não acurácia crua — o acaso concorda 50% do tempo sozinho.

E a regra de reporte honesto (módulo 5, agora com fórmula): *"win rate de X% sobre os Y% de pares em que o juiz foi consistente, juiz com Z% de acurácia em pares com gabarito"*. Sem os três números, um win rate é marketing.

---

## 4. Comparações múltiplas — a fraude acidental

Você treina 20 checkpoints, avalia todos em 200 exemplos e reporta o melhor. Parabéns: **você acabou de fabricar uma melhora.**

📐 Com 20 medições de um MESMO modelo (diferenças só de ruído), o máximo esperado fica ~1,4 desvios acima da média — com EP de ±3,4 pp em n=200, o "melhor checkpoint" parece ~5 pp melhor **por sorteio**. O lab simula exatamente isso.

O mesmo mecanismo infla: seleção de melhor prompt, melhor seed, melhor temperatura, melhor época — toda escolha feita OLHANDO para o conjunto de avaliação. Defesas, em ordem de rigor:

1. **Dois conjuntos:** desenvolvimento (para escolher) e teste lacrado (para reportar — abre-se UMA vez, módulo 4).
2. Correção de Bonferroni (dividir o α pelo número de comparações) quando os dois conjuntos não são viáveis.
3. E a versão social do problema: **benchmarks públicos saturam** porque a comunidade inteira é um grande loop de seleção sobre o mesmo conjunto de teste. Um modelo que lidera o benchmark X por 1 ponto não é melhor; é o vencedor de um sorteio com milhares de participantes (Goodhart, módulo 9, em escala civilizacional).

---

## 5. Calibração — o modelo sabe quando não sabe?

Dois modelos com 70% de acurácia podem ser radicalmente diferentes: um diz "90% de certeza" quando acerta e "55%" quando erra (calibrado — a confiança É informação); o outro diz "95%" sempre (não calibrado — a confiança é ruído).

📐 **ECE (Expected Calibration Error):** agrupe as previsões por faixa de confiança e compare a confiança média de cada faixa com a acurácia real nela; a média ponderada dos desvios é o ECE. O diagrama de confiabilidade é a versão visual.

Por que importa em produto: roteamento por incerteza (módulo 7 — casos incertos sobem para o modelo grande), abstenção (módulo 13), e priorização de revisão humana — tudo depende de a confiança significar algo. E o achado geral da literatura: **RLHF descalibra** — modelos alinhados ficam superconfiantes em relação aos base. Confiança verbalizada ("tenho 90% de certeza") é ainda menos confiável que a probabilística.

(E note o espelho: o METODO-DE-ESTUDO pede a SUA calibração nos exercícios — confiança antes do gabarito. A métrica é a mesma; o modelo é você.)

---

## 6. O sistema de avaliação de um produto sério

A pirâmide, da base ao topo:

```
   produção: monitoramento contínuo         ← amostras reais + juiz auditado + alarmes
   regressão: o golden set no CI            ← 50–200 casos LACRADOS; roda a cada mudança
   desenvolvimento: o conjunto de trabalho  ← onde se escolhe prompt/modelo/hiperparâmetro
   capacidade: benchmarks públicos          ← só para escolher o modelo base; saturados e contaminados
```

As regras que o curso já praticou, agora nomeadas:

- **A métrica antes do sistema** (módulos 5, 13) — quem define a métrica depois de ver os resultados escolhe a métrica que os favorece.
- **Teste a métrica contra exemplos de ouro** antes de avaliar qualquer modelo (módulo 5) — inclusive o juiz (seção 3) e a extração (módulo 7).
- **Inspecione os erros manualmente** — a taxa de falsos erros da extração (módulo 7) e o reward hacking invisível na curva (módulo 9) só aparecem no olho.
- **Tudo que decide precisa de IC** — e tudo que se reporta, do tamanho do conjunto.

---

## 7. Leituras

1. **Miller (2024), "Adding Error Bars to Evals"** — [arXiv:2411.00640](https://arxiv.org/abs/2411.00640). O manifesto (da Anthropic) pelo básico estatístico em avaliação de LLMs; este módulo é primo dele.
2. **Zheng et al. (2023), "Judging LLM-as-a-Judge"** — [arXiv:2306.05685](https://arxiv.org/abs/2306.05685). Os vieses do juiz, medidos.
3. **Liang et al. (2022), "HELM"** — [arXiv:2211.09110](https://arxiv.org/abs/2211.09110). O que avaliação multidimensional séria exige.
4. **Dietterich (1998), "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"** — o clássico do McNemar em ML.
5. **lm-evaluation-harness** (EleutherAI, [github](https://github.com/EleutherAI/lm-evaluation-harness)) — a ferramenta padrão; leia como ela reporta erro-padrão.

---

## 8. Checklist de saída

- [ ] Qual o IC de 95% de uma acurácia de 65% medida em 100 exemplos? E em 25?
- [ ] Por que parear as comparações multiplica o poder estatístico? O que cancela?
- [ ] No McNemar, quais perguntas carregam informação — e que princípio de design isso implica?
- [ ] Descreva o bootstrap em três passos e diga quando ele é a única opção.
- [ ] O protocolo mínimo de auditoria de um juiz, em quatro medições.
- [ ] Por que "o melhor de 20 checkpoints" é uma melhora fabricada, e qual a defesa estrutural?
- [ ] O que o ECE mede, e por que calibração vale dinheiro em produto?
- [ ] Complete com os três números: "win rate de X% sobre..."
- [ ] A conclusão "densa > BM25" do módulo 13 sobreviveu ao teste pareado com n=25? O que isso ensina?

Depois: `lab_cpu.py` (executado — inclusive a auditoria das conclusões do próprio curso), exercícios, e os cartões novos.
