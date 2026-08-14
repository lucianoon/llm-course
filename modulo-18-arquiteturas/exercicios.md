# Módulo 18 — Exercícios

Todos rodam em CPU. 💻

---

## Parte A — Conceituais

### A1. A escolha da arquitetura

Para cada caso de uso, qual arquitetura (transformer puro, Mamba/SSM, híbrido, atenção linear) e por quê?

1. Um assistente de código que precisa rastrear variáveis por um arquivo de 50k linhas.
2. Transcrição de áudio em streaming (contexto cresce indefinidamente).
3. Um modelo de DNA (sequências de milhões de bases, dependências longas mas difusas).
4. Um chatbot de conversas curtas (contexto < 4k tokens sempre).

<details><summary>Gabarito</summary>

1. **Transformer (ou híbrido com atenção suficiente)** — rastrear uma variável específica através do código é recall preciso puro (Lab 5: atenção 100%, linear 20%). SSM/linear borrariam a referência. Este é o caso onde a atenção é insubstituível.
2. **Mamba/SSM ou híbrido** — contexto que cresce sem limite mata o KV cache do transformer (O(L)); o estado fixo do SSM é a única coisa que roda indefinidamente com memória constante. Streaming é o caso ideal dos SSMs.
3. **Mamba/SSM** — sequências de milhões, dependências longas mas DIFUSAS (não exigem recall de um símbolo exato). O SSM brilha: comprime bem o que não precisa ser recuperado com precisão cirúrgica. Foi um dos primeiros sucessos do Mamba (genômica).
4. **Transformer puro** — contexto curto elimina a vantagem dos eficientes (O(L²) de 4k é trivial), e você quer o recall e a maturidade do transformer. Não complique: para contexto curto, o transformer é imbatível e não tem downside.
</details>

---

### A2. Por que o softmax importa tanto

O Lab 5 mostrou atenção 100% vs linear 20% no recall.

a) Explique mecanicamente por que o softmax permite recall preciso.
b) A atenção linear usa `φ(x) = elu(x)+1`. Por que uma feature positiva, e por que isso não recupera a capacidade do softmax?
c) Que consequência isso tem para in-context learning (módulo 16)?

<details><summary>Gabarito</summary>

a) O softmax é exponencial: `exp(score)`. Uma pequena vantagem no score de uma chave vira uma vantagem ENORME no peso — o softmax "afia" a distribuição até quase one-hot, colocando ~toda a massa na chave que casa. É seleção, não média. Recuperar o value daquela chave é então recuperar o value certo, quase puro.

b) `φ` precisa ser positiva para que `φ(Q)φ(K)ᵀ` seja um "peso" válido (não-negativo) — senão a média ponderada não faz sentido. Mas `elu+1` é aproximadamente LINEAR, não exponencial: ela não afia a distribuição. O peso de cada chave é proporcional (não exponencial) à similaridade, então a recuperação é uma média suave de vários values — borra. A capacidade de "focar" veio justamente da não-linearidade exponencial que a reordenação O(L) força a remover.

c) In-context learning depende de induction heads (módulo 16), que fazem "encontre a ocorrência anterior deste token e copie o que veio depois" — recall preciso puro. Um modelo de atenção linear/SSM puro tem few-shot learning mais fraco, porque o mecanismo de cópia exata é degradado. É uma das evidências mais fortes de por que a atenção sobrevive: ela é o substrato do ICL.
</details>

---

### A3. A conta do contexto de 10M

Um provedor anuncia "contexto de 10 milhões de tokens".

a) Estime o KV cache de um transformer denso de 8B nesse contexto.
b) Por que isso é praticamente inviável com atenção pura, e quais das arquiteturas do módulo tornam viável?
c) Que trade-off o usuário está aceitando sem saber?

<details><summary>Gabarito</summary>

a) ~128 KB/token × 10M = **1,28 TB de KV cache**. Absurdo — nem cabe na memória agregada de um nó de 8 GPUs (640 GB).

b) Inviável com atenção densa pura por dois motivos: o KV cache de 1,28 TB e o compute O(L²) de 10M² = 10¹⁴ operações por camada. Viabilizam: (1) **SSM/híbrido** — estado fixo elimina o cache; (2) **atenção linear/esparsa** — O(L) de compute; (3) **MLA + atenção esparsa/janelas** — comprime o cache e limita o alcance. Modelos de contexto ultralongo real usam combinações disso.

c) **Recall degradado no meio do contexto.** Se o provedor usa SSM/linear/atenção esparsa para viabilizar os 10M, o modelo NÃO recupera com precisão um fato enterrado no token 5.000.000 — ele "lembra" o gist, não o detalhe exato (Lab 5 + lost in the middle do módulo 13). O usuário que espera "achar a agulha no palheiro de 10M tokens" pode se decepcionar. A pergunta certa para qualquer claim de contexto gigante: *qual o recall MEDIDO no meio do contexto?* (o benchmark "needle in a haystack").
</details>

---

### A4. Multimodalidade

Um modelo recebe uma imagem e uma pergunta sobre ela.

a) Como a imagem entra na arquitetura do módulo 2 sem mudá-la?
b) Por que a atenção "mistura as modalidades de graça"?
c) Que problema do módulo 1 (tokenização) reaparece de forma diferente com imagens?

<details><summary>Gabarito</summary>

a) Um encoder de visão (ViT) corta a imagem em patches (16×16 px), projeta cada patch num vetor da MESMA dimensão dos embeddings de texto, e os intercala na sequência: `[emb_texto, emb_texto, emb_patch, emb_patch, ..., emb_texto]`. Para o transformer, são só mais tokens — ele não sabe (nem precisa saber) que alguns vieram de pixels.

b) Porque a self-attention (módulo 2) compara TODOS os tokens entre si, independente da origem. Um token de texto ("o que há na imagem?") atende naturalmente aos tokens de patches, e vice-versa. A mistura cross-modal é o comportamento default da atenção — não precisa de mecanismo especial. É por isso que a mesma arquitetura serve texto, imagem, áudio e vídeo.

c) A **granularidade da tokenização**: quantos tokens uma imagem "vale"? Poucos patches = perde detalhe (o análogo de tokenizar grosso); muitos = contexto explode (o análogo do custo do português no módulo 1). E resoluções variáveis, como o vocabulário fixo, forçam decisões de compressão. É a mesma tensão do módulo 1 — quanta informação por token — num domínio novo.
</details>

---

### A5. O transformer foi destronado?

A mídia técnica anuncia "o fim do transformer" a cada nova arquitetura. Escreva a resposta honesta e nuançada, usando os resultados do lab.

<details><summary>Gabarito</summary>

**Não — foi complementado e otimizado, não substituído.** O argumento, com evidência:

1. **A atenção continua insubstituível em recall preciso** (Lab 5: 100% vs 20%). Nenhum mecanismo O(L) iguala isso, e recall é a base do ICL, do rastreamento de referências e da recuperação de fatos. Um modelo sem atenção nenhuma é mensuravelmente pior nessas tarefas.
2. **Os "substitutos" na prática são híbridos** (Lab 6: ~12% de atenção captura o recall). Jamba, Nemotron-H etc. INCLUEM atenção — eles não a abandonaram, dosaram-na.
3. **O que mudou é real e importante:** aprendeu-se a não pagar O(L²) em TODA camada. Contexto longo barato veio de SSMs, MLA, atenção esparsa. Isso é progresso genuíno.

A formulação madura: o transformer de 2017 era "atenção em todo lugar"; a fronteira de 2026 é "atenção onde precisa, mecanismos O(L) no resto". A peça central do módulo 2 sobrevive sete anos — o que é, por si, notável. Desconfie de todo anúncio de "fim do X": em arquitetura de deep learning, o padrão é acumulação e hibridização, não substituição limpa.
</details>

---

## Parte B — Práticas

### B1. 💻 O Mamba seletivo

Estenda o SSM do Lab 2 para ser SELETIVO: faça B e C dependerem da entrada `x` (uma projeção linear de x → B_t, C_t por token). Compare o recall associativo do SSM fixo vs seletivo no teste do Lab 5.

<details><summary>Gabarito esperado</summary>

O SSM seletivo deve melhorar o recall sobre o fixo — a seletividade é exatamente o que permite ao modelo "decidir lembrar" a chave relevante. Mas ainda ficará ABAIXO da atenção: comprimir num estado fixo tem um teto de recall que a seletividade eleva, não remove.

É a lição do Mamba em miniatura: a seletividade foi o que tornou SSMs competitivos (sem ela, são RNNs fracas), mas não fecha totalmente o gap de recall — daí os híbridos. Você acabou de reproduzir a motivação de um paper inteiro num experimento de laptop.
</details>

---

### B2. 💻 A curva de crossover

Meça o ponto exato de L onde a atenção linear fica mais rápida que a quadrática, para dimensões `d` = 16, 64, 256. Plote L_crossover × d.

Por que o crossover depende de d?

<details><summary>Gabarito esperado</summary>

O crossover acontece quando o custo O(L²·d) da quadrática iguala o O(L·d²) da linear — ou seja, em `L ≈ d`. Para d pequeno, a linear ganha cedo (L baixo); para d grande, o termo d² da linear a atrasa, e o crossover sobe.

A lição prática: atenção linear só compensa quando `L ≫ d`. Em contextos curtos ou dimensões grandes, a quadrática é mais rápida (menos overhead). É por isso que atenção linear brilha em contexto ultralongo e é irrelevante em contexto curto — e por que ninguém a usa para chatbots de 2k tokens.
</details>

---

### B3. 💻 MLA do zero

Implemente a compressão do MLA: uma matriz `W_down` que projeta o KV para um latente de dimensão `r`, e `W_up` que reexpande. Meça o erro de reconstrução da atenção (vs KV cheio) para r = 128, 256, 512, e o recall associativo em cada r.

Qual r preserva o recall com a maior compressão?

<details><summary>Gabarito esperado</summary>

Espere: r maior = menos erro de reconstrução = melhor recall, com retorno decrescente. Há um "joelho" onde comprimir mais começa a doer no recall — e o DeepSeek escolheu r em torno de 512 justamente ali.

O ponto conceitual: MLA é uma aposta de que o KV vive num subespaço de baixa dimensão (a mesma hipótese de baixo posto do LoRA, módulo 6!). Se vive, comprimir é quase grátis; o r ótimo é onde essa hipótese começa a falhar. É bonito ver a mesma ideia (baixo posto) reaparecer — LoRA nos pesos, MLA no cache.
</details>

---

### B4. 💻 O híbrido, medido de verdade

Construa um mini-modelo híbrido: intercale camadas de atenção (do módulo 2) e de SSM (do Lab 2) num MiniGPT, treine no corpus de Machado (módulo 3), e compare a perplexidade e a velocidade com um transformer puro do mesmo tamanho.

<details><summary>Gabarito esperado</summary>

Num modelo e corpus de brinquedo, espere perplexidade SEMELHANTE (a tarefa não estressa o recall longo) e o híbrido mais rápido em contexto longo. O experimento não vai mostrar a vantagem completa — os benefícios do híbrido aparecem em escala e contexto que o laptop não alcança.

Mas o valor é montar a arquitetura e ver que ela TREINA — que intercalar mecanismos diferentes funciona, que o gradiente flui pelos dois. É a prova de conceito que antecede qualquer experimento sério, e o tipo de coisa que você faria antes de escalar. Documente honestamente o que a escala de brinquedo NÃO consegue mostrar (a regra do curso).
</details>

---

## Desafio — o mapa da fronteira

Este é o módulo mais próximo da pesquisa ativa. Produza um **mapa vivo da fronteira** (que você atualizará):

1. **A tabela de trade-offs:** para cada arquitetura (transformer, Mamba, linear, MLA, híbrido), as colunas: complexidade de compute, memória de estado, recall, maturidade, quem usa em produção.
2. **Leia UM paper de 2025-26** da fronteira (a lista de leituras, ou um mais recente que você achar) e adicione-o ao mapa: o que ele ataca, o que sacrifica, onde se encaixa.
3. **A previsão:** com base nos trade-offs, o que você aposta para 2027? Contexto de 100M vira comum? Híbridos dominam? Algo além de atenção resolve o recall? Escreva a aposta com a justificativa.
4. **O teste da sua compreensão:** se um paper novo anuncia "X% mais rápido que o transformer", quais três perguntas você faz antes de acreditar? (Dica: recall medido, em que L, contra qual baseline — os módulos 14 e 18 juntos.)

Este mapa é o que separa quem ACOMPANHA a fronteira de quem só ouviu falar dela. E é o aquecimento perfeito para a Fase 3, que começa lendo e reproduzindo exatamente esses papers.
