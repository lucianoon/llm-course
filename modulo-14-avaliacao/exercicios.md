# Módulo 14 — Exercícios

💻 = qualquer máquina | 🍎 = Mac

---

## Parte A — Conceituais

### A1. O leitor de papers

Um paper reporta: "nosso método atinge 71,2% contra 69,8% da baseline no benchmark X (n=500)".

a) Calcule o erro-padrão de cada acurácia e o IC aproximado da diferença NÃO pareada.
b) A diferença é significativa? Que informação faltante mudaria sua resposta?
c) Que três perguntas você faria aos autores antes de acreditar?

<details><summary>Gabarito</summary>

a) EP ≈ √(0,7·0,3/500) ≈ 2,0pp cada. A diferença (1,4pp) tem EP combinado ≈ √(2·2,0²) ≈ 2,9pp → IC95 da diferença ≈ [−4,3, +7,1] pp.

b) **Não** — o IC cruza zero com folga. O que mudaria: se a comparação for PAREADA (mesmas 500 perguntas) e eles reportarem McNemar/bootstrap pareado, 1,4pp pode ser significativo com as mesmas 500 amostras (Lab 2: o pareamento multiplica o poder). A ausência do teste pareado num setup obviamente pareável é o sinal de alerta.

c) 1) "A comparação é pareada? Cadê o teste?" 2) "Quantas configurações/checkpoints/prompts vocês avaliaram antes de escolher este?" (Lab 5 — a inflação de melhor-de-k) 3) "O conjunto de teste foi usado durante o desenvolvimento?" (contaminação de decisão, módulo 4 + Lab 5).
</details>

---

### A2. O McNemar de cabeça

Modelos A e B nas mesmas 400 perguntas: ambos acertam 280; ambos erram 60; A✓/B✗ = 20; A✗/B✓ = 40.

a) Quais são as acurácias?
b) Rode o McNemar mentalmente (z = (b−c)/√(b+c)). Significativo?
c) Agora um colega diz: "só 60 perguntas importaram das 400 — que desperdício". Corrija-o.

<details><summary>Gabarito</summary>

a) A: (280+20)/400 = 75%. B: (280+40)/400 = 80%.

b) z = (40−20)/√60 ≈ 20/7,75 ≈ **2,58** > 1,96 → significativo a 5% (p≈0,01). Cinco pontos com n=400, pareado, fecha; não pareado, o EP da diferença seria ~3pp e ficaria no limite.

c) As 340 perguntas "de plateia" não foram desperdício — elas foram necessárias PARA ENCONTRAR as 60 que discriminam (você não sabia de antemão quais seriam). Mas a lição de design é real: se você conseguir prever quais perguntas discriminam (difíceis-mas-possíveis, a fronteira do módulo 9), um conjunto de avaliação MENOR e mais discriminativo tem o mesmo poder — é exatamente por isso que benchmarks modernos (GPQA, HLE) são construídos adversarialmente contra modelos existentes.
</details>

---

### A3. O win rate desmontado

Um relatório interno: "nosso modelo novo vence o antigo em 58% das comparações (juiz: GPT-4, n=300 pares)".

Aplique o protocolo da seção 3: liste tudo o que precisa ser perguntado, na ordem, com o que cada resposta invalida.

<details><summary>Gabarito</summary>

1. **O juiz foi auditado em pares com gabarito?** Sem isso, não há teto de confiança. Se o juiz acerta 75% em pares óbvios, um win rate de 58% carrega ±erro do próprio termômetro.
2. **Cada par foi avaliado nas duas ordens?** Se não: viés de posição não medido — 58% pode ser "o novo apareceu mais vezes na posição A". Se sim: qual a taxa de inconsistência? Win rate dentro da faixa de inconsistência = ruído.
3. **IC do win rate:** EP = √(0,58·0,42/300) ≈ 2,8pp → IC ≈ [52,4%, 63,6%]. Exclui 50%, ok — MAS só depois dos itens 1–2.
4. **Empates:** foram contados como o quê? 58% vencendo com 30% de empates descartados é diferente de 58% em decisões forçadas.
5. **Viés de comprimento:** o modelo novo escreve mais longo? Se sim, parte do win rate é a preferência do juiz por comprimento (módulo 5), não qualidade.
6. **Quem escreveu os prompts de teste** e quantas versões do modelo novo foram avaliadas antes desta (Lab 5)?

Formato honesto final: "win rate de X% [IC], sobre os Y% de pares consistentes entre ordens, com juiz de Z% de acurácia em gabarito, empates = W%".
</details>

---

### A4. Desenho de conjunto de avaliação

Você vai criar o golden set de regressão de um assistente (módulo 12/13). Orçamento: 150 casos. Distribua-os e justifique com os conceitos do módulo.

<details><summary>Gabarito (uma resposta defensável)</summary>

- **~60 casos da fronteira** — os que o modelo atual acerta ~50–80% das vezes: são os que discriminam regressões (McNemar: só discordâncias informam). Fontes: os erros reais de produção corrigidos, os quase-erros.
- **~30 casos fáceis "de fumaça"** — que o modelo SEMPRE acerta: qualquer falha neles é alarme de quebra grosseira (template, EOS, grounding perdido). Baixa informação estatística, alto valor de alarme.
- **~30 casos de abstenção/segurança** — fora da base, adversariais, pedidos que devem ser recusados: medem o comportamento que ninguém testa até dar errado.
- **~30 casos de cauda** — idiomas, formatos raros, entradas longas: onde quantização e updates degradam primeiro (módulo 6: o dano é assimétrico).

E as regras: lacrado (ninguém otimiza olhando para ele), versionado, com gabarito executável (verificação automática onde possível), e IC reportado — com n=150, mudanças de menos de ~8pp num subgrupo são invisíveis; alarmes devem disparar por CASO crítico, não só por média.
</details>

---

### A5. A calibração que vale dinheiro

Seu assistente jurídico tem 78% de acurácia. Advogados revisam 100% das saídas. Com a confiança calibrada do modelo, você propõe revisar só os casos de confiança < 90%.

a) Que medição decide se a proposta é segura?
b) Qual o risco se o modelo for superconfiante como os do Lab 6?
c) Por que a confiança VERBALIZADA ("tenho 90% de certeza") não serve?

<details><summary>Gabarito</summary>

a) A **acurácia condicional na faixa de alta confiança**: dos casos com confiança ≥90%, quantos estão certos DE FATO? (o bin superior do diagrama de confiabilidade). Se acurácia@conf≥90% = 97%, a proposta libera X% do volume com 3% de erro não revisado — decisão de negócio informada. Meça também o volume da faixa (calibração perfeita com 2% dos casos acima do corte não economiza nada).

b) Superconfiança = o bin "≥90%" contém casos com acurácia real de, digamos, 80% — você libera sem revisão exatamente os erros mais perigosos (confiantes e errados). RLHF descalibra para cima: meça no SEU modelo alinhado, não assuma do base.

c) Porque é texto gerado, não probabilidade: o modelo aprendeu no RLHF que "tenho alta confiança" agrada, e a literatura mostra correlação fraca com acerto real. Use a probabilidade da resposta (logprob) ou um classificador de confiança treinado — e calibre ambos num conjunto rotulado do SEU domínio.
</details>

---

## Parte B — Práticas

### B1. 💻 O n mínimo do seu caso

Generalize o Lab 1 numa função `n_minimo(delta, acc_base, poder=0.8)` que responda por simulação: quantos exemplos PAREADOS para detectar uma melhora de `delta` com 80% de probabilidade?

Tabule para Δ ∈ {1, 3, 5, 10}pp com acc_base=0,65. Guarde a tabela — é a mais reutilizável do curso.

<details><summary>Gabarito esperado</summary>

Ordens de grandeza esperadas (pareado, com correlação típica entre modelos): Δ=10pp → poucas centenas; Δ=5pp → ~500–1.000; Δ=3pp → ~1.500–3.000; Δ=1pp → dezenas de milhares.

A leitura de engenharia: melhoras de 1–2pp — o tamanho típico de um ajuste de prompt — exigem conjuntos que quase ninguém tem. Consequência: ou você mede efeitos grandes, ou constrói conjuntos grandes, ou aceita que está decidindo por ruído. Não há quarta opção.
</details>

---

### B2. 💻 Bootstrap para MRR

O hit@1 é binário (McNemar serve); o MRR não. Implemente o bootstrap pareado para a diferença de MRR entre densa e BM25 do Lab 3 (reamostre perguntas; recompute MRR de ambos; IC da diferença).

O MRR dá veredito diferente do hit@1?

<details><summary>Gabarito esperado</summary>

Provavelmente o mesmo veredito (IC cruzando zero com n=25), mas com IC proporcionalmente mais apertado — o MRR extrai mais informação por pergunta (posição, não só acerto/erro no top-1). É o padrão geral: métricas graduadas > binárias em poder estatístico, pelo mesmo n.

Se você quiser o veredito definitivo da comparação do módulo 13: amplie o conjunto para 100+ perguntas (o desafio do módulo 13 gera mais) e rode de novo. A resposta correta para "n=25 não conclui" não é desistir — é aumentar o n.
</details>

---

### B3. 🍎 O juiz que presta

Repita a auditoria do Lab 4 com o Qwen2.5-7B-Instruct-4bit no M4 (mesmos 12 pares, duas ordens).

Compare: acurácia, consistência, e o custo por julgamento. A partir de que tamanho um juiz vira instrumento?

<details><summary>Gabarito esperado</summary>

Espere um salto grande: juízes de 7B tipicamente passam de 90% de acurácia em pares com gabarito objetivo, com consistência alta. A régua prática da área: juiz de 0.5B = moeda enviesada; 7B = utilizável para sinais grosseiros COM protocolo de duas ordens; 70B+/fronteira = padrão de leaderboard, e AINDA com vieses mensuráveis de comprimento e estilo.

A lição transferível: "usar um juiz" não é uma decisão binária — é escolher um instrumento com erro conhecido. Sem a auditoria, você não conhece o erro.
</details>

---

### B4. 💻 A inflação na prática

Simule o pipeline completo de desenvolvimento enganoso: 10 "variantes de prompt" (modelos idênticos, acc 0,70), avaliadas num dev set de n=150; a melhor é reavaliada num teste lacrado de n=500. Repita 1.000 vezes.

a) Quanto a melhor variante "ganha" no dev set, em média?
b) Quanto desse ganho sobrevive no teste lacrado?

<details><summary>Gabarito</summary>

a) ~+4–5pp sobre a média (o máximo de 10 binomiais com EP de 3,7pp).

b) **~zero** — as variantes são idênticas por construção; o ganho era 100% seleção de ruído, e o teste lacrado o revela. Este par de números é o argumento definitivo para a estrutura dev/teste: o dev set mede "qual escolher"; só o lacrado mede "quanto vale". Quem reporta números do dev set reporta a inflação junto.
</details>

---

### B5. 💻 ECE de verdade

O Lab 6 usou um bin (12 questões). Faça o ECE completo: gere 60+ questões de múltipla escolha (misture fáceis e difíceis — use o Qwen para gerar e VOCÊ valide o gabarito), meça confiança e acerto, e compute o ECE com 5 bins + o diagrama de confiabilidade em ASCII.

Por que binar as 12 questões originais seria "teatro estatístico"?

<details><summary>Gabarito</summary>

Com 12 pontos em 5 bins, cada bin tem ~2 amostras — a "acurácia do bin" é uma moeda jogada duas vezes, e o ECE resultante é ruído com aparência de rigor (o mesmo pecado do módulo 6: PPL em 30 tokens). Regra prática: ≥20 amostras por bin para o diagrama significar algo; com menos, reporte o gap global e diga que é o gap global.

No resultado com 60+: espere superconfiança (gap positivo) concentrada nas questões difíceis — o padrão clássico de modelos instruct. E guarde o método: é exatamente como você mede se pode confiar na confiança do SEU modelo em produção (A5).
</details>

---

## Desafio — o relatório de avaliação padrão-ouro

Escolha a comparação mais importante que você fez no curso até aqui (LoRA vs base do módulo 5, CoT vs direto do 7, DPO do 8 — qualquer uma) e reescreva o resultado como um **relatório estatisticamente completo**:

1. As acurácias com IC (bootstrap).
2. O teste pareado correto (McNemar para binário; bootstrap pareado para o resto).
3. O n mínimo que TERIA sido necessário para a conclusão que você quer (B1) — e o veredito honesto: o experimento conclui ou sugere?
4. Se houve juiz: a auditoria dele. Se houve seleção (de checkpoint, prompt, temperatura): a contagem de comparações e o desconto.
5. Uma frase final no formato: "com n=X, a diferença de Ypp tem IC [a, b]; a evidência {sustenta|sugere|não distingue}".

Este formato de frase final é o produto do módulo. Use-o em todo experimento seu, para sempre — inclusive fora de LLMs.
