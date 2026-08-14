# Módulo 3 — Exercícios

Os práticos usam o `lab.py` como base — importe dele ou copie as classes.

---

## Parte A — Conceituais

### A1. O orçamento de um treino

Você quer treinar um modelo de **3B de parâmetros** com **500B tokens**, alugando H100 a US$ 3,00/h, com MFU de 45%.

a) Quantos FLOPs?
b) Quantas GPU-horas?
c) Quanto custa?
d) Se você tiver 64 H100 em paralelo com escalabilidade perfeita, quantos dias?
e) Por que "escalabilidade perfeita" não existe, e o que isso faz com a resposta (d)?

<details><summary>Gabarito</summary>

a) `6 × 3e9 × 5e11 = 9,0 × 10²¹ FLOPs`

b) H100 densa = 495 TFLOP/s; com MFU 45% = `2,23 × 10¹⁴ FLOP/s`.
`9,0e21 / 2,23e14 = 4,04 × 10⁷ s = 11.220 GPU-horas`

c) `11.220 × US$ 3,00 ≈ **US$ 33.700**`

d) `11.220 / 64 ≈ 175 h ≈ **7,3 dias**`

e) Escalar para 64 GPUs exige comunicação: all-reduce de gradientes a cada passo (DDP/FSDP), all-gather de parâmetros por camada (FSDP), e sincronização. A eficiência de escala típica em 64 GPUs bem conectadas é de 85–95%, então some ~10–20% ao tempo. Além disso, MFU costuma **cair** com mais GPUs, porque a fração de tempo em comunicação cresce. Some ainda falhas de hardware, que em clusters grandes acontecem em escala de horas, e reinícios a partir de checkpoint.
</details>

---

### A2. O diagnóstico de loss

Para cada cenário, dê o diagnóstico mais provável e o primeiro teste que você faria.

1. Vocabulário 32k. Loss do primeiro batch: 10,4. Após 1.000 passos: 10,3.
2. Vocabulário 32k. Loss do primeiro batch: 24,7.
3. Loss cai bem por 500 passos, depois vira `NaN` de um passo para o outro.
4. Loss desce, mas a curva tem um serrilhado com período de exatamente 340 passos.
5. Loss de treino em 1,2; de validação em 3,8 e subindo.

<details><summary>Gabarito</summary>

1. `ln(32.000) = 10,37`. O modelo está estacionado no acaso. Causas: LR efetivamente zero (schedule mal configurado, otimizador sem os parâmetros), labels todas mascaradas, ou shift errado. **Teste:** imprima o LR real no passo 100 e a fração de labels diferentes de `-100`.

2. Muito **acima** de `ln(V)` — o modelo está sistematicamente errado, não apenas ignorante. Isso indica labels desalinhadas (shift na direção errada, ou embaralhamento entre `x` e `y`). **Teste:** decodifique `x[0]` e `y[0]` e confira visualmente se `y` é `x` deslocado de uma posição.

3. Instabilidade numérica. Suspeitos: LR alto demais para a fase, fp16 sem loss scaling adequado, ou um batch com dados corrompidos. **Teste:** verifique se há `inf`/`NaN` nos gradientes antes do clipping, e registre qual batch causou. A correção padrão é voltar ao checkpoint anterior, pular aquele intervalo de dados e reduzir o LR.

4. Dados **não embaralhados**: 340 passos é exatamente uma época, e o modelo está revendo o corpus na mesma ordem. **Teste:** `passos_por_época = len(dataset) / (batch × block)`. Corrija embaralhando a cada época.

5. Overfitting claro. Normal em SFT com poucos dados, anormal em pré-treino. **Teste:** quantas épocas você deu? Se >3 em um dataset pequeno, reduza épocas, aumente o dataset, ou aumente a regularização. Em SFT (módulo 5), 1–3 épocas é o padrão exatamente por isso.
</details>

---

### A3. β₂ e loss spikes

Explique, com o que você sabe do AdamW, por que `β₂ = 0,999` causa mais *loss spikes* em treinos com batch enorme do que `β₂ = 0,95`.

<details><summary>Gabarito</summary>

`v` é uma média móvel exponencial do gradiente ao quadrado, com meia-vida de aproximadamente `1/(1−β₂)` passos: **1.000 passos** para 0,999, **20 passos** para 0,95.

Com `β₂ = 0,999`, `v` reflete a escala do gradiente de mil passos atrás. Se a escala real cair (o que acontece continuamente durante o treino) ou mudar bruscamente (novo domínio nos dados, mudança de fase), `v` fica **grande demais** por muito tempo — passos pequenos, treino lento — ou, no caso perigoso, **pequeno demais** quando o gradiente cresce de repente. Aí o denominador `√v̂` não acompanha, o passo efetivo explode, e o modelo é jogado para fora da bacia em que estava.

Com `β₂ = 0,95`, `v` se readapta em dezenas de passos. O treino é mais reativo e absorve mudanças de escala sem produzir passos gigantes. O custo é uma estimativa mais ruidosa — irrelevante quando o batch já é de milhões de tokens e o gradiente em si tem pouco ruído.
</details>

---

### A4. Packing e SFT

No pré-treino, o packing concatena documentos e a atenção causal deixa tokens de um documento verem o anterior. Isso é tolerado. Em SFT, não é.

Por que a diferença? E como o *document masking* resolve?

<details><summary>Gabarito</summary>

**Por que é tolerado no pré-treino:** o objetivo é modelar texto em geral. Ver o fim de um documento anterior é apenas contexto ligeiramente incoerente — e a escala (trilhões de tokens, cada fronteira aparecendo uma vez) dilui o efeito. O `<eos>` ensina o modelo a tratar a fronteira como uma quebra.

**Por que é nocivo em SFT:** os exemplos são instruções completas com respostas. Se o exemplo B enxerga o exemplo A, o modelo aprende a condicionar a resposta ao exemplo anterior — que na inferência não existirá. Ele pode aprender correlações espúrias ("depois de uma pergunta sobre culinária vem uma sobre viagem") e, pior, o cálculo da loss de B fica condicionado a um contexto que jamais se repetirá. Com poucos milhares de exemplos, o efeito não é diluído.

**Document masking:** em vez de uma máscara causal triangular única para o bloco inteiro, usa-se uma máscara **bloco-diagonal**: cada exemplo só vê a si mesmo. Mantém-se todo o ganho de eficiência do packing sem contaminação. É suportado nativamente pelo TRL via `position_ids` corretos por documento e por FlashAttention com `cu_seqlens`.
</details>

---

### A5. O que fine-tuning não conserta

Um cliente pede um assistente que responda perguntas sobre a legislação tributária brasileira de 2026. Você faz SFT com 50.000 exemplos de perguntas e respostas corretas sobre o tema. O modelo passa a responder no formato certo, com tom de especialista — e continua inventando artigos de lei.

Explique o que aconteceu usando o quadro da seção 1, e diga o que fazer.

<details><summary>Gabarito</summary>

O SFT consumiu ~1% do compute e ensinou **comportamento**: formato, tom, estrutura de resposta jurídica. Ele não instalou **conhecimento** — os artigos da legislação de 2026 não estavam no pré-treino, e 50.000 exemplos não são remotamente suficientes para inserir um corpo de fatos novo de forma confiável.

Pior: o SFT tornou o problema **mais perigoso**. O modelo agora responde com a confiança e o formato de um especialista sobre um assunto que ele não domina. Você ensinou o estilo da certeza sem a substância.

O que fazer:
1. **RAG** é a resposta primária: recuperar os artigos relevantes e colocá-los no contexto. Fatos verificáveis e atualizáveis, com citação da fonte.
2. Manter o SFT, mas **retreinar** com exemplos que sempre citam o trecho recuperado — ensinando o modelo a fundamentar em vez de recitar.
3. Incluir exemplos de recusa: "não encontrei base legal para isso no material fornecido".

Regra geral: SFT ensina *como responder*; RAG fornece *o que responder*. Confundir os dois é o erro de projeto mais caro em aplicações de LLM.
</details>

---

## Parte B — Práticas

### B1. Varredura de learning rate

Treine o MiniGPT por 150 passos com LR de pico em `1e-2`, `3e-3`, `1e-3`, `3e-4` e `1e-4`. Registre a loss final e descreva a curva de cada um.

Qual é o melhor? O que acontece nos extremos?

<details><summary>Gabarito</summary>

Espere algo como: `1e-2` diverge ou fica preso alto (a loss pode até subir); `3e-3` treina rápido mas instável, com serrilhado; `1e-3` é o melhor equilíbrio; `3e-4` converge suave mas mais devagar; `1e-4` é lento demais para 150 passos.

A forma da curva "LR × loss final" é um **U assimétrico**: o lado esquerdo (LR baixo) sobe suavemente, o direito (LR alto) sobe abruptamente ou explode. Por isso a heurística prática é escolher o maior LR que ainda seja estável e depois **dividir por 2 ou 3** — a margem de segurança fica no lado barato do U.
</details>

---

### B2. O treino sem warmup

Remova o warmup (comece direto no LR de pico) e treine 3 vezes com seeds diferentes. Compare com 3 execuções com warmup.

O warmup importa neste modelo minúsculo? Se não, por quê — e por que ele é obrigatório em modelos grandes?

<details><summary>Gabarito</summary>

Em um modelo de 2M de parâmetros com batch pequeno, o warmup pode fazer pouca diferença visível — ocasionalmente uma execução sem warmup diverge ou fica pior, mas frequentemente todas convergem.

A razão de ele ser crítico em escala:
1. **Batches enormes** significam gradientes de norma muito maior no início, quando os pesos estão aleatórios e a loss é altíssima.
2. **Modelos profundos** têm mais camadas para desestabilizar em cascata — um passo ruim nas camadas iniciais se propaga.
3. **Uma única divergência** custa dias de compute e dinheiro real. Warmup é um seguro barato: 0,1–2% dos passos.

A lição metodológica é mais valiosa que o resultado: **intuições obtidas em modelos minúsculos não transferem automaticamente para a escala real.** Muitos truques só se justificam a partir de certo tamanho, e este é um deles.
</details>

---

### B3. Chinchilla no seu modelo

O Chinchilla diz ~20 tokens por parâmetro para ser compute-optimal. O MiniGPT tem ~2,1M de parâmetros, logo o ótimo seria ~42M tokens — mas o corpus tem apenas 250k.

a) Quantas épocas você precisaria para atingir 42M tokens?
b) Rode com 3 orçamentos: 200k, 800k e 3,2M tokens vistos. Plote loss final × tokens.
c) A relação segue lei de potência? Onde ela quebra, e por quê?

<details><summary>Gabarito</summary>

a) `42M / 250k ≈ **168 épocas**`.

b) Espere ganhos decrescentes claros, e o gap de validação crescendo a cada orçamento.

c) A lei de potência do Chinchilla pressupõe **dados novos** a cada token processado. Com 168 épocas sobre o mesmo corpus, você não está adicionando informação — está memorizando. A loss de **treino** continua caindo (pode chegar perto de zero); a de **validação** estaciona e depois sobe.

É a limitação fundamental que o módulo 4 vai atacar: escalar compute sem escalar dados não escala capacidade. Papers recentes sobre "repetição de dados" mostram que até ~4 épocas o dano é pequeno, e a partir de ~16 épocas o token repetido vale quase nada.
</details>

---

### B4. O modelo maior, o mesmo orçamento

Com FLOPs fixos (`6ND` constante), treine três configurações e compare a loss final:

| Config | d | camadas | ~params | tokens (para manter 6ND) |
|---|---|---|---|---|
| A | 128 | 3 | ~0,9M | 3,2M |
| B | 192 | 4 | ~2,1M | 1,4M |
| C | 320 | 6 | ~6M | 0,5M |

Qual vence? A resposta contradiz o Chinchilla?

<details><summary>Gabarito</summary>

Provavelmente a configuração do meio, ou A — mas o valor do exercício está na interpretação, não no vencedor.

Não contradiz o Chinchilla; **confirma o formato do argumento**. Chinchilla diz exatamente isto: dado um orçamento de compute fixo, existe uma alocação ótima entre tamanho do modelo e quantidade de dados, e desviar dela em qualquer direção piora o resultado. Modelo grande demais com poucos tokens fica subtreinado; modelo pequeno demais com muitos tokens satura.

Ressalva importante: seu corpus é pequeno, então C com "0,5M tokens" ainda são 2 épocas, enquanto A com 3,2M são 13 épocas. A repetição contamina o experimento — o que é, por sua vez, uma boa lição sobre o quanto é difícil rodar experimentos de scaling law limpos.
</details>

---

### B5. Reimplementando o clipping

Substitua `torch.nn.utils.clip_grad_norm_` por sua própria implementação e verifique que batem.

Depois, registre a norma **antes** do clipping ao longo do treino e responda: em quantos por cento dos passos o clipping efetivamente atuou?

<details><summary>Gabarito</summary>

```python
def clip_manual(parametros, max_norma=1.0):
    grads = [p.grad for p in parametros if p.grad is not None]
    norma_total = torch.sqrt(sum((g ** 2).sum() for g in grads))
    if norma_total > max_norma:
        escala = max_norma / (norma_total + 1e-6)
        for g in grads:
            g.mul_(escala)
    return norma_total
```

Note que a função **retorna a norma antes** do clipping — é assim que o PyTorch faz, e é por isso que você pode logar `|grad|` acima de 1,0 mesmo com clipping em 1,0. Confundir isso leva a "o clipping não está funcionando".

No treino do lab, a norma passa de 1,0 apenas nos primeiros passos (medimos 1,67 no passo 0, ~0,8 depois). Ou seja: o clipping atua em poucos por cento dos passos, e é justamente nesses que ele salva o treino. Custo próximo de zero, proteção contra o caso catastrófico — a razão de ser universal.
</details>

---

## Desafio — WSD e continuação de treino

Implemente o agendamento **WSD** (Warmup-Stable-Decay): warmup curto, platô longo em LR constante, decay nos últimos 10–20% dos passos.

1. Treine 400 passos com cosine e 400 com WSD. Compare a loss final.
2. Agora simule "consegui mais compute": a partir do **checkpoint do platô** do WSD (passo 340, antes do decay), continue por mais 400 passos e faça um novo decay.
3. Compare com treinar 800 passos de cosine do zero.

Por que o mesmo não é possível com cosine?

<details><summary>Gabarito</summary>

O cosine é definido em função do **total de passos**, conhecido de antemão. Se você treina 400 passos de cosine e depois quer continuar, tem duas opções ruins: continuar em LR mínimo (o modelo mal aprende) ou reaquecer o LR (o modelo sofre um choque e perde parte do progresso do decay).

O WSD separa as fases: durante o platô o LR é constante, então o checkpoint do platô é um ponto de continuação **natural** — nada no schedule assume quando o treino vai terminar. O decay é uma operação curta aplicada quando você decidir parar.

É por isso que o WSD ganhou tração em labs que treinam continuamente e podem receber mais compute a qualquer momento (MiniCPM, DeepSeek). A qualidade final é comparável à do cosine; a flexibilidade operacional é muito maior.

Espere que a continuação a partir do platô supere claramente "recomeçar do zero com 800 passos" em custo — você reaproveita 340 passos de trabalho.
</details>
