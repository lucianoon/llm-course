# Módulo 3 — Como um LLM é treinado

> **Pergunta central:** o que acontece, mecanicamente e financeiramente, entre um modelo com pesos aleatórios e um modelo que escreve português?

Os módulos 1 e 2 descreveram o que o modelo **calcula**. Este descreve como os pesos chegaram lá. No laboratório você vai treinar um LLM completo do zero, na sua CPU, em poucos minutos — e depois usar as mesmas fórmulas para orçar o treino de um modelo de 8B.

## Objetivos

1. Descrever as três fases de treino e qual fração do compute cada uma consome.
2. Explicar por que existe warmup, por que `β₂ = 0,95` e por que o clipping é sempre 1,0.
3. Provar que gradient accumulation é equivalente a um batch maior — e apontar o erro de normalização que quebra essa equivalência.
4. Explicar por que bf16 venceu fp16, em termos de expoente e mantissa.
5. Ler uma curva de loss e distinguir comportamento normal de bug.
6. Estimar FLOPs, GPU-horas e custo em dólares de um treino, e validar a estimativa contra números publicados.

---

## 1. As três fases

| Fase | Dados | Escala típica | % do compute | O que ensina |
|---|---|---|---|---|
| **Pré-treino** | Texto cru da web, livros, código | 1–15T tokens | **> 98%** | Língua, fatos, raciocínio, código |
| **SFT** (módulo 5) | Pares instrução→resposta | 10k–1M exemplos | ~1% | Formato, comportamento de assistente |
| **Alinhamento** (módulos 8–9) | Preferências, recompensas | 10k–500k comparações | < 1% | Tom, recusas, escolha entre boas respostas |

O desequilíbrio é o fato mais importante do quadro. **Praticamente toda a capacidade do modelo vem do pré-treino**; as fases seguintes moldam como essa capacidade se manifesta. É a origem da "hipótese do alinhamento superficial": SFT e RLHF não ensinam conhecimento novo, ensinam qual sub-distribuição do que já existe deve ser produzida.

Consequência prática direta para você: quando um fine-tuning "não funciona", quase sempre é porque a capacidade não estava lá para ser evocada. Fine-tuning não instala habilidades ausentes.

---

## 2. O objetivo, revisitado

O mesmo do módulo 1 — minimizar a cross-entropy do próximo token:

```
L(θ) = −(1/N) · Σ log p_θ(x_t | x_<t)
```

O que muda na escala do pré-treino é tudo o mais: como os dados chegam à GPU, como o gradiente é acumulado, como o learning rate evolui, e como se evita que 10.000 GPUs divirjam por causa de um lote ruim.

---

## 3. Os dados

| Corpus | Tamanho | Nota |
|---|---|---|
| The Pile (2020) | 825 GB | Primeiro corpus aberto e curado com seriedade |
| RefinedWeb (2023) | 5T tokens | Provou que web filtrada bate corpora curados |
| **FineWeb / FineWeb-Edu (2024)** | 15T / 1,3T tokens | Estado da arte aberto; o `-Edu` filtra por valor educacional |
| Llama 3 (fechado) | 15T tokens | Mais de 5% em 30+ idiomas não ingleses |

O pipeline, em ordem: extração de HTML → filtro de idioma → filtros de qualidade (heurísticos e por classificador) → **deduplicação** → decontaminação contra benchmarks.

A deduplicação é a etapa de maior impacto e a mais subestimada. Documentos repetidos fazem o modelo memorizar em vez de generalizar, e a web tem repetição massiva. O método padrão é MinHash + LSH, que encontra quase-duplicatas em trilhões de documentos sem comparar todos com todos.

O módulo 4 é inteiramente sobre isso. Aqui basta a conclusão: **na escala atual, qualidade e diversidade de dados importam mais que arquitetura**. Praticamente todos os ganhos entre gerações de modelos abertos vieram de dados melhores, com a arquitetura quase congelada desde 2023.

---

## 4. Packing — como o texto vira batches

Documentos têm comprimentos variados; a GPU quer tensores retangulares. Duas opções:

**Padding** — cada documento vira uma linha, completada com `<pad>`. Simples e desperdiçador: num corpus com comprimento médio de 500 tokens e janela de 2048, mais de 70% do batch é preenchimento. Você paga compute para processar nada.

**Packing** — concatena todos os documentos num único fluxo, separados por `<eos>`, e corta em blocos de exatamente `block_size` tokens. Desperdício ≈ 0%. É o que todo pré-treino faz. O Lab 2 mede a diferença.

> ⚠️ **Armadilha:** com packing ingênuo, um bloco contém pedaços de vários documentos, e a atenção causal permite que tokens do documento B atendam a tokens do documento A. No pré-treino isso é tolerado (o `<eos>` sinaliza a fronteira e o efeito é pequeno em escala). Em **SFT**, onde os exemplos são poucos e específicos, isso contamina o treino: um exemplo passa a condicionar o seguinte. A solução é *document masking* — máscara de atenção em blocos diagonais. Guarde isso para o módulo 5; é uma das causas de fine-tunings que "quase funcionam".

---

## 5. O loop de treino

```python
for passo in range(passos_totais):
    lr = agenda_lr(passo)                          # warmup + cosine
    for g in optim.param_groups: g["lr"] = lr

    optim.zero_grad(set_to_none=True)
    for micro in range(acumulacao):                # gradient accumulation
        x, y = proximo_batch()
        loss = modelo(x, alvos=y) / acumulacao     # ← a divisão é obrigatória
        loss.backward()                            # gradientes SOMAM

    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
    optim.step()
```

Cinco linhas conceituais, e cada uma esconde uma decisão que pode arruinar o treino. As seções seguintes tratam de cada uma.

---

## 6. 📐 AdamW

SGD usa a mesma taxa para todo parâmetro. Adam mantém, **por parâmetro**, duas médias móveis:

```
m_t = β₁·m_{t−1} + (1−β₁)·g_t              momento (direção suavizada)
v_t = β₂·v_{t−1} + (1−β₂)·g_t²             escala (magnitude típica ao quadrado)

m̂ = m_t/(1−β₁ᵗ)    v̂ = v_t/(1−β₂ᵗ)         correção de viés inicial

θ ← θ − lr · m̂/(√v̂ + ε) − lr · λ · θ       ← o último termo é o weight decay
```

A divisão por `√v̂` é a chave: parâmetros cujo gradiente é tipicamente grande recebem passos proporcionalmente menores. O passo efetivo fica aproximadamente **normalizado**, o que permite treinar camadas com escalas de gradiente muito diferentes com um único learning rate.

**Hiperparâmetros em LLMs, e por quê:**

| Parâmetro | Valor típico | Razão |
|---|---|---|
| `β₁` | 0,9 | Padrão, raramente mexido |
| `β₂` | **0,95** (não 0,999) | 0,999 tem memória de ~1000 passos; com batches enormes a escala do gradiente muda rápido e o `v` fica desatualizado, causando passos grandes demais e *loss spikes*. Padrão desde o GPT-3 |
| `ε` | 1e-8 | Estabilidade numérica |
| `weight decay` | 0,1 | Desacoplado — daí o "W" |

**Por que "W":** no Adam original, weight decay era implementado como L2 adicionada ao gradiente — e portanto **dividida por `√v̂`** junto com tudo o mais. Efeito perverso: parâmetros com gradiente pequeno sofriam decay enorme. O AdamW aplica o decay diretamente sobre `θ`, fora da normalização adaptativa. É uma correção de uma linha de código e melhora a generalização de forma consistente.

> ⚠️ **Armadilha:** não aplique weight decay a biases nem a pesos de normalização. Encolher o `γ` de um RMSNorm em direção a zero desestabiliza o treino. Toda implementação séria separa os parâmetros em dois grupos — o Lab 3 faz isso.

**A memória:** `m` e `v` são tensores do tamanho do modelo, em fp32. Com master weights e gradientes, chega-se aos **16 bytes por parâmetro** do módulo 1. É por isso que otimizadores alternativos (Adafactor, 8-bit Adam, Lion) existem — todos atacam esses 12 bytes de estado.

---

## 7. Learning rate — warmup e decay

Nenhum treino de LLM usa learning rate constante. O padrão é:

```
     lr
      │      ╭──────╮
 pico │     ╱        ╰──╮
      │    ╱             ╰────╮
      │   ╱                    ╰─────╮
 mín  │  ╱                            ╰────
      └──┴────────────────────────────────── passos
       warmup            cosine decay
```

### Por que warmup

Nos primeiros passos, `v` ainda é uma estimativa ruim baseada em pouquíssimas amostras. A correção de viés ajuda, mas não resolve: um gradiente atípico logo no início produz um passo enorme sobre pesos recém-inicializados, e o modelo pode nunca se recuperar. O warmup — subir linearmente de 0 ao pico ao longo de 0,1–2% dos passos — dá tempo para as estatísticas se estabilizarem.

Sem warmup, treinos grandes divergem. É um dos poucos truques que praticamente ninguém questiona.

### Por que decay

No fim do treino, passos grandes fazem o modelo oscilar em torno do mínimo em vez de assentar nele. O cosine decay até ~10% do pico é o padrão; o Chinchilla mostrou que o decay deve terminar **exatamente** quando os tokens acabarem — encerrar um cosine pela metade dá um modelo pior do que um cosine planejado para aquele orçamento.

**WSD (Warmup-Stable-Decay)** é a alternativa moderna: warmup, um platô longo em LR constante, e um decay curto no fim. A vantagem é prática — você pode continuar o treino a partir do platô se conseguir mais compute, o que o cosine não permite.

### Valores típicos

| Modelo | LR de pico |
|---|---|
| ~100M params | 6e-4 a 1e-3 |
| ~1B | 3e-4 |
| ~7B | 3e-4 |
| 175B (GPT-3) | 0,6e-4 |

Modelos maiores usam LR **menor**. A intuição: quanto mais larga a camada, menor deve ser o passo por parâmetro para manter a mudança na saída constante — formalizado pela parametrização µP, que permite transferir hiperparâmetros de um modelo pequeno para um grande.

---

## 8. Batch size e gradient accumulation

Pré-treinos usam batches gigantescos, medidos em tokens: GPT-3 usou 3,2M tokens por passo; Llama 3 chega a 16M. A razão é estatística — o gradiente de um batch é uma estimativa amostral, e batches maiores reduzem seu ruído, permitindo passos maiores e mais estáveis.

Existe um **critical batch size**: além dele, dobrar o batch quase não melhora a qualidade do gradiente, e você só desperdiça compute. Ele **cresce durante o treino** (o gradiente fica mais ruidoso conforme o modelo melhora), e alguns treinos aumentam o batch progressivamente por isso.

### 📐 Gradient accumulation

16M tokens não cabem na memória de nenhuma GPU. A solução: processar micro-batches, acumular gradientes, e só então dar o passo.

Funciona porque o gradiente é **linear** na loss:

```
∇ (1/N · Σᵢ Lᵢ) = 1/N · Σᵢ ∇Lᵢ
```

Como `.backward()` **soma** nos `.grad` existentes, basta chamar `backward()` em cada micro-batch e dividir por `N`.

> ⚠️ **Armadilha:** a divisão precisa acontecer, e no lugar certo. Se você esquecer o `/ acumulacao`, o gradiente fica `N` vezes maior — equivalente a multiplicar o learning rate por `N`. Com `N=8`, o treino diverge. O Lab 5 prova a equivalência numericamente e mostra o estrago do erro.

> ⚠️ **Armadilha 2:** a equivalência é exata para a loss média, mas **não** quando os micro-batches têm números diferentes de tokens válidos (padding variável). Nesse caso a média correta é ponderada pelo número de tokens, não uniforme entre micro-batches. Bug sutil e comum em SFT.

---

## 9. Mixed precision — por que bf16 venceu

| Formato | Bits (sinal/expoente/mantissa) | Faixa | Precisão relativa |
|---|---|---|---|
| fp32 | 1/8/23 | ±3,4e38 | ~7 dígitos |
| **fp16** | 1/5/10 | **±65.504** | ~3 dígitos |
| **bf16** | 1/**8**/7 | **±3,4e38** | ~2 dígitos |

O ponto: **bf16 tem o mesmo expoente do fp32**, e portanto a mesma faixa dinâmica. Sacrifica mantissa (precisão) em vez de faixa.

Isso decide a questão. Em fp16, gradientes pequenos — comuníssimos — viram **zero** (underflow), e ativações grandes viram **inf** (overflow em 65.504). O remédio era *loss scaling*: multiplicar a loss por ~2¹⁶ antes do backward e dividir depois, com detecção dinâmica de overflow e descarte de passos. Funciona, mas é frágil e cheio de casos especiais.

Com bf16 nada disso é necessário. A perda de precisão na mantissa é absorvida pelos **master weights em fp32**, que o otimizador mantém: os pesos são atualizados em fp32 e só depois convertidos para bf16 para o forward.

> 🔧 **Na prática:** bf16 exige Ampere (A100, RTX 30xx) ou mais novo. Em uma **T4 do Colab grátis** (Turing) só há fp16 — e é por isso que você vai ver `fp16=True` e loss scaling nas configurações de treino do módulo 6. Não é escolha, é limitação de hardware.

> ⚠️ **Armadilha:** RMSNorm, softmax e a acumulação da loss devem ser feitos em fp32 mesmo em treino bf16. Somar milhares de termos com 7 bits de mantissa perde informação silenciosamente.

---

## 10. Estabilidade

**Gradient clipping.** Antes do `optimizer.step()`, reescale o gradiente global se sua norma passar de um limiar — universalmente 1,0:

```python
torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
```

Um único batch anômalo (um documento de lixo, uma sequência degenerada) pode produzir um gradiente com norma centenas de vezes maior que a típica. Sem clipping, esse batch destrói pesos construídos ao longo de milhares de passos.

**Loss spikes.** Em treinos grandes, a loss ocasionalmente salta e às vezes não volta. Causas: dados corrompidos, LR alto demais para a fase, instabilidade numérica na atenção, `β₂` alto demais. Mitigações usadas na prática:

1. Pular o batch se a loss exceder um múltiplo da média móvel.
2. Voltar ao checkpoint anterior e pular alguns milhares de amostras.
3. Reduzir o LR e recomeçar do checkpoint.
4. z-loss (regularizar o logsumexp dos logits) — usado em PaLM e Chameleon.

O log de treino do OPT-175B da Meta, publicado integralmente, documenta dezenas dessas intervenções manuais. É a leitura mais honesta que existe sobre o que é treinar um modelo grande.

---

## 11. Paralelismo — como isso escala para milhares de GPUs

| Estratégia | O que divide | Comunicação | Quando usar |
|---|---|---|---|
| **DDP** | O batch (modelo replicado) | All-reduce dos gradientes | O modelo cabe numa GPU |
| **ZeRO / FSDP** | Estados do otimizador → gradientes → parâmetros | All-gather por camada | Primeira escolha quando não cabe |
| **Tensor Parallel** | Matrizes *dentro* de cada camada | All-reduce a cada camada | Dentro de um nó, com NVLink |
| **Pipeline Parallel** | Camadas entre GPUs | Ponto a ponto | Entre nós, banda limitada |
| **Sequence/Context Parallel** | A dimensão de sequência | — | Contexto muito longo |

Treinos de fronteira combinam quatro delas simultaneamente (*4D parallelism*).

A regra prática: **FSDP resolve quase tudo até algumas dezenas de bilhões de parâmetros.** Tensor parallel só compensa dentro de um nó, porque exige comunicação a cada camada — passar isso por rede Ethernet mata o desempenho. Pipeline parallel introduz a "bolha" (GPUs ociosas esperando), mitigada com micro-batches.

Nada disso é necessário para o que você vai fazer nos módulos 5–10, onde uma GPU basta. É contexto para ler papers e entender por que os números de custo são o que são.

---

## 12. Lendo uma curva de loss

Uma curva saudável de pré-treino tem fases características:

1. **Queda vertical** (primeiras dezenas de passos): de `ln(V)` para muito abaixo. O modelo aprende a distribuição de frequência dos tokens — que alguns são muitíssimo mais comuns que outros. É a fruta mais baixa.
2. **Queda rápida**: estrutura local, morfologia, palavras frequentes.
3. **Descida lenta e longa**: gramática, fatos, coerência. É onde fica 95% do tempo, e onde a curva parece "parada" num gráfico linear.
4. **Platô ou queda muito lenta**: rendimentos decrescentes.

> 🔧 **Sempre plote a loss em escala logarítmica no eixo x.** Em escala linear, todo treino parece um platô após 5% dos passos, e você perde a capacidade de diagnosticar qualquer coisa.

### O treino do laboratório, medido

Números reais do Lab 7 (MiniGPT de 2,1M parâmetros, 400 passos, 126 segundos de CPU):

| Passo | Treino | Validação | Gap | ‖grad‖ |
|---|---|---|---|---|
| 0 | 7,667 | 7,654 | −0,01 | 1,67 |
| 25 | 6,253 | 6,500 | +0,25 | 0,50 |
| 100 | 5,475 | 5,891 | +0,42 | 0,67 |
| 200 | 5,099 | 5,409 | +0,31 | 0,77 |
| 399 | 4,733 | 5,106 | +0,37 | 0,83 |

Três coisas para ler aí:

1. **A loss inicial é 7,667 e `ln(2048) = 7,625`.** O teste de sanidade do módulo 1 passou: o modelo começa exatamente como um sorteio uniforme. Se começasse muito acima, haveria bug no pipeline.
2. **A norma do gradiente começa em 1,67 e cai para ~0,8.** O clipping em 1,0 atuou de fato nos primeiros passos — foi ele que impediu o primeiro batch de destruir a inicialização.
3. **O gap de validação sobe de −0,01 para +0,37 e depois estabiliza.** O modelo passou 3,5 épocas sobre 750 KB de texto e começou a memorizar. Pré-treinos reais fazem aproximadamente **uma** época sobre trilhões de tokens e nunca entram nesse regime — esse é um luxo que só existe quando o corpus é grande demais para ser memorizado.

A perplexidade caiu de 2.137 para 114. Nas mesmas 400 iterações, o modelo aprendeu morfologia portuguesa, pontuação de diálogo e concordância local.

**Diagnóstico rápido:**

| Sintoma | Causa provável |
|---|---|
| Loss inicial ≈ `ln(vocab_size)` e não desce | Bug no pipeline: shift de labels, máscara errada, LR ≈ 0 |
| Loss inicial **muito acima** de `ln(V)` | Labels desalinhadas ou embaralhadas |
| NaN | LR alto demais, fp16 sem loss scaling, divisão por zero em norm |
| Loss desce e depois sobe | LR alto para a fase; schedule mal dimensionado |
| Serrilhado com período fixo | Dados não embaralhados — o modelo está vendo o corpus na ordem original |
| Loss de treino cai, de validação sobe | Overfitting — raro em pré-treino de 1 época, comum em SFT |

---

## 13. 📐 O custo real

```
FLOPs ≈ 6 · N · D          N = parâmetros, D = tokens
```

Origem do 6: ~2 FLOPs por parâmetro no forward (multiplicação e soma) e ~4 no backward (gradiente em relação à entrada e aos pesos).

**MFU (Model FLOPs Utilization)** = FLOPs úteis ÷ FLOPs teóricos do hardware. Valores reais ficam entre 35% e 55%; o resto se perde em comunicação, memória e kernels imperfeitos. Reportar MFU virou padrão porque é a métrica honesta de eficiência.

> ⚠️ **Armadilha:** fichas técnicas de GPU anunciam TFLOPs **com sparsity 2:4**, que é o dobro do valor denso e não se aplica a treino de LLM. A NVIDIA divulga "989 TFLOPS bf16" para a H100 SXM; o número que você deve usar é **495**. Usar o valor com sparsity subestima o custo pela metade — erro que aparece com frequência em posts de blog comparando custo de treino.

### Exemplo: Llama-3-8B

```
FLOPs   = 6 × 8e9 × 15e12                    = 7,2 × 10²³
H100 SXM bf16 (denso)                        = 495 × 10¹² FLOP/s
com MFU 40%                                  = 1,98 × 10¹⁴ FLOP/s
tempo   = 7,2e23 / 1,98e14                   = 3,65 × 10⁹ s
        ≈ 1,01 milhão de H100-horas
```

A Meta reportou **1,3 milhão de H100-horas**. A estimativa erra por 28% — e o erro tem explicação: o MFU implícito real foi de ~31%, não os 40% que assumimos.

Isso é o melhor tipo de validação: a fórmula é grosseira, você sabe exatamente **por que** ela erra, e o parâmetro que a corrige (MFU) é justamente o que se mede na prática. Em A100 o mesmo treino daria ~1,6 milhão de GPU-horas, ou cerca de **US$ 2,4 milhões** a US$ 1,50/h.

O Lab 8 transforma isso numa calculadora onde você troca modelo, hardware e preço.

> 🔧 **Na prática:** o número que interessa a você não é esse. Um QLoRA de 3 épocas sobre 10.000 exemplos num modelo de 7B custa cerca de **US$ 2 a US$ 10**. A distância entre US$ 2,4 milhões e US$ 5 é exatamente o que torna a customização viável — e é o assunto do resto do curso.

---

## 14. O que o pré-treino não ensina

Um modelo base recém-pré-treinado sabe uma quantidade enorme sobre o mundo e **não sabe conversar**. Perguntado "Qual a capital da França?", ele pode responder com mais perguntas — porque em fóruns, listas de perguntas aparecem juntas.

Falta a ele:

- o **formato** de diálogo (turnos, papéis, marcadores);
- a **disposição** de responder em vez de continuar;
- **calibração** sobre o que recusar;
- **preferência** entre respostas igualmente plausíveis.

Nada disso é conhecimento — é comportamento. E é exatamente o que os módulos 5 a 9 instalam, com menos de 1% do compute que o pré-treino consumiu.

---

## 15. Leituras

1. **Karpathy — nanoGPT** ([github](https://github.com/karpathy/nanoGPT)). O `train.py` é a referência canônica de loop de treino legível. O lab deste módulo é um primo simplificado.
2. **Brown et al. (2020), "Language Models are Few-Shot Learners" (GPT-3)** — [arXiv:2005.14165](https://arxiv.org/abs/2005.14165). Apêndice B tem todos os hiperparâmetros; é de onde vieram os defaults que todo mundo copia até hoje.
3. **Zhang et al. (2022), "OPT: Open Pre-trained Transformers"** — [arXiv:2205.01068](https://arxiv.org/abs/2205.01068), e principalmente o **logbook** publicado junto. Leia o logbook: é o relato cru de loss spikes, hardware quebrado e decisões às 3h da manhã.
4. **Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization" (AdamW)** — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
5. **Penedo et al. (2024), "FineWeb"** — [HuggingFace blog](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1). O relato mais transparente que existe de construção de corpus. Ponte natural para o módulo 4.

---

## 16. Checklist de saída

- [ ] Que fração do compute total vai para o pré-treino, e o que isso implica sobre o que o fine-tuning consegue fazer?
- [ ] Por que packing em vez de padding, e que problema o packing cria em SFT?
- [ ] O que `m` e `v` guardam no Adam, e por que `β₂ = 0,95` em LLMs?
- [ ] O que o "W" do AdamW corrige, exatamente?
- [ ] Quais parâmetros **não** devem receber weight decay?
- [ ] Por que existe warmup? O que acontece sem ele?
- [ ] Por que modelos maiores usam learning rate menor?
- [ ] Escreva a equivalência de gradient accumulation e diga onde entra a divisão.
- [ ] Por que bf16 dispensa loss scaling e fp16 não?
- [ ] Sua loss inicial é 11,8 num modelo com vocab de 128k e não desce. Diagnóstico?
- [ ] Estime o custo de treinar um modelo de 3B em 500B tokens numa H100 a US$ 3/h com MFU 45%.

Depois, abra o `lab.py` — você vai treinar um LLM de verdade.
