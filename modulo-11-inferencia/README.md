# Módulo 11 — Mixture of Experts, quantização e inferência

> **Pergunta central:** como isso tudo vira produção que cabe no orçamento?

Os módulos 5–10 produziram modelos customizados. Este módulo é sobre servi-los: as três alavancas que separam um demo de um sistema economicamente viável — **arquitetura esparsa** (MoE), **precisão reduzida** (quantização, agora do lado de servir) e **engenharia de inferência** (batching, cache, decodificação especulativa).

## Objetivos

1. Explicar o MoE: por que substituir só o MLP, como o roteador decide, e o que "parâmetros ativos" significa no preço.
2. Reproduzir o **colapso de roteamento** e consertá-lo com a loss de balanceamento.
3. Implementar **decodificação especulativa** do zero e medir a taxa de aceitação.
4. Conhecer o mapa da quantização de inferência: GPTQ/AWQ/GGUF, e o papel dos outliers.
5. Ler as métricas que importam: TTFT, TPOT, throughput — e o que o batching faz com cada uma.
6. Fazer a conta de custo por token de um deployment.

---

## 1. Recapitulando o problema (módulo 1, agora com preço)

| Fase | Gargalo | Consequência econômica |
|---|---|---|
| Prefill | compute (O(n²)) | prompts longos custam compute, mas paralelizam bem |
| **Decode** | **banda de memória** | cada token exige ler os pesos INTEIROS da memória |

No decode, a GPU passa a maior parte do tempo esperando memória — utilização de compute de um dígito percentual é normal. As três alavancas do módulo atacam exatamente isso:

- **MoE**: ler só uma fração dos pesos por token.
- **Quantização**: os pesos que se leem são menores.
- **Batching/especulação**: amortizar cada leitura entre mais tokens.

---

## 2. Mixture of Experts

### A ideia

Do módulo 2: o MLP é 87,7% dos parâmetros de um bloco. O MoE substitui **esse MLP** por `E` cópias ("experts") e um **roteador** — uma projeção linear minúscula que decide, por token, quais `k` experts processam:

```
router(x) = softmax(W_r · x)          # [E] scores
y = Σ_{i ∈ top-k} p_i · expert_i(x)   # só k dos E experts executam
```

A atenção continua densa e compartilhada. Só o MLP é esparso — porque é onde estão os parâmetros, e porque a atenção precisa ver tudo (é ela que mistura posições; especializá-la fragmentaria o contexto).

### Parâmetros totais vs ativos

| Modelo | Totais | Ativos/token | Experts |
|---|---|---|---|
| Mixtral 8x7B | 47B | 13B | 8, top-2 |
| DeepSeek-V3 | 671B | **37B** | 256 finos + 1 compartilhado, top-8 |
| Qwen3-30B-A3B | 30B | **3B** | 128, top-8 |

A conta que define o MoE: **custo de compute de um modelo pequeno, capacidade de um grande** — pagando com memória (todos os experts residem na RAM/VRAM, mesmo os inativos). É a troca inversa da quantização, e por isso as duas se combinam tão bem.

> 🔧 Para os seus 16 GB: um MoE de 30B totais em 4-bit ocupa ~17 GB — não cabe. Um de 14B totais (Qwen1.5-MoE-A2.7B) em 4-bit, ~8 GB — cabe, e roda com o custo de decode de um modelo de 2,7B. MoE + memória unificada é uma combinação particularmente boa para Macs.

### 📐 O problema do roteador: colapso

O roteador é treinado junto com o resto, e tem um ponto fixo patológico: se um expert começa ligeiramente melhor, recebe mais tokens, treina mais, fica melhor ainda — **winner-take-all**. Os demais experts nunca treinam e viram peso morto; o MoE degenera num modelo denso pequeno e caro.

A defesa padrão é a **loss auxiliar de balanceamento** (Switch Transformer):

```
L_aux = α · E · Σ_i f_i · P_i
```

onde `f_i` = fração de tokens roteados ao expert `i` e `P_i` = probabilidade média do roteador para `i`. O produto é mínimo quando a distribuição é uniforme — a loss empurra contra a concentração. `α` típico: 0,01.

O Lab 2 mede o fenômeno. Resultado da execução de referência (MiniGPT-MoE, 4 experts top-1, 300 passos):

| | Utilização por expert | PPL |
|---|---|---|
| Sem loss auxiliar | 32% / 20% / 31% / **17%** (spread 2×, crescendo) | 230,7 |
| Com α=0,01 | 23% / 25% / 27% / 25% | 235,2 |

Em 300 passos vê-se a **deriva** (spread de 2×), não o colapso terminal — o colapso é realimentação composta ao longo de treinos longos. E note o custo honesto: a loss auxiliar cobra um imposto na loss principal (PPL 235 vs 231); o retorno vem de manter todos os experts treinando quando há dados de verdade.

No mesmo lab, MoE vs denso com o mesmo compute ativo: o denso venceu (222,6 vs 252,8) — com 4 experts top-1 e 300 passos, cada expert recebe ~¼ dos gradientes, e a capacidade extra (2,6× os parâmetros) não teve tempo de pagar. É o resultado esperado nessa escala e o aviso de sempre: MoE é aposta de **escala** — o retorno da capacidade extra chega com trilhões de tokens, não com milhões.

Refinamentos modernos: experts *finos* em maior número (DeepSeek: 256 pequenos > 8 grandes — mais combinações), um **expert compartilhado** sempre ativo (conhecimento comum não precisa ser roteado), e balanceamento sem loss auxiliar (bias ajustável por expert, DeepSeek-V3).

> ⚠️ **MoE e fine-tuning:** LoRA sobre MoE funciona, mas o roteador é sensível — a prática comum é congelá-lo e adaptar só os experts (ou só a atenção). Fine-tune agressivo do roteador desfaz o balanceamento aprendido no pré-treino.

---

## 3. Quantização, do lado de servir

O módulo 6 cobriu NF4 para *treinar* (QLoRA). Servir tem um cardápio próprio:

| Método | Ideia | Quando |
|---|---|---|
| **RTN** (round-to-nearest) | Arredondar por bloco, sem dados | O `mlx_lm.convert -q`; rápido e razoável |
| **GPTQ** | Quantiza coluna a coluna **corrigindo o erro** nas colunas seguintes, com dados de calibração | 4-bit de qualidade em GPU |
| **AWQ** | Identifica os ~1% de pesos *salientes* (pelas ativações) e os protege reescalando | Robusto, popular em produção |
| **GGUF (llama.cpp)** | Formatos K-quant (Q4_K_M...) com bits variáveis por bloco | O ecossistema local/Mac |

Dois princípios que unificam tudo:

1. **Outliers de ativação são o inimigo.** Do módulo 2: attention sinks produzem ativações com magnitude 100× a típica. Um único outlier num bloco arrasta a escala e esmaga a resolução dos demais valores. GPTQ/AWQ existem, em grande parte, para lidar com isso — e é por isso que quantização por bloco (módulo 6) é universal.
2. **Nem tudo se quantiza.** Normas (1.792 parâmetros críticos, módulo 2), frequentemente embeddings e `lm_head`, ficam em precisão alta. O custo é desprezível; o dano de quantizá-los, não.

E o lembrete do módulo 6, que vale dobrado em produção: **meça a degradação no SEU domínio e idioma** — a degradação em português literário foi 4× a do inglês.

---

## 4. Engenharia de inferência

### KV cache é o recurso escasso

Do módulo 1: em batch, o KV cache — não os pesos — limita quantos usuários simultâneos cabem. **PagedAttention** (vLLM) trata o cache como memória virtual: blocos de tamanho fixo, alocados sob demanda, sem fragmentação — tipicamente 2–4× mais usuários na mesma GPU.

### Continuous batching

Batching estático espera o batch inteiro terminar; uma resposta longa mantém a GPU refém. O contínuo troca sequências terminadas por novas **a cada passo de decode** — throughput 2–10× maior em cargas reais. É o default de vLLM, TGI, SGLang.

### 📐 Decodificação especulativa

A joia do módulo, porque amarra três módulos anteriores:

O decode é memory-bound: verificar `k` tokens de uma vez custa quase o mesmo que gerar 1 (é um prefill curto). Então: um **draft** pequeno propõe `k` tokens baratos; o modelo grande **verifica todos em um forward**; aceita-se o prefixo compatível.

O critério de aceitação (rejection sampling): aceite o token `t` do draft com probabilidade `min(1, p_alvo(t)/p_draft(t))`; na primeira rejeição, reamostre da distribuição residual `max(0, p_alvo − p_draft)` normalizada.

**A garantia matemática: a distribuição final é EXATAMENTE a do modelo grande.** Não é aproximação — é o mesmo texto que o alvo geraria, mais rápido. O speedup depende da **taxa de aceitação**: quanto melhor o draft imita o alvo... e "pequeno que imita o grande" é literalmente o módulo 10. Draft models são alunos destilados.

```
speedup ≈ (1 − a^{k+1}) / ((1 − a)(k·c + 1))    a = taxa de aceitação, c = custo relativo do draft
```

Com `a = 0,8`, `k = 4` e draft ~10× menor: ~2,5–3× mais rápido, de graça na qualidade.

Medido no Lab 4 (alvo de 8,1M, draft de 0,9M — 11% do alvo — ambos treinados em Machado):

| Métrica | Valor |
|---|---|
| Taxa de aceitação | **59%** |
| Tokens por forward do alvo | **3,37** (normal = 1,00) |
| Speedup de parede (até em CPU, com overhead Python) | 1,83× |

E a prova de equivalência: a distância de variação total entre a distribuição empírica do especulativo (3.000 amostras) e a exata do alvo deu **0,139 — abaixo do piso de ruído da amostragem direta (0,17–0,19)**. O especulativo é estatisticamente indistinguível de amostrar do alvo. (Com uma lição de método embutida: a primeira versão do teste usava o piso `1/√N ≈ 0,018`, que vale para um evento binário e não para distribuições de alta entropia — o piso honesto é o controle empírico.)

### As métricas

| Métrica | O que mede | Quem se importa |
|---|---|---|
| **TTFT** (time to first token) | latência do prefill | UX de chat |
| **TPOT** (time per output token) | velocidade do decode | UX de leitura |
| **Throughput** (tokens/s agregado) | custo por token | a fatura |

O trade-off central: **batch maior melhora throughput e piora TPOT** — cada usuário divide a banda de memória com os demais. Servir é escolher um ponto nessa curva; um SLA de latência define o teto do batch, e o teto do batch define o custo.

---

## 5. A conta final

Custo por milhão de tokens de saída, de primeiro princípio:

```
custo/Mtok ≈ preço_gpu_hora / (throughput_tok_s × 3600) × 10⁶
```

Uma A100 a US$ 1,50/h servindo um 8B em 4-bit com continuous batching faz ~2.000–5.000 tok/s agregados → **US$ 0,08–0,21 por Mtok** — contra US$ 0,30–10 de APIs comerciais. A margem existe, e é ela que paga toda a customização do curso; mas só aparece com utilização alta (GPU ociosa custa igual) e engenharia de serving decente. Em volume baixo, a API vence — a mesma conta do desafio do módulo 10.

---

## 6. Leituras

1. **Fedus et al. (2021), "Switch Transformers"** — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961). MoE moderno + a loss de balanceamento do lab.
2. **DeepSeek-AI (2024), "DeepSeek-V3 Technical Report"** — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437). O estado da arte em MoE eficiente (e MLA para o KV cache).
3. **Leviathan et al. (2022), "Fast Inference via Speculative Decoding"** — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192). A prova da equivalência exata.
4. **Kwon et al. (2023), "PagedAttention/vLLM"** — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
5. **Frantar et al. (2022), "GPTQ"** — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323); **Lin et al. (2023), "AWQ"** — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978).

---

## 7. Checklist de saída

- [ ] Por que o decode é memory-bound, e como cada uma das três alavancas ataca isso?
- [ ] Por que o MoE substitui só o MLP? (Duas razões — uma de parâmetros, uma de função.)
- [ ] O que "3B ativos de 30B totais" significa para compute e para memória, separadamente?
- [ ] Descreva o mecanismo do colapso de roteamento e a loss que o previne.
- [ ] Por que outliers de ativação são o inimigo da quantização, e de onde eles vêm (módulo 2)?
- [ ] O que NÃO se quantiza, e por quê?
- [ ] Enuncie a garantia da decodificação especulativa. Por que ela não degrada a qualidade?
- [ ] Qual a conexão entre draft models e o módulo 10?
- [ ] Batch maior: o que melhora, o que piora, e quem decide o ponto?
- [ ] Faça a conta: H100 a US$ 3/h, 3.500 tok/s agregados — custo por Mtok?

Depois: `lab_cpu.py` (MoE do zero + colapso + decodificação especulativa, executados), `lab_mlx.py` (MoE real, níveis de quantização e speculative no M4).
