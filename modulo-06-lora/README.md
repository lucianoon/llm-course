# Módulo 6 — Fine-tuning eficiente: LoRA e QLoRA

> **Pergunta central:** como treinar um modelo de 7B numa máquina de 16 GB, se os estados do otimizador sozinhos pedem 122 GB?

Este módulo resolve o problema que os módulos 1 e 3 deixaram armado. E, diferente de quase tudo que se lê sobre LoRA, aqui as afirmações vêm com medição — inclusive as que **não** confirmam a narrativa usual.

## Objetivos

1. Enunciar e **testar** a hipótese de baixo posto, em vez de aceitá-la.
2. Implementar LoRA do zero e provar as três propriedades que o definem.
3. Escolher `rank`, `alpha` e alvos com justificativa quantitativa.
4. Explicar por que NF4 bate int4, e medir a diferença em pesos reais.
5. Montar o orçamento de memória de qualquer treino, antes de rodá-lo.
6. Saber quando LoRA **não** é a escolha certa.

---

## 1. O problema, em números

Do módulo 3, full fine-tune com AdamW em precisão mista custa **16 bytes por parâmetro**:

| Item | Bytes/param |
|---|---|
| Pesos bf16 | 2 |
| Gradientes bf16 | 2 |
| Master weights fp32 | 4 |
| Adam `m` fp32 | 4 |
| Adam `v` fp32 | 4 |

Medido no Lab 9, para os seus ~10 GB úteis:

| Modelo | Full FT | LoRA | QLoRA | Inferência 4-bit |
|---|---|---|---|---|
| Qwen2.5-0.5B | 7,9 GB | 1,0 GB | 0,3 GB | 0,2 GB |
| Qwen2.5-1.5B | **24,6 GB** | 3,2 GB | 0,9 GB | 0,8 GB |
| Qwen2.5-3B | 49,4 GB | 6,5 GB | 1,9 GB | 1,5 GB |
| Qwen2.5-7B | 121,9 GB | **16,0 GB** | 4,6 GB | 3,8 GB |
| Llama-3-8B | 128,5 GB | 16,9 GB | **4,8 GB** | 4,0 GB |

Leia a linha do 1.5B: **full fine-tune de um modelo de 1,5B não cabe no seu Mac.** Com LoRA, cabe com folga. Com QLoRA, o 8B cabe.

Duas ideias independentes produzem essa redução, e elas se compõem:

- **LoRA** ataca os 14 bytes de *estados* — gradientes e otimizador — treinando pouquíssimos parâmetros.
- **Quantização** ataca os 2 bytes dos *pesos congelados*, comprimindo-os para 0,5.

---

## 2. A hipótese de baixo posto — e o que a medição diz

A premissa do LoRA (Hu et al., 2021): a **atualização** `ΔW = W_final − W_inicial` tem posto intrínseco baixo, mesmo que `W` seja de posto cheio. Se for verdade, `ΔW ≈ B·A` com `B ∈ R^{out×r}`, `A ∈ R^{r×in}` e `r` pequeno.

O Lab 1 testa isso diretamente: fine-tuna o MiniGPT num subdomínio, calcula o SVD de `ΔW` e conta quantos valores singulares carregam 90% da energia.

| Camada | Posto máximo | r para 50% | r para 90% |
|---|---|---|---|
| `attn.qkv` | 192 | 5–6 | **26–29** |
| `attn.saida` | 192 | 6–8 | 26–38 |
| `mlp.portao` | 192 | 8–16 | 43–84 |
| `mlp.abaixo` | 192 | 15–18 | 66–76 |

**O resultado é mais matizado do que a literatura popular sugere.** Há concentração real — 5 direções bastam para metade da energia em `qkv`, contra 192 possíveis. Mas capturar 90% exige `r ≈ 26–84`, muito acima do `r=8` que todo mundo usa.

Três observações honestas sobre isso:

1. **Capturar 90% da energia de `ΔW` não é o objetivo.** O objetivo é capturar a parte de `ΔW` que **importa para a tarefa**. Direções de baixa energia podem ser ruído do otimizador.
2. **Modelos maiores são mais favoráveis.** Este experimento usa um modelo de 2M de parâmetros treinado do zero, num subdomínio muito distante do original — o caso mais adverso possível. Em modelos de bilhões, a redundância é muito maior e a mesma medição dá números proporcionalmente menores.
3. **O que decide é o resultado da tarefa, não o espectro.** E é isso que a seção 6 mede.

> 📐 Note que a hipótese é sobre `ΔW`, **não** sobre `W`. Os pesos originais são de posto cheio e carregam todo o pré-treino. O que é de baixo posto é a *correção*.

---

## 3. 📐 LoRA

Congele `W`. Aprenda duas matrizes finas:

```
y = Wx + (α/r) · B·A·x
```

| Símbolo | Forma | Inicialização |
|---|---|---|
| `W` | `[out, in]` | pré-treinada, **congelada** |
| `A` | `[r, in]` | aleatória (Kaiming) |
| `B` | `[out, r]` | **zeros** |
| `α/r` | escalar | fator de escala |

Parâmetros treináveis: `r·(in + out)` em vez de `in·out`. Medido no Lab 3:

| Matriz | Full | LoRA r=8 | Razão |
|---|---|---|---|
| Qwen 0.5B `q_proj` (896×896) | 802.816 | 14.336 | 56× |
| Llama-3-8B `q_proj` (4096×4096) | 16.777.216 | 65.536 | **256×** |
| Llama-3-8B `down_proj` (14336×4096) | 58.720.256 | 147.456 | **398×** |

Quanto maior a matriz, maior a economia — LoRA escala melhor justamente onde é mais necessário.

### As três propriedades, verificadas no Lab 2

1. **Identidade no passo 0.** Como `B = 0`, temos `BA = 0` e a saída é *exatamente* a do modelo base. O fine-tuning parte do modelo original sem choque. (Se você inicializasse ambas aleatoriamente, o primeiro forward já seria um modelo diferente e pior.)
2. **Só `A` e `B` recebem gradiente.** `W` está congelada, então não há estados de otimizador para ela — a economia de 14 bytes/param.
3. **Merge sem perda.** `W' = W + (α/r)·B·A` é uma matriz comum. Verificado com erro máximo de **2,1×10⁻⁵** em pesos reais do Qwen. Em produção, **LoRA tem custo de latência zero.**

### `alpha` e `rank`

`α/r` escala a contribuição do adaptador. A convenção comum é `α = 2r`, dando escala 2.

> ⚠️ **Convenções diferentes entre frameworks.** No PEFT você configura `lora_alpha` e o multiplicador é `lora_alpha/r`. No **MLX você configura `scale` diretamente** — ele *é* o multiplicador. O default do MLX (`scale: 20.0`, `rank: 8`) equivale a `lora_alpha = 160`, muito mais agressivo que o típico. Converta com `scale_mlx = lora_alpha / r`.

---

## 4. Onde aplicar

Do módulo 2, num bloco do Qwen2.5-0.5B: atenção = 12,3% dos parâmetros, MLP = 87,7%. Medido no Lab 6, em parâmetros de adaptador:

| Alvos | r=8 | r=64 | % do modelo (r=8) |
|---|---|---|---|
| `q_proj`, `v_proj` — *default do MLX* | 540.672 | 4.325.376 | **0,109%** |
| Atenção completa | 1.081.344 | 8.650.752 | 0,219% |
| Todas as lineares — *recomendação do QLoRA* | 4.399.104 | 35.192.832 | 0,890% |

Mesmo adaptando **todas** as camadas lineares com `r=8`, você treina menos de 1% do modelo.

**A escolha prática:**

- **Estilo, formato, tom** → `q_proj`, `v_proj` bastam. É o caso do módulo 5.
- **Conhecimento novo, tarefas complexas** → todas as lineares. O paper do QLoRA é explícito: adaptar todas as camadas foi necessário para igualar full fine-tune.
- **Memória apertada** → menos camadas (`--num-layers` no MLX), começando pelas finais.

---

## 5. Quantização

### Por que NF4 e não int4

Quantizar para 4 bits significa representar cada peso com um de **16 valores**. A pergunta é onde colocá-los.

- **int4 linear** — 16 níveis igualmente espaçados entre `−absmax` e `+absmax`.
- **NF4** — 16 níveis nos **quantis de uma normal**.

A justificativa: pesos de redes neurais são aproximadamente gaussianos, concentrados perto de zero. Níveis uniformes gastam metade da resolução em regiões quase vazias.

Medido no Lab 8, em pesos reais do Qwen2.5-0.5B:

| Camada | Erro NF4 (RMSE) | Erro int4 | NF4 melhor |
|---|---|---|---|
| `layers.0.self_attn.q_proj` | 0,006459 | 0,008268 | 1,28× |
| `layers.0.mlp.gate_proj` | 0,002142 | 0,002558 | 1,19× |
| `layers.0.mlp.down_proj` | 0,001725 | 0,002139 | 1,24× |
| **média** | | | **1,22×** |

A premissa se confirma nos dados: `gate_proj` tem média −0,00003, desvio 0,023, e seus quantis seguem de perto os de uma normal. A curtose medida é **4,16** (normal = 3,0) — caudas um pouco mais pesadas que gaussianas, o que explica por que o ganho é 1,22× e não mais.

### Blocos e double quantization

Quantiza-se em **blocos** (64 pesos), cada um com sua constante `absmax`. Um único outlier em toda a matriz arruinaria a escala global; por bloco, o dano fica contido.

Mas as constantes custam memória: um `float32` por bloco de 64 = **0,5 bit por peso**. A *double quantization* do QLoRA quantiza também essas constantes, reduzindo para ~0,127 bit/peso.

### O custo real da quantização

Aqui está o resultado mais importante do módulo para você. Lab 8, Qwen2.5-0.5B com 168 camadas quantizadas para NF4:

| Texto avaliado | Tokens | PPL fp32 | PPL NF4 | Degradação |
|---|---|---|---|---|
| **Literatura em português** | 1.293 | 43,51 | 51,07 | **+17,4%** |
| **Diálogo em português** | 1.017 | 39,08 | 43,04 | +10,1% |
| Técnico em português | 365 | 2,14 | 2,24 | +4,7% |
| Inglês | 187 | 2,09 | 2,18 | **+4,3%** |
| Código | 294 | 1,48 | 1,58 | +6,6% |
| **média** | | | | **+8,6%** |

**A quantização degrada 4× mais em português literário que em inglês.** Faz sentido e é importante: o modelo é mais fraco nesses domínios, opera com margem menor, e o ruído da quantização o empurra para fora. Quantização não degrada uniformemente — ela **amplia as fraquezas existentes**.

Se seu caso de uso é português, meça a degradação **em português**. O número que os papers reportam é quase sempre em inglês.

> ⚠️ **Como não medir.** A primeira versão deste lab usou uma única frase de 30 tokens e obteve **−13%** — a quantização teria melhorado o modelo, o que é impossível. Perplexidade em textos curtos tem variância maior que o efeito medido. É o erro mais comum em posts sobre quantização. Meça em milhares de tokens, em domínios variados, e reporte a faixa.

---

## 6. QLoRA — e o trade-off que ninguém mostra

**QLoRA** = base quantizada em NF4 (congelada) + adaptadores LoRA em bf16.

Funciona porque as duas técnicas atacam coisas diferentes: a base congelada não precisa de precisão para receber gradientes, e os adaptadores — que precisam — são minúsculos.

### O experimento honesto

Lab 4: mesmo modelo base, mesmo subdomínio, mesmos 150 passos.

| Método | Treináveis | % | Loss no alvo | Loss geral | Esquecimento |
|---|---|---|---|---|---|
| *(modelo base)* | — | — | 4,999 | 5,351 | — |
| **Full fine-tune** | 2.164.416 | 98,3% | **3,562** (−28,7%) | 6,218 | **+16,2%** |
| LoRA r=1 | 4.608 | 0,21% | 4,614 (−7,7%) | 5,554 | +3,8% |
| LoRA r=4 | 18.432 | 0,84% | 4,545 (−9,1%) | 5,600 | +4,6% |
| LoRA r=8 | 36.864 | 1,67% | 4,555 (−8,9%) | 5,586 | +4,4% |
| LoRA r=32 | 147.456 | 6,70% | 4,500 (−10,0%) | 5,609 | +4,8% |

**Duas leituras, e a segunda é a que raramente aparece:**

1. **LoRA esquece 4× menos.** Full fine-tune degradou o desempenho geral em 16,2%; LoRA, em ~4%. Os pesos originais estão congelados — o esquecimento vem só da perturbação de baixo posto. É a mitigação de catastrophic forgetting do módulo 5, agora quantificada.

2. **Neste experimento, LoRA aprendeu bem menos.** Full fine-tune ganhou 28,7% no alvo; o melhor LoRA, 10%. **LoRA não é gratuito** — é um trade-off entre capacidade de adaptação e preservação.

A literatura reporta LoRA se aproximando de full fine-tune, e o meu resultado não contradiz isso: este é o caso mais adverso possível (modelo de 2M de parâmetros treinado do zero, subdomínio muito distante, poucos passos). Em modelos grandes, com fine-tunes de comportamento e não de domínio, a lacuna é bem menor.

Mas a lição de engenharia vale: **verifique no seu caso.** Se o seu LoRA está muito atrás do que você espera, aumentar o rank é a primeira tentativa — e note na tabela que o retorno é decrescente: de `r=1` para `r=32`, 32× mais parâmetros compraram 2,3 pontos percentuais.

---

## 7. Variantes

| Variante | Ideia | Quando |
|---|---|---|
| **DoRA** | Decompõe em magnitude e direção, adaptando cada uma | Ganho consistente sobre LoRA em rank baixo; disponível no MLX (`--fine-tune-type dora`) |
| **rsLoRA** | Escala `α/√r` em vez de `α/r` | Ranks altos (≥64), onde LoRA padrão satura |
| **LoRA+** | Learning rate maior para `B` que para `A` | Convergência mais rápida, custo zero |
| **PiSSA** | Inicializa `A`,`B` com os componentes principais de `W` | Convergência mais rápida |
| **QA-LoRA** | Mantém a quantização após o merge | Quando o alvo é servir quantizado |

> ⚠️ **A armadilha do merge em QLoRA.** Você treinou sobre uma base NF4, mas o merge produz pesos em precisão cheia. Se depois você requantizar o modelo fundido, o resultado **não** é o que você treinou — a base requantizada difere daquela usada no treino. Ou sirva com o adaptador acoplado (sem merge), ou use QA-LoRA. Este é um erro silencioso e comum.

---

## 8. Quando LoRA não é a escolha

- **Idioma ou domínio muito fora da distribuição.** Adaptar um modelo majoritariamente inglês para um idioma pouco representado exige mudar representações profundas — full fine-tune (ou continued pre-training) costuma ser necessário.
- **Vocabulário novo.** Se você adiciona tokens, as linhas novas da matriz de embeddings precisam ser treinadas de verdade.
- **Você tem GPU sobrando e dados abundantes.** Aí full fine-tune é simplesmente melhor, como a tabela da seção 6 mostra.

---

## 9. Leituras

1. **Hu et al. (2021), "LoRA"** — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). A seção 7.2 é a análise de posto que o Lab 1 replica.
2. **Dettmers et al. (2023), "QLoRA"** — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314). NF4, double quantization e paged optimizers.
3. **Liu et al. (2024), "DoRA"** — [arXiv:2402.09353](https://arxiv.org/abs/2402.09353).
4. **Biderman et al. (2024), "LoRA Learns Less and Forgets Less"** — [arXiv:2405.09673](https://arxiv.org/abs/2405.09673). Confirma em escala real exatamente o trade-off que o Lab 4 mediu. Leia depois de rodar o lab.

---

## 10. Checklist de saída

- [ ] Por que a hipótese de baixo posto é sobre `ΔW` e não sobre `W`?
- [ ] O que a medição do espectro mostrou, e por que `r=8` funciona apesar dela?
- [ ] Por que `B` é inicializada em zero? O que aconteceria se as duas fossem aleatórias?
- [ ] Por que LoRA tem custo de latência zero em produção?
- [ ] Converta `LoraConfig(r=16, lora_alpha=32)` para o YAML do MLX.
- [ ] Por que NF4 bate int4, e quanto — no Qwen2.5-0.5B?
- [ ] Por que se quantiza em blocos, e quanto custam as constantes?
- [ ] A quantização degradou mais em qual domínio, e por quê?
- [ ] Cite os dois lados do trade-off LoRA vs full fine-tune, com números.
- [ ] Por que fundir um adaptador QLoRA e requantizar é problemático?
- [ ] Full fine-tune do Qwen2.5-1.5B cabe nos seus 16 GB? E QLoRA do 8B?

---

Comece pelo **`lab_cpu.py`** — ele roda no Windows e no Mac, e é onde tudo isso foi verificado. Depois o **`lab_mlx.py`**, que aplica no seu M4 com modelos de verdade.
