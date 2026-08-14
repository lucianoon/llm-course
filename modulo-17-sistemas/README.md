# Módulo 17 — Sistemas de treino em escala

> **Pergunta central:** como se treina em 1.000 GPUs o que não cabe em uma — e por que juntar GPUs não dá aceleração proporcional?

O módulo 3 mostrou que um 7B custa 112 GB só de estados e milhões de dólares. Este módulo abre a caixa da engenharia que torna isso possível: o motor de gradientes por baixo do `.backward()`, o particionamento que faz o impossível caber, e a comunicação que decide se 1.000 GPUs rendem como 1.000 ou como 300.

Não dá para alugar um cluster num laptop — mas dá para **implementar o motor do zero e simular as contas que governam o cluster**. E entender essas contas é o que separa "chamei `trainer.train()`" de "sei por que meu treino distribuído está lento".

## Objetivos

1. Implementar autograd do zero e explicar o que `.backward()` faz.
2. Derivar backprop de uma MLP na mão, incluindo a derivada da cross-entropy.
3. Entender o gradient checkpointing e seu trade-off — o `--grad-checkpoint` que você já usou.
4. Explicar a progressão do ZeRO/FSDP e por que ela faz o que não cabe caber.
5. Calcular o volume de comunicação e prever quando ele domina o passo.
6. Explicar a bolha do pipeline e por que N GPUs não dão N× de velocidade.

---

## 1. 📐 Autograd — o motor por baixo de tudo

Todo treino do curso chamou `.backward()`. O que ele faz é surpreendentemente simples: constrói um **grafo** das operações do forward, e depois percorre esse grafo de trás para frente aplicando a **regra da cadeia** em cada nó.

A regra da cadeia, para uma composição `L = f(g(x))`:
```
dL/dx = dL/df · df/dg · dg/dx
```

Cada operação sabe duas coisas: como calcular sua saída (forward) e como propagar o gradiente para suas entradas (backward). O autograd só precisa:
1. Registrar cada operação e seus "filhos" (as entradas) num grafo.
2. Ordenar o grafo topologicamente (cada nó depois de todos que dependem dele).
3. Propagar de trás para frente, cada nó acumulando o gradiente nos filhos.

O Lab 1 implementa isso em ~40 linhas (o "micrograd" do Karpathy) e verifica contra o PyTorch: os gradientes batem **exatamente** (em float64). Aquelas 40 linhas SÃO o deep learning — PyTorch, JAX e o MLX do seu Mac são essa ideia escalada para tensores e GPUs.

O ponto que muda sua relação com o framework: `.backward()` não é mágica. É a regra da cadeia mecanizada sobre um grafo. Quando um treino dá NaN (módulo 3), o gradiente explodiu ou sumiu em algum nó desse grafo — e saber isso é o começo de depurá-lo.

---

## 2. 📐 A derivada que aparece em todo LLM

O Lab 2 deriva backprop de uma MLP com matrizes e confirma na mão. Uma derivada merece destaque, porque governa a loss de todo o curso:

```
∂(cross-entropy ∘ softmax)/∂logits = softmax(logits) − onehot(y)
```

"A probabilidade que o modelo deu, menos a que deveria dar (1 no token certo, 0 no resto)." É uma das expressões mais elegantes do ML, e a razão de softmax e cross-entropy virem **sempre juntos** — a composição delas tem gradiente trivial de calcular. Todo passo de treino do curso, do MiniGPT ao GRPO, empurra os logits por essa fórmula.

Saber derivá-la na mão é o que permite responder "por que meu gradiente é enorme aqui?" sem ser refém do framework — a diferença entre usar e entender.

---

## 3. Gradient checkpointing — o trade-off que você já usou

O backward precisa das **ativações** de cada camada, guardadas desde o forward (para aplicar a regra da cadeia). Numa rede profunda, isso domina a memória — mais que os próprios pesos, em contextos longos.

Checkpointing troca memória por compute: **não guarda** as ativações intermediárias e as **recalcula** no backward, refazendo o forward de cada bloco quando precisa dele. Medido no Lab 3:

| Camadas | Ativações sem ckpt | Com √n checkpoints | Economia |
|---|---|---|---|
| 24 | 3,1 MB | 0,7 MB | 79% |
| 48 | 6,3 MB | 0,9 MB | 85% |
| 96 | 12,6 MB | 1,3 MB | **90%** |

A memória de ativações cai de O(n) para O(√n) — quanto mais profunda a rede, maior a economia. O custo é ~um forward extra no backward: **+30–50% de tempo** em GPU com blocos reais.

> ⚠️ **O lab mediu 30× mais lento, não 30%** — e isso é uma distorção honesta de medir em CPU com camadas `Linear` minúsculas, onde o overhead de orquestração do checkpoint domina o recompute real. O número que TRANSFERE para o seu hardware é a economia de memória (independente de plataforma); o fator de tempo real, com blocos transformer numa GPU, é +30–50%. É a lição recorrente do curso: saiba qual número transfere e qual é artefato da escala de brinquedo.

É por isso que checkpointing é padrão em treino de LLM, e é exatamente o `--grad-checkpoint` do módulo 6: no seu M4 de 16 GB, trocar compute barato por memória escassa quase sempre compensa.

---

## 4. 📐 ZeRO/FSDP — como o que não cabe cabe

O módulo 3: um 7B pede 112 GB de estados. Como 8 GPUs de 80 GB treinam isso? Não replicando — **particionando**. O ZeRO (Rajbhandari et al.) tem estágios, cada um dividindo mais estado entre as GPUs. Medido no Lab 4 (7B, 8 GPUs, memória por GPU):

| Estágio | O que particiona | Por GPU | Cabe em 80 GB? |
|---|---|---|---|
| DDP | nada (replica tudo) | **112 GB** | ❌ |
| ZeRO-1 | estados do otimizador | 38,5 GB | ✅ |
| ZeRO-2 | + gradientes | 26,2 GB | ✅ |
| **ZeRO-3 / FSDP** | + parâmetros | **14 GB** | ✅ folgado |

A progressão é a resposta inteira para "treinar o que não cabe": DDP replica os 112 GB em cada GPU e estoura; FSDP divide TUDO por 8 e sobra memória. FSDP (Fully Sharded Data Parallel, o nome do ZeRO-3 no PyTorch) é a primeira escolha moderna. O preço está na próxima seção.

---

## 5. 📐 Comunicação — o custo escondido

Particionar não é grátis: as GPUs precisam trocar dados para sincronizar a cada passo. As duas operações coletivas centrais:

- **All-reduce** (DDP): soma os gradientes de todas as GPUs e distribui o resultado. Cada GPU acaba com a soma.
- **All-gather** (FSDP): junta os pedaços dos parâmetros de todas as GPUs antes de usar cada camada (a GPU só guarda seu pedaço, mas precisa do parâmetro inteiro para o forward).

Medido no Lab 5 (7B, 8 GPUs, comunicação por GPU por passo):

| Estratégia | Volume | Em NVLink (400 GB/s) | Em Ethernet (50 GB/s) |
|---|---|---|---|
| DDP | 24,5 GB | 61 ms | **490 ms** |
| FSDP | 36,8 GB | 92 ms | **735 ms** |

FSDP comunica ~50% mais que DDP (troca parâmetros além de gradientes). E o número que decide arquitetura de cluster: **em NVLink é barato; em Ethernet, domina o passo.** É por isso que FSDP escala lindamente DENTRO de um nó (GPUs conectadas por NVLink) e sofre ENTRE nós sem rede rápida (InfiniBand). A topologia da rede não é detalhe — é o que determina qual paralelismo usar onde.

---

## 6. 📐 A bolha do pipeline

Pipeline parallelism divide as CAMADAS entre GPUs (GPU 0: camadas 1–8, GPU 1: 9–16...). O problema: enquanto a GPU 0 processa o micro-batch, as outras esperam pelo resultado dela. Essa ociosidade é a **bolha**. Medido no Lab 6 (4 estágios):

| Micro-batches | Eficiência (GPUs úteis) |
|---|---|
| 1 | 25% |
| 4 | 57% |
| 8 | 73% |
| 16 | 84% |
| 32 | **91%** |

Com 1 micro-batch, 3 das 4 GPUs ficam ociosas na maior parte do tempo — **4 GPUs rendendo como 1**. A bolha se dilui com mais micro-batches (enquanto uma GPU processa o micro-batch 5, a anterior já processa o 6). É por isso que treinos reais usam dezenas de micro-batches por passo, e por que pipeline é a última escolha de paralelismo — só entre nós, quando a rede é lenta demais para FSDP.

---

## 7. O mapa completo (o módulo 3, agora com os custos)

| Estratégia | Divide | Custo | Quando |
|---|---|---|---|
| **DDP** | o batch (replica o modelo) | all-reduce de gradientes | o modelo cabe numa GPU |
| **FSDP / ZeRO-3** | todos os estados | all-gather a cada camada | não cabe (1ª escolha) |
| **Tensor Parallel** | matrizes dentro da camada | all-reduce a CADA camada | dentro de um nó, NVLink |
| **Pipeline** | camadas entre GPUs | a bolha | entre nós, rede lenta |

Treinos de fronteira combinam as quatro (**4D parallelism**). A regra prática: FSDP resolve quase tudo até dezenas de bilhões de parâmetros; as outras entram quando a rede ou a escala forçam. Nada disso é necessário para o que você faz no curso (uma GPU/o M4 bastam) — é o mapa para ler papers e entender por que os números de custo são o que são.

---

## 8. Leituras

1. **Karpathy, "The spelled-out intro to neural networks and backprop (micrograd)"** — [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0). O Lab 1 é isto. Assista se a seção 1 não fez clique.
2. **Rajbhandari et al. (2020), "ZeRO"** — [arXiv:1910.02054](https://arxiv.org/abs/1910.02054). Os estágios do Lab 4.
3. **Chen et al. (2016), "Training Deep Nets with Sublinear Memory Cost"** — [arXiv:1604.06174](https://arxiv.org/abs/1604.06174). O gradient checkpointing e o O(√n).
4. **Narayanan et al. (2021), "Efficient Large-Scale Language Model Training (Megatron-LM 3D parallelism)"** — [arXiv:2104.04473](https://arxiv.org/abs/2104.04473). O 4D parallelism na prática.
5. **HuggingFace, "The Ultra-Scale Playbook"** — o guia moderno e visual de treino distribuído; a referência prática que fecha este módulo.

---

## 9. Checklist de saída

- [ ] O que `.backward()` faz, em três passos (grafo, ordenação, propagação)?
- [ ] Qual a derivada de cross-entropy∘softmax, e por que as duas vêm sempre juntas?
- [ ] O que o gradient checkpointing troca por quê, e qual a economia (O(?) → O(?))?
- [ ] A progressão DDP → ZeRO-1 → 2 → 3: o que cada estágio particiona?
- [ ] Por que FSDP escala dentro de um nó e sofre entre nós?
- [ ] All-reduce vs all-gather: qual estratégia usa cada um?
- [ ] O que é a bolha do pipeline, e como se dilui?
- [ ] Qual paralelismo você escolheria para: modelo que cabe numa GPU / 70B em um nó de 8×NVLink / treino entre dois datacenters?

Depois: `lab_cpu.py` (executado — o autograd e as contas de escala) e os cartões em `revisao/baralho-02-expansao.tsv`.
