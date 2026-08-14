# Módulo 17 — Exercícios

Todos rodam em CPU (é o motor e as contas, não o cluster). 💻

---

## Parte A — Conceituais

### A1. O grafo de gradientes

Para `L = (a·b + c)·d`, com a=2, b=3, c=1, d=4:

a) Calcule L.
b) Desenhe o grafo e calcule ∂L/∂a, ∂L/∂b, ∂L/∂c, ∂L/∂d na mão pela regra da cadeia.
c) Por que o autograd precisa da ordenação topológica?

<details><summary>Gabarito</summary>

a) L = (2·3 + 1)·4 = 7·4 = **28**.

b) Seja `u = a·b + c = 7`. `L = u·d`.
- ∂L/∂d = u = **7**
- ∂L/∂u = d = 4
- ∂L/∂a = ∂L/∂u · ∂u/∂a = 4·b = **12**
- ∂L/∂b = 4·a = **8**
- ∂L/∂c = 4·1 = **4**

c) Porque um nó só pode propagar seu gradiente depois que TODOS os nós que dependem dele já propagaram os deles para ele (senão seu `.grad` estaria incompleto). A ordenação topológica garante essa ordem: processar do fim (a loss) para o início, cada nó depois de todos os seus "pais". Sem ela, um nó com múltiplos caminhos até a loss (comum em redes reais) teria gradiente errado.
</details>

---

### A2. Onde a memória vai

Um treino de um modelo de 3B em fp16/AdamW, batch 8, seq 2048, 32 camadas, d=2560.

a) Estime a memória de pesos + gradientes + otimizador.
b) Estime a memória de ATIVAÇÕES (aprox.: batch × seq × d × n_camadas × 2 bytes × ~4 tensores/camada).
c) Qual domina, e o que o checkpointing faz com cada um?

<details><summary>Gabarito</summary>

a) Estados: 3e9 × 16 bytes = **48 GB** (pesos 2 + grad 2 + otim 12).

b) Ativações ≈ 8 × 2048 × 2560 × 32 × 2 × 4 ≈ 8 × 2048 × 2560 × 32 × 8 ≈ **~86 GB** (grosseiro; o fator de tensores/camada varia com a arquitetura).

c) As ativações DOMINAM em contexto/batch grandes — e é exatamente o que muita gente não percebe (foca nos 48 GB de estados e esquece os 86 de ativações). Checkpointing corta as ativações para O(√32)·(...) ≈ ~15 GB, **sem tocar nos 48 GB de estados**. É por isso que checkpointing é a primeira alavanca quando o treino estoura a memória com sequências longas: ele ataca o termo que cresce com batch×seq, que os outros truques (LoRA, quantização) não atacam.
</details>

---

### A3. A escolha do paralelismo

Para cada cenário, qual estratégia (DDP, FSDP, Tensor, Pipeline, ou combinação)?

1. Modelo de 1B, você tem 4 GPUs de 24 GB conectadas por PCIe.
2. Modelo de 70B, um nó com 8 GPUs de 80 GB e NVLink.
3. Modelo de 400B, 4 nós de 8 GPUs, InfiniBand entre nós, NVLink dentro.
4. Fine-tuning LoRA de um 7B, uma GPU de 24 GB.

<details><summary>Gabarito</summary>

1. **DDP** — o 1B cabe folgado numa GPU de 24 GB (2 GB pesos + estados); só replique e divida o batch. A comunicação (all-reduce) em PCIe é tolerável para um modelo pequeno.
2. **FSDP** dentro do nó — 70B não cabe numa GPU (140 GB em bf16); FSDP divide os estados pelas 8, ~virando ~30-40 GB/GPU com otimizador, e o NVLink torna o all-gather barato. Talvez + tensor parallel se a latência de camada importar.
3. **4D: FSDP/tensor dentro do nó (NVLink), pipeline entre nós (InfiniBand)** — 400B força a combinação. Pipeline entre nós porque a bolha é preferível ao all-gather de FSDP sobre a rede mais lenta; tensor+FSDP dentro do nó onde a banda é alta.
4. **Nenhum paralelismo** — uma GPU basta com LoRA/QLoRA (módulo 6). Distribuir seria complexidade sem ganho. A resposta certa às vezes é "não distribua".
</details>

---

### A4. A conta do cluster

Você vai treinar um 13B em 15T tokens. Tem acesso a 256 H100 (NVLink dentro de nós de 8, InfiniBand entre nós), MFU esperado de 40%.

a) Quantas GPU-horas (fórmula do módulo 3)?
b) Quantos dias com escalabilidade perfeita? E com 85% de eficiência de escala?
c) Onde os 15% de perda vão?

<details><summary>Gabarito</summary>

a) FLOPs = 6 × 13e9 × 15e12 = 1,17e24. H100 densa = 495 TFLOP/s × 40% = 1,98e14 FLOP/s. Tempo = 1,17e24 / 1,98e14 = 5,9e9 s = **1,64 milhão de GPU-horas**.

b) Perfeita: 1,64e6 / 256 / 24 = **267 dias**... espera, isso está errado para um 13B. Recalculando o wall-clock: 1,64e6 GPU-horas / 256 GPUs = 6.400 horas = 267 dias por GPU?? Não — 1,64e6 horas ÷ 256 = 6.400 horas de wall-clock = **267 dias**. (13B em 15T tokens é de fato um treino de meses; é por isso que só grandes labs os fazem.) Com 85% de eficiência: 267 / 0,85 ≈ **314 dias**.

c) Os 15%: comunicação (all-gather/all-reduce que não se sobrepõe ao compute), a bolha do pipeline, desbalanceamento de carga entre GPUs, e — em clusters grandes — **falhas de hardware e reinícios** (numa escala de 256 GPUs por meses, GPUs morrem; cada falha custa o tempo desde o último checkpoint). O logbook do OPT (módulo 3) documenta isso. A lição: eficiência de escala é uma métrica de engenharia tão importante quanto o MFU.
</details>

---

### A5. O NaN às 3 da manhã

Seu treino distribuído roda por 6 horas e a loss vira NaN. Usando o que o módulo ensinou sobre o grafo de gradientes, liste a ordem de investigação.

<details><summary>Gabarito</summary>

1. **Onde no grafo?** Registre a norma do gradiente POR camada (hook no backward). NaN costuma nascer em um ponto específico e se propagar; achar a primeira camada com NaN localiza a causa.
2. **Overflow numérico?** Em fp16 (não bf16), ativações ou gradientes podem estourar 65.504 (módulo 3). Verifique se é fp16 sem loss scaling adequado; a norma explodindo antes do NaN confirma.
3. **Divisão por zero?** RMSNorm/LayerNorm com variância zero (uma sequência de tokens idênticos), ou softmax com todos -inf (máscara errada). O upcast para fp32 nas normas (módulo 2) previne parte disso.
4. **Um batch podre?** Dados corrompidos produzindo gradiente gigante; o gradient clipping (módulo 3) deveria pegar — verifique se está ativo e ANTES do step.
5. **Dessincronização entre GPUs?** Em treino distribuído, uma GPU com estado divergente (checkpoint corrompido, ordem de dados diferente) contamina o all-reduce. Compare a loss por GPU.

A metodologia é a do módulo: o NaN é um nó do grafo onde a regra da cadeia produziu inf/nan. Localize o nó, entenda a operação, corrija a causa. E — a lição operacional — **checkpoints frequentes** transformam "perdi 6 horas" em "perdi 20 minutos".
</details>

---

## Parte B — Práticas

### B1. 💻 Autograd com mais operações

Estenda o `Valor` do Lab 1 com `__pow__`, `exp`, `log` e `relu` (cada uma com seu `_backward`). Verifique cada uma contra o PyTorch (float64). Depois construa um neurônio (`w·x + b` seguido de relu) e treine-o com descida de gradiente para aprender uma função simples.

<details><summary>Gabarito esperado</summary>

As derivadas: `x^n → n·x^(n-1)`; `exp(x) → exp(x)·grad`; `log(x) → grad/x`; `relu(x) → grad·(x>0)`.

O valor do exercício é fechar o ciclo: com essas operações você tem um framework de deep learning completo (escalar) — forward, backward, e um loop de treino. É literalmente o que o PyTorch faz, e implementá-lo remove para sempre a sensação de que `.backward()` é mágica. Se você conseguir treinar até um XOR (a função não-linear clássica), construiu uma rede neural do zero, do autograd ao treino.
</details>

---

### B2. 💻 A curva do checkpointing

Meça (tempo E a economia de memória calculada) do checkpointing para redes de 12, 24, 48, 96 camadas. Plote memória × profundidade para as duas estratégias (todas as ativações vs √n checkpoints).

Confirme a lei O(n) vs O(√n) e explique por que a economia CRESCE com a profundidade.

<details><summary>Gabarito esperado</summary>

A memória sem checkpoint cresce linearmente (O(n)); com √n checkpoints, cresce como √n. A economia relativa `1 − √n/n = 1 − 1/√n` cresce com n: 79% em 24 camadas, 90% em 96, tendendo a 100% para redes muito profundas.

A intuição: checkpointing guarda √n "âncoras" e recalcula os √n blocos entre elas sob demanda. Quanto mais profunda a rede, mais a razão entre "guardar tudo" e "guardar as âncoras" pende para a economia. É por isso que modelos muito profundos (dezenas de bilhões, centenas de camadas) dependem criticamente dele.
</details>

---

### B3. 💻 O simulador de cluster

Escreva `simular_treino(n_params, n_gpus, banda_gbs, estrategia)` que estime o tempo por passo somando compute + comunicação, e use-o para responder: para um 30B, a partir de quantas GPUs a comunicação (em Ethernet de 50 GB/s) passa a dominar o compute?

<details><summary>Gabarito esperado</summary>

O compute por GPU CAI com mais GPUs (o trabalho se divide); a comunicação por GPU é ~constante ou cresce (mais GPUs = mais sincronização). Existe um ponto de cruzamento onde adicionar GPUs para de acelerar — a lei de Amdahl da parte sequencial/comunicação.

Em Ethernet lento, esse ponto chega cedo (dezenas de GPUs); em NVLink/InfiniBand, muito mais tarde (milhares). O simulador torna concreta a regra da seção 5: a topologia da rede define a escala máxima útil. É a conta que um engenheiro de infra de ML faz antes de dimensionar um cluster — e a razão de a interconexão custar tanto quanto as GPUs em supercomputadores de IA.
</details>

---

### B4. 💻 A bolha, visualizada

Implemente a simulação do pipeline como uma grade tempo × GPU (quem está processando o quê em cada slot), e desenhe-a em ASCII para 4 estágios com 1, 2 e 8 micro-batches. Marque as células ociosas (a bolha).

Confirme a fórmula de eficiência do Lab 6 contando as células úteis.

<details><summary>Gabarito esperado</summary>

Você vai desenhar o clássico diagrama de "enchimento e esvaziamento" do pipeline: no início, só a GPU 0 trabalha (as outras esperam o forward chegar); no fim, só a última (as outras já terminaram). O losango de células úteis no meio, cercado por triângulos de bolha nas pontas.

Contar as células confirma `eficiência = M/(M+E-1)` onde M=micro-batches, E=estágios. Ver o diagrama torna óbvio por que mais micro-batches ajudam: eles alongam o losango útil sem alongar os triângulos de bolha (que dependem só de E). É o mesmo insight de todo sistema de pipeline — de CPUs a linhas de montagem.
</details>

---

## Desafio — o playbook do seu treino

Você não vai treinar um modelo de fronteira, mas vai enfrentar decisões de escala no seu M4 e em GPUs alugadas. Produza um **playbook de decisão** de uma página:

1. **A árvore de memória:** dado (params, batch, seq, otimizador), calcule os quatro termos (pesos, grad, otim, ativações) e diga qual domina. Onde cada alavanca (LoRA, quantização, checkpointing, batch menor) ataca.
2. **A árvore de paralelismo:** dado (tamanho do modelo, nº e tipo de GPUs, interconexão), a estratégia — com a conta de comunicação que a justifica.
3. **O checklist do NaN** (A5) e o de checkpoints frequentes.
4. **A conta de custo** (módulo 3): GPU-horas, dias, dólares, para o seu treino realista.

Aplique ao caso concreto: fine-tunar um 7B no seu M4 (o que o módulo 6 já decidiu — mas agora com a árvore de memória explícita mostrando POR QUE QLoRA é a única opção) e ao caso hipotético de escalá-lo para um 70B em GPUs alugadas.

Este playbook é o que um engenheiro de ML consulta antes de cada treino. Tê-lo escrito, com as contas que você mesmo derivou, é a diferença entre seguir tutoriais e tomar decisões.
