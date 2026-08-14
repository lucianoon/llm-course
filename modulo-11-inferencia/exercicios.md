# Módulo 11 — Exercícios

💻 = qualquer máquina | 🍎 = Mac

---

## Parte A — Conceituais

### A1. A anatomia do MoE

O Qwen3-30B-A3B tem 30B parâmetros totais, ~3B ativos, 128 experts com top-8.

a) Quanta memória para servi-lo em 4 bits? Cabe no seu M4?
b) Qual o custo de *compute* por token, comparado a um denso de 30B? E a banda de memória lida por token?
c) Por que "cabe na memória" e "roda rápido" são perguntas independentes num MoE?

<details><summary>Gabarito</summary>

a) `30e9 × 0,5 byte ≈ 15 GB` + overhead → **não cabe** confortavelmente em 16 GB com sistema e KV cache. (O Qwen1.5-MoE-A2.7B, 14B totais ≈ 8 GB, cabe — por isso o lab usa ele.)

b) Compute: ~3B ativos → ~1/10 do denso de 30B, próximo de um denso de 3B. Banda por token: idem — só os experts ativos são lidos (com a ressalva da localidade: tokens diferentes ativam experts diferentes, então o conjunto de trabalho real por *batch* é maior que 3B).

c) Porque a memória paga os parâmetros **totais** (todos residentes) e a velocidade paga os **ativos** (lidos por token). MoE é a arquitetura que separa as duas contas — e por isso é ideal onde memória sobra (servidores com muita RAM, Macs grandes) e ruim onde ela é o teto (o seu caso com 30B).
</details>

---

### A2. O colapso, passo a passo

Descreva o ciclo de feedback do colapso de roteamento em 4 passos, e aponte em qual passo a loss auxiliar `α·E·Σ f_i·P_i` intervém.

<details><summary>Gabarito</summary>

1. Por inicialização aleatória, o expert `j` é marginalmente melhor para os tokens típicos.
2. O roteador, treinado para minimizar a loss, envia mais tokens a `j`.
3. `j` recebe mais gradientes e melhora mais rápido; os demais ficam para trás.
4. O gap cresce; o roteador concentra ainda mais; os outros experts nunca mais treinam. Ponto fixo: um denso pequeno pagando memória de E.

A loss auxiliar intervém no **passo 2**: ela penaliza a própria concentração (`f_i` alto × `P_i` alto), adicionando um gradiente que empurra o roteador para espalhar ANTES que o gap de qualidade dos passos 3–4 se consolide. Por isso `α` precisa estar ligado desde o início — depois do colapso, reativá-la não recupera experts mortos (eles são piores de fato, e espalhar tokens para eles piora a loss principal).
</details>

---

### A3. A garantia do especulativo

a) Prove informalmente que a aceitação com `min(1, p_alvo/p_draft)` + reamostragem da residual produz amostras exatas de `p_alvo`.
b) Se a garantia é exata, por que a escolha do draft importa?
c) Por que o mesmo tokenizer é requisito duro?

<details><summary>Gabarito</summary>

a) Para um token `t`: P(sair `t`) = P(draft propõe `t`)·P(aceitar) + P(rejeitar algo)·P(residual = `t`)
= `p_d(t)·min(1, p_a(t)/p_d(t)) + (Σ_s p_d(s)·max(0, 1−p_a(s)/p_d(s)))·residual(t)`.
O primeiro termo dá `min(p_d, p_a)(t)`; a residual é `max(0, p_a−p_d)/Z` com `Z = Σ max(0, p_a−p_d)` — que é exatamente a massa rejeitada. Somando: `min(p_d,p_a)(t) + max(0, p_a−p_d)(t) = p_a(t)`. ∎ (O Lab 4 verifica empiricamente: distância de variação total na ordem do ruído amostral.)

b) A garantia é sobre a **distribuição**, não sobre a **velocidade**. Draft ruim → taxa de aceitação baixa → você paga o draft e quase todo token vem da reamostragem do alvo → *mais lento* que o alvo sozinho. A qualidade nunca degrada; o ganho de tempo pode virar perda.

c) A comparação `p_alvo(t)/p_draft(t)` exige que `t` seja o MESMO evento nas duas distribuições — o mesmo id sobre o mesmo vocabulário. Tokenizers diferentes segmentam o texto diferente; não há correspondência token a token para comparar. (Existem variantes por bytes, mas são exóticas.)
</details>

---

### A4. O cardápio de quantização

Escolha o método e justifique:

1. Servir um 7B no seu M4 para uso pessoal.
2. Servir um 70B em produção numa A100, com 50 exemplos de calibração disponíveis.
3. Distribuir um modelo para usuários rodarem em llama.cpp/Ollama.
4. Um modelo cujo desempenho em português é crítico.

<details><summary>Gabarito</summary>

1. **RTN via `mlx_lm.convert -q`** — sem dados de calibração, qualidade suficiente, um comando. O que o curso inteiro usa.
2. **AWQ ou GPTQ** — os 50 exemplos de calibração pagam qualidade: AWQ protege os pesos salientes pelas ativações; GPTQ corrige o erro coluna a coluna. Em 70B/4-bit para produção, a diferença sobre RTN é mensurável.
3. **GGUF K-quants** (Q4_K_M como default) — é o formato do ecossistema; e o `mlx_lm.fuse --export-gguf` faz a ponte.
4. Qualquer um dos acima **com avaliação em português antes e depois** — a lição medida do módulo 6 (+17% em literatura PT vs +4% em inglês no mesmo modelo). E calibração (caso 2) com dados EM PORTUGUÊS: calibrar em inglês otimiza o erro na distribuição errada.
</details>

---

### A5. O dimensionamento

Um produto precisa servir 200 usuários simultâneos de chat, SLA de TPOT ≤ 50 ms (20 tok/s por usuário), respostas médias de 300 tokens, modelo de 8B.

a) Que throughput agregado o sistema precisa?
b) Uma H100 com vLLM faz ~3.000–6.000 tok/s agregados num 8B-FP8 com batching. Quantas GPUs?
c) Que papel o KV cache tem na conta — e qual otimização de arquitetura (módulo 2) ajuda?

<details><summary>Gabarito</summary>

a) `200 × 20 = 4.000 tok/s agregados` — mantendo o TPOT individual de 50 ms.

b) 1–2 H100s *se* o batch de 200 couber na memória e o TPOT se sustentar nesse batch — que é a pegadinha: throughput agregado cresce com o batch, mas o TPOT individual degrada. O dimensionamento certo procura o maior batch cujo TPOT ainda ≤ 50 ms, e divide os 200 usuários pelas GPUs necessárias para isso. Com margem para picos: 2–3.

c) 200 sequências × KV cache por token × contexto médio: num 8B com GQA (módulo 1: ~128 KB/token), 200 × 2.000 tokens ≈ **51 GB só de cache** — mais que os pesos! É o KV cache que limita o batch, logo o throughput, logo o custo. GQA (já no modelo) reduz 4×; MLA (DeepSeek) ainda mais; PagedAttention elimina o desperdício de alocação. A conta inteira do serving gira em torno do cache.
</details>

---

## Parte B — Práticas

### B1. 💻 Top-2 e a granularidade

O lab treina com top-1 (Switch). Treine com top-2 e com 8 experts menores (`d_ff` na metade, top-2) — mesmo compute ativo nos três.

Compare PPL e utilização. O que a granularidade maior (mais experts, menores) compra?

<details><summary>Gabarito esperado</summary>

Top-2 sobre 4 experts: mais estável (cada token tem um segundo voto; gradientes fluem para mais experts), PPL igual ou melhor. 8 experts finos com top-2: mais **combinações** possíveis (28 pares vs 6) — especialização mais fina com o mesmo compute.

É a tendência da área em miniatura: DeepSeek-V3 usa 256 experts finos + top-8 exatamente por isso. O custo: roteamento mais difícil de balancear (mais experts para colapsar) — observe a utilização.
</details>

---

### B2. 💻 O expert compartilhado

Implemente a variante DeepSeek: um expert **sempre ativo** somado ao top-1 roteado (ajuste `d_ff` para manter o compute). Compare com top-2 puro.

<details><summary>Gabarito esperado</summary>

O compartilhado captura o "conhecimento comum" (sintaxe, tokens frequentes) e libera os roteados para especializar — espere utilização mais equilibrada dos roteados (a pressão de winner-take-all cai, porque o generalista já existe) e PPL igual ou melhor.

É uma das ideias mais elegantes do DeepSeekMoE: transformar o colapso de "todo mundo quer ser o generalista" em arquitetura — nomeie um generalista e acabou a disputa.
</details>

---

### B3. 💻 A curva de aceitação

No especulativo do lab, varie: (a) a qualidade do draft (treine drafts de 100, 400 e 800 passos) e (b) o `k` (2, 4, 8). Meça taxa de aceitação e tokens/forward-do-alvo para cada célula.

Onde está o ótimo? Por que `k` grande com draft fraco é o pior canto?

<details><summary>Gabarito esperado</summary>

Tokens/forward ≈ `(1−a^{k+1})/(1−a)` cresce com `k` só enquanto `a` é alto. Com draft fraco (a ≈ 0,4), a chance de aceitar 8 seguidos é ~0,07% — você paga 8 forwards do draft para aproveitar ~1,6 tokens. O ótimo prático: draft forte com k=4–5.

A curva medida deve mostrar: melhorar o draft desloca o ótimo de `k` para cima. É por isso que os drafts de produção são destilados do próprio alvo (módulo 10) — cada ponto de aceitação vale mais que qualquer ajuste de `k`.
</details>

---

### B4. 🍎 A escada completa

Estenda o Lab 1 do lab_mlx com 3 bits (`--q-bits 3`, se a versão suportar) e grupos diferentes (`--q-group-size 32` vs `64` vs `128`) no 4-bit.

Monte a tabela completa bits×grupo → memória, tok/s, PPL-PT. Onde está o joelho da curva qualidade/custo?

<details><summary>Gabarito esperado</summary>

Espere: 8→4 bits quase sem dano de PPL (o joelho clássico); 4→3 com dano visível e crescente. Grupos menores (32) melhoram a PPL do 4-bit ao custo de mais metadados (módulo 6, B3: 32/grupo = 1 bit/peso de constantes!) — memória e velocidade pioram um pouco.

O joelho para uso geral: **4-bit com grupo 64** — que é exatamente o default do ecossistema, agora justificado pelos seus números.
</details>

---

### B5. 🍎 O custo por milhão do seu Mac

Feche a conta da seção 5 do README com números seus:

1. Meça o decode tok/s do modelo que você de fato usaria (o 1.5B fine-tunado do curso, ou o 7B-4bit).
2. Atribua um custo/hora ao M4 (energia ~10-30W no decode + depreciação, ou o custo de oportunidade que preferir).
3. Calcule o custo/Mtok e compare com 2–3 APIs comerciais.
4. Encontre o volume mensal em que uma GPU alugada (A100 spot a ~US$ 1,50/h) supera as duas opções.

<details><summary>Gabarito esperado</summary>

Ordem de grandeza esperada: M4 fazendo 20–60 tok/s num 1.5B–7B ≈ 70–200 Mtok/mês rodando 24/7 — a um custo marginal de energia de ~R$ 10–30/mês. Para uso pessoal e desenvolvimento, imbatível.

O crossover para GPU alugada chega quando (a) o volume exige throughput que o M4 não tem, ou (b) a latência importa comercialmente. A resposta certa do exercício não é um número — é o hábito de fazer esta conta antes de qualquer decisão de infra.
</details>

---

## Desafio — o relatório de serving

Escolha o modelo final de um dos seus experimentos do curso (o LoRA de suporte do módulo 5, o destilado do 10, ou outro) e produza um **relatório de deployment** de uma página:

1. **Requisitos:** volume esperado, SLA de TTFT/TPOT, orçamento.
2. **A escolha de servir:** M4 local / GPU alugada / API — com a conta do B5 para os três.
3. **Formato do modelo:** bits, grupo, com ou sem adaptador fundido (módulo 6: a armadilha do merge+requantização!).
4. **Otimizações aplicáveis:** especulativo (tem draft da mesma família?), prefix caching (o system prompt é longo e fixo?), batching (o volume justifica?).
5. **Medições:** TTFT, TPOT e PPL/qualidade no SEU domínio, antes e depois de cada otimização.
6. **O ponto de operação escolhido** e o custo/Mtok resultante.

Este relatório é o entregável que o módulo 12 vai pedir como componente — e é, na prática profissional, o documento que separa "treinei um modelo" de "entreguei um sistema".
