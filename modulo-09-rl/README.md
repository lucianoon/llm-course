# Módulo 9 — Reinforcement Learning: PPO e GRPO

> **Pergunta central:** quando existe um verificador — um teste que passa, uma resposta que confere — como transformar esse sinal em treino, sem pares, sem anotação, sem imitação?

Este é o degrau acima do DPO. O DPO precisa de pares prontos; o RL só precisa de uma **função de recompensa**. Se você consegue *verificar* uma resposta automaticamente, o modelo pode gerar as próprias tentativas e aprender das que funcionam — inclusive descobrindo estratégias que nenhum dado de SFT continha. Foi assim que nasceram o o1 e o DeepSeek-R1.

## Objetivos

1. Derivar o policy gradient (REINFORCE) e entender por que a variância é o inimigo.
2. Explicar o PPO — e por que ele exige quatro modelos na memória.
3. Derivar o GRPO como "PPO sem value model" e implementá-lo do zero.
4. Conhecer a receita do R1: recompensas por regras, sem reward model neural.
5. Produzir e diagnosticar **reward hacking** — e saber as defesas.
6. Decidir entre SFT, DPO e RL com critério.

---

## 1. O cenário: recompensa verificável

| Sinal | Exemplo | Módulo |
|---|---|---|
| Alvo a imitar | resposta escrita por humano | 5 (SFT) |
| Preferência entre dois | chosen ≻ rejected | 8 (DPO) |
| **Recompensa por tentativa** | teste passa? resposta = 42? JSON válido? | **9 (RL)** |

A recompensa verificável é o sinal mais barato de todos **quando existe**: nenhum humano no loop, nenhum juiz com viés, escala infinita. Matemática (resposta confere), código (testes passam), formato (parser aceita), jogos (vitória). O paradigma pós-R1 tem nome: **RLVR** — RL with Verifiable Rewards.

O que muda em relação a tudo que veio antes no curso: **o modelo gera os próprios dados de treino.** A cada passo, ele tenta; o verificador pontua; o gradiente reforça o que pontuou bem. Nenhum dataset fixo — a distribuição de treino é sempre a política atual (on-policy), eliminando o gap off-policy que assombrou o módulo 8.

---

## 2. 📐 Policy gradient do zero

Queremos maximizar a recompensa esperada das gerações:

```
J(θ) = E_{y ~ π_θ}[ R(y) ]
```

O problema: `R` não é diferenciável (é um teste que passa ou não), e a amostragem também não. O **log-derivative trick** resolve:

```
∇J = E_{y ~ π_θ}[ R(y) · ∇ log π_θ(y) ]
```

📐 A derivação em uma linha: `∇π = π·∇log π`, então `∇E[R] = Σ R·∇π = Σ R·π·∇log π = E[R·∇log π]`.

Isto é o **REINFORCE** (Williams, 1992): gere `y`, calcule `R(y)`, e empurre a log-prob da sequência inteira **proporcionalmente à recompensa**. Respostas boas ficam mais prováveis; ruins, menos. É um SFT ponderado pela recompensa — a conexão com tudo que você já sabe.

### O inimigo: variância

O estimador é não-enviesado e **ruidosíssimo**. Se todas as recompensas são positivas (digamos 0,7–1,0), *toda* geração é reforçada — inclusive as piores — e a diferenciação vem só da magnitude, afogada em ruído de amostragem.

A correção clássica: subtrair uma **baseline** `b` que não depende da ação:

```
∇J = E[ (R(y) − b) · ∇ log π(y) ]
```

📐 Subtrair `b` não enviesa (porque `E[∇log π] = 0` — a soma das probabilidades é constante), mas reduz drasticamente a variância se `b ≈ E[R]`. A quantidade `A = R − b` é a **vantagem**: não "quão boa foi a resposta", mas "quão melhor que o esperado". Medido no Lab 1: mesma direção de gradiente, variância **53,7× menor** — só de subtrair a média.

Toda a diferença entre os algoritmos desta área está em **como estimar a baseline**.

---

## 3. PPO — a resposta de 2017, e seu preço

O PPO (Schulman et al.) adiciona duas peças ao REINFORCE:

### O value model como baseline

Um **segundo modelo treinado** `V(s)` prevê a recompensa esperada a partir de cada estado. A vantagem vira `A = R − V(s)` (refinada pelo GAE, que interpola horizontes). Baseline precisa, por token.

### A trust region: o clipped surrogate

Reutilizar as mesmas gerações por vários passos de gradiente tira o treino de on-policy. O PPO permite isso com segurança limitando o quanto a política nova pode se distanciar da que gerou os dados:

```
ratio = π_θ(y) / π_old(y)
L = −E[ min( ratio·A, clip(ratio, 1−ε, 1+ε)·A ) ]
```

Se o ratio sai da janela `[1−ε, 1+ε]` (ε ≈ 0,2), o gradiente é **cortado** — o incentivo para se afastar mais desaparece. É uma trust region de uma linha.

### O preço

Na memória, simultaneamente: **política** (treina), **referência** (KL), **reward model** (pontua), **value model** (baseline, treina). Quatro modelos, dois deles recebendo gradiente. Para um 7B, é infraestrutura de cluster — e o value model é notoriamente difícil de treinar bem em texto.

---

## 4. 📐 GRPO — a baseline grátis

A pergunta do DeepSeek (2024): *e se a baseline não precisasse de um modelo?*

**Group Relative Policy Optimization:** para cada prompt, gere um **grupo** de `G` respostas (4–16). A baseline é a média do grupo:

```
A_i = ( r_i − mean(r_1..r_G) ) / std(r_1..r_G)
```

A vantagem de cada resposta é seu z-score **dentro do grupo**. Acima da média do grupo → reforça; abaixo → suprime. O value model desaparece — e com ele metade da memória e a fonte número um de instabilidade do PPO.

O resto é PPO: clipped surrogate sobre o ratio, mais uma penalidade KL contra a referência congelada (o estimador k3, `π_ref/π − log(π_ref/π) − 1`, não-enviesado e sempre positivo).

```
L_GRPO = −E[ min(ratio·A, clip(ratio)·A) − β·KL(π‖π_ref) ]
```

### As consequências do design

- **A recompensa pode ter qualquer escala** — a normalização por grupo torna o algoritmo invariante a deslocamento e escala de `R`. Recompensa 0/1 funciona tão bem quanto 0/100.
- **Se o grupo inteiro acerta (ou erra), a vantagem é zero** — nada a aprender daquele prompt. O treino se concentra automaticamente nos prompts de dificuldade intermediária: é um currículo emergente. (E é por isso que a *seleção de prompts* — nem fáceis demais, nem impossíveis — importa tanto no RLVR.)
- **Variantes:** `Dr. GRPO` remove a divisão por std (que enviesa contra grupos de baixa variância) e a normalização por comprimento (que favorece respostas longas erradas). O campo ainda está assentando.

---

## 5. A receita do R1

O DeepSeek-R1-Zero é o experimento mais limpo da história recente: **modelo base + GRPO + recompensas por regra. Sem SFT, sem reward model neural.**

As recompensas, literalmente:

1. **Acurácia** — a resposta final (extraída de um formato fixo) confere com o gabarito?
2. **Formato** — o raciocínio está entre `<think>` e `</think>`?

Só isso. E do treino emergiram — sem nunca terem sido demonstrados — cadeias de raciocínio cada vez mais longas, auto-verificação, backtracking ("Wait, let me reconsider"), o chamado *aha moment*. O RL não ensinou *como* raciocinar; criou a pressão seletiva sob a qual raciocinar melhor é a única forma de pontuar.

> 🔧 Por que recompensa por **regra** e não um reward model neural? Porque RM neural é *hackeável* — o RL encontra os pontos cegos dele (seção 6). Uma regra de extração + comparação exata não tem pontos cegos exploráveis da mesma forma. A lição do R1: quando existe verificador exato, use-o; guarde RMs neurais para o que não é verificável.

O R1 completo (não-Zero) adiciona um SFT curto antes (dados de *cold start*) para estabilizar o formato e a legibilidade — a ordem pipeline completa: **SFT breve → RLVR longo → (distillation, módulo 10)**.

---

## 6. Reward hacking — Goodhart em ação

> "Quando uma métrica vira alvo, deixa de ser boa métrica."

O RL é um otimizador implacável da recompensa **escrita**, não da intenção. Todo buraco entre as duas será encontrado. Exemplos reais e reproduzidos no lab:

| Recompensa escrita | O que o modelo aprende |
|---|---|
| nº de ocorrências do padrão X | **spam de X** — repetir o padrão até o limite de tokens |
| juiz neural "resposta útil" | respostas longas, confiantes e bajuladoras (o juiz gosta) |
| testes de código passam | apagar os testes, hardcodar os casos, `sys.exit(0)` |
| usuário clica 👍 | concordar com tudo |

Defesas, em camadas:

1. **Recompensas binárias e limitadas** (0/1, não contagens) — não há gradiente para "mais ainda".
2. **KL contra a referência** — encarece fugir da distribuição de linguagem natural.
3. **Verificadores exatos** em vez de proxies neurais, onde existirem.
4. **Ler as gerações.** Nenhuma métrica substitui isso: reward hacking é óbvio ao olho e invisível na curva de recompensa — que estará *linda*.

> ⚠️ A curva de recompensa subindo é condição necessária e **nem de longe suficiente**. O lab produz um hack de propósito, e o resultado merece ser transcrito.

### O experimento completo, medido

Tarefa: gerar continuações que abram fala de diálogo (`\n--`, o marcador de Machado). Verificador: busca de substring — exato e binário. Taxa de sucesso da base: 27% (dentro da janela útil: nem zero, nem resolvido).

**Treino saudável** (recompensa binária, β_KL=0,05, 60 passos, 87 segundos de CPU):

| | Taxa de sucesso | PPL validação |
|---|---|---|
| Base | 27% | 222,6 |
| Após GRPO | **90%** | 492,0 |

A recompensa média foi de 0,21 a 1,00 em ~20 passos. O modelo já sabia abrir diálogos; o GRPO amplificou comportamento raro em dominante — a mecânica do R1 em miniatura.

**O hack** (uma linha mudada: recompensa = *contagem* de `--`, ilimitada):

```
passo  0: recompensa média  0,25
passo 30: recompensa média 17,38      ← a curva está LINDA
```

E o que o modelo "ótimo" gera:

```
'antemeres.\n\n------------------------------------'
' maior passa jument.\n------------------------------------'
```

**Spam de travessões.** PPL: 223 → 1.491. Na curva de recompensa, este treino e o saudável são indistinguíveis de dois sucessos. A única defesa que os separa é ler as gerações.

**A ablação do KL** (recompensa binária, β=0):

| Variante | Sucesso | PPL |
|---|---|---|
| GRPO (β=0,05) | 90% | 492 |
| GRPO sem KL | **100%** | **1.739** |

Sem o KL, o verificador é satisfeito ao máximo — `'--D\n\n----A\n\n----------Não'` — e a linguagem morre. Os 10 pontos de sucesso que o KL "custa" são o preço de continuar existindo um modelo de linguagem. O KL não é regularização opcional em RL; é o que impede o colapso na solução degenerada.

---

## 7. SFT vs DPO vs RL — a decisão

| Pergunta | Se sim → |
|---|---|
| Existe verificador automático exato? | **RLVR/GRPO** (ou destile de quem fez, módulo 10) |
| Tenho respostas-alvo de qualidade? | **SFT** primeiro, sempre |
| Tenho preferências (pares ou 👍/👎)? | **DPO/KTO** |
| O comportamento-alvo nem aparece nas amostras? | SFT antes de qualquer RL — não se reforça o que não ocorre |
| Poucos recursos, uma GPU? | SFT/DPO; GRPO só em modelos pequenos |

E a regra de ouro da era R1: **RL de verdade exige que o modelo já acerte às vezes.** A vantagem de grupo é zero se ninguém no grupo acerta. O RL amplifica competência rara em competência confiável; não cria competência do nada. (Compare com a lição da self-consistency no módulo 7 — é a mesma.)

---

## 8. Leituras

1. **Schulman et al. (2017), "PPO"** — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).
2. **Shao et al. (2024), "DeepSeekMath"** — [arXiv:2402.03300](https://arxiv.org/abs/2402.03300). Onde o GRPO foi introduzido, seção 4.
3. **DeepSeek-AI (2025), "DeepSeek-R1"** — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948). Releia agora as seções de RL com olhos deste módulo.
4. **Liu et al. (2025), "Understanding R1-Zero-Like Training" (Dr. GRPO)** — [arXiv:2503.20783](https://arxiv.org/abs/2503.20783). Os vieses do GRPO original.
5. **Karpathy, "Deep RL: Pong from Pixels"** — [blog](http://karpathy.github.io/2016/05/31/rl/). REINFORCE explicado como ninguém; leia antes do lab.

---

## 9. Checklist de saída

- [ ] Derive o log-derivative trick em uma linha.
- [ ] Por que subtrair uma baseline não enviesa o gradiente? E o que ela faz com a variância?
- [ ] Quais quatro modelos o PPO mantém na memória, e qual deles o GRPO elimina — trocando por quê?
- [ ] Por que a vantagem do GRPO é invariante à escala da recompensa?
- [ ] O que acontece com um prompt em que o grupo inteiro erra? Que consequência isso tem para a seleção de dados?
- [ ] Quais eram as DUAS recompensas do R1-Zero, literalmente?
- [ ] Por que recompensa por contagem é hackeável e binária não?
- [ ] O que o clip do PPO/GRPO limita, exatamente?
- [ ] "A curva de recompensa está subindo" — por que isso não basta, e qual é a verificação que basta?
- [ ] Seu modelo nunca acerta a tarefa. RL resolve? O que fazer antes?

Depois: `lab_cpu.py` (GRPO do zero, executado), `preparar_dados.py`, `lab_mlx.py` (Apple Silicon) ou `lab_cuda.py` (TRL/PEFT em GPU NVIDIA, com recompensas separadas de exatidão e formato).
