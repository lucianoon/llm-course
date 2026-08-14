# Módulo 16 — Interpretabilidade mecanicista

> **Pergunta central:** o que acontece DENTRO do modelo — e como sabemos, sem nos enganar?

Todos os módulos anteriores trataram o modelo como uma função a ser treinada, avaliada ou usada. Este o abre. Interpretabilidade mecanicista é a engenharia reversa das computações internas: não "o modelo acerta?" mas "qual circuito de neurônios e cabeças produz esta resposta, e o que acontece se eu intervir nele?".

É a maior lacuna do curso frente à formação de elite (a ARENA constrói metade do programa aqui), e — surpresa boa — **roda inteiro em CPU**: é análise de ativações de um modelo já treinado, não treino. O fio condutor vem do módulo 2, agora com método: **correlação sugere, só a intervenção prova.**

## Objetivos

1. Ler o residual stream com o logit lens — e saber por que o cru falha.
2. Executar activation patching, o método causal padrão-ouro, e entender por que é causal.
3. Localizar componentes importantes por ablação.
4. Identificar induction heads — a base mecanicista do in-context learning.
5. Treinar probes lineares (representação) e construir steering vectors (intervenção).
6. Distinguir, em todo achado, o que é evidência correlacional do que é causal.

---

## 1. Por que isto importa (além da beleza)

Três razões concretas, todas ligadas ao resto do curso:

- **Segurança e controle.** Se comportamentos (recusa, sicofância, veracidade, engano) são direções no espaço de ativações, dá para detectá-los e ajustá-los em tempo de inferência — sem retreinar. É a base de uma abordagem de alinhamento complementar ao DPO/RLHF (módulos 8–9), que age nos *dados*; interpretabilidade age nos *mecanismos*.
- **Depuração.** "Por que o modelo alucinou aqui?" tem, em princípio, uma resposta mecanicista. Reward hacking (módulo 9), injeção (módulo 15), a fidelidade do CoT (módulo 7) — todos são perguntas sobre mecanismo interno.
- **Ciência de verdade.** É uma das poucas frentes da área que produz *entendimento* em vez de só capacidade. E o Anthropic, DeepMind e a comunidade de safety contratam pesadamente aqui.

O framework fundador é o **residual stream** (Elhage et al., 2021, o transformer-circuits que o módulo 2 já citou): o vetor de dimensão `d` que atravessa o modelo é um *barramento de comunicação*; cada cabeça e cada MLP **lê** de subespaços dele e **escreve** de volta. As camadas não transformam a representação — elas a editam incrementalmente. Toda técnica abaixo explora essa estrutura.

---

## 2. Logit lens — a resposta se formando

Se o residual stream é editado incrementalmente rumo à resposta, então projetar o estado de uma camada intermediária pela `lm_head` mostra "o que o modelo preveria se parasse aqui". A resposta ganhando forma, camada a camada.

O Lab 1 mostra funcionando — e mostra falhando, que é a lição:

> ⚠️ **O logit lens cru é ruído nas camadas intermediárias de modelos pequenos.** Medido em "The capital of France is": as camadas 1–21 preveem lixo (`'\';";\n'`, `' ____'`, fragmentos de código), e **"Paris" só emerge na camada 22** (27,6% → 33,7% na 23). Curiosamente a camada 24 volta para o genérico `' the'` — o modelo já "decidiu" na 22–23 e a última camada recalibra. Não é bug — o logit lens assume que toda camada escreve no "dialeto" da `lm_head`, o que só vale perto do fim. O **tuned lens** (Belrose et al., 2023) treina uma projeção por camada para corrigir isso, e existe exatamente porque o ingênuo não basta.

Onde o logit lens é honesto e útil: ver a confiança da resposta *subir* nas últimas camadas, e comparar prompts (um fato memorizado emerge cedo; algo que exige composição emerge tarde). É uma janela, não um raio-X.

---

## 3. 📐 Activation patching — o padrão-ouro causal

O módulo 2 martelou: mapas de atenção são correlação. Uma cabeça pode atender fortemente a um token sem que isso afete a saída. **Interpretabilidade séria é causal**, e o instrumento é o patching.

A receita (também chamada *causal tracing*):

1. **Run limpo:** um prompt que produz a resposta certa (`"...France is"` → Paris). Salve as ativações.
2. **Run corrompido:** um prompt de estrutura idêntica, resposta diferente (`"...Russia is"` → Moscow). A resposta certa fica improvável.
3. **Run com patch:** rode o corrompido, mas **transplante** a ativação de uma posição/camada específica do run limpo. Meça se a resposta certa volta.

Se transplantar a ativação `(camada L, posição p)` recupera "Paris", então **aquela ativação carrega causalmente a informação do país**. Não "correlaciona com" — *causa*. Você intervju e mediu o efeito, o padrão da ciência experimental.

O Lab 2 mede exatamente isso — patchar "France is" no run "Russia is", camada a camada:

| Camada patchada | Recuperação de "Paris" |
|---|---|
| 1–4 | 0–2% |
| 5–17 | 6–12% |
| **18** | **46%** ← o salto |
| 21 | 67% |
| **22–23** | **100%** |

Um platô baixo até a camada 17, depois um **salto abrupto na 18 e recuperação total na 22–23**. Ali é onde o modelo transporta o fato do país para a posição da resposta — e a afirmação é CAUSAL ("transplantar esta ativação FAZ 'Paris' voltar"), não correlacional. É como Meng et al. (ROME) localizaram fatos e depois os EDITARAM.

Variações de granularidade: patchar o residual stream inteiro (grosso), uma cabeça (fino), o output de um MLP (fino), ou traçar o caminho entre componentes (**path patching** — o mais fino, exercício B3).

---

## 4. Ablação — os componentes que importam

Uma versão barata e complementar: **desligue** um componente (zere a saída de uma cabeça) e meça o dano à tarefa. As cabeças cuja ablação mais machuca são as que a tarefa usa.

Distinção sutil mas importante:
- **Zero ablation** (zerar) é fácil mas impreciso — pode tirar o componente da distribuição de formas artificiais.
- **Mean ablation** (substituir pela ativação média) é mais honesto — remove a informação específica mantendo a estatística geral.
- **Patching** é o mais preciso — substitui por uma ativação real de outro contexto.

O Lab 3 usa zero ablation para achar candidatos rapidamente; um circuito confirmado exige patching. A ordem de trabalho real: ablar para triar, patchar para provar.

---

## 5. Induction heads — o circuito mais famoso

Olsson et al. (2022) identificaram o mecanismo por trás do in-context learning: cabeças que **completam padrões**. Dado `[A][B] ... [A]`, uma induction head atende ao token que SEGUIU `[A]` da última vez (`[B]`) e aumenta sua probabilidade. Em uma frase: "isto já aconteceu antes; o que veio depois?".

Por que é grande:
- É a base mecanicista do **few-shot learning** (módulo 1) — o modelo "copia" a estrutura dos exemplos do prompt via essas cabeças, sem treinar.
- Sua formação durante o treino coincide com um **salto abrupto** na capacidade de in-context learning — um dos poucos casos em que uma habilidade emergente foi rastreada até um circuito específico.

O Lab 4 as caça com um padrão repetido explícito e um "score de indução". Medido: a cabeça **16/3** (camada 16, cabeça 3) tem score **0,98** contra uma média de 0,070 — **14× a média**. Ela atende, em quase toda posição da repetição, exatamente ao token que veio depois na primeira ocorrência. Você vê o circuito, não só a capacidade — e as próximas do ranking (11/12, 9/13, outras da camada 16) são candidatas ao mesmo circuito.

---

## 6. 📐 Probing e steering — representação vs intervenção

### Linear probing (representação)

Um *probe* é um classificador linear treinado sobre as ativações de uma camada para ler um conceito (idioma, sentimento, verdade/mentira). Se o probe acerta, o conceito está **linearmente representado** ali — é uma direção no espaço de `d` dimensões. É como se descobriu que modelos têm direções de "verdade" e de "recusa".

> ⚠️ **Probe alto prova representação, NÃO uso.** O modelo pode codificar o idioma sem que isso influencie a saída — a armadilha correlação-vs-causa do módulo 2, de novo. Provar uso exige intervir.

### Steering vectors (intervenção)

A carga causal: se um conceito é uma direção `v`, então **somar** `α·v` às ativações durante a geração deve empurrar o comportamento naquela direção — controle sem tocar num peso. O Lab 6 constrói `v` = (média das ativações PT − média das EN) e injeta.

Steering (Turner et al., 2023) é o teste que o probe não é: probe mostra que a direção EXISTE; steering prova que ela CAUSA comportamento. E o Lab 6 é o resultado mais dramático do módulo. Injetando o vetor "português − inglês" na camada 12, sobre o prompt inglês *"My favorite thing about the weekend is"*:

```
força  0: " the opportunity to get out and explore the outdoors. I love..."   (inglês normal)
força  6: " quebrantando o tempo. Quebrantando o tempo é uma atividade..."    (VIROU PORTUGUÊS)
força 12: " a chance devoir. I am um de um of 100000000000..."                 (degenera)
```

**Uma direção somada às ativações trocou o idioma da geração — sem tocar num único peso.** E a força 12 mostra a outra ponta: steering é uma alavanca cega, e forte demais degenera (a mesma curva da temperatura do módulo 1 e do β do módulo 8). Probe provou que a direção do idioma existe; steering provou que ela CONTROLA a saída. Representação + intervenção = a evidência completa. É uma linha de segurança prática — direções de recusa, veracidade e sicofância podem ser reforçadas ou suprimidas em inferência, mais barato e mais cirúrgico que fine-tuning.

**Representação + intervenção = a evidência completa.** É a lição metodológica do módulo inteiro: nenhum achado de interpretabilidade está completo sem o teste causal.

---

## 7. A fronteira: SAEs e a superposição

O problema que domina a interpretabilidade atual: **superposição.** Modelos representam MAIS conceitos que dimensões, empacotando-os em direções que se sobrepõem — então um neurônio raramente significa uma coisa só (é *polissemântico*). Isso quebra a esperança ingênua de "um neurônio, um conceito".

A aposta da vez: **Sparse Autoencoders (SAEs).** Treina-se um autoencoder que reexpressa as ativações numa base MUITO maior e esparsa, na esperança de que cada dimensão dessa base seja *monossemântica* (um conceito só). Anthropic (2024) escalou isso para o Claude 3 Sonnet e encontrou features interpretáveis — de "a ponte Golden Gate" a "código com bug de segurança" — e as usou para steering. É a fronteira aberta, cara de treinar (o SAE é maior que a camada que ele explica), e provavelmente o próximo grande capítulo da área.

Fica como leitura e exercício avançado (B5): o método é o do curso — o SAE é treinado com a loss do módulo 3 e a esparsidade é o `α·||z||₁` que você já viu em espírito.

---

## 8. Leituras

1. **Elhage et al. (2021), "A Mathematical Framework for Transformer Circuits"** — [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html). O framework do residual stream. Difícil e fundamental.
2. **Olsson et al. (2022), "In-context Learning and Induction Heads"** — [transformer-circuits.pub](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). O Lab 4.
3. **Meng et al. (2022), "Locating and Editing Factual Associations (ROME)"** — [arXiv:2202.05262](https://arxiv.org/abs/2202.05262). Patching que vira edição.
4. **Turner et al. (2023), "Activation Addition (steering)"** — [arXiv:2308.10248](https://arxiv.org/abs/2308.10248). O Lab 6.
5. **Templeton et al. (2024), "Scaling Monosemanticity" (SAEs no Claude 3)** — [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/). A fronteira.
6. **ARENA** ([arena.education](https://www.arena.education/)) — o currículo prático de referência; a seção de interpretabilidade é o padrão-ouro de labs.

---

## 9. Checklist de saída

- [ ] O que é o residual stream, e por que "as camadas editam em vez de transformar"?
- [ ] Por que o logit lens cru falha nas camadas intermediárias de modelos pequenos?
- [ ] Descreva os três runs do activation patching e por que o resultado é CAUSAL.
- [ ] Zero vs mean vs patching como ablação: qual é mais honesto e por quê?
- [ ] O que uma induction head faz, e qual capacidade do módulo 1 ela explica?
- [ ] O que um probe alto prova — e o que NÃO prova?
- [ ] Como um steering vector transforma uma direção em controle? Por que é o teste que o probe não é?
- [ ] O que é superposição, e por que ela motiva os SAEs?
- [ ] O fio condutor: que tipo de evidência sugere, e que tipo prova?

Depois: `lab_cpu.py` (executado — as seis técnicas no Qwen) e os cartões em `revisao/baralho-02-expansao.tsv`.
