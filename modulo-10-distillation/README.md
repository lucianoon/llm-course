# Módulo 10 — Model Distillation

> **Pergunta central:** como transferir a capacidade de um modelo grande para um pequeno — e o que se perde na mudança?

O módulo 9 terminou com uma divisão de trabalho: **RL descobre, distillation dissemina.** O R1 gastou o compute do RL uma vez, num modelo enorme; depois transferiu o resultado para modelos de 1,5B a 70B por SFT simples — e o 1,5B destilado supera modelos 100× maiores em matemática. Este módulo é sobre essa transferência: a matemática, as duas famílias de técnica, e as armadilhas.

## Objetivos

1. Distinguir as duas famílias — logits (white-box) e dados (black-box) — e saber quando cada uma é possível.
2. Explicar o que os *soft targets* carregam que os rótulos duros não carregam.
3. Entender forward vs reverse KL — e por que a direção importa para modelos generativos.
4. Executar uma destilação de logits completa e medir o ganho sobre treinar do zero.
5. Executar o pipeline black-box do R1 em miniatura: gerar, filtrar, treinar.
6. Orçar uma destilação: custo do professor vs custo do treino.

---

## 1. As duas famílias

| | **White-box (logits)** | **Black-box (dados)** |
|---|---|---|
| O que o professor fornece | A distribuição completa por token | Só o texto gerado |
| Requisitos | Acesso aos logits **e mesmo tokenizer** | Só a API |
| Sinal por token | `V` números (soft target) | 1 token (hard target) |
| Técnica | Minimizar KL entre distribuições | SFT nos dados gerados |
| Exemplos | Hinton 2015, DistilBERT, Gemma (2→1B), MiniLLM | **R1-Distill**, Alpaca, quase tudo em LLMs |

Na prática de LLMs, a black-box domina — porque os melhores professores são fechados ou têm tokenizer diferente, e porque SFT em texto gerado é trivial de implementar. Mas a white-box extrai mais sinal por token, e quando os dois modelos são seus (mesma família), ela é o método certo.

> ⚠️ **A restrição do tokenizer é dura.** A loss de logits compara distribuições sobre o **mesmo vocabulário**, posição a posição. Qwen (151k tokens) não destila logits para Llama (128k) sem técnicas de alinhamento de vocabulário que são frágeis e recentes. Verifique o tokenizer antes de planejar qualquer white-box.

---

## 2. 📐 Soft targets e dark knowledge

O rótulo duro para "O gato subiu no ___" é `telhado`, one-hot. A distribuição do professor é:

```
telhado 0,52   muro 0,31   árvore 0,12   sofá 0,04   ...   sintaxe_impossível ~0
```

O soft target carrega o que Hinton chamou de **dark knowledge**: a estrutura de similaridade da tarefa inteira. `muro` é quase tão bom; `sofá` é plausível mas raro; um verbo ali é impossível. O aluno que treina nisso recebe, em cada posição, um mapa completo do espaço de respostas — não um único ponto.

### A temperatura

Distribuições de professores treinados são afiadas (o topo concentra quase tudo — módulo 1). A temperatura as suaviza antes da comparação:

```
p_i(T) = softmax(z_i / T)
L_KD = T² · KL( p_professor(T) ‖ p_aluno(T) )
```

`T` entre 1 e 4 é o usual: alto o bastante para as relações entre classes aparecerem, baixo o bastante para não virar uniforme. O fator `T²` compensa o encolhimento dos gradientes (que escalam com `1/T²`).

A loss completa mistura KD com a cross-entropy dos dados reais:

```
L = α · L_KD + (1−α) · L_CE          α típico: 0,5–0,9
```

### O experimento central, medido

Professor de 8,1M de parâmetros (PPL 124,2); aluno de 0,9M — **11% do tamanho** — treinado com o mesmo número de passos por três receitas (Lab 3):

| Receita | PPL do aluno | Tempo |
|---|---|---|
| Hard labels (do zero, sem professor) | 197,8 | 87 s |
| KD puro (α=1, T=2) | 166,8 | 270 s |
| Mistura (α=0,7) | **165,5** | 281 s |

**A destilação melhorou a PPL em 16,3%** sobre treinar do zero — mesmos dados, mesmos passos, mesma arquitetura; a única diferença é o sinal por token. A afirmação de Hinton, confirmada aqui. E o custo escondido na coluna de tempo: cada passo de KD paga um forward do professor (3× mais lento) — a dark knowledge não é grátis.

Duas surpresas honestas da execução, ambas documentadas no lab:

- **A temperatura ótima foi T=1**, monotonicamente pior até T=8 — contrariando o folclore "T=2–4", que vem da classificação de imagens. Em LLMs a distribuição por token já é suave (entropia ~6 nats no Lab 1); suavizar mais dilui. E a PPL é avaliada em T=1 — treinar em T alta descasa treino e avaliação.
- **O black-box ingênuo foi um desastre**: aluno treinado em 15k tokens gerados pelo professor fraco terminou com PPL **3.190** (16× pior que o baseline). Autópsia no lab: professor fraco + 53 épocas sobre corpus minúsculo + sem filtragem — o model collapse do módulo 4 em dose concentrada, e a negação exata das três regras que fazem o pipeline R1 funcionar (professor forte, escala, rejection sampling).

---

## 3. 📐 A direção do KL — o detalhe que muda tudo

KL não é simétrico, e a escolha da direção define o comportamento do aluno quando ele **não tem capacidade** de imitar o professor inteiro (que é sempre — o aluno é menor):

**Forward KL — `KL(p ‖ q)`** (professor ‖ aluno), o padrão da classificação:
- Pune `q ≈ 0` onde `p > 0` → o aluno é forçado a **cobrir todos os modos** do professor.
- Sem capacidade para todos, ele se espalha: massa em regiões medianas, *mode-covering*.
- Em geração, isso produz texto de qualidade média baixa — o aluno "tenta ser tudo".

**Reverse KL — `KL(q ‖ p)`** (aluno ‖ professor):
- Pune `q > 0` onde `p ≈ 0` → o aluno **nunca pode gerar o que o professor considera impossível**.
- Sem capacidade, ele **escolhe alguns modos e os faz bem**: *mode-seeking*.
- Para modelos generativos, é normalmente o que se quer — qualidade sobre cobertura. É a base do MiniLLM (2023) e de boa parte da destilação white-box moderna (com o custo de exigir amostragem do aluno durante o treino — parentesco direto com o RL do módulo 9).

O Lab 2 torna isso visível: ajustar uma gaussiana a uma mistura bimodal (modos em ±2, largura 0,6) pelas duas direções:

| Direção | Resultado medido |
|---|---|
| Forward KL | μ=0,00, σ=2,13 — **no vale**, alargada para cobrir os dois modos |
| Reverse KL | μ=+2,00, σ=0,60 — **um modo, com a largura exata dele** |

Com uma ressalva medida que a literatura raramente menciona: o mode-seeking do reverse KL é a geometria do objetivo, **não uma garantia do otimizador**. Inicializando a gaussiana larga demais (σ₀=1,0), o reverse também fica preso no centro (mínimo local: cobrir os dois modos "de longe" é localmente melhor que atravessar o vale). Só com σ₀≤0,5 ele escolhe. Em destilação real isso reaparece como a prática do MiniLLM: **inicializar o aluno com SFT antes do reverse KL** — começar perto de um modo.

### On-policy KD (GKD)

O refinamento de 2023: em vez de destilar só sobre texto do professor (que o aluno talvez nunca gere), amostre do **aluno** e peça ao professor para corrigir a distribuição em cada posição. Ataca o *exposure bias* — o aluno aprende nas situações em que ele mesmo se coloca. É o análogo destilatório do online DPO.

---

## 4. O pipeline black-box — a receita do R1-Distill

```
1. Professor gera respostas para um conjunto de prompts     (caro, uma vez)
2. FILTRAR: rejection sampling                              (a etapa que separa)
   - resposta final confere com o gabarito? (verificável!)
   - legível, língua certa, comprimento razoável?
3. SFT do aluno nos traços aprovados                        (módulo 5, nada novo)
4. Avaliar contra o professor E contra o aluno original
```

Números do R1 real: ~800k amostras filtradas, SFT puro (sem RL nenhum nos alunos), e o resultado da tabela do módulo 7. O paper é explícito: aplicar o RL diretamente nos modelos pequenos funcionou **pior** que destilar — o modelo pequeno não tem de onde "descobrir"; imitar quem descobriu rende mais.

As decisões que importam:

- **Filtragem por gabarito** é o controle de qualidade grátis quando a tarefa é verificável — e introduz o viés já visto no módulo 7 (problemas difíceis são descartados desproporcionalmente).
- **Diversidade de prompts** vale mais que volume por prompt (módulo 4 inteiro se aplica).
- **Licença do professor** (módulo 4, seção 6): destilar GPT-4/Claude para uso comercial viola termos. R1 e Qwen permitem explicitamente — e é por isso que o ecossistema aberto destila deles.

---

## 5. O que se perde

Distillation **transfere, raramente supera**. O aluno herda:

- o teto do professor (erros incluídos — inclusive os hacks de recompensa que o RL do professor tenha aprendido);
- os vieses de estilo e de distribuição;
- e uma versão *comprimida*: a cauda de capacidades raras é o primeiro sacrifício (compare com o model collapse do módulo 4 — mesma mecânica, dose menor).

Regra de bolso do que sobrevive bem à compressão: comportamento, formato, procedimentos frequentes. O que sofre: conhecimento de cauda longa, robustez fora de distribuição, capacidades emergentes do tamanho.

---

## 6. Leituras

1. **Hinton, Vinyals & Dean (2015), "Distilling the Knowledge in a Neural Network"** — [arXiv:1503.02531](https://arxiv.org/abs/1503.02531). Dez páginas, fundou a área.
2. **DeepSeek-AI (2025), "DeepSeek-R1"**, seção de distillation — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948). A tabela 5 (RL direto vs distillation em modelos pequenos) é o resultado central para este módulo.
3. **Gu et al. (2023), "MiniLLM"** — [arXiv:2306.08543](https://arxiv.org/abs/2306.08543). Reverse KL para LLMs.
4. **Agarwal et al. (2023), "GKD"** — [arXiv:2306.13649](https://arxiv.org/abs/2306.13649). On-policy KD.
5. **Sanh et al. (2019), "DistilBERT"** — [arXiv:1910.01108](https://arxiv.org/abs/1910.01108). O clássico de produção.

---

## 7. Checklist de saída

- [ ] Quando a white-box é impossível, por duas razões independentes?
- [ ] O que um soft target carrega que o hard target não carrega? Dê um exemplo concreto.
- [ ] Para que serve a temperatura, e por que o fator `T²`?
- [ ] Forward KL força o quê? Reverse KL proíbe o quê? Qual você quer num gerador, e por quê?
- [ ] Onde a gaussiana ajustada por forward KL assenta numa bimodal — e por que isso é a pior escolha para geração?
- [ ] Descreva o pipeline do R1-Distill em quatro passos e aponte a etapa de controle de qualidade.
- [ ] Por que o RL direto nos modelos pequenos perdeu para a distillation (R1, tabela 5)?
- [ ] Que viés a filtragem por gabarito introduz?
- [ ] O que sobrevive bem à compressão e o que morre primeiro?

Depois: `lab_cpu.py` (executado — logits, KL dos dois lados, e uma destilação real), `lab_mlx.py` (o pipeline R1-Distill no seu M4).
