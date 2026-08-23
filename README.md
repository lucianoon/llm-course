# 🧠 Customização de LLMs Curso Completo

Um curso completo e autônomo sobre como modelos de linguagem funcionam e como customizá-los — **construído de baixo para cima e verificado por medição.** Todo número na teoria saiu de um laboratório que roda; quando a medição desmentiu o que estava escrito (aconteceu dezenas de vezes), o material foi corrigido e **o erro virou conteúdo.**

Não é uma coletânea de tutoriais. É a área inteira reconstruída peça por peça — tokenizer, transformer, treino, fine-tuning, RL, RAG, agentes, interpretabilidade — cada uma implementada do zero e medida, com atenção especial ao que muda **em português**.

> **21 módulos · 3 fases · ~50 laboratórios executáveis · 140 cartões de revisão**
> Fundamentos e ciência rodam em **CPU** (qualquer máquina). Customização com modelos reais roda no **Mac Apple Silicon** (MLX).

---

## Por que este curso é diferente

**Tudo é medido, nada é decorado.** A maioria do material sobre LLMs repete números de papers sem nunca reproduzi-los. Aqui, quando o texto diz *"LoRA esquece 4× menos"* ou *"a quantização degrada 4× mais em português"*, existe um lab que você roda e obtém o número. Alguns destaques, todos medidos nos labs:

| Achado | Módulo |
|---|---|
| Uma camada do Qwen2.5 reconstruída do zero bate **bit a bit** com o HuggingFace | 2 |
| A quantização 4-bit degrada **+17% em português literário** contra +4% em inglês | 6 |
| Chain-of-thought torna a resposta certa **38× mais provável** — medido sem gerar um token | 7 |
| GRPO leva a taxa de acerto de **27% → 90%**; e o *reward hacking* produzido em cativeiro | 9 |
| Um agente com calculadora vai de **0% → 87%** na aritmética que o CoT não resolve | 15 |
| *Steering*: somar uma direção às ativações **troca o idioma da geração** sem treinar | 16 |
| A auditoria estatística **rebaixou uma conclusão do próprio curso** (n=25 não conclui) | 14 |

**O erro é first-class.** Os melhores trechos são as armadilhas: o teste de EOS mal construído, os TFLOPs com sparsity, a métrica de degeneração medida no modo de decoding errado, a simulação que desmentiu o próprio README. Aprender onde a intuição falha vale mais que decorar onde ela acerta.

---

## As três fases

### 🔵 Fase 1 — Fundação (módulos 1–12) · *o pipeline completo, do zero*

| # | Módulo | Pergunta central | HW |
|---|---|---|---|
| 1 | [Fundamentos de LLMs](modulo-01-fundamentos/) | O que um LLM realmente calcula? | CPU |
| 2 | [Transformers, Attention e QKV](modulo-02-attention/) | Como o contexto vira representação? | CPU |
| 3 | [Como um LLM é treinado](modulo-03-treino/) | Pré-treino, objetivo, escala, custo | CPU |
| 4 | [Curadoria de datasets](modulo-04-dados/) | Por que dado é o gargalo real | CPU |
| 5 | [Supervised Fine-Tuning](modulo-05-sft/) | Como se ensina um formato | Mac |
| 6 | [LoRA e QLoRA](modulo-06-lora/) | Como treinar 7B em 16 GB | Mac |
| 7 | [Reasoning](modulo-07-reasoning/) | O que muda quando o modelo "pensa" | Mac |
| 8 | [Alinhamento (DPO)](modulo-08-dpo/) | Como se ensina preferência sem RL | Mac |
| 9 | [RL: PPO e GRPO](modulo-09-rl/) | Quando a recompensa é verificável | Mac |
| 10 | [Distillation](modulo-10-distillation/) | Transferir capacidade para modelos menores | Mac |
| 11 | [MoE, quantização e inferência](modulo-11-inferencia/) | Como isso vira produção que cabe no orçamento | Mac |
| 12 | [Projeto final](modulo-12-projeto/) | Fechar o ciclo ponta a ponta | Mac |

### 🟢 Fase 2 — Expansão (módulos 13–18) · *o que as melhores formações têm* — tudo em CPU

| # | Módulo | Pergunta central |
|---|---|---|
| 13 | [RAG e conhecimento externo](modulo-13-rag/) | Como dar ao modelo o que ele não sabe, sem treinar? |
| 14 | [Avaliação como disciplina](modulo-14-avaliacao/) | Quantas amostras para afirmar que A é melhor que B? |
| 15 | [Agentes e tool use](modulo-15-agentes/) | O que muda quando o modelo age em vez de responder? |
| 16 | [Interpretabilidade mecanicista](modulo-16-interpretabilidade/) | O que acontece DENTRO do modelo — e como intervir? |
| 17 | [Sistemas de treino em escala](modulo-17-sistemas/) | Como treinar em 1.000 GPUs o que não cabe em uma? |
| 18 | [Fronteira de arquiteturas](modulo-18-arquiteturas/) | O que vem depois do transformer? |

### 🟣 Fase 3 — Maestria (módulos 19–21) · *de aluno a pesquisador*

[**FASE-3-MAESTRIA.md**](FASE-3-MAESTRIA.md) — reproduzir papers · contribuir e publicar · pesquisa própria · a trilha contínua.

O currículo se mede contra **Stanford CS336/CS224N, ARENA, Berkeley CS294 e Karpathy** — ver o mapa completo em [PLANO-MESTRE.md](PLANO-MESTRE.md).

---

## Como cada módulo é organizado

Cada pasta `modulo-NN-*/` tem:

| Arquivo | O que é |
|---|---|
| **`README.md`** | A aula: teoria, matemática e armadilhas, com os números **medidos**. Leia primeiro. |
| **`lab_cpu.py`** (ou `lab.py`) | O algoritmo do zero + verificações numéricas. Roda em qualquer máquina — **executado e validado na autoria.** |
| **`lab_mlx.py`** | A receita de produção no Mac (MLX), com modelos reais. |
| **`exercicios.md`** | Exercícios com gabarito escondido. Faça sem olhar o lab. |
| **`dados.py`** | Baixa os datasets, de forma idempotente (quando o módulo usa). |

Os labs estão em formato *percent* (`# %%`) — legíveis como script e conversíveis em notebook com `python tools/build_notebooks.py`.

### O sistema de apoio (vale para o curso inteiro)

| Arquivo | Para quê |
|---|---|
| 📖 [**GLOSSARIO.md**](GLOSSARIO.md) | ~130 termos explicados **para leigos**: analogia → precisão → onde aparece. Leia como mini-curso ou consulte pelo índice. |
| 🔧 [**GUIA-DE-CODIGO.md**](GUIA-DE-CODIGO.md) | O "docstring humano" dos labs: os 11 padrões que se repetem, com *o que faz / por que / onde tropeça*. |
| 🧭 [**PLANO-MESTRE.md**](PLANO-MESTRE.md) | O norte: currículo comparado às melhores formações do mundo. |
| 🎯 [**METODO-DE-ESTUDO.md**](METODO-DE-ESTUDO.md) | Como estudar isto para **reter** — ciência cognitiva aplicada, protocolo semanal. |
| 🗂️ [**revisao/**](revisao/) | Baralho Anki de **140 cartões** + diário de erros. Conteúdo sem retenção é entretenimento. |

---

## Começando

**1. Escolha o ambiente (a leitura de 5 min que evita 90% dos problemas):**
- Módulos de CPU (fundamentos + toda a Fase 2) → [`00-setup.md`](00-setup.md)
- Customização com modelos reais no Mac → [`00-setup-mac.md`](00-setup-mac.md)

**2. Clone e prepare (no Mac, após instalar o `gh`):**
```bash
gh repo clone lucianoon/llm-course
cd llm-course
python tools/build_notebooks.py      # gera os notebooks a partir dos labs
```
Os `dados.py` de cada módulo baixam os datasets na primeira execução — nada precisa ser versionado.

**3. Para cada módulo, na ordem:** leia o `README.md` → rode o `lab_cpu.py` **prevendo cada saída antes** → faça os `exercicios.md` sem olhar o lab → escreva a explicação Feynman. Não avance com menos de 80% no checklist de saída. (O porquê de cada passo está no [método de estudo](METODO-DE-ESTUDO.md).)

**4. Estude para reter:** importe [`revisao/baralho-*.tsv`](revisao/) no [Anki](https://apps.ankiweb.net) e faça 15 min por dia. É o piso que impede o conhecimento de escorrer.

---

## Convenções do material

- **⚠️ Armadilha** — erro que quase todo mundo comete na primeira vez.
- **📐 Matemática** — a derivação vale a pena; pulá-la é entender o *o quê* mas não o *porquê* dos hiperparâmetros.
- **🔧 Na prática** — o que muda quando isso vai para produção.
- Números concretos (VRAM, tokens, custo) sempre que possível. Conceito sem ordem de grandeza não decide nada.

## Nota de honestidade

Os labs de **CPU** (`lab_cpu.py`, todos os fundamentos e toda a Fase 2) foram **executados e validados** durante a escrita — e essa execução pegou dezenas de erros que teriam passado. Os labs **`lab_mlx.py`** (Apple Silicon, módulos 5–11) foram escritos com os comandos verificados contra a documentação oficial, mas **não foram executados** no ambiente de autoria (Windows sem GPU). Espere que alguns precisem de pequenos ajustes de versão na primeira execução no Mac — eles estão marcados no topo de cada arquivo.

O fio que atravessa tudo, e o que vale acima de qualquer técnica: **desconfie de todo número que você não mediu.**

---

*Material de estudo pessoal, construído de forma autônoma. Feito para ser trabalhado, não só lido.*
