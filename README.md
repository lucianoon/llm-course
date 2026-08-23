# 🧠 Customização de LLMs Curso Completo

Um curso completo e autônomo sobre como modelos de linguagem funcionam e como customizá-los — **construído de baixo para cima e verificado por medição.** Todo número na teoria saiu de um laboratório que roda; quando a medição desmentiu o que estava escrito (aconteceu dezenas de vezes), o material foi corrigido e **o erro virou conteúdo.**

Não é uma coletânea de tutoriais. É a área inteira reconstruída peça por peça — tokenizer, transformer, treino, fine-tuning, RL, RAG, agentes, interpretabilidade — cada uma implementada do zero e medida, com atenção especial ao que muda **em português**.

> **Fase 0 + 21 módulos · trilha essencial + especializações · ~50 laboratórios executáveis**
> Fundamentos e ciência rodam em **CPU** (qualquer máquina). Customização com modelos reais roda no **Mac Apple Silicon** (MLX).

> 🌱 **Nunca programou?** Comece pela [Fase 0 — iniciante absoluto](00-iniciante-zero/)
> e siga somente a [trilha essencial do zero ao primeiro nível profissional](TRILHA-ESSENCIAL.md).
> Você não precisa concluir todos os temas avançados para começar a trabalhar bem.

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

## O mapa completo

### ⚪ Fase 0 — Alfabetização técnica · *para quem nunca programou*

Python, terminal, testes, Git e a matemática mínima aparecem juntos em
[`00-iniciante-zero/`](00-iniciante-zero/). O laboratório começa com variáveis e termina
com um preditor de próxima palavra, sem baixar modelos.

### A rota profissional essencial

Para chegar ao primeiro nível profissional, siga esta ordem:

`Fase 0 → módulos 1–6 → módulo 11 → módulos 13–15 → projeto (módulo 12)`

Essa rota ensina a construir, testar, avaliar, documentar e entregar sistemas de LLM.
Os demais módulos continuam disponíveis como especializações. Veja os gates e projetos
em [TRILHA-ESSENCIAL.md](TRILHA-ESSENCIAL.md).

### 🔵 Fase 1 — Fundação (módulos 1–12) · *o pipeline completo, do zero*

| # | Módulo | Pergunta central | Rota | HW |
|---|---|---|---|---|
| 1 | [Fundamentos de LLMs](modulo-01-fundamentos/) | O que um LLM realmente calcula? | Essencial | CPU |
| 2 | [Transformers, Attention e QKV](modulo-02-attention/) | Como o contexto vira representação? | Essencial | CPU |
| 3 | [Como um LLM é treinado](modulo-03-treino/) | Pré-treino, objetivo, escala, custo | Essencial | CPU |
| 4 | [Curadoria de datasets](modulo-04-dados/) | Por que dado é o gargalo real | Essencial | CPU |
| 5 | [Supervised Fine-Tuning](modulo-05-sft/) | Como se ensina um formato | Essencial | Mac |
| 6 | [LoRA e QLoRA](modulo-06-lora/) | Como treinar 7B em 16 GB | Essencial | Mac |
| 7 | [Reasoning](modulo-07-reasoning/) | O que muda quando o modelo "pensa" | Especialização | Mac |
| 8 | [Alinhamento (DPO)](modulo-08-dpo/) | Como se ensina preferência sem RL | Especialização | Mac |
| 9 | [RL: PPO e GRPO](modulo-09-rl/) | Quando a recompensa é verificável | Especialização | Mac |
| 10 | [Distillation](modulo-10-distillation/) | Transferir capacidade para modelos menores | Especialização | Mac |
| 11 | [MoE, quantização e inferência](modulo-11-inferencia/) | Como isso vira produção que cabe no orçamento | Essencial | Mac |
| 12 | [Projeto final](modulo-12-projeto/) | Fechar o ciclo ponta a ponta | Essencial, após o 15 | Mac |

### 🟢 Fase 2 — Expansão (módulos 13–18) · *o que as melhores formações têm* — tudo em CPU

| # | Módulo | Pergunta central | Rota |
|---|---|---|---|
| 13 | [RAG e conhecimento externo](modulo-13-rag/) | Como dar ao modelo o que ele não sabe, sem treinar? | Essencial |
| 14 | [Avaliação como disciplina](modulo-14-avaliacao/) | Quantas amostras para afirmar que A é melhor que B? | Essencial |
| 15 | [Agentes e tool use](modulo-15-agentes/) | O que muda quando o modelo age em vez de responder? | Essencial |
| 16 | [Interpretabilidade mecanicista](modulo-16-interpretabilidade/) | O que acontece DENTRO do modelo — e como intervir? | Especialização |
| 17 | [Sistemas de treino em escala](modulo-17-sistemas/) | Como treinar em 1.000 GPUs o que não cabe em uma? | Especialização |
| 18 | [Fronteira de arquiteturas](modulo-18-arquiteturas/) | O que vem depois do transformer? | Especialização |

### 🟣 Fase 3 — Pesquisa (módulos 19–21) · *especialização opcional*

[**FASE-3-MAESTRIA.md**](FASE-3-MAESTRIA.md) — reproduzir papers · contribuir e publicar · pesquisa própria · a trilha contínua.

O currículo se mede contra **Stanford CS336/CS224N, ARENA, Berkeley CS294 e Karpathy** — ver o mapa completo em [PLANO-MESTRE.md](PLANO-MESTRE.md).

---

## Como cada módulo é organizado

Cada pasta `modulo-NN-*/` tem a estrutura abaixo. A Fase 0 usa a mesma convenção com
`README.md`, `lab.py`, `lab.ipynb` gerado e `exercicios.md`.

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

**1. Escolha seu ponto de entrada:**
- Nunca programou ou não conhece testes, Git e tensores → [`00-iniciante-zero/`](00-iniciante-zero/)
- Já programa e quer a rota mais curta até projetos profissionais → [`TRILHA-ESSENCIAL.md`](TRILHA-ESSENCIAL.md)
- Já domina os fundamentos e quer pesquisa → escolha uma especialização no mapa acima

**2. Escolha o ambiente (a leitura de 5 min que evita 90% dos problemas):**
- Módulos de CPU (fundamentos + toda a Fase 2) → [`00-setup.md`](00-setup.md)
- Customização com modelos reais no Mac → [`00-setup-mac.md`](00-setup-mac.md)

**3. Clone e prepare (no Mac, após instalar o `gh`):**
```bash
gh repo clone lucianoon/llm-course
cd llm-course
python tools/build_notebooks.py      # gera os notebooks a partir dos labs
```
Os `dados.py` de cada módulo baixam os datasets na primeira execução — nada precisa ser versionado.

**4. Para cada módulo da sua rota:** leia o `README.md` → rode o lab **prevendo cada saída antes** → faça os `exercicios.md` sem olhar o lab → escreva a explicação Feynman. Não avance com menos de 80% no checklist de saída. (O porquê de cada passo está no [método de estudo](METODO-DE-ESTUDO.md).)

**5. Estude para reter:** importe [`revisao/baralho-*.tsv`](revisao/) no [Anki](https://apps.ankiweb.net) e faça 15 min por dia a partir do módulo 1. Na Fase 0, priorize executar, errar e corrigir o código.

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
