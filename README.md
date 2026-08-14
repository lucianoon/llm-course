# Customização de LLMs — Curso de Estudo

Material de estudo aprofundado espelhando a grade do curso *Advanced AI: Customização de LLMs* (I2A2), construído para ser trabalhado de forma autônoma: teoria com a matemática que importa, código que roda, exercícios com critério de correção.

**Um princípio atravessa os 12 módulos:** todo número citado na teoria foi medido pelos labs — e quando a medição desmentiu o que estava escrito (aconteceu uma dúzia de vezes), o material foi corrigido e o erro virou conteúdo. Os READMEs marcam esses casos; são as melhores partes.

## O sistema completo

| Camada | Arquivo |
|---|---|
| **O norte** — o currículo comparado às melhores formações do mundo e as 3 fases até a maestria | [PLANO-MESTRE.md](PLANO-MESTRE.md) |
| **O método** — como estudar isto para RETER (ciência cognitiva aplicada, protocolo semanal) | [METODO-DE-ESTUDO.md](METODO-DE-ESTUDO.md) |
| **A revisão** — baralho Anki de 140 cartões + diário de erros + textos Feynman | [revisao/](revisao/) |
| **A maestria** — como virar o conhecimento em carreira (reproduzir, contribuir, pesquisar) | [FASE-3-MAESTRIA.md](FASE-3-MAESTRIA.md) |

## Como usar

Dois companheiros de leitura valem para o curso inteiro — mantenha-os abertos ao lado:

| Arquivo | Para quê |
|---|---|
| **[GLOSSARIO.md](GLOSSARIO.md)** | Todo termo técnico explicado **para leigos**: analogia primeiro, precisão depois, e onde aparece no curso. Organizado na ordem dos módulos (dá para ler como um mini-curso) + índice alfabético para consulta. |
| **[GUIA-DE-CODIGO.md](GUIA-DE-CODIGO.md)** | O "docstring humano" dos labs: os 11 padrões de código que se repetem (o loop de treino, o shift, a LoRALinear, os loops de DPO e GRPO...), cada um com *o que faz, por que é assim, e onde tropeça quem lê pela primeira vez*. |

Cada módulo é uma pasta com estas peças:

| Arquivo | O que é |
|---|---|
| `README.md` | A aula. Teoria, matemática, decisões de engenharia e armadilhas — com os números **medidos** pelos labs. Leia primeiro. |
| `lab.py` (módulos 1–5) | O laboratório único, executável em CPU. |
| `lab_cpu.py` (módulos 6–11) | O algoritmo implementado do zero + verificações numéricas. Roda em **qualquer máquina** (Windows/Mac, CPU) — foi executado e validado na autoria. |
| `lab_mlx.py` (módulos 6–11) | A receita de produção no **Mac M4** (MLX), com modelos reais. |
| `exercicios.md` | Exercícios com gabarito escondido. Faça sem olhar o lab. |
| `dados.py` / `preparar_dados.py` | Quando o módulo usa dataset: baixa e prepara, de forma idempotente. |

O `lab.py` está em formato *percent* (`# %%`), legível e versionável. O `.ipynb` é gerado a partir dele:

```powershell
python tools/build_notebooks.py
```

Rode o lab só depois de ler o README do módulo — o código assume os conceitos.

## Trilha

| # | Módulo | Pergunta central | Hardware |
|---|---|---|---|
| 1 | [Fundamentos de LLMs](modulo-01-fundamentos/) | O que um LLM realmente calcula? | CPU |
| 2 | [Transformers, Attention e QKV](modulo-02-attention/) | Como o contexto vira representação? | CPU |
| 3 | [Como um LLM é treinado](modulo-03-treino/) | Pré-treino, objetivo, escala, custo | CPU |
| 4 | [Preparação e curadoria de datasets](modulo-04-dados/) | Por que dado é o gargalo real | CPU |
| 5 | [Supervised Fine-Tuning e Instruction Tuning](modulo-05-sft/) | Como se ensina um formato | Mac M4 |
| 6 | [Fine-tuning eficiente (LoRA e QLoRA)](modulo-06-lora/) | Como treinar 7B em 16GB | Mac M4 |
| 7 | [Reasoning e dados de raciocínio](modulo-07-reasoning/) | O que muda quando o modelo "pensa" | Mac M4 |
| 8 | [Alinhamento por preferências (DPO)](modulo-08-dpo/) | Como se ensina preferência sem RL | Mac M4 |
| 9 | [Reinforcement Learning (PPO e GRPO)](modulo-09-rl/) | Quando a recompensa é verificável | Mac M4 |
| 10 | [Model Distillation](modulo-10-distillation/) | Como transferir capacidade para modelos menores | Mac M4 |
| 11 | [MoE, quantização e inferência](modulo-11-inferencia/) | Como isso vira produção que cabe no orçamento | Mac M4 |
| 12 | [Projeto final de customização](modulo-12-projeto/) | Fechar o ciclo ponta a ponta | Mac M4 |
| 13 | [RAG e conhecimento externo](modulo-13-rag/) | Como dar ao modelo o que ele não sabe, sem treinar? | CPU |
| 14 | [Avaliação como disciplina](modulo-14-avaliacao/) | Quantas amostras para afirmar que A é melhor que B? | CPU |
| 15 | [Agentes e tool use](modulo-15-agentes/) | O que muda quando o modelo age em vez de responder? | CPU |
| 16 | [Interpretabilidade mecanicista](modulo-16-interpretabilidade/) | O que acontece DENTRO do modelo, e como intervir? | CPU |
| 17 | [Sistemas de treino em escala](modulo-17-sistemas/) | Como treinar em 1.000 GPUs o que não cabe em uma? | CPU |
| 18 | [Fronteira de arquiteturas](modulo-18-arquiteturas/) | O que vem depois do transformer? | CPU |

## Antes de começar

Módulos 1–4: leia [`00-setup.md`](00-setup.md) — ambiente local em CPU.

Módulos 5+: leia [`00-setup-mac.md`](00-setup-mac.md) — migração para o Mac M4 e ambiente MLX.

## Convenções do material

- **⚠️ Armadilha** — erro que quase todo mundo comete na primeira vez.
- **📐 Matemática** — a derivação vale a pena; se você pular, entende o *o quê* mas não o *por quê* dos hiperparâmetros.
- **🔧 Na prática** — o que muda quando isso vai para produção.
- Números concretos (VRAM, tokens, custo) aparecem sempre que possível. Decorar conceito sem ordem de grandeza não ajuda a decidir nada.
