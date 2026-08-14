# Plano Mestre — de curso espelhado a formação de elite

O objetivo mudou: não é mais acompanhar um curso de 36h, é **se tornar referência na área**. Este documento é o novo norte: o currículo comparado com as melhores formações do mundo, as fases até a maestria, e o que separa quem *usa* a área de quem a *empurra*.

## O benchmark — contra quem estamos nos medindo

| Formação | O que ela tem de melhor | Status no nosso curso |
|---|---|---|
| **Stanford CS336** (Language Modeling from Scratch) | Implementar TUDO do zero: tokenizer, transformer, treino distribuído, kernels | ✅ Módulos 1–3 cobrem o núcleo; falta a trilha de sistemas (fase 2) |
| **Stanford CS224N** (NLP with Deep Learning) | Fundamentos teóricos rigorosos, história das ideias | ✅ Coberto nos módulos 1–2, com mais medição que o original |
| **Karpathy — Zero to Hero / nanoGPT / nanochat** | O padrão-ouro de "construir para entender" | ✅ É o DNA dos nossos labs; nanochat é leitura da fase 3 |
| **ARENA** (Alignment Research Engineer Accelerator) | Interpretabilidade mecanicista e RL com labs de nível de pesquisa | ❌ Nossa maior lacuna — vira o módulo 16 |
| **CMU 11-667** (Large Language Models) | Sistemas de inferência e treino em escala | ⚠️ Módulo 11 cobre inferência; falta treino distribuído (módulo 17) |
| **Berkeley CS294** (LLM Agents) | Agentes, tool use, planejamento | ❌ Vira o módulo 15 |
| **fast.ai** | Pedagogia top-down: o todo funcional primeiro, os detalhes depois | ⚠️ Parcial — nosso curso é bottom-up; a fase 2 alterna |
| **HuggingFace smol-course / EleutherAI** | Prática aberta, avaliação, comunidade | ⚠️ Falta o módulo de avaliação (14) e a prática de contribuição (fase 3) |

**Leitura honesta da tabela:** a Fase 1 (módulos 1–12) já cobre o núcleo do CS336 + CS224N com um diferencial raro (tudo medido) e um déficit conhecido (escala de brinquedo). O que falta para o nível "melhor do mundo" está nas lacunas: interpretabilidade, agentes, avaliação, sistemas, e — acima de tudo — **as habilidades de pesquisa da fase 3**, que nenhum curso ensina e todo pesquisador de elite tem.

---

## As três fases

### FASE 1 — Fundação ✅ (módulos 1–12, completa)

O pipeline inteiro: fundamentos → treino → dados → SFT → LoRA/QLoRA → reasoning → DPO → RL → distillation → inferência → projeto. Com glossário, guia de código e método de estudo.

**Pendência única:** executar os 7 labs MLX no M4 (semana 1 da fase 2).

### FASE 2 — Expansão (módulos 13–18) ✅ COMPLETA

O que as melhores formações têm e nós ainda não:

| # | Módulo | Pergunta central | Inspiração | Hardware |
|---|---|---|---|---|
| 13 ✅ | **[RAG e conhecimento externo](modulo-13-rag/)** | Como dar ao modelo o que ele não sabe, sem treinar? | prática da indústria | M4 |
| 14 ✅ | **[Avaliação como disciplina](modulo-14-avaliacao/)** | Como saber — com rigor estatístico — se um modelo é melhor que outro? | HELM, lm-eval-harness, Eleuther | M4 |
| 15 ✅ | **[Agentes e tool use](modulo-15-agentes/)** | O que muda quando o modelo age em vez de só responder? | Berkeley CS294 | M4 |
| 16 ✅ | **[Interpretabilidade mecanicista](modulo-16-interpretabilidade/)** | O que acontece DENTRO do modelo — e como intervir? | ARENA, transformer-circuits | M4 (CPU!) |
| 17 ✅ | **[Sistemas de treino em escala](modulo-17-sistemas/)** | Como se treina em 1.000 GPUs — autograd, DDP/FSDP, kernels | CS336 (trilha de sistemas), CMU | M4 + simulação |
| 18 ✅ | **[Fronteira de arquiteturas](modulo-18-arquiteturas/)** | O que vem depois do transformer? SSMs/Mamba, MLA, híbridos, multimodal | papers 2024–2026 | M4 |

Mesmo padrão da fase 1: teoria medida, `lab_cpu`/`lab_mlx`, exercícios com gabarito, tudo alimentando o glossário e o baralho de revisão.

### FASE 3 — Maestria ✅ ([FASE-3-MAESTRIA.md](FASE-3-MAESTRIA.md))

O que separa os melhores não é conhecer mais técnicas — é o ciclo de pesquisa. Três músculos, treinados em sequência:

| # | Módulo | O músculo |
|---|---|---|
| 19 ✅ | **Ler e reproduzir papers** | Escolher um paper recente e **reproduzi-lo de ponta a ponta** — o rito de passagem de todo pesquisador. Aprende-se a ler criticamente, a identificar o que o paper esconde, e a distância entre "entendi" e "fiz funcionar". |
| 20 ✅ | **Contribuir e publicar** | Uma contribuição real a um projeto aberto (mlx-lm, vllm, lm-eval-harness) + escrever tecnicamente em público (blog/relatórios). Quem é referência na área é **visível** na área. |
| 21 ✅ | **Pesquisa própria** | Uma pergunta sua, não respondida na literatura, atacada com o método do curso: hipótese → experimento mínimo → medição → escrita. O projeto do módulo 12, elevado a pesquisa. |

E a **trilha contínua**, sem fim: 2 papers/semana (método do módulo 19), o baralho de revisão diário, e o diário de erros — para sempre. Os melhores da área não "terminaram de estudar"; institucionalizaram o estudo.

---

## O cronograma realista

Assumindo ~10–12 h/semana de estudo deliberado:

| Período | O quê |
|---|---|
| Semanas 1–2 | Migração ao M4 + execução de TODOS os labs MLX (com correções) + início do método de estudo (revisão espaçada da fase 1) |
| Semanas 3–10 | Fase 2, um módulo a cada ~10 dias, com intercalação da fase 1 (ver MÉTODO) |
| Semanas 11–14 | Módulo 19 (reprodução de um paper) |
| Contínuo a partir da semana 3 | Baralho diário (15 min) + sabatinas espaçadas |
| Semana 15+ | Módulos 20–21 — e aqui o "curso" acaba e a carreira de especialista começa |

**Sobre o I2A2 (out/2026–jan/2027):** com este plano, quando ele começar você estará na fase 3. Se mantiver a matrícula, ele vira o que universidades chamam de *seminário de auditoria* — você compara, questiona e extrai dos instrutores o que material nenhum contém. Se cancelar, nada essencial se perde.

---

## Princípios de construção (herdados da fase 1, agora explícitos)

1. **Nada entra na teoria sem ser medido** — e quando a medição desmentir, o erro vira conteúdo.
2. **Todo módulo produz artefatos de revisão** — cartões novos no baralho, verbetes no glossário, padrões no guia de código.
3. **Escala de brinquedo com aviso de escala** — todo resultado de MiniGPT declara explicitamente o que transfere e o que não.
4. **Português como cidadão de primeira classe** — os efeitos específicos do idioma (tokenização, quantização, avaliação) são medidos, não assumidos.
5. **O método de estudo é parte do curso** — ver [`METODO-DE-ESTUDO.md`](METODO-DE-ESTUDO.md). Conteúdo sem retenção é entretenimento.
