# Mapa de módulos

Use esta tabela para escolher o próximo passo. O tempo é uma estimativa de estudo e execução
em ritmo individual; o gate, e não o relógio, define quando avançar.

| Módulo | Pré-requisito | Tempo | Dificuldade | Ambiente | Evidência de saída |
|---|---|---:|---|---|---|
| Fase 0 | Nenhum | 20–30 h | iniciante | CPU | classificador por regras testado |
| 1 — Fundamentos | Fase 0 ou Python básico | 4–6 h | iniciante | CPU | comparação de tokenização e decoding |
| 2 — Attention | Módulo 1 | 6–8 h | iniciante+ | CPU | attention causal reconstruída |
| 3 — Treino | Módulos 1–2 | 8–12 h | intermediário | CPU | curva de treino interpretada |
| 4 — Dados | Módulo 3 | 6–10 h | intermediário | CPU | dataset auditado e manifesto |
| 5 — SFT | Módulos 1–4 | 8–12 h | intermediário | CPU no núcleo; Mac/GPU opcional | baseline, treino e métricas |
| 6 — LoRA/QLoRA | Módulo 5 | 8–12 h | intermediário | CPU no núcleo; Mac/GPU opcional | adapter e comparação de memória |
| 7 — Reasoning | Módulos 3–5 | 8–12 h | avançado | CPU no núcleo; Mac/GPU opcional | hipótese sobre raciocínio testada |
| 8 — DPO | Módulos 4–6 | 8–12 h | avançado | CPU no núcleo; Mac/GPU opcional | preferências auditadas |
| 9 — RL | Módulos 5–8 | 10–16 h | avançado | CPU no núcleo; Mac/GPU opcional | recompensa e efeitos colaterais medidos |
| 10 — Distillation | Módulos 5–6 | 8–12 h | avançado | CPU no núcleo; Mac/GPU opcional | professor/aluno comparados |
| 11 — Inferência | Módulos 1–6 | 8–12 h | intermediário | CPU; Mac/GPU opcional | memória, latência e custo |
| 13 — RAG | Módulos 4 e 11 | 6–10 h | intermediário | CPU | recuperação com citação e abstenção |
| 14 — Avaliação | Módulos 3–5 | 6–10 h | intermediário | CPU | baseline, IC e análise de erro |
| 15 — Agentes | Módulos 13–14 | 6–10 h | intermediário+ | CPU | ferramenta segura e trilha auditável |
| 16 — Interpretabilidade | Módulos 1–3 | 10–16 h | avançado | CPU; acelerador opcional | intervenção e limitação documentadas |
| 17 — Sistemas | Módulo 3 | 10–16 h | avançado | CPU para conceitos; GPU opcional | estratégia de escala justificada |
| 18 — Arquiteturas | Módulos 1–3 | 8–12 h | avançado | CPU | arquitetura comparada por hipótese |
| 19 — Produção | Módulos 11, 13–15 | 6–10 h | intermediário+ | CPU | serviço com p50/p95, custo e rollback |
| 12 — Projeto final | Módulos 13–15 e 19 | 2–3 semanas | intermediário+ | conforme o projeto | repositório reproduzível e banca |

## Regras de navegação

- **Primeira vez com Python:** Fase 0 → 1 → 2 → 3 → 4.
- **Já programa e quer trabalhar:** 1 → 2 → 3 → 4 → 5 → 6 → 11 → 13 → 14 → 15 → 19 → 12.
- **Quer pesquisa:** conclua a rota essencial antes de escolher 7–10 ou 16–18.
- **Travou:** não avance para outro tema; volte ao gate do módulo e registre o erro em
  [`revisao/diario-de-erros.md`](revisao/diario-de-erros.md).

As horas não incluem downloads lentos, espera por GPU ou a leitura opcional de papers.
