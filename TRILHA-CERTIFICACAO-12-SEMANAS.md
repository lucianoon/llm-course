# Trilha de certificação profissional — 12 semanas

Esta trilha seleciona a rota profissional essencial: 36 horas de encontros guiados,
mais preparação individual e o projeto final. O restante do repositório é aprofundamento.
Cada encontro produz uma evidência, não apenas leitura. Os gates conceituais rodam em CPU;
o laboratório de SFT da semana 5 exige Mac Apple Silicon ou GPU NVIDIA. Veja a instalação em
[`AMBIENTES.md`](AMBIENTES.md) antes de iniciar.

| Semana | Preparação | Encontro/lab obrigatório | Entregável |
|---|---|---|---|
| 1 | Módulo 1 §§1–7 | `modulo-01-fundamentos/lab.py` | comparação de tokenização e decoding |
| 2 | Módulo 2 §§1–7 | `modulo-02-attention/lab.py` | atenção causal reconstruída e teste de bug |
| 3 | Módulo 3 §§2–10 | `modulo-03-treino/lab.py` | curva de treino interpretada |
| 4 | Módulo 4 + governança | `modulo-04-dados/lab.py` e auditoria do dataset | manifesto de dataset sem PII pendente |
| 5 | Módulo 5 | `lab.py` em Mac/GPU + leitura do caminho acelerado | baseline, modelo, métricas e run registrada |
| 6 | Módulo 6 | `lab_cpu.py` + `lab_adapters.py` | comparação de memória e manifesto do adapter |
| 7 | Módulo 11 | `lab_cpu.py` + benchmark disponível | memória, quantização e curva concorrência/latência |
| 8 | Módulo 13 | `lab_cpu.py` | RAG com citação, abstenção e teste de recuperação |
| 9 | Módulo 14 | `lab_cpu.py` | baseline, amostra, intervalo de confiança e análise de erro |
| 10 | Módulo 15 | `lab_cpu.py` | ferramenta segura, limite de passos e trilha auditável |
| 11 | Módulo 19 | `lab_cpu.py` | p50/p95, custo, logs, disjuntor e rollback |
| 12 | Módulo 12 | projeto e banca | repositório, relatório, apresentação e revisão |

## Estrutura das três horas

- 30 min: recuperação ativa e diagnóstico.
- 45 min: conceito e decisões.
- 90 min: laboratório em pares.
- 15 min: registro de evidência e próximos passos.

## Critério de certificação

- 70%: projeto final reproduzível e medido.
- 20%: evidências semanais executadas.
- 10%: revisão técnica de outro projeto.

Não há aprovação se o projeto não tiver baseline justa, manifesto de dados, revisão
imutável do modelo, teste sem vazamento, métricas antes/depois e limitações explícitas. A
[rubrica completa](CERTIFICACAO.md) define os pesos, mínimos e o badge verificável. A rota
obrigatória é majoritariamente executável em CPU, mas a evidência de SFT real da semana 5
requer hardware acelerado; um dry-run não será apresentado como treino executado.

## Depois da certificação

Os módulos 7–10 (reasoning, DPO, RL e distillation) são especializações. Faça-os quando o
projeto exigir preferência, recompensa verificável ou transferência de capacidade. A trilha de
pesquisa e os módulos 16–18 continuam disponíveis em [FASE-3-MAESTRIA.md](FASE-3-MAESTRIA.md).
