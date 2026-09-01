# Trilha intensiva de customização — 12 semanas, 36 horas

Esta trilha seleciona o subconjunto que cabe em 3 horas semanais. O restante do
repositório é aprofundamento. Cada encontro produz uma evidência, não apenas leitura.

> Esta é uma trilha de conclusão autônoma, não uma certificação profissional emitida,
> supervisionada ou acreditada. “Concluir” abaixo significa preservar as evidências e
> satisfazer a rubrica; uma certificação formal exigiria avaliação por uma banca identificada.

| Semana | Preparação | Encontro/lab obrigatório | Entregável |
|---|---|---|---|
| 1 | Módulo 1 §§1–7 | `modulo-01-fundamentos/lab.py` | comparação de tokenização e decoding |
| 2 | Módulo 2 §§1–7 | `modulo-02-attention/lab.py` | atenção causal reconstruída e teste de bug |
| 3 | Módulo 3 §§2–10 | `modulo-03-treino/lab.py` | curva de treino interpretada |
| 4 | Módulo 4 + governança | `modulo-04-dados/lab.py` e auditoria do dataset | manifesto de dataset sem PII pendente |
| 5 | Módulo 5 | `lab_cuda.py --metodo lora` | baseline, modelo, métricas e run registrada |
| 6 | Módulo 6 | repetir `lora`/`qlora` + `lab_adapters.py` | comparação de memória e manifesto do adapter |
| 7 | Módulo 7 | `lab_cpu.py` + `lab_process_supervision.py` | outcome vs processo e análise de risco |
| 8 | Módulo 8 | `lab_cpu.py`; DPO/ORPO em GPU quando disponível | preferência, PPL e inspeção de saídas |
| 9 | Módulo 9 | `lab_cuda.py` | curva de recompensa, KL e reward hacking auditado |
| 10 | Módulo 10 | `lab_avancado.py` + `lab_cuda.py` | teacher/student antes/depois e conta de custo |
| 11 | Módulo 11 | `lab_moe_cuda.py` + `benchmark_vllm.py` | utilização de experts e curva concorrência/latência |
| 12 | Módulo 12 | projeto e banca | repositório, modelo, relatório e apresentação |

## Estrutura das três horas

- 30 min: recuperação ativa e diagnóstico.
- 45 min: conceito e decisões.
- 90 min: laboratório em pares.
- 15 min: registro de evidência e próximos passos.

## Critério de conclusão

- 70%: projeto final reproduzível e medido.
- 20%: evidências semanais executadas.
- 10%: revisão técnica de outro projeto.

Não há aprovação se o projeto não tiver baseline justa, manifesto de dados, revisão
imutável do modelo, teste sem vazamento, métricas antes/depois e limitações explícitas.
