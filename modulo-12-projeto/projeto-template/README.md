# Projeto template — módulo 12

Esqueleto **mínimo e executável** do contrato do projeto final. Não é o seu projeto —
é o ponto de partida. Clone esta pasta, renomeie e preencha com o seu problema.

```text
projeto-template/
  config.json          ← a decisão em dados (problema, métrica, baselines, modelo, serving)
  scripts/
    preparar.py         coleta, limpeza, splits, manifesto (mód. 4)
    treinar.py          plano de treino; épocas calculadas (mód. 5)
    avaliar.py          gera o resultados.json (mód. 14)
    servir.py           endpoint + guardião/disjuntor (mód. 19)
  model-card.md         ficha técnica do modelo
  relatorio.md          o relatório do mód. 12 (seção 7 obrigatória)
  resultados.json       ← gerado por avaliar.py
```

## Como o esqueleto se conecta ao curso

| Arquivo | O que invoca | Módulo |
|---|---|---|
| `preparar.py` | manifesto de dados, auditoria de PII | 4 |
| `treinar.py` | cálculo de épocas (passos × batch ÷ exemplos) | 5 |
| `avaliar.py` | compare os resultados com baselines e IC | 14 |
| `servir.py` | guardião de custo + disjuntor + extrato p50/p95 | 19 |

## Rodar (dry-run)

```bash
uv run python modulo-12-projeto/projeto-template/scripts/preparar.py --dry-run
uv run python modulo-12-projeto/projeto-template/scripts/treinar.py --dry-run
uv run python modulo-12-projeto/projeto-template/scripts/avaliar.py
uv run python modulo-12-projeto/projeto-template/scripts/servir.py --prompt "minha pergunta"
```

Os scripts são *skeletons validados*: compilam, passam no ruff e imprimem o plano —
mas o corpo (o treino, a geração, a avaliação reais) é seu, do módulo correspondente.

## Critério de prontidão

O projeto está concluído quando os cinco pontos do revisor técnico em
[`../VALIDACAO.md`](../VALIDACAO.md) têm resposta: *por que esta solução, como sabemos
que funciona, quanto custa, como falha, outra pessoa reproduz?*
