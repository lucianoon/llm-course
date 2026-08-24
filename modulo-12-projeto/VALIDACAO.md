# Validação do projeto final

O diretório do projeto deve conter:

```text
projeto/
  README.md
  config.json
  dataset-manifest.json
  model-card.md
  relatorio.md
  resultados.json
  scripts/
    preparar.py
    treinar.py
    avaliar.py
    servir.py
```

`resultados.json` precisa declarar `baseline`, `modelo_final`, `n`, `metrica`, intervalo
de confiança quando aplicável, commit, revisão do modelo, seed e hardware. A banca deve
conseguir regenerá-lo executando os scripts, sem editar caminhos à mão.

## Rubric de banca (100 pontos)

| Critério | Pontos | Falha eliminatória |
|---|---:|---|
| Problema e métrica definidos antes do treino | 10 | métrica não representa o produto |
| Governança, licença, PII e split | 15 | uso sem autorização ou vazamento |
| Baseline esforçada | 15 | baseline deliberadamente fraca |
| Método e hiperparâmetros | 10 | decisão sem justificativa |
| Reprodutibilidade | 15 | resultado não regenerável |
| Avaliação quantitativa e IC | 10 | conjunto de teste usado para otimizar |
| Inspeção qualitativa e efeitos colaterais | 10 | nenhuma leitura de erros |
| Serving, latência, memória e custo | 10 | só mede qualidade offline |
| Limitações e comunicação | 5 | omite resultado negativo material |

O projeto reprova independentemente da nota se violar uma falha eliminatória.
