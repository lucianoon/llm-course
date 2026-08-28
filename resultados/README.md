# Resultados — o lugar das medições reproduzidas

Este diretório guarda as **reproduções** dos resultados citados no curso, no schema que
[`EVIDENCIAS.md`](../EVIDENCIAS.md) exige. Uma alegação só passa de "executada na autoria"
para **reproduzida** quando há aqui um arquivo com commit, ambiente, hardware, seed e
métricas preservadas.

## Como registrar uma reprodução

```python
from pathlib import Path
from tools.reproducao import registrar_reproducao

arquivo = registrar_reproducao(
    Path("."),
    experimento="modulo-02/lab",
    comando="uv run python modulo-02-attention/lab.py",
    seed=0,
    metricas={"bit_exact": True},
    observacoes="Qwen2.5-0.5B, float32/eager",
)
```

## Estrutura

```text
resultados/
  modulo-NN/<nome-do-lab>/   ← experimento = "modulo-NN/nome-do-lab"
    <AAAA-MM-DD>-<commit8>.json ← uma execução por arquivo (mesma data/commit = reexecução)
```

O JSON guarda as métricas estruturadas; stdout, tabelas completas ou gráficos ficam ao
lado. Um resultado só é **reproduzido** quando outra execução preservada confirma o
mesmo número dentro da tolerância declarada (o espelho do que o módulo 14 ensina: todo
número que decide algo carrega o tamanho da amostra e o intervalo).
