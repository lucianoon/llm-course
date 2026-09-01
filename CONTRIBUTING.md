# Como contribuir

Obrigado por ajudar a tornar este curso mais correto, reproduzível e acessível.

## Contribuições prioritárias

- reproduções independentes dos experimentos listados em `EVIDENCIAS.md`;
- correções conceituais acompanhadas de fonte primária ou experimento;
- relatos de instalação em Linux, macOS, Apple Silicon e CUDA;
- melhorias de acessibilidade, exercícios e explicações em português;
- testes que impeçam a volta de um erro já corrigido.

## Antes de abrir um pull request

1. Abra uma issue para mudanças grandes de currículo ou arquitetura.
2. Mantenha cada pull request focado em um problema.
3. Não versione modelos, datasets baixados, chaves, credenciais ou dados pessoais.
4. Para resultados experimentais, registre ambiente, versões, seed, amostra e limitações.
5. Execute as verificações locais:

```bash
uv sync --extra dev --extra test --locked
uv run python tools/build_notebooks.py
uv run ruff check .
uv run python -m pytest
uv run python tools/eval_ci.py
```

## Reproduções

Use `tools.reproducao.registrar_reproducao` e grave o resultado sob
`resultados/<experimento>/<data>-<commit>.json`. Consulte `resultados/README.md` para o contrato.
Uma reprodução deve informar também qualquer desvio do protocolo original.

## Estilo

- Prefira português claro; defina o termo técnico em inglês na primeira ocorrência.
- Diferencie observação, inferência e recomendação.
- Não generalize um resultado além do modelo, dados e ambiente medidos.
- Labs devem ser determinísticos quando isso for compatível com o objetivo pedagógico.

Ao enviar uma contribuição, você concorda que ela será distribuída sob a licença Apache-2.0
do repositório.
