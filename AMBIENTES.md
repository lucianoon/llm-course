# Matriz de ambientes

Este arquivo resume qual instalação usar para cada rota do curso. Todas as rotas
usam Python 3.11 ou 3.12 e devem ser instaladas dentro do ambiente gerenciado
por `uv`.

## Rotas suportadas

| Rota | Sistema recomendado | Hardware | Extras | Status de validação |
|---|---|---|---|---|
| Fundamentos e Fase 2 | Linux, macOS ou Windows | CPU | `cpu`, `dev`, `test` | CI em Ubuntu e macOS |
| Customização | macOS em Apple Silicon | GPU integrada Apple | `cpu`, `mlx`, `rl` | validação estática + `--dry-run`; execução depende de Mac |
| Customização CUDA | Linux | NVIDIA com suporte a bf16 | `gpu`, `dev` | validação estática + `--dry-run`; execução depende de GPU |
| Serving com vLLM | Linux | NVIDIA compatível com vLLM | `serving` | contrato e métricas do módulo 19 |
| Tracking opcional | qualquer rota compatível | — | `tracking` | integração depende de credenciais W&B |

As faixas de VRAM são apenas um orçamento inicial; sequência, batch, quantização e arquitetura
podem alterar o consumo. A tabela operacional detalhada está em [`00-setup-gpu.md`](00-setup-gpu.md).

## Instalação mínima por rota

### CPU

```bash
uv sync --extra cpu --extra dev --extra test --locked
```

Use esta rota para a Fase 0, módulos 1–4 e 13–19. Ela não baixa modelos
automaticamente durante a instalação.

### Apple Silicon / MLX

```bash
uv sync --extra cpu --extra mlx --extra rl --extra dev --locked
```

Use os scripts `lab_mlx.py` nos módulos que oferecem essa variante. Não instale
`bitsandbytes` no Mac; essa dependência pertence à rota CUDA.

### NVIDIA / CUDA

```bash
uv sync --extra gpu --extra dev --locked
```

Para os benchmarks de serving, crie o ambiente separado recomendado pelo módulo:

```bash
uv sync --extra serving --locked
```

Os scripts CUDA devem ser testados primeiro com `--dry-run`. Esse modo verifica
argumentos e configuração sem baixar modelos nem iniciar treinamento.

## O que o CI garante

O CI verifica Python 3.11 e 3.12 em Ubuntu e macOS, lint, compilação, geração
de notebooks, testes sem download de modelos e gates CPU selecionados. Ele não
substitui uma execução em Apple Silicon, GPU NVIDIA ou um servidor vLLM.

Para saber se um resultado experimental foi realmente reproduzido, consulte
[`EVIDENCIAS.md`](EVIDENCIAS.md).

Para validar localmente os contratos baratos dos laboratórios:

```bash
uv run python tools/validate_labs.py
```

Esse comando analisa a sintaxe dos 35 labs e executa os seis scripts acelerados
em modo `--dry-run`, sem baixar modelos ou datasets.

Para diagnosticar uma instalação GPU ou serving:

```bash
uv run python tools/doctor.py --profile gpu --require-cuda
uv run python tools/doctor.py --profile serving
```

O diagnóstico diferencia pacote ausente, falha de importação nativa e ausência de
GPU. Um resultado `OK` não substitui a execução reproduzida do laboratório.
