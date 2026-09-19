# Setup GPU NVIDIA — CUDA, PEFT, TRL e vLLM

Esta é a rota portátil para os labs de modelos reais. MLX continua disponível no Mac,
mas os scripts `lab_cuda.py` reproduzem o workflow usado em servidores NVIDIA.

## Ambiente

Requisitos: Linux, Python 3.11 ou 3.12, driver NVIDIA recente e GPU com suporte a bf16
preferencialmente. Em Colab/Runpod, comece numa imagem PyTorch atual.

Como ponto de partida, reserve pelo menos 20 GB de disco para ambiente, cache e modelos:

| VRAM disponível | Rota realista | Observação |
|---:|---|---|
| 16 GB | modelos pequenos e LoRA | reduza batch, sequência e número de passos |
| 24 GB | QLoRA de modelos pequenos/médios | valide o orçamento antes de baixar pesos |
| 40–48 GB | LoRA/QLoRA mais largos e serving pequeno | ainda depende da arquitetura e do contexto |
| 80 GB ou mais | experimentos CUDA mais pesados | não elimina o custo de dados e avaliação |

Esses valores são faixas de planejamento, não garantias de execução. Confirme o hardware
antes da instalação:

```bash
nvidia-smi
```

```bash
git clone https://github.com/lucianoon/llm-course
cd llm-course
pip install uv
uv sync --extra gpu --extra dev --locked
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
uv run python tools/doctor.py --profile gpu --require-cuda
```

O extra `gpu` instala Transformers, PEFT, TRL, Accelerate e bitsandbytes. O serving possui
restrições próprias de CUDA/PyTorch. No mesmo ambiente, instale os extras juntos para que o
`uv sync` não remova as dependências GPU:

```bash
uv sync --extra gpu --extra serving --extra dev --locked
```

Se preferir isolar o vLLM, crie um segundo ambiente virtual e use apenas `serving` nele.
O fluxo completo está em [`modulo-11-inferencia/SERVING.md`](modulo-11-inferencia/SERVING.md).

O registro local não exige conta externa. Se quiser espelhar métricas, instale
`uv sync --extra gpu --extra tracking --extra dev --locked` e passe `--report-to wandb` aos labs
SFT/GRPO. O suporte opcional a MLflow foi retirado temporariamente enquanto não existe versão
corrigida para o alerta de SSRF `GHSA-h7x2-h6g9-p789`.

## Smoke tests sem gastar GPU

Cada lab CUDA possui `--dry-run`: valida argumentos e mostra exatamente o que seria
executado sem importar PyTorch ou baixar modelos.

```bash
uv run python modulo-05-sft/lab_cuda.py --dry-run
uv run python modulo-09-rl/lab_cuda.py --dry-run
uv run python tools/validate_labs.py
```

Para uma reprodução, fixe as revisões do Hub em vez de aceitar a resolução atual:

```bash
uv run python modulo-05-sft/lab_cuda.py --metodo lora --revision <commit-do-modelo>
uv run python modulo-09-rl/lab_cuda.py --revision <commit-do-modelo>
uv run python modulo-10-distillation/lab_cuda.py \
  --revision-professor <commit-professor> --revision-aluno <commit-aluno>
```

`RESOLVIDA_EM_RUNTIME` é o padrão conveniente para exploração, mas não é suficiente
para declarar uma reprodução.

## Ordem recomendada

1. `uv run python modulo-05-sft/preparar_dados.py`
2. Audite os arquivos preparados (preencha origem, licença e finalidade reais):
   `uv run python tools/auditar_dataset.py modulo-05-sft/suporte/train.jsonl modulo-05-sft/suporte/valid.jsonl --nome suporte --origem "fonte do dataset" --licenca "licença verificada" --finalidade "SFT de suporte"`
3. `uv run python modulo-05-sft/lab_cuda.py --metodo lora`
4. Repita com `--metodo qlora`; full fine-tuning só quando a conta de VRAM permitir.
5. `uv run python modulo-09-rl/preparar_dados.py`
6. `uv run accelerate launch modulo-09-rl/lab_cuda.py`

Todos os treinos salvam metadados e métricas em `runs/`. O commit e a revisão imutável
do modelo são registrados. Não publique datasets, traces ou adapters antes de executar
a auditoria de governança e preencher seus manifestos.

## Limites honestos

Os scripts CUDA foram verificados estaticamente contra as APIs oficiais, mas continuam
com status **GPU pendente** até uma execução ser preservada em `resultados/`. Uma saída
sem hardware, versão, commit, seed e revisão do modelo não muda esse status.
