# Setup GPU NVIDIA — CUDA, PEFT, TRL e vLLM

Esta é a rota portátil para os labs de modelos reais. MLX continua disponível no Mac,
mas os scripts `lab_cuda.py` reproduzem o workflow usado em servidores NVIDIA.

## Ambiente

Requisitos: Linux, Python 3.11 ou 3.12, driver NVIDIA recente e GPU com suporte a bf16
preferencialmente. Em Colab/Runpod, comece numa imagem PyTorch atual.

```bash
git clone https://github.com/lucianoon/llm-course
cd llm-course
pip install uv
uv sync --extra gpu --extra dev --locked
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
```

O extra `gpu` instala Transformers, PEFT, TRL, Accelerate e bitsandbytes. O serving fica
separado porque vLLM possui restrições próprias de CUDA/PyTorch:

```bash
uv sync --extra serving --locked
```

O registro local não exige conta externa. Se quiser espelhar métricas, instale
`uv sync --extra gpu --extra tracking --locked` e passe `--report-to wandb` ou
`--report-to mlflow` aos labs SFT/GRPO.

## Smoke tests sem gastar GPU

Cada lab CUDA possui `--dry-run`: valida argumentos e mostra exatamente o que seria
executado sem importar PyTorch ou baixar modelos.

```bash
python modulo-05-sft/lab_cuda.py --dry-run
python modulo-09-rl/lab_cuda.py --dry-run
```

## Ordem recomendada

1. `python modulo-05-sft/preparar_dados.py`
2. `python modulo-05-sft/lab_cuda.py --metodo lora`
3. Repita com `--metodo qlora`; full fine-tuning só quando a conta de VRAM permitir.
4. `python modulo-09-rl/preparar_dados.py`
5. `accelerate launch modulo-09-rl/lab_cuda.py`

Todos os treinos salvam metadados e métricas em `runs/`. O commit e a revisão imutável
do modelo são registrados. Não publique datasets, traces ou adapters antes de executar
a auditoria de governança e preencher seus manifestos.

## Limites honestos

Os scripts CUDA foram verificados estaticamente contra as APIs oficiais, mas continuam
com status **GPU pendente** até uma execução ser preservada em `resultados/`. Uma saída
sem hardware, versão, commit, seed e revisão do modelo não muda esse status.
