# 00b — Migração e setup no Mac M4 (16 GB)

> 🌱 **Começando sem experiência em programação?** Prepare o ambiente aqui e faça a
> [`Fase 0 — ponte para iniciantes`](00-iniciante-zero/) antes do módulo 1.

A partir do módulo 5 o curso roda no seu **MacBook M4 com 16 GB de memória unificada**, usando **MLX** — o framework de ML da Apple para Apple Silicon.

Este documento cobre a migração da pasta, o ambiente, e — mais importante — o **orçamento de memória realista** de 16 GB, que determina quais modelos você pode tocar.

---

## 1. Migração da pasta

A pasta `llm-course` tem ~250 KB de código e ~23 MB de dados baixados. Três opções, da melhor para a mais simples:

### Git (recomendado)

No Windows:

```powershell
cd $HOME\llm-course
git init
```

Crie um `.gitignore` antes do primeiro commit — os dados baixados não precisam ir junto (o `dados.py` os rebaixa):

```
data/
__pycache__/
*.ipynb
.venv/
adapters/
fused_model/
```

Depois:

```powershell
git add -A
git commit -m "curso de LLM: modulos 1-4"
gh repo create llm-course --private --source=. --push
```

No Mac:

```bash
gh repo clone <seu-usuario>/llm-course
cd llm-course
```

O `.ipynb` está ignorado de propósito: notebooks são derivados do `lab.py` e você os regenera com `python tools/build_notebooks.py`. Versionar notebook gera conflito de merge inútil.

### Alternativas

- **iCloud Drive** — arraste a pasta. Simples, mas cuidado: o iCloud pode "otimizar" (remover localmente) arquivos grandes.
- **Zip + AirDrop** — mais direto para uma migração única.

---

## 2. Ambiente no Mac

```bash
# Homebrew, se ainda não tiver
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.12 git-lfs
cd ~/llm-course
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Dependências

Instalação reproduzível pelo `uv.lock`, incluindo as dependências de treino do MLX e os
módulos de preferência/RL:

```bash
pip install uv
uv sync --extra cpu --extra mlx --extra rl --locked
source .venv/bin/activate
```

Sem `uv`, instale pelas faixas declaradas no `pyproject.toml`:

```bash
pip install -e ".[cpu,mlx,rl]"
```

Equivalente manual:

```bash
# MLX — a trilha principal a partir do módulo 5
pip install mlx "mlx-lm[train]" mlx-lm-lora

# PyTorch (backend MPS) — para os labs dos módulos 1-4 e comparações
pip install torch transformers tokenizers

# utilitários
pip install jupyter matplotlib
```

**Não instale `bitsandbytes`.** Ele é CUDA-only; no Mac falha ou instala um stub inútil. Quem faz a quantização aqui é o próprio MLX.

### Verificação

```bash
python -c "
import mlx.core as mx, platform
print('chip      :', platform.processor(), platform.machine())
print('mlx device:', mx.default_device())
a = mx.random.normal((2048, 2048)); mx.eval(a @ a)
print('mlx ok    : matmul 2048x2048 executou na GPU')
"
python -c "import torch; print('torch mps :', torch.backends.mps.is_available())"
```

Espere `Device(gpu, 0)` no MLX e `True` no MPS.

---

## 3. 📐 O orçamento de 16 GB

Este é o número que decide tudo o que vem a seguir.

```
16 GB totais
 −4 a 6 GB   macOS, navegador, apps
 ─────────────────────────────────
 ≈10 a 11 GB disponíveis para o modelo
```

E dentro desses ~10 GB precisam caber, simultaneamente: os **pesos**, os **estados do otimizador**, os **gradientes**, as **ativações** e o **KV cache**.

### O que cabe

| Tarefa | Memória | Cabe em 16 GB? |
|---|---|---|
| Inferência Qwen2.5-1.5B bf16 | ~3,1 GB | ✅ folgado |
| Inferência Qwen2.5-7B **4-bit** | ~4,2 GB | ✅ confortável |
| Inferência 14B 4-bit | ~8 GB | ✅ apertado |
| **LoRA em 1.5B bf16** | ~5–6 GB | ✅ **alvo principal do curso** |
| **LoRA em 3B bf16** | ~8–9 GB | ✅ com `--grad-checkpoint` |
| **QLoRA em 7B 4-bit** | ~6–8 GB | ✅ com batch 1 e poucas camadas |
| LoRA em 7B bf16 | ~16 GB | ❌ |
| Full fine-tune de qualquer coisa ≥1B | ≥16 GB | ❌ |

O último caso é o módulo 3 batendo à porta: full fine-tune custa ~16 bytes/parâmetro. Um modelo de 1B já pede 16 GB só de estados. **É exatamente por isso que o módulo 6 existe.**

### Modelos-alvo do curso

Sempre prefira as versões já convertidas da `mlx-community` — elas baixam prontas, sem conversão local:

| Modelo | Uso |
|---|---|
| `mlx-community/Qwen2.5-1.5B-Instruct-bf16` | **Padrão dos labs.** Rápido, cabe com folga |
| `mlx-community/Qwen2.5-0.5B-Instruct-bf16` | Iteração rápida, mesmo modelo dos módulos 1–2 |
| `mlx-community/Qwen2.5-3B-Instruct-4bit` | Quando quiser mais capacidade |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | O teto confortável da máquina |

### Aumentando o limite da GPU (opcional)

O macOS limita quanto da memória unificada a GPU pode reservar (~75% por padrão). Para elevar:

```bash
sudo sysctl iogpu.wired_limit_mb=13000    # 13 GB, em 16 GB totais
```

> ⚠️ Isso não é permanente (volta ao reiniciar) e **é possível travar o sistema** se você deixar pouco para o macOS. Use só quando um treino específico estourar por pouco, e nunca passe de ~80% da RAM total.

---

## 4. Os comandos do MLX que você vai usar

```bash
# gerar texto
mlx_lm.generate --model mlx-community/Qwen2.5-1.5B-Instruct-bf16 \
                --prompt "Explique o que é fine-tuning." --max-tokens 200

# quantizar um modelo do HuggingFace para 4 bits
mlx_lm.convert --hf-path Qwen/Qwen2.5-3B-Instruct -q --mlx-path ./qwen3b-4bit

# treinar LoRA (QLoRA é automático se o modelo for quantizado)
mlx_lm.lora --model mlx-community/Qwen2.5-1.5B-Instruct-bf16 \
            --train --data ./dados --iters 600 --batch-size 2 \
            --num-layers 16 --mask-prompt

# avaliar no split de teste
mlx_lm.lora --model <modelo> --adapter-path adapters --data ./dados --test

# gerar com o adaptador acoplado
mlx_lm.generate --model <modelo> --adapter-path adapters --prompt "..."

# fundir o adaptador nos pesos base
mlx_lm.fuse --model <modelo> --adapter-path adapters
```

### Formato dos dados

O MLX espera uma **pasta** com `train.jsonl` e `valid.jsonl` (e opcionalmente `test.jsonl`). Quatro formatos aceitos:

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"prompt": "Qual a capital da França?", "completion": "Paris."}
{"text": "Texto cru para continuação."}
```

O formato `messages` aplica o chat template do modelo automaticamente — é o que você quer para SFT, e resolve de graça a armadilha do template e do EOS do módulo 4.

> 🔧 A flag `--mask-prompt` calcula a loss **apenas na resposta** — é o masking com `-100` do módulo 4, seção 9. Ela não é o default; ligue explicitamente.

---

## 5. MLX × CUDA — o mapa de tradução

Você vai encontrar tutoriais escritos para CUDA. Esta tabela traduz:

| Conceito | CUDA (indústria) | MLX (seu Mac) |
|---|---|---|
| Framework | PyTorch + Transformers | MLX + mlx-lm |
| LoRA | `peft.LoraConfig` | `--fine-tune-type lora` |
| Quantização 4-bit | `bitsandbytes` NF4 | `mlx_lm.convert -q` (grupos de 64, 4 bits) |
| Treino SFT | `trl.SFTTrainer` | `mlx_lm.lora --train` |
| Masking do prompt | `completion_only_loss=True` | `--mask-prompt` |
| Atenção rápida | FlashAttention 2 | kernels Metal nativos |
| Onde os pesos ficam | VRAM dedicada | memória unificada |
| Camadas adaptadas | `target_modules=[...]` | `lora_parameters.keys` |

**Uma diferença que importa:** o default do MLX é `keys: ["self_attn.q_proj", "self_attn.v_proj"]` — LoRA apenas nas projeções de query e value. No módulo 2 medimos que a atenção inteira é **12,3%** dos parâmetros de um bloco, e `q_proj`+`v_proj` são pouco mais da metade disso. O QLoRA original recomenda adaptar **todas** as camadas lineares. Você vai comparar as duas escolhas no módulo 6.

---

## 6. Os módulos 1–4 no Mac

Rodam sem alteração — são CPU e PyTorch. Regere os notebooks e siga:

```bash
cd ~/llm-course
python tools/build_notebooks.py
python modulo-03-treino/dados.py     # rebaixa o corpus
python modulo-04-dados/dados.py      # rebaixa o Alpaca
```

O treino do módulo 3 deve ficar **mais rápido** que os 126 s medidos no Windows — os P-cores do M4 são substancialmente melhores que a CPU anterior. Se quiser, troque `torch.device("cpu")` por `"mps"` e compare; para um modelo de 2M de parâmetros o MPS pode até perder, porque o overhead de despachar kernels domina.

---

## 7. Aviso metodológico

Nos módulos 1 a 4, cada lab foi **executado** antes de ser entregue, e isso pegou vários erros — contagens de tokens erradas, TFLOPs com sparsity, um teste de EOS mal construído.

A partir do módulo 5 isso não é possível: o código é para macOS/Metal e o ambiente de autoria é Windows sem GPU. A mitigação:

- Todos os comandos e flags do MLX foram verificados contra a **documentação oficial** (`mlx_lm/LORA.md` e `examples/lora_config.yaml`), não escritos de memória.
- Os labs marcam explicitamente o que é **verificável por você** (asserções, formas de tensores, contagens) para que um erro apareça cedo, e não depois de vinte minutos de treino.
- Quando algo falhar, me mande a mensagem de erro completa — é mais rápido corrigir do que adivinhar.
