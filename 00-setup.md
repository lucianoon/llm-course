# 00 — Setup do ambiente (Windows / CPU)

> ℹ️ **Este documento cobre os módulos 1–4 (CPU) e a estratégia de GPU em nuvem.** A trilha
> principal dos módulos 5–12 roda no **Mac M4 com MLX** — veja [`00-setup-mac.md`](00-setup-mac.md).
> A seção 3 abaixo (Colab/Runpod) permanece como alternativa CUDA, útil se você quiser
> reproduzir as receitas da indústria ou treinar modelos acima do teto do M4.

Seu ambiente detectado: **Windows 11**, **Python 3.11**, **sem GPU NVIDIA local**.

Isso define a estratégia do curso: módulos 1–4 rodam inteiros na sua máquina (CPU basta para tokenizadores, inspeção de modelos pequenos e curadoria de dados). A partir do módulo 5 (SFT), você precisa de GPU — e a decisão de *qual* GPU é ela própria conteúdo do curso.

---

## 1. Ambiente local (módulos 1–4)

```powershell
cd $HOME\llm-course
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Se o `Activate.ps1` for bloqueado pela política de execução:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Dependências

O repositório declara as dependências em `pyproject.toml` e fixa a resolução completa
em `uv.lock`. Para reproduzir exatamente o ambiente validado, na raiz do curso:

```powershell
pip install uv
uv sync --extra cpu --locked
```

Sem `uv`, a instalação convencional continua disponível, mas resolve as versões dentro
das faixas declaradas em vez de usar o lockfile:

```powershell
pip install -e ".[cpu]"
```

Equivalente manual, se você não quiser instalar o projeto em modo editável:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers tokenizers datasets accelerate
pip install jupyter matplotlib numpy pandas
```

`torch` de CPU é ~200MB em vez dos ~2.5GB da build CUDA. Como não há GPU aqui, instalar a build CUDA só desperdiça disco.

> ⚠️ **Armadilha:** não instale `bitsandbytes` localmente. Ele exige CUDA e falha ou instala um stub inútil no Windows. Ele entra só no ambiente de GPU, no módulo 6.

### Verificação

```powershell
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print('cuda:', torch.cuda.is_available())"
```

Espere `cuda: False`. Está correto.

> **Você já tem `torch` e `transformers` instalados no Python global** (verificado). O venv acima é recomendado para isolar o curso, mas se preferir rodar direto no global, só faltam `jupyter` e `matplotlib`:
> ```powershell
> pip install jupyter matplotlib
> ```

---

## 2. Modelos usados nos labs locais

Os labs de fundamentos usam modelos pequenos o bastante para CPU:

| Modelo | Params | Download | Para quê |
|---|---|---|---|
| `gpt2` | 124M | ~550MB | Arquitetura clássica, tokenizer BPE de 50k, sem chat template |
| `Qwen/Qwen2.5-0.5B-Instruct` | 494M | ~1GB | Modelo moderno: GQA, RoPE, vocabulário de 151k, chat template |

O cache do HuggingFace fica em `C:\Users\<você>\.cache\huggingface`. Para mudar:

```powershell
$env:HF_HOME = "D:\hf-cache"
```

> **Aviso de symlink no Windows:** o `huggingface_hub` avisa que sua máquina não suporta symlinks e que o cache ficará "em versão degradada". Na prática significa que arquivos duplicados ocupam espaço duas vezes — irrelevante neste curso. Para silenciar:
> ```powershell
> $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
> ```
> Se quiser resolver de verdade, ative o Modo Desenvolvedor do Windows.

### Conta HuggingFace

Não é obrigatória para os modelos acima, mas é para modelos *gated* (Llama, Gemma) que aparecem depois:

```powershell
pip install huggingface_hub
huggingface-cli login
```

Crie o token em https://huggingface.co/settings/tokens com escopo *read*.

---

## 3. GPU a partir do módulo 5

O curso original oferece GPU. Sem ela, as opções reais, em ordem de custo:

| Opção | Custo | VRAM | Serve para |
|---|---|---|---|
| **Google Colab** (grátis) | R$ 0 | T4 16GB, sessão instável | QLoRA em modelos ≤ 3B, provas de conceito |
| **Colab Pro** | ~US$ 10/mês | L4 24GB / A100 40GB | QLoRA 7–8B confortável |
| **Runpod / Vast.ai** (spot) | ~US$ 0,20–0,40/h por A5000 24GB | 24–80GB | Treinos longos, controle total do ambiente |
| **Lambda / Modal** | ~US$ 1,10/h por A100 40GB | 40–80GB | Full fine-tune, treinos de MoE |

Recomendação prática: **Colab grátis para os módulos 5–8** (QLoRA em modelos de 0.5B–3B ensina tudo o que importa) e **Runpod por hora nos módulos 9–12**, quando RL e distillation exigem runs mais longos.

> 🔧 **Na prática:** o erro de custo mais comum não é escolher a GPU errada — é deixar uma instância ligada ociosa. Em Runpod, sempre `terminate`, não `stop`, quando terminar; volume parado continua cobrando.

### Regra de bolso de VRAM

Guarde estes números, eles decidem quase tudo no curso:

| Tarefa | VRAM aproximada |
|---|---|
| Inferência bf16 | `2 × params` bytes → 7B ≈ 14GB |
| Inferência 4-bit | `0.5 × params` + overhead → 7B ≈ 5GB |
| Full fine-tune (AdamW, mixed precision) | `~16 × params` bytes → 7B ≈ 112GB **só de estados** |
| LoRA sobre base bf16 | `2 × params` + ~1% → 7B ≈ 16GB |
| QLoRA (base 4-bit) | `0.5 × params` + ~1% → 7B ≈ 6–8GB |

A distância entre 112GB e 7GB na mesma linha "treinar 7B" é a razão de existirem os módulos 6 e 11.

---

## 4. Gerar os notebooks

```powershell
python tools\build_notebooks.py
jupyter notebook
```

O script converte todo `modulo-*/lab.py` em `lab.ipynb`. Se você editar o `.ipynb` e quiser voltar para o `.py`, edite o `.py` — ele é a fonte da verdade; o notebook é derivado e será sobrescrito.
