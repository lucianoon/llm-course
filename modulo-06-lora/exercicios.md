# Módulo 6 — Exercícios

Os práticos marcados com 💻 rodam no `lab_cpu.py` (qualquer máquina); os marcados com 🍎 exigem o Mac.

---

## Parte A — Conceituais

### A1. O orçamento

Você quer fine-tunar o Llama-3-8B. Disponível: um Mac M4 de 16 GB.

a) Full fine-tune com AdamW: quanta memória só de pesos e estados?
b) LoRA r=16 sobre base bf16?
c) QLoRA r=16 sobre base NF4?
d) Qual cabe, e o que falta na conta?

<details><summary>Gabarito</summary>

a) `8,03e9 × 16 bytes = **128,5 GB**`. Oito Macs.

b) Base bf16 = `8,03e9 × 2 = 16,1 GB`, mais os adaptadores e seus estados (~0,8 GB). **≈16,9 GB** — não cabe, e o problema é a base, não o adaptador.

c) Base NF4 = `8,03e9 × 0,5 = 4,0 GB`, mais adaptadores (~0,8 GB). **≈4,8 GB** — cabe.

d) Só QLoRA. E a conta ignora **ativações** e **KV cache**, que dependem de `batch_size` e `max_seq_length` e podem somar vários GB. Trate a tabela do README como piso: 4,8 GB de pesos com `max_seq_length=2048` e batch 4 pode facilmente virar 9–10 GB no total.
</details>

---

### A2. A inicialização

Por que `B = 0` e `A` aleatória? Analise três alternativas:

a) Ambas aleatórias.
b) Ambas zero.
c) `A = 0` e `B` aleatória.

<details><summary>Gabarito</summary>

a) **Ambas aleatórias:** `BA ≠ 0` no passo 0, então o modelo inicial já é diferente — e pior — que a base. Você perde a propriedade de "partir exatamente do modelo pré-treinado" e introduz ruído em toda a rede antes do primeiro gradiente.

b) **Ambas zero:** `BA = 0` corretamente, mas o gradiente também é zero para as duas. `∂(BA)/∂A = Bᵀ = 0` e `∂(BA)/∂B = Aᵀ = 0`. O adaptador nunca sai de zero — simetria que nada quebra.

c) **`A = 0`, `B` aleatória:** funciona, é o espelho da convenção. Na prática dá quase no mesmo; a assimetria escolhida (A aleatória, B zero) é a do paper e a que todos os frameworks adotam.

O requisito real é: **o produto começa em zero, mas pelo menos um dos fatores é não nulo** — para que o gradiente flua.
</details>

---

### A3. A degradação assimétrica

O Lab 8 mediu, ao quantizar o Qwen2.5-0.5B para NF4:

| Domínio | Degradação |
|---|---|
| Literatura PT | +17,4% |
| Inglês | +4,3% |

a) Por que a diferença?
b) Que implicação isso tem para você, que trabalha em português?
c) Como você mediria isso corretamente para um caso de uso real?

<details><summary>Gabarito</summary>

a) O modelo é mais fraco em português literário — perplexidade base de 43,5 contra 2,1 em inglês. Onde ele já opera com margem estreita e representações menos robustas, o ruído da quantização o empurra para fora com mais facilidade. Quantização **amplia fraquezas existentes**, não degrada uniformemente.

Some a isso a tokenização: do módulo 1, o português consome mais tokens por caractere, então cada previsão carrega menos informação e há mais oportunidades de erro acumulado.

b) Que os números de degradação publicados em papers — quase sempre medidos em inglês, em WikiText ou C4 — **subestimam o custo no seu caso**. "4-bit degrada só 2%" pode significar 10% no seu domínio.

c) Monte um conjunto de avaliação com textos do **seu** domínio, com milhares de tokens, e meça antes/depois. Melhor ainda: meça na tarefa final (acurácia, aderência de formato, win rate), não em perplexidade. Perplexidade é um proxy conveniente, não o objetivo.
</details>

---

### A4. Traduzindo configurações

Você tem esta configuração de um tutorial CUDA:

```python
LoraConfig(r=32, lora_alpha=64, lora_dropout=0.1,
           target_modules=["q_proj","k_proj","v_proj","o_proj",
                           "gate_proj","up_proj","down_proj"])
```

a) Escreva o YAML equivalente do `mlx_lm.lora`.
b) Por que não dá para passar isso por flags de linha de comando?

<details><summary>Gabarito</summary>

a)
```yaml
fine_tune_type: lora
lora_parameters:
  keys: ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
  rank: 32
  scale: 2.0        # lora_alpha / r = 64 / 32
  dropout: 0.1
```

Note os prefixos `self_attn.` e `mlp.` — o MLX casa pelo caminho do módulo, não pelo nome curto.

b) Porque `build_parser()` do `mlx_lm/lora.py` **não define** argumentos para `rank`, `scale`, `dropout` ou `keys`. Eles vivem apenas em `CONFIG_DEFAULTS` e só são sobrescritos por `--config`.

O perigo prático: passar `--rank 32` pode não gerar erro visível, e você treina com o default `rank: 8` convencido de que mudou. Sempre imprima o YAML antes de treinar.
</details>

---

### A5. O trade-off

O Lab 4 mediu, num fine-tune de subdomínio:

| Método | Ganho no alvo | Esquecimento |
|---|---|---|
| Full fine-tune | −28,7% | +16,2% |
| LoRA r=8 | −8,9% | +4,4% |

Um colega conclui: "LoRA é pior, use full fine-tune sempre que couber na memória". Avalie.

<details><summary>Gabarito</summary>

A conclusão é apressada por três razões:

1. **Esquecimento é um custo real**, não um detalhe. Se o modelo precisa continuar bom em tarefas gerais — quase sempre o caso de um assistente — os 16,2% de degradação podem inviabilizar o resultado, mesmo com ganho maior no alvo.

2. **O experimento é o caso mais adverso para LoRA:** modelo de 2M de parâmetros treinado do zero, subdomínio muito distante, 150 passos. Em modelos de bilhões, com fine-tune de comportamento, a lacuna é bem menor — é o que Biderman et al. (2024) mediram em escala real, no paper cujo título é literalmente *"LoRA Learns Less and Forgets Less"*.

3. **Memória não é a única restrição.** LoRA dá adaptadores de poucos MB, permite servir vários clientes com uma base compartilhada, e torna trivial reverter ou trocar comportamentos.

A conclusão defensável: **se você tem memória sobrando, dados abundantes, e o esquecimento não importa, full fine-tune tende a ser melhor no alvo.** Fora dessas condições — que raramente se acumulam — LoRA vence por engenharia, não só por economia.
</details>

---

## Parte B — Práticas

### B1. 💻 O espectro em modelos maiores

O Lab 1 mediu o espectro de `ΔW` no MiniGPT (2M de parâmetros). Refaça com um MiniGPT maior (`d=384`, 6 camadas) e compare a razão `r_90 / posto_máximo`.

A hipótese de baixo posto fica mais forte ou mais fraca com o tamanho?

<details><summary>Gabarito esperado</summary>

Espere que a razão **diminua** — modelos maiores têm mais redundância, e a atualização se concentra numa fração menor do espaço disponível.

É exatamente por isso que LoRA funciona bem em modelos de bilhões de parâmetros e menos bem no MiniGPT. Se você quiser ser rigoroso, plote `r_90/posto` contra `d` para três ou quatro tamanhos.
</details>

---

### B2. 💻 Rank alto e rsLoRA

O Lab 5 mostrou retorno decrescente do rank. Investigue se isso é limitação do rank ou da **escala**:

1. Treine com `r` = 8, 32, 128, mantendo `alpha = 2r` (escala constante = 2).
2. Repita com escala `alpha/√r` (rsLoRA).
3. Compare.

<details><summary>Gabarito</summary>

A tese do rsLoRA é que a escala `α/r` **penaliza demais** ranks altos: conforme `r` cresce, cada direção contribui menos, e o adaptador satura. Com `α/√r`, a magnitude da atualização se mantém estável.

Espere que rsLoRA ajude em `r ≥ 64` e faça pouca diferença em `r ≤ 16`. Se o seu experimento não mostrar diferença nenhuma, verifique se o treino está longo o bastante para o rank alto ser usado — ranks maiores tipicamente precisam de mais passos.
</details>

---

### B3. 💻 Quantização por bloco

O Lab 8 usou blocos de 64. Meça o erro de reconstrução com blocos de 16, 32, 64, 128, 256 e sem blocos (escala única para a matriz).

a) Como o erro varia?
b) Qual o custo de memória de cada escolha, em bits por peso?
c) Onde está o ponto ótimo, e por que o QLoRA escolheu 64?

<details><summary>Gabarito</summary>

a) Blocos menores → erro menor (a escala se adapta melhor a variações locais), com retorno decrescente.

b) Uma constante `float32` por bloco custa `32/bloco` bits por peso:
- bloco 16 → 2,0 bits/peso (**metade do custo dos próprios pesos!**)
- bloco 64 → 0,5 bit/peso
- bloco 256 → 0,125 bit/peso

c) O bloco 64 equilibra: erro já próximo do mínimo, custo de 0,5 bit/peso — que a *double quantization* reduz para ~0,127 quantizando as próprias constantes em blocos de 256.

O exercício mostra por que a double quantization existe: sem ela, blocos pequenos (que dão bom erro) custariam caro demais em metadados.
</details>

---

### B4. 🍎 Rank × memória × tempo, medidos

Rode o Lab 4 do `lab_mlx.py` e monte a tabela completa: rank, tamanho do adaptador, memória de pico (Monitor de Atividade), tempo, e test loss.

Qual rank você escolheria para o dataset de suporte? Justifique com os três eixos.

<details><summary>Gabarito esperado</summary>

Para uma tarefa de **formato** com 120 exemplos, espere que ranks baixos (4–8) já saturem. Rank 64 provavelmente não melhora o test loss e gasta mais memória e tempo.

A justificativa completa precisa dos três eixos: se `r=4` empata em qualidade com `r=64`, escolha `r=4` — sobra memória para aumentar `batch_size` ou `max_seq_length`, que provavelmente ajudam mais.
</details>

---

### B5. 🍎 O merge que quebra

Demonstre empiricamente a armadilha do merge em QLoRA:

1. Treine QLoRA sobre `qwen15b-4bit`.
2. Avalie com `--adapter-path` (sem merge).
3. Funda com `mlx_lm.fuse`, avalie o modelo fundido.
4. Requantize o fundido para 4 bits e avalie de novo.

Compare os três. Onde está a perda?

<details><summary>Gabarito</summary>

Espere: (2) e (3) muito próximos — o merge é matematicamente exato, como o Lab 7 do `lab_cpu.py` verificou com erro de 2×10⁻⁵.

O passo (4) é onde aparece a perda. Você treinou o adaptador para corrigir os erros de **uma base NF4 específica**. Ao fundir e requantizar, a nova base quantizada difere daquela — a correção aprendida não corresponde mais ao que ela deveria corrigir, e parte do treino é desperdiçada.

A prática correta: sirva com o adaptador acoplado (o custo é desprezível), ou use QA-LoRA, que mantém a quantização durante todo o processo.
</details>

---

## Desafio — reproduzir o paper do LoRA

O paper (Hu et al., 2021), seção 7.2, faz três análises. Reproduza-as no MiniGPT:

1. **Quais matrizes adaptar.** Com orçamento fixo de parâmetros, compare adaptar só `W_q`, só `W_v`, `W_q+W_v` com metade do rank, e as quatro de atenção com um quarto. Qual vence?

2. **Subespaço entre ranks.** Treine `r=8` e `r=64` e meça a similaridade de subespaço de Grassmann entre as direções principais aprendidas. As direções de `r=8` aparecem dentro das de `r=64`?

3. **ΔW amplifica o quê?** Projete `W` no subespaço de `ΔW` e compare com a projeção em direções aleatórias. `ΔW` amplifica direções que já estavam em `W` mas eram de baixa magnitude?

<details><summary>Notas sobre cada análise</summary>

1. O paper conclui que **espalhar por mais matrizes com rank menor bate concentrar numa só com rank alto**. Se o seu resultado confirmar, é um argumento forte para adaptar todas as lineares com `r` pequeno em vez de `q,v` com `r` grande.

2. Use similaridade de subespaço de Grassmann:
   ```
   φ(A, B, i, j) = ||U_A[:, :i]ᵀ U_B[:, :j]||_F² / min(i, j)
   ```
   O paper encontra que as direções de topo de `r=8` estão contidas nas de `r=64`, sugerindo que o rank alto adiciona sobretudo ruído. Esse achado é o que justifica ranks baixos funcionarem.

3. Esta é a análise mais interessante e a menos citada. O paper mostra que `ΔW` **não** aponta para as direções principais de `W` — ele amplifica direções de baixa magnitude, específicas da tarefa. É a resposta mecanicista para "o que o fine-tuning faz": não reforça o que o modelo já faz bem, e sim promove capacidades latentes.
</details>
