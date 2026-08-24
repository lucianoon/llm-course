# Módulo 5 — Supervised Fine-Tuning e Instruction Tuning

> **Pergunta central:** como se ensina um comportamento a um modelo que já sabe a língua?

Aqui o curso vira. Nos módulos 1–4 você construiu entendimento; a partir de agora você **modifica modelos reais**, no seu M4, com resultados que dá para ver e medir.

E a boa notícia do módulo 3 se concretiza: o pré-treino custou US$ 2,4 milhões e consumiu 98% do compute. O que você vai fazer neste módulo custa **zero** e leva minutos.

## Objetivos

1. Explicar por que SFT muda comportamento e não conhecimento — e prever quando isso vai frustrar você.
2. Preparar um dataset de instrução no formato correto, com masking e EOS.
3. Treinar LoRA com `mlx_lm.lora` e ler o log de treino.
4. Medir o resultado com mais rigor do que "olhei e pareceu melhor".
5. Detectar e quantificar *catastrophic forgetting*.
6. Escolher hiperparâmetros com justificativa, não por cópia.

---

## 1. O que o SFT faz, mecanicamente

Nada de novo: **é a mesma cross-entropy do módulo 3**.

```
L = −(1/N) · Σ log p_θ(x_t | x_<t)
```

A diferença está inteiramente nos dados e no masking:

| | Pré-treino | SFT |
|---|---|---|
| Dados | Texto cru da web | Pares instrução → resposta |
| Formato | Fluxo contínuo com packing | Conversas com chat template |
| Loss em | Todos os tokens | Só nos tokens da resposta |
| Tokens | 1–15 trilhões | 1–100 milhões |
| Learning rate | 3e-4 (pico) | 1e-5 (full) ou 1e-4 (LoRA) |
| Épocas | ~1 | 1–3 |

Não há algoritmo novo. **SFT é pré-treino continuado sobre um corpus muito específico, com learning rate muito menor.** Se você entendeu o módulo 3, já entendeu o mecanismo; o que resta é engenharia de dados e de hiperparâmetros.

### Por que funciona com tão poucos dados

O modelo base já sabe responder perguntas — ele viu milhões de pares pergunta/resposta na web. O que ele não sabe é **que é isso que deve fazer agora**. Um prompt de usuário, para um modelo base, é apenas texto a ser continuado; continuar com mais perguntas é tão plausível quanto responder.

O SFT não instala a capacidade. Ele **desloca a massa de probabilidade** para a região onde a capacidade já vive. Mil exemplos consistentes bastam para isso — é a hipótese do alinhamento superficial, e é o que o LIMA demonstrou (módulo 4).

> ⚠️ O corolário, que custa caro a quem ignora: se a capacidade **não** está no modelo base, o SFT não a cria. Ele ensina o modelo a *soar* como quem tem a capacidade. Um modelo que não sabe direito a legislação tributária, após SFT em respostas tributárias, passa a inventar artigos de lei com muito mais confiança e formatação profissional.

---

## 2. O dado de SFT, na prática

### Formato

O MLX aceita quatro formatos de `.jsonl`. Use **`messages`**:

```jsonl
{"messages": [{"role": "user", "content": "Qual a capital da França?"}, {"role": "assistant", "content": "Paris."}]}
```

Ele aplica o chat template do modelo automaticamente, o que resolve de graça duas das três armadilhas do módulo 4 — o template certo e o EOS no lugar. A terceira, o masking, é a flag `--mask-prompt`.

### Multi-turn

Conversas com vários turnos são um único exemplo, e a loss é calculada em **todos** os turnos do assistente:

```jsonl
{"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Cada resposta do assistente é condicionada a todo o histórico anterior — que é exatamente a situação de inferência. Não quebre a conversa em pares independentes: isso ensina o modelo a ignorar o contexto anterior.

### Quantos exemplos

Do módulo 4, agora com o contexto de execução:

| Objetivo | Exemplos | Iterações típicas no MLX |
|---|---|---|
| Formato de saída fixo, tom | 200–1.000 | 200–600 |
| Tarefa específica | 1.000–10.000 | 600–2.000 |
| Domínio amplo | 10.000+ | 2.000+ |

---

## 3. Hiperparâmetros — e o que cada um faz

Os defaults do `mlx_lm.lora` (de `examples/lora_config.yaml`):

```yaml
num_layers: 16              # quantas camadas recebem adaptador (do fim para o início)
batch_size: 4
iters: 1000
learning_rate: 1e-5
max_seq_length: 2048
grad_accumulation_steps: 1
lora_parameters:
  keys: ["self_attn.q_proj", "self_attn.v_proj"]
  rank: 8
  scale: 20.0
  dropout: 0.0
```

### Learning rate

**O default de `1e-5` é conservador para LoRA.** Ele é o valor típico de *full fine-tune*; adaptadores LoRA, por terem poucos parâmetros e partirem do zero, costumam usar **1e-4 a 3e-4**.

A razão: em full fine-tune você está ajustando pesos que já carregam todo o conhecimento — passos grandes destroem. Em LoRA, a matriz `B` é inicializada em zero e precisa **aprender do nada**; com 1e-5 ela mal sai do lugar em 600 iterações.

Comece em **1e-4**. Se a loss oscilar, reduza; se não descer, aumente.

### Épocas

```
épocas = (iters × batch_size) / número_de_exemplos
```

Fique entre **1 e 3**. Acima disso o modelo memoriza — e como o dataset de SFT é pequeno, a memorização chega rápido. Calcule esse número explicitamente antes de treinar; é o erro mais fácil de cometer e o mais fácil de evitar.

### `num_layers`

Quantas camadas recebem adaptador, contadas **do fim para o início**. Com `16` num modelo de 28 camadas, as 12 primeiras ficam intocadas.

A intuição (e há evidência para ela): camadas iniciais codificam sintaxe e características de baixo nível, que você não quer mudar; camadas finais codificam comportamento e estilo, que é o alvo do SFT. Adaptar menos camadas economiza memória — crítico nos seus 16 GB.

### `rank` e `scale`

Assunto do módulo 6, mas o essencial: `rank` é a capacidade do adaptador (8 basta para formato e estilo; 32–64 para tarefas complexas).

> ⚠️ **Diferença de convenção importante.** No MLX, `scale` multiplica diretamente a saída do adaptador. No PEFT (CUDA), o que se configura é `lora_alpha`, e o multiplicador efetivo é `lora_alpha / r`. O default do MLX, `scale: 20.0`, corresponde a um `lora_alpha` de **160** com `rank=8` — muito mais agressivo que o típico `alpha=16, r=8` (multiplicador 2) do PEFT. Ao traduzir hiperparâmetros de um tutorial CUDA, converta: `scale_mlx = lora_alpha / r`.

### `max_seq_length`

Sequências mais longas custam memória quadrática na atenção. Meça o p95 do seu dataset e use esse valor, não 2048 por inércia. Nos seus 16 GB, isso pode ser a diferença entre treinar e receber um erro de memória.

---

## 4. Catastrophic forgetting

Treinar num domínio estreito degrada o desempenho em tudo o mais. O modelo não tem noção de "aprender além" — ele desloca pesos, e o que estava codificado neles se move junto.

**Como medir** (e o Lab 6 faz isso): calcule a perplexidade do modelo em texto **fora** do domínio de fine-tuning, antes e depois. Se subiu muito, você esqueceu algo.

**Como mitigar**, em ordem de eficácia:

1. **Use LoRA em vez de full fine-tune.** Os pesos originais ficam congelados; o adaptador é uma perturbação de baixo posto. É a mitigação mais forte, e você a tem de graça.
2. **Menos épocas, learning rate menor.** Quase todo esquecimento vem de treinar demais.
3. **Misture dados gerais** — 5–20% de exemplos de instrução genérica no seu dataset específico.
4. **Adapte menos camadas** (`num_layers` menor).

> 🔧 **Na prática:** com LoRA e 1–2 épocas, o esquecimento costuma ser pequeno o bastante para não importar. Ele vira problema real em full fine-tune, ou em LoRA muito agressivo (rank alto, muitas épocas, todas as camadas).

---

## 5. Avaliação — a parte que quase todo mundo faz mal

"Rodei três prompts e pareceu melhor" não é avaliação. Quatro níveis, do mais barato ao mais confiável:

### Nível 1 — Loss de validação/teste
Grátis, automática, e **fraca**. Mede se o modelo prevê bem as respostas do seu conjunto de teste. Não mede se as respostas são boas, úteis ou corretas. Serve para detectar overfitting e comparar checkpoints do mesmo treino — nada além disso.

### Nível 2 — Métricas de formato
Se o objetivo é formato, meça formato: o JSON é válido? A resposta tem as três seções? O comprimento está na faixa? São métricas objetivas, baratas e frequentemente **as que de fato importam** — e o Lab 5 constrói uma.

### Nível 3 — LLM-as-judge
Um modelo mais forte compara respostas do modelo A e do B e escolhe. Produz *win rate*. Correlaciona razoavelmente com julgamento humano, e tem vieses conhecidos e mensuráveis:

- **Viés de posição** — a primeira opção tende a ganhar. Mitigação: avalie nas duas ordens e some.
- **Viés de comprimento** — respostas longas parecem melhores. Mitigação: controle o comprimento ou instrua o juiz.
- **Viés de auto-preferência** — modelos preferem o próprio estilo.

### Nível 4 — Avaliação humana
Cara, lenta, e o padrão-ouro. Para uma aplicação real, 100 comparações humanas bem feitas valem mais que qualquer benchmark.

> ⚠️ **Sempre compare contra a baseline certa.** O comparativo não é "meu modelo fine-tuned vs. o modelo base sem prompt nenhum" — isso é fácil demais e infla o resultado. É "meu modelo fine-tuned vs. **o modelo base com o melhor prompt que eu consegui escrever**". Se um prompt melhor teria resolvido, seu fine-tuning não provou nada.

---

## 6. Quando o SFT falha

| Sintoma | Diagnóstico | Ação |
|---|---|---|
| Responde certo e não para de falar | EOS ausente nos dados | Use formato `messages`; verifique o template |
| Ignora a instrução, continua o texto | Template errado, ou faltou `add_generation_prompt` | Imprima a string final antes de treinar |
| Formato ótimo, conteúdo inventado | Problema de conhecimento, não de comportamento | RAG (módulo 4, seção A5 dos exercícios) |
| Loss desce, saída piora | Overfitting; ou a loss não mede o que importa | Menos épocas; avalie no nível 2 ou 3 |
| Loss não desce | LR baixo demais, ou `num_layers` pequeno demais | LR para 1e-4; suba `num_layers` |
| Estilo bom, mas ficou "burro" em geral | Catastrophic forgetting | Menos épocas, mistura de dados gerais |
| Repete os exemplos de treino literalmente | Épocas demais em poucos dados | Reduza épocas; aumente ou diversifique o dataset |

---

## 7. O pipeline completo

```
1. Definir o objetivo         → o que exatamente deve mudar no comportamento?
2. Definir a avaliação         → ANTES de coletar dados. Como você saberá que funcionou?
3. Coletar/gerar dados         → módulo 4
4. Limpar e curar              → módulo 4 (dedup, filtros, seleção)
5. Separar treino/val/teste    → antes de qualquer outra coisa
6. Formatar (messages, JSONL)  → uma pasta com train/valid/test
7. Baseline                    → medir o modelo base COM bom prompt
8. Treinar                     → mlx_lm.lora
9. Avaliar                     → mesma métrica da baseline
10. Iterar                     → hiperparâmetros, depois dados
```

**Os passos 2 e 7 são os que se pulam, e são os que decidem se o projeto tem sentido.** Sem uma métrica definida antes, você vai racionalizar qualquer resultado; sem baseline com bom prompt, você não sabe se o fine-tuning contribuiu com alguma coisa.

---

## 8. Leituras

1. **Ouyang et al. (2022), "InstructGPT"** — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155). A seção 3 descreve o SFT que originou o ChatGPT. Note quão pequeno é o dataset: ~13k exemplos.
2. **Taori et al. (2023), "Stanford Alpaca"** — [github](https://github.com/tatsu-lab/stanford_alpaca). O experimento que popularizou o SFT barato.
3. **Zhou et al. (2023), "LIMA"** — [arXiv:2305.11206](https://arxiv.org/abs/2305.11206). Releia à luz deste módulo.
4. **`mlx_lm/LORA.md`** — [github](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md). A documentação da ferramenta que você vai usar. Leia inteira, é curta.
5. **Zheng et al. (2023), "Judging LLM-as-a-Judge (MT-Bench)"** — [arXiv:2306.05685](https://arxiv.org/abs/2306.05685). Onde os vieses do juiz são medidos.

---

## 9. Checklist de saída

- [ ] Qual a diferença algorítmica entre pré-treino e SFT? (Cuidado: é uma pergunta com pegadinha.)
- [ ] Por que 1.000 exemplos bastam para SFT, mas não bastariam para ensinar um domínio novo?
- [ ] Por que o LR de LoRA é ~10× o de full fine-tune?
- [ ] Como se calcula o número de épocas a partir de `iters`, `batch_size` e do tamanho do dataset?
- [ ] O que `scale: 20.0` no MLX corresponde no PEFT, com `rank=8`?
- [ ] Como se mede catastrophic forgetting, concretamente?
- [ ] Por que a loss de validação é uma métrica fraca de qualidade de SFT?
- [ ] Cite dois vieses do LLM-as-judge e como mitigá-los.
- [ ] Qual é a baseline correta para comparar seu modelo fine-tuned?
- [ ] Seu modelo responde certo e continua falando sozinho. Causa e correção?

Depois, `preparar_dados.py` e `lab.py`. Em GPU NVIDIA, use `lab_cuda.py` para comparar full fine-tuning, LoRA e QLoRA com revisão imutável do modelo e registro local do experimento.
