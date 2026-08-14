# Módulo 1 — Fundamentos de Large Language Models

> **Pergunta central:** o que exatamente um LLM calcula, e quais decisões de engenharia decorrem disso?

Este módulo é a fundação de todos os outros. Praticamente todo hiperparâmetro que você vai ajustar nos módulos 5 a 11 — learning rate, comprimento de sequência, temperatura, tamanho de batch, escolha de quantização — só faz sentido se você entender o que está sendo otimizado aqui.

## Objetivos

Ao final, você deve conseguir:

1. Explicar o objetivo de treino de um LLM em uma equação e dizer o que ele **não** otimiza.
2. Prever quantos tokens um texto em português vai custar e por que isso muda com o modelo.
3. Ler `config.json` de qualquer modelo e estimar VRAM de inferência, de treino e de KV cache.
4. Implementar geração de texto do zero (greedy, temperature, top-k, top-p) sem chamar `.generate()`.
5. Calcular perplexidade corretamente — incluindo o shift de labels que quase todo mundo erra.
6. Decidir, diante de um problema real, qual técnica de customização atacar primeiro.

---

## 1. A única coisa que um LLM faz

Um LLM autorregressivo é uma função que recebe uma sequência de tokens e devolve uma distribuição de probabilidade sobre o próximo token:

```
p_θ(x_t | x_1, ..., x_{t-1})
```

Só isso. Tudo o mais — responder perguntas, escrever código, "raciocinar", seguir instruções — é comportamento que emerge de aplicar essa função repetidamente e de ter treinado θ em dados onde esses comportamentos aparecem.

Isso tem consequências que valem mais do que parecem:

- **O modelo não tem estado.** Cada chamada reprocessa todo o contexto. "Memória" de conversa é você reenviando o histórico.
- **O modelo não sabe o que não escreveu.** Ele não planeja a resposta inteira antes de emitir o primeiro token. Cadeia de raciocínio (módulo 7) é a maneira de dar a ele espaço para "pensar em voz alta" porque o pensamento *é* a geração.
- **Fluência e veracidade são o mesmo mecanismo.** O modelo maximiza plausibilidade, não verdade. Alucinação não é um bug do sistema; é o comportamento esperado de um otimizador de verossimilhança fora da distribuição de treino.

O pipeline completo, do texto à resposta:

```
texto  →  [tokenizer]  →  ids  →  [embedding]  →  vetores
                                                     ↓
                                            [N blocos transformer]
                                                     ↓
                              logits  ←  [lm_head]  ←  vetores contextualizados
                                 ↓
                          [softmax + sampling]  →  próximo token  →  (repete)
```

As seções seguintes percorrem essa cadeia estágio por estágio.

---

## 2. Tokenização

O modelo não vê caracteres nem palavras. Vê inteiros que indexam um vocabulário fixo.

### Por que não caracteres, por que não palavras

| Unidade | Vocabulário | Problema |
|---|---|---|
| Caractere | ~100 | Sequências longas demais; custo de atenção explode (é O(n²)) |
| Palavra | 500k+ e ainda assim incompleto | Toda palavra fora do vocabulário vira `<unk>`; typos e código destroem o modelo |
| **Subpalavra (BPE)** | 32k–256k | Cobertura total via bytes, sequências curtas, palavras frequentes viram 1 token |

**BPE (Byte-Pair Encoding)** parte de bytes individuais e, iterativamente, funde o par de símbolos adjacentes mais frequente no corpus, até atingir o tamanho de vocabulário alvo. Byte-level BPE (GPT-2 em diante) garante que *qualquer* sequência de bytes é representável — não existe `<unk>`.

### O que isso causa na prática

**Português custa mais caro que inglês.** O vocabulário é aprendido do corpus de treino, majoritariamente inglês. Palavras portuguesas com acento se quebram em mais pedaços:

Medido (os números abaixo saem do Lab 1, não são estimativa):

| Texto | GPT-2 | Qwen2.5 |
|---|---|---|
| `The implementation is straightforward` | 4 | 4 |
| `A implementação é direta` | **8** | **6** |
| `The organization's conclusion was unnecessary` | 6 | 6 |
| `A conclusão da organização era desnecessária` | **14** | **10** |

O mesmo conteúdo custa **2,3× mais tokens** no GPT-2 e **1,7× mais** no Qwen2.5 quando escrito em português. A diferença entre os dois tokenizers é o corpus de treino: o Qwen2.5 viu português e aprendeu `implementação` como dois pedaços; o GPT-2 quebra em cinco.

Regra de bolso: inglês ≈ 4 caracteres/token; português ≈ 3–3,5 caracteres/token em tokenizers modernos, e ~2 em tokenizers antigos. Você paga isso três vezes: em custo de API, em janela de contexto consumida e em passos de geração.

> ⚠️ **Armadilha:** contar tokens com `len(texto.split())` ou com uma regra de "1 token ≈ 0,75 palavras" tirada de blog em inglês subestima português em 30–60%. Sempre conte com o tokenizer do modelo que você vai usar. O lab mede isso.

**Números são tokenizados de forma inconsistente.** Medido no Lab 1: o GPT-2 quebra `1234` em `['12','34']` e `3.14159` em `['3','.','14','159']` — agrupamentos arbitrários, herdados da frequência no corpus. É uma das razões pelas quais LLMs erram aritmética: não veem os dígitos alinhados por casa decimal. Modelos modernos (Qwen2.5, Llama 3) forçam **dígitos individuais**: `1234` → `['1','2','3','4']`. Custa mais tokens e melhora a aritmética.

**Espaços pertencem ao token seguinte.** Em BPE byte-level, `" gato"` (com espaço) e `"gato"` são tokens diferentes. Isso quebra prompts que terminam em espaço — o modelo fica forçado a continuar um token que ele preferiria não ter começado.

> ⚠️ **Armadilha:** nunca termine um prompt com espaço em branco. `"Resposta: "` é pior que `"Resposta:"`, porque o token `" "` isolado é raro no treino e joga o modelo para fora da distribuição.

**Tokenizer e modelo são um par indissociável.** Trocar o tokenizer de um modelo treinado é o equivalente a embaralhar o dicionário: os ids passam a apontar para embeddings errados. Se você adicionar tokens novos (módulo 5), precisa redimensionar a matriz de embeddings e treinar as linhas novas.

---

## 3. Embeddings — de inteiro a vetor

A matriz de embeddings `E` tem forma `[V, d]`, onde `V` é o tamanho do vocabulário e `d` é a dimensão do modelo. A operação é uma consulta de linha: token id `i` → `E[i]`, um vetor de `d` dimensões.

Os números importam:

| Modelo | V | d | Params de embedding | % do total |
|---|---|---|---|---|
| GPT-2 small | 50.257 | 768 | 38,6M | 31% de 124M |
| Qwen2.5-0.5B | 151.936 | 896 | 136M | 27% de 494M |
| Llama-3-8B | 128.256 | 4096 | 525M | 6,5% de 8B |

> ⚠️ **Armadilha:** `config.vocab_size` e `tokenizer.vocab_size` **não são o mesmo número**. No Qwen2.5-0.5B o config diz 151.936 e o tokenizer diz 151.643 — a matriz de embeddings é propositalmente maior, arredondada para um múltiplo conveniente (alinhamento de kernel de GPU) e com espaço sobrando para tokens especiais. As linhas excedentes nunca foram treinadas. Se você adicionar tokens novos no módulo 5, é dessas linhas que eles saem — e elas começam como ruído.

Duas leituras disso:

1. **Em modelos pequenos, o vocabulário domina.** Quase um terço do "cérebro" do GPT-2 é tabela de consulta. Isso limita o que modelos ≤1B conseguem aprender de fato — parte da capacidade está gasta só em representar o vocabulário.
2. **Vocabulários grandes barateiam a geração.** 151k tokens comprimem melhor o texto, então cada passo de decode produz mais caracteres. O preço é uma `lm_head` maior e um softmax mais caro.

**Weight tying:** muitos modelos (GPT-2, Qwen2.5-0.5B) compartilham a matriz de embeddings com a camada de saída — `lm_head.weight` é literalmente `embedding.weight` transposta. Economiza parâmetros e costuma melhorar o resultado em modelos pequenos. O flag no config é `tie_word_embeddings`.

> ⚠️ **Armadilha:** com weight tying ativo, tocar na `lm_head` durante o fine-tuning modifica os embeddings de entrada também. Se você resize o vocabulário, faça pela API `model.resize_token_embeddings()`, nunca manipulando os tensores na mão.

---

## 4. O que acontece no meio (visão de mapa)

Entre embeddings e logits está a pilha de blocos transformer. O módulo 2 abre isso em detalhe; aqui basta o mapa e o vocabulário:

Cada bloco faz duas coisas, cada uma envolvida por uma conexão residual e uma normalização:

1. **Self-attention** — cada posição olha para as posições anteriores e mistura informação delas. É onde o contexto entra. Máscara causal garante que a posição `t` só vê `≤ t`; sem ela o modelo trapacearia lendo o futuro.
2. **MLP (feed-forward)** — transformação posição a posição, tipicamente expandindo para `4d` e voltando. É onde mora a maior parte dos parâmetros e, pela evidência atual, boa parte do conhecimento factual.

Um bloco de `d=4096` tem aproximadamente `4d² (attention) + 8d² a 12d² (MLP)` parâmetros. Em Llama-3-8B: 32 blocos × ~218M ≈ 7,5B, mais 525M de embeddings.

O que sai da pilha é um vetor por posição — a mesma forma que entrou, `[seq, d]`, mas agora **contextualizado**: o vetor da posição `t` codifica a sequência inteira até `t`, não apenas o token `t`.

---

## 5. Logits — a saída bruta

A `lm_head` projeta o último vetor de volta para o espaço do vocabulário:

```
z = h_t · W_out^T      z ∈ R^V
```

`z` são os **logits**: um número real por token do vocabulário, sem escala fixa, podendo ser negativo. Não são probabilidades. Viram probabilidades pelo softmax:

### 📐 Softmax com temperatura

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

O parâmetro `T` (temperatura) controla o achatamento da distribuição:

- `T → 0`: a distribuição colapsa no argmax. Determinístico, repetitivo, seguro.
- `T = 1`: a distribuição que o modelo de fato aprendeu.
- `T > 1`: achata; tokens improváveis ganham chance. Criativo, e depois de certo ponto, incoerente.
- `T → ∞`: uniforme sobre o vocabulário. Ruído puro.

Note que `T` divide os logits **antes** da exponencial. É por isso que o efeito não é linear: dobrar `T` não dobra a "criatividade", ele reescala razões de probabilidade exponencialmente.

Detalhe de implementação que aparece em todo código de produção: softmax é calculado como `exp(z - max(z))` para evitar overflow em float. Matematicamente idêntico, numericamente estável.

---

## 6. Decoding — de distribuição a texto

Ter a distribuição não basta; é preciso escolher. As estratégias, em ordem histórica:

### Greedy
Pega o argmax a cada passo. Determinístico. Produz texto que degenera em repetição (`"o o o o"`) porque loops de alta probabilidade são atratores.

### Sampling puro com temperatura
Amostra da distribuição completa. O problema: a cauda tem dezenas de milhares de tokens, cada um com probabilidade ínfima, mas somados representam massa não desprezível. Cedo ou tarde você sorteia lixo — e um único token ruim contamina todo o resto da geração, porque ele entra no contexto.

### Top-k
Zera tudo fora dos `k` tokens mais prováveis e renormaliza. Corta a cauda. Defeito: `k` é fixo, mas a incerteza real varia muito entre posições. Depois de `"a capital da França é"` o modelo está quase certo — `k=50` inclui 49 candidatos absurdos. No meio de uma frase livre, `k=50` pode ser restritivo demais.

### Top-p (nucleus sampling)
Ordena por probabilidade decrescente, acumula, e mantém o menor conjunto cuja massa acumulada ≥ `p`. O tamanho do conjunto se **adapta** à confiança do modelo: 1 token onde ele está certo, centenas onde está incerto. É o padrão atual (`p` entre 0,9 e 0,95).

### Min-p
Mais recente: mantém tokens com `p_i ≥ min_p × p_max`. Limiar relativo ao token mais provável. Robusto a temperaturas altas, popular em modelos abertos.

### Penalidades
`repetition_penalty` divide os logits de tokens já emitidos; `frequency_penalty` e `presence_penalty` (nomenclatura OpenAI) subtraem valor proporcional à contagem. Todas são hacks que combatem sintoma, não causa — use com moderação, pois penalizar tokens já vistos degrada código e texto estruturado, onde repetição é legítima.

> ⚠️ **Armadilha:** a ordem de aplicação importa. Temperatura antes ou depois do corte top-k produz distribuições diferentes. HuggingFace aplica os *logits processors* na ordem em que estão registrados; a convenção padrão é penalidades → temperatura → top-k → top-p.

> ⚠️ **Armadilha:** `temperature=0` **não garante reprodutibilidade bit a bit**. Kernels de GPU não são determinísticos entre tamanhos de batch, e a ordem de redução em ponto flutuante muda. Dois logits empatados na quinta casa decimal podem inverter a ordem. Em produção, `seed` fixa + `temperature=0` reduz a variação, não a elimina.

**Escolha prática:**

| Tarefa | Configuração |
|---|---|
| Extração, classificação, JSON estruturado | `temperature=0` (greedy) |
| Código | `temperature` 0,1–0,3, `top_p=0,95` |
| Chat / assistente | `temperature` 0,6–0,8, `top_p=0,9` |
| Escrita criativa | `temperature` 0,9–1,1, `top_p=0,95` |

---

## 7. Como isso é treinado — cross-entropy e perplexidade

### 📐 A loss

O objetivo de pré-treino é maximizar a verossimilhança dos dados, o que equivale a minimizar a cross-entropy média por token:

```
L = -(1/T) · Σ_{t=1}^{T} log p_θ(x_t | x_{<t})
```

Para cada posição, pegue a probabilidade que o modelo atribuiu ao token que *de fato* veio, tire o log, negativo, e tire a média. Se o modelo dá probabilidade 1 ao token correto, a contribuição é 0. Se dá probabilidade ~0, a contribuição explode.

Um detalhe estrutural: essa loss é calculada **em paralelo para todas as posições** durante o treino. O modelo não gera nada no treino; ele vê a sequência inteira e prevê, em cada posição, o token seguinte, com a máscara causal impedindo trapaça. É por isso que treinar é eficiente e gerar é lento — treino é uma passada, geração são `T` passadas.

### 📐 Perplexidade

```
PPL = exp(L)
```

Interpretação: **o número efetivo de tokens igualmente prováveis entre os quais o modelo está hesitando** a cada posição.

- `PPL = 1` → certeza absoluta e correta em toda posição.
- `PPL = 50.257` → o modelo é indistinguível de um sorteio uniforme no vocabulário do GPT-2.
- GPT-2 small em texto em inglês: ~25–30. Modelos modernos de 7B: ~6–10 nos mesmos benchmarks.

> ⚠️ **Armadilha:** perplexidade **não é comparável entre tokenizers diferentes**. Um modelo com vocabulário de 151k prevê menos tokens para o mesmo texto, e cada previsão é sobre um conjunto maior. Comparar a PPL do Qwen com a do GPT-2 é comparar unidades diferentes. Para comparação justa, normalize por byte ou por caractere (*bits per byte*).

### ⚠️ O shift de labels

O erro mais comum ao implementar treino ou avaliação na mão. Os logits da posição `t` predizem o token `t+1`. Portanto:

```python
# ERRADO — compara a previsão da posição t com o token da posição t
loss = F.cross_entropy(logits.view(-1, V), input_ids.view(-1))

# CERTO — desloca em uma posição
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = input_ids[:, 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1))
```

Quando você passa `labels=input_ids` para um modelo HuggingFace, **ele faz esse shift internamente**. Se você fizer o shift *e* passar labels, terá deslocado duas vezes. O sintoma é uma loss que estaciona alta e nunca converge direito.

O valor `-100` é o `ignore_index` padrão do PyTorch: posições marcadas com `-100` não contribuem para a loss. É assim que se mascaram tokens de padding e, no módulo 5, os tokens de *prompt* quando você quer treinar só a resposta.

---

## 8. Contexto, KV cache e o custo real

### Prefill vs decode

Gerar uma resposta tem duas fases com perfis de custo opostos:

| Fase | O que faz | Gargalo | Escala |
|---|---|---|---|
| **Prefill** | Processa o prompt inteiro de uma vez | Compute (FLOPs) | O(n²) na atenção |
| **Decode** | Gera 1 token por passo | Largura de banda de memória | O(n) por token, O(n²) acumulado |

Isso explica um fenômeno que confunde muita gente: um prompt de 4.000 tokens é processado quase tão rápido quanto um de 500, mas gerar 500 tokens de resposta demora dez vezes mais que gerar 50. No decode, a GPU passa a maior parte do tempo *lendo os pesos da memória*, não calculando. Por isso quantização (módulo 11) acelera inferência mesmo sem reduzir operações: menos bytes para ler.

### 📐 KV cache

Para não recomputar a atenção sobre todo o prefixo a cada token, guardam-se as matrizes K e V de cada camada. O custo de memória:

```
bytes = 2 (K e V) × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes_por_valor
```

Llama-3-8B (32 camadas, 8 KV heads por GQA, head_dim 128, bf16):

```
2 × 32 × 8 × 128 × 2 bytes = 131.072 bytes por token ≈ 128 KB/token
```

Com 8.000 tokens de contexto: **1 GB**. Com batch 16: **16 GB** — mais que o próprio modelo quantizado.

Se o modelo não usasse GQA (32 KV heads em vez de 8), seriam 512 KB/token e 4 GB por sequência. **Essa é a razão de existir o GQA**, e é um exemplo perfeito de decisão de arquitetura motivada por custo de inferência, não por qualidade.

> 🔧 **Na prática:** ao dimensionar um servidor de inferência, o KV cache — não os pesos — costuma ser o que limita quantos usuários simultâneos cabem. É o que vLLM ataca com PagedAttention.

---

## 9. Base, instruct, reasoning — e o chat template

Três estágios do mesmo modelo, e confundi-los custa caro:

| Tipo | Treinado com | Comportamento | Uso |
|---|---|---|---|
| **Base** (`Qwen2.5-0.5B`) | Só pré-treino: prever o próximo token em texto cru | Completa texto. Se você perguntar algo, ele pode responder com *mais perguntas* — é o que apareceria num fórum | Ponto de partida para fine-tuning próprio |
| **Instruct** (`Qwen2.5-0.5B-Instruct`) | Base + SFT (módulo 5) + alinhamento (módulo 8) | Segue instruções, responde em formato de diálogo | Uso direto, ou base para especialização |
| **Reasoning** (DeepSeek-R1, o-series) | Instruct + RL sobre tarefas verificáveis (módulo 9) | Gera cadeia de raciocínio longa antes da resposta | Matemática, código, planejamento |

### Chat template

Modelos instruct esperam um formato exato de marcação de turnos. O Qwen2.5 usa:

```
<|im_start|>system
Você é um assistente útil.<|im_end|>
<|im_start|>user
Qual a capital da França?<|im_end|>
<|im_start|>assistant
```

O template vive dentro do tokenizer, como um script Jinja em `tokenizer_config.json`, e se aplica assim:

```python
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Qual a capital da França?"}],
    tokenize=False,
    add_generation_prompt=True,   # abre o turno do assistente
)
```

> ⚠️ **Armadilha — a mais cara do curso inteiro:** usar template errado, ou esquecer `add_generation_prompt=True`, degrada a qualidade de forma dramática e silenciosa. O modelo não dá erro; só responde pior. E no fine-tuning, se você treinar com um template e servir com outro, **todo o treino é desperdiçado**. Sempre imprima a string final que vai para o modelo, ao menos uma vez, antes de rodar um treino de horas.

---

## 10. Scaling laws — por que os modelos são do tamanho que são

### O orçamento de compute

FLOPs de treino, aproximação padrão:

```
C ≈ 6 · N · D
```

`N` = parâmetros, `D` = tokens de treino. O 6 vem de ~2 FLOPs por parâmetro no forward e ~4 no backward, por token.

### Kaplan (2020) e Chinchilla (2022)

Kaplan et al. mostraram que a loss cai como lei de potência em relação a parâmetros, dados e compute — previsivelmente, ao longo de várias ordens de grandeza. Isso transformou treinar LLM de aposta em projeto de engenharia orçável.

Hoffmann et al. (Chinchilla) corrigiram a alocação: dado um orçamento fixo `C`, o ponto ótimo é escalar `N` e `D` **na mesma proporção**, resultando em aproximadamente **20 tokens de treino por parâmetro**. Modelos da era GPT-3 (175B params, 300M tokens/param... ou melhor, 1,7 tokens/param) estavam gravemente subtreinados.

### Por que ninguém segue Chinchilla hoje

Llama-3-8B foi treinado com 15 trilhões de tokens: **1.875 tokens por parâmetro**, quase 100× além do compute-optimal. Por quê?

Porque Chinchilla otimiza o **custo de treino**, e quem serve um modelo para milhões de usuários paga muito mais em **inferência** ao longo da vida do modelo. Um modelo menor e sobretreinado custa mais para treinar uma vez e economiza para sempre depois. A regra virou:

> *Compute-optimal ≠ deployment-optimal.* Treine pequeno por muito mais tempo do que o Chinchilla sugere.

Isso é diretamente relevante para você: significa que modelos abertos de 1B–8B hoje são muito melhores do que a contagem de parâmetros faria supor, e são justamente os que cabem em uma GPU acessível para customização.

### 📐 A memória de treino

Full fine-tune com AdamW em mixed precision, por parâmetro:

| Item | Bytes |
|---|---|
| Pesos bf16 | 2 |
| Gradientes bf16 | 2 |
| Master weights fp32 | 4 |
| Adam momento `m` fp32 | 4 |
| Adam variância `v` fp32 | 4 |
| **Total** | **16** |

Para 7B: **112 GB só de estados**, antes de qualquer ativação. Uma A100 de 80GB não dá conta sozinha.

Guarde esse número. Ele é a justificativa inteira dos módulos 6 (LoRA/QLoRA reduzem a ~7 GB) e 11 (quantização). O curso, daqui para frente, é em grande parte a história de como espremer 112 GB em 16.

---

## 11. O mapa da customização

Fechando o módulo e abrindo o curso: diante de um problema real, em que ordem atacar? Do mais barato para o mais caro:

| Sintoma | Diagnóstico | Ferramenta | Módulo |
|---|---|---|---|
| Resposta no formato errado, tom errado | Falta de especificação | Prompt melhor, few-shot | — |
| "Não sei", ou fatos desatualizados/proprietários | Falta de **conhecimento** | RAG | — |
| Erra em tarefa que exige passos | Falta de **espaço para pensar** | Chain-of-thought, reasoning | 7 |
| Formato inconsistente mesmo com prompt bom; prompt gigante e caro | Falta de **comportamento** | SFT / LoRA | 5, 6 |
| Acerta o formato mas escolhe mal entre respostas plausíveis | Falta de **preferência** | DPO | 8 |
| Existe métrica automática de acerto (teste passa / não passa) | Sinal **verificável** disponível | GRPO / PPO | 9 |
| Funciona, mas caro ou lento demais em produção | Problema de **custo** | Distillation, quantização, MoE | 10, 11 |

> 🔧 **Na prática:** a ordem importa financeiramente. Fine-tuning é frequentemente escolhido para resolver problemas que eram de prompt ou de RAG — gastando semanas de GPU para consertar o que uma reformulação de contexto resolveria. Antes de treinar qualquer coisa, prove que o problema é de *comportamento*, não de *conhecimento*. O teste é simples: se colocar a informação no prompt resolve, você tem um problema de conhecimento (RAG). Se nem com a informação no prompt o modelo faz certo, você tem um problema de comportamento (SFT).

---

## 12. Leituras

Ordem sugerida, do essencial ao complementar:

1. **Karpathy — "Let's build GPT: from scratch"** ([vídeo](https://www.youtube.com/watch?v=kCc8FmEb1nY)). Se você assistir uma coisa só neste curso inteiro, que seja esta. Implementa tudo o que este módulo descreveu, do zero.
2. **Vaswani et al. (2017), "Attention Is All You Need"** — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). Leia agora as seções 1–3; volte a ele no módulo 2.
3. **Hoffmann et al. (2022), "Training Compute-Optimal LLMs" (Chinchilla)** — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556). Leia o abstract e a seção 3.
4. **Sennrich et al. (2016), "Neural Machine Translation of Rare Words with Subword Units"** — [arXiv:1508.07909](https://arxiv.org/abs/1508.07909). A origem do BPE em NLP.
5. **Holtzman et al. (2019), "The Curious Case of Neural Text Degeneration"** — [arXiv:1904.09751](https://arxiv.org/abs/1904.09751). De onde vem o top-p, e por que sampling puro degenera.

---

## 13. Checklist de saída

Antes de ir para o módulo 2, você deve conseguir responder sem consultar:

- [ ] Por que sequências de caracteres seriam inviáveis, em uma frase que envolva O(n²)?
- [ ] Quantos tokens custa "implementação" no tokenizer do Qwen2.5, e por que difere do GPT-2?
- [ ] Qual a diferença entre logit e probabilidade, e o que a temperatura faz com cada um?
- [ ] Por que top-p se adapta melhor que top-k?
- [ ] Qual o shift correto entre logits e labels, e o que acontece se você aplicar duas vezes?
- [ ] Por que perplexidade não é comparável entre modelos com tokenizers diferentes?
- [ ] Quantos GB de KV cache o Llama-3-8B usa com 8k de contexto, e por que GQA existe?
- [ ] Por que 16 bytes por parâmetro, e o que isso implica para treinar 7B?
- [ ] Diante de "meu assistente não sabe as políticas internas da empresa", RAG ou fine-tuning? Por quê?

Depois, abra o `lab.py`.
