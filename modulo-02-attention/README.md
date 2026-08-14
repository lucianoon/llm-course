# Módulo 2 — Transformers, Attention e QKV

> **Pergunta central:** como o contexto vira representação? O que exatamente acontece entre o embedding e os logits?

O módulo 1 tratou a pilha de blocos como caixa-preta. Aqui ela é aberta — e, no laboratório, reconstruída peça por peça até que a sua implementação bata com a do HuggingFace até a sexta casa decimal.

Este é o módulo mais denso do curso. Vale o esforço: praticamente toda decisão de arquitetura que aparece nos módulos 6, 9 e 11 (onde o LoRA se acopla, o que a quantização degrada, por que o MoE substitui só o MLP) é ininteligível sem ele.

## Objetivos

Ao final, você deve conseguir:

1. Derivar `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` e explicar cada símbolo, incluindo por que existe o `√d_k`.
2. Implementar atenção causal multi-head do zero, sem `nn.MultiheadAttention`.
3. Explicar MHA → MQA → GQA e calcular o KV cache de cada um.
4. Explicar RoPE e por que ele permite estender a janela de contexto de um modelo já treinado.
5. Contar exatamente os parâmetros de um bloco e dizer onde eles estão (spoiler: 87% no MLP).
6. Reconstruir uma camada completa a partir dos pesos e validar contra o forward oficial.

---

## 1. O problema que a atenção resolve

Antes de 2017, sequências eram processadas por RNNs: um estado oculto `h_t` atualizado token a token. Dois defeitos fatais:

- **Gargalo de informação.** Todo o passado precisa caber num vetor de tamanho fixo. Informação da posição 5 chega à posição 500 depois de 495 sobrescritas sucessivas.
- **Sequencialidade.** `h_t` depende de `h_{t-1}`. Não há como paralelizar o treino ao longo do tempo — e paralelismo é exatamente o que GPUs oferecem.

A atenção resolve os dois de uma vez: cada posição acessa **diretamente** qualquer posição anterior, e todas as posições são computadas **simultaneamente**. O custo é trocar `O(n)` de compute por `O(n²)`. Em 2017 essa troca parecia cara; com GPUs, trocar operações sequenciais por operações paralelas foi o melhor negócio da década.

---

## 2. QKV — a metáfora da recuperação

A ideia é um sistema de busca associativa, diferenciável.

Imagine um dicionário Python. Você tem uma **chave de busca** e um conjunto de **chaves armazenadas**; a busca compara sua chave com as armazenadas e devolve o **valor** correspondente. Isso é discreto — a chave bate ou não bate.

A atenção é a versão contínua: em vez de bater ou não bater, cada chave armazenada recebe um **peso** proporcional à similaridade, e o resultado é a média ponderada de todos os valores.

Cada token produz três vetores, por projeções lineares aprendidas:

| Vetor | Símbolo | Papel | Intuição |
|---|---|---|---|
| **Query** | `q = x·W_Q` | O que esta posição procura | "sou um verbo, procuro meu sujeito" |
| **Key** | `k = x·W_K` | O que esta posição oferece | "sou um substantivo, posso ser sujeito" |
| **Value** | `v = x·W_V` | O que esta posição entrega se for atendida | a informação de fato transportada |

A separação entre `k` e `v` é o ponto sutil: **o critério de ser encontrado é independente do conteúdo entregue**. Um token pode ser fácil de encontrar por uma razão e carregar informação sobre outra. Se `k` e `v` fossem o mesmo vetor, a atenção só conseguiria buscar por similaridade de conteúdo.

---

## 3. 📐 Scaled dot-product attention

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

Passo a passo, com formas — `Q, K, V` têm forma `[n, d_k]` para uma sequência de `n` tokens:

1. `S = Q Kᵀ` → `[n, n]`. `S[i,j]` é a afinidade bruta entre a query da posição `i` e a key da posição `j`.
2. `S / √d_k` → normalização de escala (adiante).
3. `+ máscara causal` → posições futuras recebem `-∞`.
4. `A = softmax(S, dim=-1)` → `[n, n]`, cada **linha** soma 1. `A[i,j]` = fração da atenção de `i` dedicada a `j`.
5. `A V` → `[n, d_k]`. Cada saída é a média dos values, ponderada pela atenção.

### Por que `√d_k`

Suponha componentes de `q` e `k` independentes, média 0 e variância 1. O produto escalar é a soma de `d_k` termos independentes, logo:

```
Var(q·k) = d_k        →        desvio-padrão = √d_k
```

Com `d_k = 128`, os logits de atenção teriam desvio-padrão ~11. Jogados num softmax, isso produz uma distribuição praticamente one-hot: um peso ≈ 1, todos os outros ≈ 0.

O problema não é apenas ser "duro demais" — é o **gradiente**. A derivada do softmax saturado é essencialmente zero, então nenhum sinal de treino volta pelas projeções `W_Q` e `W_K`. Dividir por `√d_k` devolve a variância a 1 e mantém o softmax numa região onde ele aprende. O Lab 2 mede exatamente isso.

> 📐 Note que isso é o mesmo fenômeno da temperatura do módulo 1: `√d_k` é uma temperatura fixa, escolhida para manter a entropia do softmax numa faixa útil.

### A máscara causal

Sem máscara, a posição `i` veria o token `i+1` — e prever o próximo token seria trivial, com loss indo a zero e generalização nula. A máscara é uma matriz triangular superior de `-∞` somada antes do softmax (`-∞` vira exatamente 0 depois da exponencial, sem hack numérico).

> ⚠️ **Armadilha:** máscara causal e máscara de padding são coisas diferentes e ambas necessárias. A causal impede ver o futuro; a de padding impede atender a tokens de preenchimento. Esquecer a de padding contamina as representações com lixo — e o sintoma é sutil: o modelo treina, mas fica pior conforme o batch fica mais heterogêneo em comprimento.

> ⚠️ **Armadilha:** padding à **esquerda** vs à **direita**. Para *treino* use padding à direita; para *geração em batch* use padding à esquerda, senão a última posição — de onde sai o próximo token — cai em cima de um token de padding. `tokenizer.padding_side = "left"` antes de gerar em batch. Este é um dos bugs mais comuns em código de inferência caseiro.

---

## 4. Multi-head attention

Uma única atenção calcula **uma** média ponderada por posição. Mas uma posição precisa de várias coisas ao mesmo tempo: o sujeito do verbo, o antecedente do pronome, o delimitador do bloco de código.

A solução: rodar `h` atenções em paralelo, cada uma em um subespaço de dimensão `d_head = d / h`, e concatenar:

```
MultiHead(X) = Concat(head_1, ..., head_h) · W_O
onde head_i = Attention(X·W_Q^i, X·W_K^i, X·W_V^i)
```

O custo total é o mesmo de uma atenção de dimensão `d` — as cabeças **dividem** a dimensão, não a multiplicam. Você ganha especialização de graça.

Na implementação, não existem `h` matrizes separadas: há uma projeção única `[d, d]` e um `reshape` para `[batch, seq, h, d_head]` seguido de `transpose` para `[batch, h, seq, d_head]`. As cabeças são uma **visão** do tensor, não objetos distintos. O Lab 4 faz esse malabarismo de eixos explicitamente — é onde mais se erra.

> 🔧 **Na prática:** cabeças se especializam de fato, e o fenômeno é estudado (*induction heads*, cabeças de cópia, cabeças posicionais). Mas cuidado com a interpretação: mapas de atenção bonitos são péssima evidência causal. Uma cabeça pode atender fortemente a um token sem que isso influencie a saída, porque o `value` daquele token pode ser quase nulo. Atenção mostra *para onde se olha*, não *o que se usa*.

---

## 5. MHA → MQA → GQA

O módulo 1 mostrou que o KV cache, não os pesos, costuma limitar quantos usuários cabem numa GPU. A evolução da arquitetura foi guiada por isso:

| Variante | KV heads | KV cache | Qualidade |
|---|---|---|---|
| **MHA** (2017) | `h` (uma por query head) | Baseline | Baseline |
| **MQA** (2019) | **1**, compartilhada por todas | `h×` menor | Perda perceptível |
| **GQA** (2023) | `g` grupos, `h/g` queries cada | `h/g×` menor | Perda desprezível |

GQA é o meio-termo que venceu: praticamente todo modelo moderno usa. Llama-3-8B tem 32 query heads e 8 KV heads (fator 4). Qwen2.5-0.5B tem **14 query heads e 2 KV heads** (fator 7).

Na implementação, o `repeat_kv` replica cada KV head para as queries do seu grupo. É replicação lógica, não memória extra no cache — o cache guarda apenas os `g` heads reais. Essa distinção é a fonte da economia.

**A conta que importa** (Qwen2.5-0.5B, 24 camadas, head_dim 64, bf16):

```
GQA (2 kv heads):  2 × 24 × 2 × 64 × 2 bytes  =  12 KB/token
MHA (14 kv heads): 2 × 24 × 14 × 64 × 2 bytes =  84 KB/token
```

Sete vezes menos. Em 32k de contexto: 0,4 GB contra 2,8 GB.

---

## 6. Posição — o transformer é cego a ordem

Ponto crucial e frequentemente esquecido: **a atenção é permutacionalmente equivariante**. Embaralhe os tokens de entrada e as saídas saem embaralhadas do mesmo jeito, com os mesmos valores. Sem informação posicional, `"o cão mordeu o homem"` e `"o homem mordeu o cão"` são idênticos para o modelo.

A evolução das soluções:

| Método | Como | Quem usa | Limitação |
|---|---|---|---|
| **Absoluta aprendida** | Vetor treinado por posição, somado ao embedding | GPT-2, BERT | Não extrapola: posição 2048 nunca foi treinada, é ruído |
| **Senoidal** | Senos/cossenos de frequências variadas | Transformer original | Extrapola mal na prática |
| **RoPE** | Rotaciona `q` e `k` por ângulo proporcional à posição | Llama, Qwen, Mistral, quase tudo hoje | — |
| **ALiBi** | Penalidade linear na distância, direto nos logits | BLOOM, MPT | Menos expressivo |

### 📐 RoPE (Rotary Position Embedding)

A ideia: em vez de **somar** posição ao vetor, **rotacionar** o vetor por um ângulo proporcional à posição.

Tome os componentes de `q` aos pares, `(q_0,q_1), (q_2,q_3), ...`, e trate cada par como um número complexo / um ponto no plano. Para a posição `m`, rotacione o par `i` pelo ângulo `m·θ_i`:

```
θ_i = base^(-2i/d)          base tipicamente 10.000, ou 1.000.000 em modelos de contexto longo
```

Pares no início do vetor rotacionam rápido (alta frequência, distinguem posições próximas); pares no fim rotacionam devagar (baixa frequência, distinguem regiões distantes). É um relógio de múltiplos ponteiros.

**A propriedade que faz tudo funcionar:**

```
⟨R_m q, R_n k⟩ = ⟨R_{m-n} q, k⟩
```

O produto interno de dois vetores rotacionados depende **apenas da diferença de posições** `m − n`. Ou seja: você injeta posição absoluta em cada vetor, e a atenção enxerga automaticamente posição **relativa**. Nada na fórmula da atenção precisa mudar. O Lab 6 verifica essa identidade numericamente.

**Por que isso importa para customização:** como a posição está nos ângulos e não nos pesos, dá para **esticar a janela de contexto de um modelo já treinado** mexendo só na base `θ`:

- **Position interpolation** — comprime as posições para dentro da faixa vista no treino.
- **NTK-aware scaling / YaRN** — aumenta a `base` de forma não uniforme, preservando as altas frequências.

É assim que modelos de 4k viram modelos de 32k com um fine-tuning curto. Impossível com posições absolutas aprendidas.

> ⚠️ **Armadilha:** RoPE é aplicado a `q` e `k`, **nunca a `v`**. A posição entra no *critério de busca*, não no *conteúdo transportado*. Aplicar em `v` é um bug clássico de reimplementação — e silencioso, porque o modelo ainda treina, só que pior.

---

## 7. O bloco completo

```
        x ──────────────────────────┐
        │                           │
    [Norm]                          │   ← pre-norm
        │                           │
   [Attention]                      │
        │                           │
        ├───────────────────────────┘   ← residual
        │
        ├───────────────────────────┐
    [Norm]                          │
        │                           │
      [MLP]                         │
        │                           │
        ├───────────────────────────┘   ← residual
        │
        ▼
```

### Conexões residuais

`x + f(x)`, e não `f(x)`. Duas razões:

1. **Gradiente.** A derivada de `x + f(x)` em relação a `x` é `1 + f'(x)`. Aquele `1` é um caminho direto pelo qual o gradiente flui sem atenuação, por 80 camadas se preciso. Sem residual, transformers profundos simplesmente não treinam.
2. **Residual stream.** A interpretação moderna: existe um "barramento" de dimensão `d` que atravessa o modelo inteiro, e cada bloco **lê** dele, computa algo e **soma de volta**. Os blocos não transformam a representação — eles a editam incrementalmente. Essa visão é a base de quase toda a interpretabilidade mecanicista.

### Pre-norm vs post-norm

O transformer original normalizava **depois** do residual (`Norm(x + f(x))`). Isso exigia warmup cuidadoso de learning rate e ficava instável com profundidade. Todos os modelos modernos usam **pre-norm** (`x + f(Norm(x))`), que mantém o caminho residual como identidade pura e treina de forma muito mais estável.

### LayerNorm vs RMSNorm

```
LayerNorm(x) = γ · (x − μ) / σ + β          (subtrai média, tem bias)
RMSNorm(x)   = γ · x / √(mean(x²) + ε)      (só reescala)
```

RMSNorm descarta a centralização e o bias. Empiricamente, a média não fazia falta, e o resultado é ~10–15% mais rápido. Padrão em Llama, Qwen, Mistral.

> ⚠️ **Armadilha:** RMSNorm é calculado em **float32** mesmo em modelos bf16. `mean(x²)` sobre 4096 dimensões estoura a precisão de bf16. Toda implementação séria faz upcast para fp32, normaliza, e volta. Ignorar isso produz NaN em treino longo.

### O MLP e o SwiGLU

O MLP clássico é `W_2 · GELU(W_1 x)`, expandindo para `d_ff = 4d` e voltando. Os modelos atuais usam **SwiGLU**, com três matrizes e um portão multiplicativo:

```
MLP(x) = W_down · ( SiLU(W_gate · x) ⊙ (W_up · x) )
```

`SiLU(z) = z·σ(z)`. O caminho `gate` decide *quanto* de cada canal do caminho `up` passa — multiplicação, não soma. Custa uma matriz a mais, e por isso `d_ff` costuma ser reduzido (frequentemente `⁸⁄₃·d`) para manter a contagem de parâmetros.

**Por que o MLP importa tanto:** é onde está a maior parte dos parâmetros, e a evidência atual sugere que é onde mora a maior parte do conhecimento factual (linha de pesquisa de *knowledge editing*, ROME/MEMIT). No módulo 11, quando o MoE substituir **apenas o MLP** por especialistas, será por isso.

---

## 8. Onde estão os parâmetros

Qwen2.5-0.5B: `d=896`, `h=14`, `head_dim=64`, `kv_heads=2`, `d_ff=4864`, 24 camadas.

Por bloco:

| Componente | Forma | Parâmetros |
|---|---|---|
| `q_proj` (+bias) | 896 × 896 | 803.712 |
| `k_proj` (+bias) | 896 × 128 | 114.816 |
| `v_proj` (+bias) | 896 × 128 | 114.816 |
| `o_proj` | 896 × 896 | 802.816 |
| **subtotal atenção** | | **1.836.160  (12,3%)** |
| `gate_proj` | 896 × 4864 | 4.358.144 |
| `up_proj` | 896 × 4864 | 4.358.144 |
| `down_proj` | 4864 × 896 | 4.358.144 |
| **subtotal MLP** | | **13.074.432  (87,7%)** |
| 2 × RMSNorm | 896 × 2 | 1.792 |
| **total por bloco** | | **14.912.384** |

24 blocos = 357.897.216, mais 136.134.656 de embeddings = **494.032.768**. Fecha com o nome do modelo. (Todos os números desta tabela saem do Lab 9 — confira você mesmo.)

Três conclusões que valem para o curso inteiro:

1. **O MLP domina** (87,7%). LoRA aplicado só às projeções de atenção toca **12,3%** do modelo — parte da razão de LoRA em `q,v` apenas ser tão econômico, e também de às vezes não bastar (módulo 6).
2. **GQA encolhe `k` e `v` drasticamente**: 128 colunas contra 896 de `q`. Sete vezes menos parâmetros nessas projeções.
3. **As normas são desprezíveis** em parâmetros (1.792) e absolutamente críticas em estabilidade. Nunca as quantize (módulo 11).

---

## 9. Complexidade e FlashAttention

| Recurso | Custo |
|---|---|
| Compute da atenção | `O(n² · d)` |
| Memória da matriz de atenção (naive) | `O(n²)` |
| Compute do MLP | `O(n · d²)` |

Para `n=2048, d=896`, o MLP ainda domina o compute. O cruzamento acontece quando `n ≈ d`. Em contextos de 32k+, a atenção passa a dominar tudo.

Mas o gargalo real não é o compute — é a **memória**. Materializar `A` de forma `[batch, heads, n, n]` com `n=8192` e 14 cabeças custa, em bf16:

```
14 × 8.192² × 2 bytes ≈ 1,9 GB     por camada, por sequência do batch
```

E o forward precisa de pelo menos duas dessas matrizes vivas ao mesmo tempo (os scores e as probabilidades pós-softmax), sem contar o que o backward guarda. Multiplique por 24 camadas e por um batch qualquer: insustentável.

**FlashAttention** (Dao et al., 2022) resolve sem aproximar nada: processa a atenção em blocos que cabem na SRAM da GPU, usando softmax online (estatísticas acumuladas incrementalmente), e **nunca materializa a matriz `n×n`** na memória principal. Resultado: memória `O(n)`, e 2–4× mais rápido — não por fazer menos contas, mas por fazer menos idas à HBM. O resultado é numericamente idêntico à atenção padrão (a menos de arredondamento).

> 🔧 **Na prática:** no PyTorch moderno isso está em `F.scaled_dot_product_attention`, que seleciona o melhor kernel disponível automaticamente. Use-o em vez de escrever a sua atenção — a implementação manual do lab é para **entender**, não para produção.

---

## 10. Attention sinks — um detalhe que vira problema

Ao inspecionar mapas de atenção reais (Lab 7), você vai ver uma faixa vertical brilhante na **primeira coluna**: quase toda cabeça, em quase toda posição, dedica atenção substancial ao primeiro token — mesmo quando ele é irrelevante.

A explicação: o softmax é obrigado a somar 1. Se uma cabeça não tem nada a buscar naquela posição, ela precisa despejar a massa em algum lugar. O primeiro token, visível a todas as posições, vira o ralo — o *attention sink* (Xiao et al., 2023).

A escala do fenômeno surpreende. Medido no Qwen2.5-0.5B (Lab 7), média sobre todas as cabeças e posições:

| Camada | Massa de atenção no token 0 | Massa na diagonal (auto-atenção) |
|---|---|---|
| 0 | 10,5% | 29,3% |
| 8 | 40,0% | 18,3% |
| **16** | **92,1%** | 8,0% |
| 20 | 88,2% | 9,3% |

Nas camadas intermediárias, **mais de 90% da atenção vai para um único token** que quase sempre é irrelevante ao conteúdo. O modelo não está "prestando atenção" na maior parte do tempo — está descansando.

Duas consequências práticas:

- **Streaming.** Ao descartar tokens antigos para caber na janela, se você jogar fora os primeiros, a qualidade despenca. StreamingLLM preserva os 4 primeiros tokens sempre.
- **Quantização.** Os ativações do sink têm magnitude muito maior que o resto — são *outliers*. É por isso que quantização ingênua para 8 bits degrada tanto, e por que métodos como LLM.int8() tratam outliers separadamente (módulo 11).

---

## 11. Leituras

1. **Vaswani et al. (2017), "Attention Is All You Need"** — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). Agora leia inteiro. Note quanta coisa envelheceu: post-norm, senoidal, encoder-decoder.
2. **Alammar, "The Illustrated Transformer"** — [jalammar.github.io](https://jalammar.github.io/illustrated-transformer/). As melhores figuras que existem sobre o assunto.
3. **Su et al. (2021), "RoFormer" (RoPE)** — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864). Seção 3.4 é a derivação da propriedade relativa.
4. **Ainslie et al. (2023), "GQA"** — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245). Curto e direto.
5. **Dao et al. (2022), "FlashAttention"** — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135). Leia a seção 1 e a figura 1; o resto é engenharia de kernel.
6. **Elhage et al. (2021), "A Mathematical Framework for Transformer Circuits"** — [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html). Onde a ideia de residual stream é formalizada. Difícil e recompensador.

---

## 12. Checklist de saída

- [ ] Por que `k` e `v` são vetores separados? O que se perderia se fossem o mesmo?
- [ ] Por que dividir por `√d_k` — e por que o problema é de *gradiente*, não de "distribuição feia"?
- [ ] Qual a forma da matriz de atenção, e o que uma **linha** dela representa?
- [ ] Multi-head multiplica ou divide a dimensão por cabeça?
- [ ] Quantas vezes o GQA do Qwen2.5-0.5B reduz o KV cache, e por quê?
- [ ] Por que o transformer é cego a ordem, e o que RoPE rotaciona exatamente?
- [ ] Por que RoPE não se aplica a `v`?
- [ ] Que fração dos parâmetros de um bloco está no MLP?
- [ ] Por que RMSNorm precisa de float32?
- [ ] FlashAttention muda o resultado da atenção? Por que é mais rápido, então?
- [ ] O que é um attention sink e por que ele complica quantização?

Depois, abra o `lab.py`. O Lab 8 reconstrói o modelo inteiro a partir dos pesos e compara com o forward oficial — em float32 o resultado costuma ser **bit-exact** (erro `0.0`), porque é literalmente o mesmo cálculo. Em seguida ele repete a comparação com quatro bugs propositais (RoPE em `v`, sem máscara, sem `√d_k`, sem residual) para provar que a verificação detecta problemas reais. Nenhum dos quatro lança exceção — todos apenas tornam o modelo pior em silêncio, que é exatamente como esses erros aparecem na vida real.
