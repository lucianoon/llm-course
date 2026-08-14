# Módulo 18 — Fronteira de arquiteturas

> **Pergunta central:** o que vem depois do transformer — e por que, sete anos depois, ele ainda está no centro?

O curso inteiro tratou o transformer como dado. Este módulo, que fecha a Fase 2, olha para os desafiantes: os State Space Models (Mamba), a atenção linear, a atenção latente (MLA), o mundo multimodal. E chega a uma conclusão honesta que a mídia técnica costuma exagerar: o transformer não foi destronado — foi **complementado**. Entender por quê é entender a força e o limite de cada arquitetura.

O fio é o mesmo do curso: **a arquitetura evolui por pressão de custo.** Cada desafiante ataca um custo específico do transformer — e paga um preço específico em troca. O lab mede os dois lados de cada troca.

## Objetivos

1. Nomear o gargalo O(L²) que motiva toda a fronteira, e medi-lo.
2. Explicar SSMs/Mamba: estado fixo, seletividade, parallel scan.
3. Entender a reordenação da atenção linear e o que ela sacrifica.
4. Explicar o MLA e por que é a escolha "conservadora".
5. Medir o trade-off central: recall preciso vs eficiência.
6. Entender por que os modelos de fronteira são híbridos, e como multimodalidade se encaixa.

---

## 1. O pecado original: O(L²)

Do módulo 2: a atenção compara cada token com todos os anteriores. A matriz `QKᵀ` é `L×L` — **dobrar o contexto quadruplica compute e memória.** Medido no Lab 1: de L=1024 para 2048, o tempo salta ~5×; a matriz de atenção de uma cabeça em L=128k pesa gigabytes.

Some a isso o KV cache (módulo 1), que cresce **linearmente** com o contexto e domina a memória de inferência em conversas longas. Esses dois custos — O(L²) de compute, O(L) de cache — são o que toda arquitetura deste módulo tenta atacar. A pergunta que organiza a fronteira: *dá para processar sequências longas sem pagar o quadrado?*

---

## 2. SSMs e Mamba — estado de tamanho fixo

A ideia dos **State Space Models**: em vez de guardar todo o passado (KV cache), resumi-lo num **estado de tamanho fixo** `h`, atualizado token a token:

```
h_t = A·h_{t-1} + B·x_t      o estado resume tudo que veio antes
y_t = C·h_t                   a saída lê o estado
```

Isto é uma RNN (módulo 2) — e as RNNs foram abandonadas por dois defeitos: não paralelizavam no treino, e sofriam com dependências longas. O Mamba (Gu & Dao, 2023) resolve os dois:

- **Seletividade:** os parâmetros A, B, C **dependem da entrada** — o modelo escolhe, por token, o que lembrar e o que esquecer. É o que faltava às RNNs para competir com a atenção em qualidade.
- **Parallel scan:** um algoritmo de prefix-sum treina a recorrência em paralelo na GPU, matando o defeito fatal das RNNs.

A propriedade que muda tudo, medida no Lab 2:

| Contexto | KV cache (transformer) | Estado (SSM) |
|---|---|---|
| 8.192 | 1,0 MB | 0,002 MB |
| 131.072 | 16,8 MB | 0,002 MB |
| **1.000.000** | **128 MB** | **0,002 MB** |

**O estado do SSM é CONSTANTE no comprimento.** Em contextos de milhões de tokens, é a diferença entre caber e não caber — e é por isso que SSMs empolgaram a área. O custo dessa mágica está na seção 5.

---

## 3. 📐 Atenção linear — a reordenação

Um ataque diferente, que mantém a forma de atenção mas remove o softmax. A atenção é `softmax(QKᵀ)V`; o O(L²) vem de materializar `QKᵀ` (matriz `L×L`) **antes** de multiplicar por V. Sem o softmax, a **associatividade** da multiplicação de matrizes permite reordenar:

```
(φ(Q)·φ(K)ᵀ)·V   =   φ(Q)·(φ(K)ᵀ·V)
  ↑ L×L, O(L²)          ↑ d×d, O(L)!
```

`φ(K)ᵀ·V` é uma matriz `d×d` — **independente de L**. Você processa a sequência inteira acumulando essa matriz pequena, em tempo linear. Medido no Lab 3:

| L | Quadrática | Linear | Speedup |
|---|---|---|---|
| 512 | 1,1 ms | 0,8 ms | 1,4× |
| 2.048 | 12,5 ms | 0,8 ms | 15× |
| 8.192 | 283 ms | 4,3 ms | **66×** |

O speedup explode com o contexto — exatamente o que se quer. E o preço, mais uma vez, na seção 5.

---

## 4. MLA — a escolha conservadora

O DeepSeek-V3 fez a aposta oposta às duas anteriores: **não trocou de arquitetura.** Manteve a atenção (e todo o seu recall) e atacou só o KV cache. Em vez de guardar K e V cheios, guarda uma **projeção latente comprimida** por token, e a reexpande na hora de calcular a atenção.

Medido no Lab 4 (contexto de 32k, tipo Llama-3-8B):

| Método | KV cache |
|---|---|
| GQA (8 kv heads) | 4,29 GB |
| **MLA (latente 512)** | **2,15 GB** (−50%) |

MLA (Multi-head Latent Attention) é a escolha "conservadora" e foi a certa para produção: mantém o mecanismo que comprovadamente funciona e comprime só o que dói. É uma das razões de o DeepSeek servir contexto longo tão barato — junto com o MoE do módulo 11. Note o padrão: GQA (módulo 2) já era compressão de KV; MLA é o próximo passo da mesma pressão.

---

## 5. 📐 O trade-off que decide tudo: recall vs eficiência

Aqui está a pergunta central da fronteira, e a medição mais importante do módulo. As arquiteturas eficientes (SSM, linear) sacrificam a capacidade de **recuperar um token específico do passado** — justamente o que a atenção faz bem.

O teste clássico é *associative recall*: dado "A→1, B→2, C→3..." no contexto, e depois "B→?", lembrar "2". Medido no Lab 5:

| Mecanismo | Recall associativo |
|---|---|
| **Atenção** | **100%** |
| Atenção linear | **20%** |

A diferença é o **softmax**. A atenção pode "focar" — colocar quase toda a massa na chave exata que casa com a query (o attention sink do módulo 2 é a mesma capacidade, usada mal). Sem softmax, a recuperação vira uma média ponderada suave: ela *borra* os valores em vez de selecionar um. SSMs sofrem de uma versão do mesmo problema — comprimir tudo num estado fixo perde a capacidade de apontar para um token arbitrário do passado.

**Este é o trade-off que impede o transformer de ser destronado.** Recall preciso é insubstituível para: recuperar um fato do prompt, seguir uma variável através de um código, in-context learning (as induction heads do módulo 16 dependem disso). Nenhum mecanismo O(L) faz isso tão bem quanto a atenção O(L²) — ainda.

---

## 6. Híbridos e multimodalidade — a resposta real

Como cada mecanismo tem uma força e uma fraqueza opostas, a resposta da indústria é **combiná-los**. Modelos de fronteira que usam SSM (Jamba, Nemotron-H, Falcon-Mamba) **intercalam** camadas: muitas de Mamba (eficiência) e poucas de atenção (recall). Medido no Lab 6 (custo de compute em contexto longo):

| % de camadas de atenção | Custo relativo |
|---|---|
| 100% (transformer puro) | 100% |
| 25% | 25% |
| **~12% (1 a cada 8)** | **~9%** |
| 0% (Mamba puro) | ~0% |

Um híbrido com ~1 camada de atenção a cada 6–8 de Mamba captura quase toda a eficiência do Mamba E quase todo o recall do transformer. É o "melhor dos dois mundos" que tirou os SSMs do laboratório.

### Multimodalidade — a mesma arquitetura, outros tokens

O padrão que transformou "LLM" em "modelo de fundação": um **encoder** de imagem (ViT), áudio ou vídeo projeta a entrada em vetores no MESMO espaço dos embeddings de texto — vira "tokens" que a arquitetura já sabe processar. O transformer não muda; só a fonte dos tokens muda. GPT-4V, Gemini, Qwen-VL: todos são a arquitetura do módulo 2 recebendo tokens de imagem intercalados com os de texto. A atenção mistura as modalidades de graça — é para isso que ela serve.

### A leitura honesta da fronteira

O transformer não foi destronado; foi **complementado e otimizado**. A atenção continua sendo o melhor mecanismo de recall preciso que existe; o que a área aprendeu é a usá-la com **parcimônia** — delegando o processamento de contexto longo a mecanismos O(L) e reservando a atenção para onde o recall importa. Sete anos e a peça central do módulo 2 continua central. Isso, por si, é um resultado científico: raramente uma arquitetura sobrevive tanto.

---

## 7. Leituras

1. **Gu & Dao (2023), "Mamba"** — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). O SSM seletivo. Denso; a seção 3 é o coração.
2. **Katharopoulos et al. (2020), "Transformers are RNNs (linear attention)"** — [arXiv:2006.16236](https://arxiv.org/abs/2006.16236). A reordenação do Lab 3.
3. **DeepSeek-AI (2024), "DeepSeek-V2/V3"** — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434). O MLA.
4. **Lieber et al. (2024), "Jamba"** — [arXiv:2403.19887](https://arxiv.org/abs/2403.19887). O híbrido Mamba-Transformer em produção.
5. **Arora et al. (2023), "Zoology: Measuring and Improving Recall in Efficient Language Models"** — [arXiv:2312.04927](https://arxiv.org/abs/2312.04927). O trade-off do Lab 5, medido a sério.
6. **Dosovitskiy et al. (2020), "An Image is Worth 16×16 Words (ViT)"** — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929). Como imagens viram tokens.

---

## 8. Checklist de saída

- [ ] Qual o gargalo O(L²), e por que dobrar o contexto quadruplica o custo?
- [ ] Como um SSM resume o passado, e por que seu estado é constante no comprimento?
- [ ] O que o Mamba adiciona ao SSM clássico (dois itens) que salvou a ideia de RNN?
- [ ] Qual reordenação torna a atenção linear O(L), e o que ela remove para consegui-la?
- [ ] O que o MLA comprime, e por que é a escolha "conservadora"?
- [ ] O trade-off central, com os números: recall da atenção vs linear, e por que a diferença?
- [ ] Por que os modelos de fronteira são híbridos, e qual a proporção típica?
- [ ] Como uma imagem vira algo que o transformer processa?
- [ ] A conclusão honesta: o transformer foi destronado? Justifique.

Depois: `lab_cpu.py` (executado — os desafiantes medidos) e os cartões em `revisao/baralho-02-expansao.tsv`. **Isto fecha a Fase 2** — a próxima parada é a Fase 3, onde você para de aprender o que existe e passa a produzir.
