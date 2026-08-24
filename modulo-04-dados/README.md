# Módulo 4 — Preparação e curadoria de datasets

> **Pergunta central:** por que dado é o gargalo real, e o que exatamente se faz com ele antes de treinar?

Este é o módulo menos glamouroso e o de maior retorno prático. A arquitetura de LLMs está essencialmente congelada desde 2023 — RoPE, GQA, SwiGLU, RMSNorm, pre-norm. Praticamente todo o ganho entre gerações de modelos abertos veio de **dados melhores**.

Para você, que vai customizar modelos e não pré-treiná-los, a conclusão é ainda mais direta: **a qualidade do seu dataset determina o resultado do fine-tuning muito mais do que qualquer hiperparâmetro**. No Lab 7 você vai medir isso, não apenas ler.

## Objetivos

1. Descrever o pipeline de curadoria de pré-treino e o que cada etapa remove.
2. Implementar MinHash + LSH e explicar por que a deduplicação exata é insuficiente.
3. Aplicar filtros heurísticos de qualidade e medir o que sobra.
4. Detectar contaminação de benchmark por sobreposição de n-gramas.
5. Formatar um dataset de SFT corretamente — chat template, masking de prompt e o EOS que quase todo mundo esquece.
6. Decidir quantos exemplos você precisa, e quando parar de coletar e começar a limpar.

---

## 1. A evidência de que dados dominam

| Modelo | Ano | Params | Tokens | MMLU |
|---|---|---|---|---|
| GPT-3 | 2020 | 175B | 300B | 43,9 |
| Llama 1 | 2023 | 7B | 1,0T | 35,1 |
| Llama 2 | 2023 | 7B | 2,0T | 45,3 |
| Llama 3 | 2024 | 8B | 15T | 66,6 |
| Qwen2.5 | 2024 | 7B | 18T | 74,2 |

Entre Llama 1 e Llama 3, o número de parâmetros mudou de 7B para 8B — praticamente nada — e a arquitetura mudou pouquíssimo (ganhou GQA e um vocabulário maior). O que mudou foi **15× mais tokens, muito melhor filtrados**. O desempenho quase dobrou.

O experimento mais limpo nessa direção é o **FineWeb-Edu**: a mesma web, filtrada por um classificador de "valor educacional" treinado com anotações do Llama-3-70B. Reduziu 15T tokens para 1,3T — descartou **91% do corpus** — e produziu modelos melhores em benchmarks de conhecimento. Menos dados, melhores resultados.

### A versão em miniatura, que você vai rodar

O Lab 7 reproduz esse achado em escala de laboratório: mesmo modelo, mesmo compute, mesmo tokenizer, mesma validação limpa. Muda só o corpus.

| Corpus de treino | Tokens | Loss de treino | Loss na validação limpa | PPL |
|---|---|---|---|---|
| Limpo (Machado) | 213k | 5,337 | **5,627** | 278 |
| Poluído (+37% lixo de web) | 1.023k | **3,003** | 6,068 (+7,8%) | 432 |
| Filtrado (poluído + Gopher) | 334k | 4,439 | **5,612** (−0,3%) | 274 |

Três leituras, e a segunda é a mais importante:

1. **Poluir o corpus piora o modelo em 7,8%**, mesmo tendo dado a ele 5× mais tokens de treino. Capacidade gasta modelando menus de navegação é capacidade perdida.
2. **O modelo poluído tem a MENOR loss de treino de todos — 3,00 contra 5,34 do limpo — e o pior desempenho real.** Lixo de web é trivialmente previsível: depois da vigésima repetição de "Comprar celular barato...", prever a vigésima primeira é fácil. A loss de treino mede o quão previsível é o seu corpus, não o quão bom é o seu modelo. **Nunca compare modelos treinados em corpora diferentes pela loss de treino.**
3. **A filtragem recupera todo o dano** e ainda fica marginalmente à frente do corpus limpo original — apesar de descartar 34% dos documentos bons junto com o lixo. Esse é o trade-off real de todo filtro, e ele compensa.

A prova mais convincente não está na tabela, e sim no texto gerado. Pedindo aos três a continuação de *"Uma noite destas"*:

> **limpo:** *Uma noite destas de cho. Aque, e era, a mem, não eu a a gara...*
> **poluído:** *Uma noite destas. Comprar celular barato... Melhor celular barato... Top 10 notebook*
> **filtrado:** *Uma noite destas, mas elle de opiso. Este site utiliza cookies para melhorar sua experiência...*

Nenhum é bom — são modelos de 2M de parâmetros. Mas o poluído aprendeu a ser um site de e-commerce, e o filtrado ainda carrega o aviso de cookies que sobreviveu ao filtro. **Você vê exatamente o que o corpus ensinou.**

---

## 2. O pipeline de pré-treino

```
HTML bruto (petabytes)
   ↓ extração de texto (trafilatura, resiliparse)
   ↓ filtro de idioma (fastText, limiar de confiança)
   ↓ filtros heurísticos de qualidade  ← seção 3
   ↓ classificador de qualidade         ← seção 4
   ↓ DEDUPLICAÇÃO                       ← seção 5, a etapa de maior impacto
   ↓ decontaminação de benchmarks       ← seção 7
   ↓ remoção de PII
corpus de treino (trilhões de tokens)
```

Cada etapa descarta uma fração enorme. Do Common Crawl bruto ao corpus final, sobrevive tipicamente **menos de 10%**. O trabalho de um time de dados de LLM é, em grande parte, decidir o que jogar fora.

---

## 3. Filtros heurísticos

As regras do **Gopher** (DeepMind, 2021) viraram o padrão de facto. Um documento é descartado se:

| Regra | Limiar | O que pega |
|---|---|---|
| Número de palavras | < 50 ou > 100.000 | Fragmentos e despejos |
| Comprimento médio de palavra | fora de [3, 10] chars | Código minificado, gibberish |
| Razão símbolo/palavra (`#`, `...`) | > 0,1 | Boilerplate, texto truncado |
| Linhas começando com bullet | > 90% | Menus de navegação |
| Linhas terminando em `...` | > 30% | Listagens truncadas de SEO |
| Palavras com ao menos uma letra | < 80% | Tabelas de números, hashes |
| Contém stop words comuns | < 2 delas | Texto que não é prosa |

O **C4** (Google) acrescenta regras de linha: manter apenas linhas que terminam em pontuação, descartar linhas com `javascript`, `lorem ipsum`, `{`, e documentos inteiros que contenham termos de política de cookies.

> 🔧 **Na prática:** filtros heurísticos são baratos, transparentes e auditáveis — você consegue explicar por que cada documento foi removido. Classificadores são mais poderosos e opacos. O padrão da indústria é usar heurísticas primeiro (removem a maior parte do lixo por quase nada de compute) e o classificador depois, sobre o que sobrou.

> ⚠️ **Armadilha:** esses limiares foram calibrados em **inglês**. "Comprimento médio de palavra entre 3 e 10" descarta agressivamente idiomas aglutinantes (alemão, finlandês, turco); a lista de stop words em inglês elimina praticamente todo o corpus em português.

Medido no Lab 3, sobre 924 parágrafos de Machado de Assis:

| Lista de stop words | Documentos aprovados |
|---|---|
| Inglês (`the, be, to, of, and, that, have, with`) | **2 de 924 (0,2%)** |
| Português (`de, a, o, que, e, do, da, em, um, para, com, não`) | 922 de 924 (99,8%) |

**Machado de Assis reprovado por um filtro de qualidade** — 99,8% do corpus descartado como "não sendo prosa". Trocar uma lista de oito palavras é a diferença entre um corpus e nada.

Isso não é curiosidade acadêmica: é uma explicação direta de por que tantos modelos têm português pior do que o volume de texto em português na web faria supor. O filtro que garantiu a qualidade do inglês jogou fora o resto.

---

## 4. Filtragem por classificador

Três abordagens, em ordem de custo:

1. **Classificador de qualidade** — um modelo leve (fastText, ou um encoder pequeno) treinado para distinguir "texto bom" (Wikipedia, livros) de web aleatória. Usado no GPT-3 e no Llama 1.
2. **Perplexity filtering** — rodar um modelo de linguagem pequeno sobre cada documento e descartar os de perplexidade muito alta (lixo) **e muito baixa** (texto repetitivo, boilerplate). Descartar as duas caudas é o detalhe que se esquece.
3. **LLM-as-judge** — pedir a um modelo grande que pontue documentos, treinar um classificador leve nessas anotações, e aplicar em escala. É o método do FineWeb-Edu e o estado da arte atual.

---

## 5. Deduplicação — a etapa de maior impacto

A web repete tudo. Um mesmo artigo aparece no site original, em agregadores, em scrapers, em traduções automáticas de volta ao original.

Três níveis:

### Exata
Hash do documento inteiro. Trivial, rápido, e **insuficiente** — basta um cabeçalho diferente para dois documentos idênticos escaparem.

### 📐 Near-duplicate: MinHash + LSH

O problema: comparar 10 bilhões de documentos par a par são 5×10¹⁹ comparações. Impossível.

**MinHash** resolve estimando a similaridade de Jaccard sem comparar conjuntos inteiros:

```
J(A,B) = |A ∩ B| / |A ∪ B|
```

Para cada documento, extraia seus n-gramas (shingles). Aplique `k` funções de hash independentes e guarde, para cada uma, **apenas o menor valor**. A assinatura resultante tem `k` números.

A propriedade mágica:

```
P(minhash_i(A) = minhash_i(B)) = J(A,B)
```

A probabilidade de dois documentos terem o mesmo mínimo, para uma função de hash qualquer, é **exatamente** a similaridade de Jaccard. Logo, a fração de posições coincidentes entre duas assinaturas estima `J`, com erro `~1/√k`. Com `k=128`, erro de ~9%.

**LSH (Locality-Sensitive Hashing)** evita comparar todas as assinaturas: divida os `k` valores em `b` bandas de `r` linhas (`k = b·r`) e indexe cada banda numa tabela hash. Dois documentos viram candidatos se colidirem em **ao menos uma** banda:

```
P(candidatos) = 1 − (1 − J^r)^b
```

Essa função tem forma de S com o joelho aproximadamente em `J ≈ (1/b)^(1/r)`. Escolhendo `b` e `r` você sintoniza o limiar de similaridade. O Lab 2 implementa tudo isso do zero — são ~30 linhas.

### Substring
Remove sequências repetidas de 50+ tokens **dentro** de documentos diferentes, via suffix array. Pega o caso do parágrafo copiado.

### O impacto

Lee et al. (2021) mediram: deduplicar reduz a **memorização em até 10×** (o modelo regurgita muito menos texto de treino literalmente), reduz a contaminação de benchmarks, e **melhora** a perplexidade em dados novos. É a rara etapa que melhora tudo ao mesmo tempo.

---

## 6. Datasets de SFT — outro jogo

Pré-treino quer **volume e diversidade**. SFT quer **qualidade e consistência**. As técnicas divergem completamente.

### A evidência do LIMA

Zhou et al. (2023) treinaram o LIMA com **1.000 exemplos** meticulosamente curados à mão. Ele bateu modelos treinados com 52.000 exemplos gerados automaticamente (Alpaca).

A interpretação é a "hipótese do alinhamento superficial": o SFT não ensina conhecimento — ele ensina qual sub-distribuição do pré-treino ativar. Para isso, mil exemplos consistentes e bem escritos bastam. Cinquenta mil exemplos inconsistentes ensinam inconsistência.

> 🔧 Isto é a coisa mais acionável do módulo. Se você tem 5.000 exemplos de qualidade duvidosa, **selecionar os 500 melhores tende a superar usar todos os 5.000**. Curadoria bate coleta.

### De onde vêm os dados de SFT

| Origem | Custo | Qualidade | Riscos |
|---|---|---|---|
| **Humana** (no_robots, Dolly) | Alto | Alta | Escala limitada |
| **Sintética por self-instruct** (Alpaca) | Baixo | Variável | Alucinações herdadas, pouca diversidade |
| **Evol-Instruct** (WizardLM) | Médio | Boa | Complexidade artificial |
| **Destilação** de um modelo forte | Baixo | Alta | **Licença** — quase sempre proibido comercialmente |
| **Logs de produção** | Baixo | Ótima (é a distribuição real!) | PII, precisa de anotação |

> ⚠️ **Armadilha jurídica:** os termos de uso da OpenAI e da Anthropic proíbem usar as saídas dos modelos para treinar modelos concorrentes. O Alpaca é distribuído como CC BY-NC 4.0 **e** carrega essa restrição, porque foi gerado com `text-davinci-003`. Muita gente treina modelos "abertos" sobre dados destilados e descobre a restrição tarde demais. Se o destino for comercial, verifique a licença de cada dataset — inclusive a proveniência dos dados sintéticos.

### Model collapse

Shumailov et al. (2024) mostraram que treinar recursivamente sobre saídas do próprio modelo degenera a distribuição: as caudas somem, a diversidade colapsa, e depois de algumas gerações o modelo produz um subconjunto cada vez mais estreito.

A ressalva importante, frequentemente omitida: o colapso ocorre quando dados sintéticos **substituem** os reais. Trabalhos posteriores mostram que **acumular** (real + sintético) não colapsa. Dados sintéticos são uma ferramenta legítima; substituir integralmente sua fonte de verdade por eles não é.

---

## 7. Contaminação de benchmark

Se o corpus de treino contém o conjunto de teste do MMLU, a pontuação no MMLU mede memorização, não capacidade.

Isso acontece **o tempo todo**: benchmarks são publicados na web, e crawls os capturam.

**Detecção padrão:** sobreposição de n-gramas. O GPT-3 usou 13-gramas — se um documento de treino compartilha uma sequência de 13 palavras com um item de teste, é marcado. O Lab 5 implementa isso.

**Métodos mais recentes** comparam a perplexidade do item na ordem canônica contra ordens embaralhadas: um modelo que viu o benchmark atribui probabilidade anomalamente alta à ordem original.

> 🔧 **Na prática:** quando você fizer fine-tuning e avaliar, garanta que seus exemplos de avaliação **não** estejam no treino. Parece óbvio e é a fonte número um de resultados bons demais para ser verdade. Separe o split de teste **antes** de qualquer deduplicação ou augmentação, e deduplique o teste contra o treino, não só o treino contra si mesmo.

---

## 8. Quantos exemplos você precisa

| Objetivo | Ordem de grandeza |
|---|---|
| Formato de saída, tom, estilo | 100 – 1.000 |
| Tarefa específica bem definida | 1.000 – 10.000 |
| Domínio complexo, múltiplas tarefas | 10.000 – 100.000 |
| Idioma ou domínio muito fora da distribuição | 100.000+ |

A curva de ganho é **logarítmica**: dobrar os dados dá um ganho aproximadamente constante e pequeno. O corolário prático: passado o primeiro milhar, o retorno de aumentar a *qualidade média* costuma superar o de aumentar a quantidade.

**O procedimento recomendado:** comece com 200–500 exemplos de altíssima qualidade, treine, meça. Só então decida se o gargalo é quantidade (o modelo acerta o formato mas erra casos variados) ou qualidade (o modelo aprendeu inconsistências que estão nos dados).

---

## 9. O formato final — como o dataset chega ao treino

Três decisões que decidem se o fine-tuning funciona:

### Chat template
Sempre `tokenizer.apply_chat_template`. Nunca monte a string na mão. (Módulo 1, seção 9.)

### Masking do prompt
O padrão é calcular a loss **apenas nos tokens da resposta**, mascarando o prompt com `-100`:

```
<|im_start|>user\nQual a capital?<|im_end|>\n<|im_start|>assistant\n   ← labels = -100
Paris.<|im_end|>                                                       ← labels = ids reais
```

A lógica: você quer ensinar o modelo a **responder**, não a gerar instruções de usuário. Há debate — treinar no prompt inteiro também funciona e, com poucos dados, às vezes regulariza. Mas o padrão (e o default do TRL para `completion_only_loss`) é mascarar.

### ⚠️ O EOS

**A armadilha número um do SFT.** Se os exemplos de treino não terminam com o token de fim de sequência, o modelo nunca aprende a parar. Em produção ele gera a resposta correta e continua — inventa o próximo turno, responde a si mesmo, degenera — até bater o limite de tokens.

O sintoma é inconfundível e a causa é uma linha de código. Verifique **sempre**, imprimindo os últimos tokens de um exemplo tokenizado, que o EOS está lá. O Lab 6 mostra o certo e o errado lado a lado.

---

## 10. Leituras

1. **Penedo et al. (2024), "FineWeb"** — [HuggingFace](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1). O melhor relato existente de construção de corpus, com ablações medidas para cada decisão de filtragem.
2. **Zhou et al. (2023), "LIMA: Less Is More for Alignment"** — [arXiv:2305.11206](https://arxiv.org/abs/2305.11206). Curto e transformador.
3. **Lee et al. (2021), "Deduplicating Training Data Makes Language Models Better"** — [arXiv:2107.06499](https://arxiv.org/abs/2107.06499).
4. **Rae et al. (2021), "Gopher"** — [arXiv:2112.11446](https://arxiv.org/abs/2112.11446). O apêndice A tem as regras de filtragem completas.
5. **Wang et al. (2022), "Self-Instruct"** — [arXiv:2212.10560](https://arxiv.org/abs/2212.10560). A receita que gerou o Alpaca.
6. **Shumailov et al. (2024), "AI models collapse when trained on recursively generated data"** — [Nature](https://www.nature.com/articles/s41586-024-07566-y).

---

## 11. Checklist de saída

- [ ] Qual etapa do pipeline de pré-treino tem o maior impacto medido, e quais três coisas ela melhora ao mesmo tempo?
- [ ] Por que a deduplicação exata não basta?
- [ ] Enuncie a propriedade que faz o MinHash funcionar, em uma linha.
- [ ] O que `b` e `r` controlam no LSH, e como se calcula o limiar aproximado?
- [ ] Por que aplicar os filtros do Gopher a um corpus em português é perigoso?
- [ ] Por que se descartam as **duas** caudas na filtragem por perplexidade?
- [ ] O que o LIMA demonstrou, e o que isso implica para o seu dataset de 5.000 exemplos duvidosos?
- [ ] Qual o risco jurídico de treinar com Alpaca para uso comercial?
- [ ] Sob que condição dados sintéticos causam model collapse — e sob qual não causam?
- [ ] Quais três coisas precisam estar certas na formatação de um exemplo de SFT?
- [ ] Seu modelo responde certo e depois continua falando sozinho. Diagnóstico?

Depois, abra o `lab.py`. O Lab 7 é um experimento controlado: mesmo modelo, mesmo compute, corpus limpo contra corpus poluído. Antes de usar dados externos, aplique o gate de proveniência, licença, checksum e PII descrito em [`../GOVERNANCA-DE-DADOS.md`](../GOVERNANCA-DE-DADOS.md).
