# Módulo 13 — RAG e conhecimento externo

> **Pergunta central:** como dar ao modelo o que ele não sabe — sem treinar nada?

A fase 1 apontou para cá uma dúzia de vezes: *"problema de conhecimento? RAG"*. Este módulo paga a dívida. E fecha o quadro conceitual do curso: **fine-tuning muda o que o modelo É; RAG muda o que ele VÊ.** Comportamento se treina; conhecimento se entrega no contexto.

O lab constrói o sistema sobre a base de conhecimento mais útil disponível: **o próprio curso** — um assistente de estudo sobre os 12 módulos, com avaliação verificável por passagens rotuladas que contêm a evidência de cada resposta.

## Objetivos

1. Justificar RAG vs fine-tuning pela anatomia do problema (volátil? volumoso? citável?).
2. Fazer chunking com critério — e saber o que ele quebra.
3. Implementar BM25 do zero e busca densa com bi-encoder, e **medi-los na mesma métrica**.
4. Combinar os dois (RRF) e avaliar com hit@k e MRR.
5. Medir o valor do contexto em probabilidade — e o **custo do contexto errado**.
6. Detectar perguntas fora da base (abstenção) e conhecer os limites do score.

---

## 1. Por que RAG — a anatomia da decisão

O teste do módulo 4, agora formalizado: *se colocar a informação no prompt resolve, o problema é de conhecimento.* E conhecimento tem três propriedades que o fine-tuning atende mal:

| Propriedade | Fine-tuning | RAG |
|---|---|---|
| **Volátil** (muda toda semana) | retreinar a cada mudança | atualizar um documento no índice |
| **Volumoso** (milhões de fatos de cauda longa) | 50k exemplos não instalam um catálogo | o índice guarda tudo; o contexto recebe só o relevante |
| **Citável** (precisa apontar a fonte) | o modelo "sabe", mas não sabe de onde | cada resposta carrega os chunks de origem |

E a assimetria de risco que o módulo 5 mediu: SFT em domínio factual **aumenta a confiança sem aumentar a precisão** — o formato de especialista sem a substância. RAG erra diferente: quando a recuperação falha, dá para detectar (score baixo, fonte ausente) e abster.

A arquitetura em uma linha:

```
pergunta → [recuperar os k trechos mais relevantes do índice] → prompt(contexto + pergunta) → modelo
```

Todo o resto do módulo são as decisões dentro desses colchetes.

---

## 2. Chunking — a decisão subestimada

O índice não guarda documentos; guarda **pedaços**. O tamanho deles é um trade-off real:

- **Grandes** (1000+ palavras): carregam contexto completo, mas o embedding vira uma média borrada de vários assuntos — a busca perde precisão — e enchem o prompt de irrelevância.
- **Pequenos** (50 palavras): busca cirúrgica, mas o trecho chega órfão — sem o entorno que o torna interpretável.

As decisões defensáveis, na ordem:

1. **Corte por estrutura semântica** — títulos, seções, parágrafos: as fronteiras que o autor já marcou. (É o que o lab faz com os `##` dos READMEs.)
2. **Sobreposição** (10–20%) — para o fato que mora na fronteira entre dois chunks não ser cortado ao meio.
3. **Metadados junto** — módulo, título da seção: baratos no índice, valiosos na citação e no filtro.
4. Variantes avançadas quando o simples falha: *parent-child* (busca no pedaço pequeno, entrega o pai grande), janelas por sentença, chunking semântico por embedding.

---

## 3. 📐 As duas famílias de busca

### BM25 — a baseline de 50 anos

Palavras em comum, com dois refinamentos que a tornaram imbatível por décadas:

```
score(q, d) = Σ_{termo ∈ q} IDF(termo) · TF_saturada(termo, d)

IDF = ln((N − df + 0,5)/(df + 0,5) + 1)      termos raros valem muito
TF_saturada = f·(k₁+1) / (f + k₁·(1−b+b·|d|/média))   a 10ª ocorrência ≈ a 3ª
```

Forças: termos exatos e raros (`--mask-prompt`, `NF4`, nomes próprios), zero treino, interpretável, rápido. Fraqueza: **vocabulário literal** — "como impedir o modelo de esquecer" não encontra "catastrophic forgetting".

### Busca densa — o bi-encoder

Um modelo de embeddings (treinado contrastivamente: pares pergunta-documento verdadeiros aproximados, falsos afastados) converte pergunta e chunks para o **mesmo espaço vetorial**; relevância = cosseno. O "banco vetorial" é, na essência, uma matriz `[n_chunks, dim]` e um produto de matrizes — todo o resto (FAISS, HNSW, pgvector) é engenharia para quando `n_chunks` tem seis dígitos ou mais.

Forças: paráfrase, multilíngue, sinônimos. Fraquezas: termos exatos fora do treino do encoder, números, códigos — exatamente onde o BM25 brilha.

> ⚠️ **A armadilha de template, versão embeddings:** modelos da família e5 exigem os prefixos `query:` / `passage:` — foram treinados assim. Omiti-los degrada silenciosamente, sem erro. O mesmo padrão do chat template do módulo 1: sempre verifique a convenção de uso do encoder.

### Híbrido — RRF

Como as fraquezas são complementares, a fusão é quase grátis: **Reciprocal Rank Fusion** soma `1/(60 + posição)` de cada documento nos dois rankings. Não precisa calibrar escalas de score (BM25 dá 0–20, cosseno dá 0,7–0,9 — incomensuráveis); usa só as posições. É o default sensato de produção.

### O andar de cima: reranking

O bi-encoder comprime pergunta e documento **separadamente** — barato, mas cego às interações palavra a palavra. O **cross-encoder** lê os dois juntos e pontua com toda a atenção — caro demais para o índice inteiro, perfeito para reordenar o top-30 do estágio barato. Pipeline padrão: híbrido recupera 30 → reranker escolhe os 5 → modelo recebe 5. (Exercício B4.)

---

## 4. Avaliar a recuperação — antes e separada da geração

O erro clássico: avaliar o sistema inteiro lendo respostas finais. Quando a resposta sai errada, foi a busca ou a geração? Sem separar, não se sabe o que consertar.

**A recuperação se avalia sozinha, com gabarito de passagem:** o módulo de origem é
metadado, não relevância. Um chunk qualquer do módulo correto não conta como acerto;
ele precisa conter a evidência rotulada para a resposta.

| Métrica | Pergunta que responde |
|---|---|
| **hit@k** | a fonte certa está entre os k primeiros? (k = quantos chunks você manda ao modelo) |
| **MRR** | em que posição o primeiro acerto aparece, em média (1/rank) |
| nDCG | versão com relevância graduada — quando há "muito relevante" vs "relevante" |

> **Correção metodológica:** a primeira versão do lab usava apenas o módulo de origem
> como gabarito e reportava BM25 84%, densa 92% e RRF 88% em hit@1. Esse critério podia
> contar um chunk irrelevante do módulo correto como hit. Os números foram retirados da
> teoria; rode `lab_cpu.py` para obter a medição atual com gabarito no nível de passagem.
> O episódio é parte do conteúdo: antes da estatística, valide o que o rótulo significa.

---

## 5. A geração — e os dois fenômenos que a governam

### O valor do contexto, medido

A técnica do módulo 7 transplantada: `log P(resposta correta | pergunta + contexto)` em três condições. Resultado em 8 perguntas:

| Condição | log P típico |
|---|---|
| Sem contexto | −8 a −36 |
| **Com o chunk recuperado (top-1 real)** | ganho médio **+5,72 nats = resposta 306× mais provável** |
| Com chunk irrelevante | frequentemente **pior que sem contexto** (ex.: −8,7 → −11,5) |

Dois achados na tabela completa do lab:

1. **Contexto errado é pior que nenhum contexto** na maioria das linhas. O modelo confia no que você entregou — o mesmo mecanismo de "seguir a cadeia" do módulo 7 (fidelidade causal), agora como risco de produto. RAG com recuperação ruim não é RAG neutro; é um gerador de alucinações com citação.
2. **Em 2 das 8 perguntas, até o chunk CERTO atrapalhou um pouco** (a do `ln 2`, a do GSM8K). O chunk continha a informação, mas com fraseado distante da resposta-alvo — e um contexto longo dilui a atenção. Nem todo acerto de recuperação vira acerto de geração; é por isso que as duas etapas se avaliam separadas.

### Grounding e abstenção

Produção exige duas defesas:

1. **Instrução de grounding:** "responda APENAS com base no contexto; se a resposta não estiver nele, diga que não sabe". Modelos instruct modernos respeitam isso razoavelmente — e o lab_mlx mede o quanto.
2. **Abstenção por score:** perguntas fora da base tendem a ter similaridade top-1 menor. **Medido: o limiar simples FALHOU.** Dentro da base: 0,869–0,910; fora: 0,778–0,874 — os intervalos se sobrepõem ("qual o melhor framework de frontend?" pontuou 0,874, acima do mínimo interno). Cossenos de bi-encoder vivem numa faixa comprimida e não são detectores de domínio confiáveis. As soluções reais: calibração num conjunto rotulado, rerankers (scores mais discriminativos), ou — a mais robusta — a instrução de grounding no gerador, que o lab_mlx mede. O resultado negativo é o conteúdo: quem confia num limiar de cosseno sem medir a separação embarca alucinação em produção.

### Lost in the middle

Modelos dão mais atenção ao início e ao fim do contexto; informação no meio de um prompt longo é sistematicamente pior utilizada (Liu et al., 2023). Consequências práticas: mande **poucos chunks bons** (5, não 20), ponha o mais relevante primeiro ou último, e desconfie de "contexto de 128k resolve tudo" — resolver *caber* não resolve *usar*.

---

## 6. O mapa completo (e o que ficou para os exercícios)

```
                    ┌─ reescrita da consulta (HyDE, multi-query)     [B5]
pergunta ──────────►│
                    ▼
índice ──► híbrido (BM25 + denso) ──► reranker cross-encoder [B4] ──► top-k
                    ▲                                                   │
chunking [B1-B2] ───┘                                                   ▼
                                              prompt com grounding ──► modelo ──► resposta + fontes
                                                                        │
avaliação: hit@k/MRR na recuperação  +  fidelidade na geração [lab_mlx]┘
```

E a fronteira além do módulo: GraphRAG (índice como grafo de entidades — para perguntas que atravessam documentos), agentic RAG (o modelo decide *quando* e *o que* buscar — ponte para o módulo 15), e fine-tuning de embeddings no seu domínio (quando o e5 genérico não basta).

---

## 7. Leituras

1. **Lewis et al. (2020), "Retrieval-Augmented Generation"** — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401). A origem do termo (num contexto que envelheceu; o padrão atual é mais simples que o paper).
2. **Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond"** — o BM25 pelos autores.
3. **Wang et al. (2022), "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (E5)** — [arXiv:2212.03533](https://arxiv.org/abs/2212.03533). O encoder do lab.
4. **Liu et al. (2023), "Lost in the Middle"** — [arXiv:2307.03172](https://arxiv.org/abs/2307.03172).
5. **Cormack et al. (2009), "Reciprocal Rank Fusion"** — 2 páginas, e o método continua imbatível pela simplicidade.

---

## 8. Checklist de saída

- [ ] Enuncie o teste que separa problema de conhecimento de problema de comportamento.
- [ ] Quais as três propriedades do conhecimento que favorecem RAG sobre fine-tuning?
- [ ] O trade-off do tamanho de chunk, nos dois extremos — e a primeira estratégia defensável.
- [ ] O que o IDF e a saturação de TF fazem no BM25?
- [ ] Onde o BM25 vence a busca densa, e vice-versa? Por que o híbrido é quase grátis?
- [ ] Por que o RRF usa posições em vez de scores?
- [ ] hit@k e MRR: o que cada um responde, e por que se avalia a recuperação SEPARADA da geração?
- [ ] O que a medição de log-prob mostrou sobre contexto errado vs nenhum contexto?
- [ ] O que é lost in the middle e quais as duas consequências práticas?
- [ ] Quando o bi-encoder não basta e o cross-encoder entra — e por que não usá-lo no índice inteiro?

Depois: `lab_cpu.py` (executado — o assistente sobre o próprio curso), `lab_mlx.py` (geração de ponta a ponta com grounding), e os cartões novos em `revisao/baralho-02-expansao.tsv`.
