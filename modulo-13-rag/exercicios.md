# Módulo 13 — Exercícios

💻 = qualquer máquina | 🍎 = Mac

---

## Parte A — Conceituais

### A1. RAG ou fine-tuning — o diagnóstico

Para cada caso, RAG, fine-tuning, os dois, ou nenhum — com o teste que justifica:

1. Assistente jurídico que precisa citar artigos da legislação vigente.
2. Modelo que deve responder sempre em JSON com schema fixo.
3. Chatbot que precisa saber o cardápio do restaurante (muda toda semana) E falar no tom da marca.
4. Modelo que erra contas de juros compostos.
5. Buscador interno que precisa achar documentos parecidos com uma consulta — sem gerar nada.

<details><summary>Gabarito</summary>

1. **RAG** — as três propriedades juntas: volátil (leis mudam), volumoso (milhares de artigos), citável (requisito do domínio). Fine-tuning aqui aumenta confiança sem precisão (módulo 5).
2. **Fine-tuning** (ou constrained decoding) — formato é comportamento puro, o caso ideal do SFT. Não há conhecimento a recuperar.
3. **Os dois** — a divisão de trabalho canônica: RAG entrega o cardápio (conhecimento volátil), SFT instala o tom (comportamento estável). É o padrão da maioria dos produtos reais.
4. **Nenhum dos dois** — é capacidade de cálculo, não conhecimento nem formato. A resposta de engenharia: ferramenta (calculadora/Python) via tool use — módulo 15.
5. **Só a metade de recuperação do RAG** — busca densa/BM25 sem gerador. Nem todo problema com embeddings precisa de um LLM na frente; às vezes o produto É o retrieval.
</details>

---

### A2. A tabela que surpreendeu

Rode o lab com o gabarito de passagens e compare a híbrida com a densa pura.

a) Explique o mecanismo pelo qual o RRF pode piorar o sistema.
b) Em que condições o híbrido paga — e como você DETECTARIA essas condições antes de adotar?
c) Proponha uma variante do RRF que mitigue o problema.

<details><summary>Gabarito</summary>

a) O RRF dá peso igual aos dois rankings. Quando um sistema domina o outro, a fusão pode promover os erros do mais fraco: um chunk irrelevante bem ranqueado no BM25 pode roubar a posição 1 de uma passagem relevante da busca densa. Confirme se isso acontece na execução atual antes de concluir.

b) O híbrido paga quando as falhas são **complementares** — cada sistema acerta perguntas que o outro erra. Detecção: rode os dois separados no seu conjunto de avaliação e monte a matriz de concordância (ambos acertam / só A / só B / ambos erram). Se "só BM25 acerta" é quase vazio, a fusão só puxa para baixo. É uma tabela de 4 células que quase ninguém faz antes de adotar "híbrido é melhor".

c) RRF ponderado: `score = w·1/(c+rank_denso) + (1−w)·1/(c+rank_bm25)` com `w` calibrado na validação; ou usar o BM25 apenas como *fallback* quando o score denso do top-1 é baixo; ou fundir só quando os dois discordam fortemente.
</details>

---

### A3. O chunk certo que atrapalhou

Em 2 das 8 perguntas do Lab 6, o contexto CORRETO reduziu a probabilidade da resposta.

a) Liste três mecanismos que explicam isso.
b) O que isso implica para a frase "melhorei o hit@1, logo o sistema melhorou"?

<details><summary>Gabarito</summary>

a) 1) **Fraseado distante**: o chunk contém a informação com outras palavras; a resposta-alvo exata ("ln 2") pode ficar *menos* provável que uma paráfrase — a métrica de log-prob da string exata pune isso. 2) **Diluição de atenção**: 200 palavras de contexto competem com a pergunta; a resposta curta fica mais longe do fim do prompt. 3) **Conflito de formato**: o chunk pode induzir o modelo a continuar o texto do chunk em vez de responder.

b) Que recuperação e geração se avaliam SEPARADAS e o sistema, de ponta a ponta: hit@1 é condição necessária, não suficiente. Mesmo uma passagem rotulada como relevante pode ser mal utilizada pelo gerador — e a otimização real de RAG frequentemente está em COMO o chunk entra no prompt (posição, formato, compressão), não só em qual chunk.
</details>

---

### A4. A abstenção que falhou

O limiar de cosseno não separou dentro/fora da base (0,869–0,910 vs 0,778–0,874).

a) Por que cossenos de bi-encoder são maus detectores de "fora do domínio"?
b) Ordene as quatro defesas por robustez: limiar de cosseno, instrução de grounding, reranker como detector, calibração supervisionada.

<details><summary>Gabarito</summary>

a) O treino contrastivo otimiza o RANKING dentro de uma consulta, não a magnitude absoluta entre consultas. A anisotropia do espaço comprime todos os cossenos numa faixa estreita (~0,75–0,95 no e5), e "parecido em estilo" (uma pergunta em português sobre tecnologia) já pontua alto mesmo sem a resposta existir. O score compara superfícies, não responde "a resposta está aqui?".

b) Da mais fraca à mais forte: **limiar de cosseno** (falhou no lab) < **calibração supervisionada** (funciona no domínio calibrado, degrada fora) < **reranker como detector** (o cross-encoder lê pergunta+chunk juntos e pontua "este trecho responde isto?" — sinal muito mais discriminativo) < **instrução de grounding no gerador** (a última linha de defesa: mesmo com lixo no contexto, o modelo pode recusar). Em produção séria: as duas últimas juntas.
</details>

---

### A5. Dimensionando um RAG real

Sua empresa tem 40.000 documentos internos (~200M de palavras). Estime, com as contas do curso:

a) Quantos chunks e o tamanho do índice denso (e5-small, 384 dims, fp32)?
b) O custo de embedar tudo uma vez (o lab mediu 201 chunks/29s em CPU).
c) A busca por produto de matrizes ainda basta, ou precisa de índice aproximado (HNSW/FAISS)?

<details><summary>Gabarito</summary>

a) 200M palavras ÷ ~180 palavras/chunk ≈ **1,1M chunks**. Índice: 1,1M × 384 × 4 bytes ≈ **1,7 GB** — cabe folgado em RAM.

b) CPU do lab: ~7 chunks/s → 1,1M/7 ≈ 44 horas de CPU. Numa GPU modesta (~100×): ~30 min. Conclusão: embedar 1M de chunks é trabalho de UMA GPU-hora, não um projeto — o custo real é o pipeline de LIMPEZA dos 40k documentos (módulo 4 inteiro se aplica).

c) Produto de matrizes denso: 1,1M × 384 multiplicações por consulta ≈ 0,4 GFLOP — dezenas de ms em CPU, sub-ms em GPU. **Para 1M de chunks, busca exata ainda basta** para latências normais; HNSW/FAISS entram na casa de dezenas de milhões de vetores ou de milhares de consultas/s. A lição: o "banco vetorial" como infraestrutura pesada é frequentemente prematuro — comece com uma matriz.
</details>

---

## Parte B — Práticas

### B1. 💻 A curva do chunking

Rode a avaliação com `alvo_palavras` ∈ {80, 150, 220, 400} e `overlap` ∈ {0, 40}. Monte a tabela hit@1 × configuração.

Qual venceu? O overlap pagou o custo (mais chunks = índice maior)?

<details><summary>Gabarito esperado</summary>

Espere um ótimo interior (150–250 palavras) — chunks de 80 fragmentam explicações; de 400, diluem o embedding. O overlap tende a ajudar pouco AQUI (as seções dos READMEs já são fronteiras limpas) — o ganho dele aparece em texto corrido sem estrutura. Conclusão transferível: chunking por estrutura semântica > tamanho fixo, e cada corpus tem sua curva — meça a sua.
</details>

---

### B2. 💻 Estressando o índice

O hit@3 deu 100% — o corpus é pequeno e limpo demais. Degrade-o de propósito: adicione ao índice os chunks do GLOSSARIO.md e do GUIA-DE-CODIGO.md (que falam dos MESMOS temas com outras palavras) e reavalie.

O hit@1 cai? As fontes "concorrentes" roubam o top-1 de qual sistema — BM25 ou densa?

<details><summary>Gabarito esperado</summary>

O glossário pode "roubar" posições mesmo contendo respostas válidas. Antes de medir, amplie o gabarito com as passagens relevantes do novo documento; caso contrário, você estará penalizando uma recuperação útil. Isso expõe uma decisão real: o gabarito deve representar relevância para o usuário, não apenas a origem editorial preferida.
</details>

---

### B3. 🍎 Quanto vale a instrução de grounding

No lab_mlx, rode as duas medições (acerto na base + abstenção fora) em três condições: com a instrução completa, sem a frase de abstenção, e sem instrução nenhuma (só contexto + pergunta).

<details><summary>Gabarito esperado</summary>

Espere: a taxa de abstenção fora da base desabar sem a frase explícita (o modelo responde da memória paramétrica por cima do contexto irrelevante), e o acerto na base mudar pouco. Conclusão: a instrução de grounding custa zero e é a defesa de abstenção mais eficaz do sistema — mas só funciona se estiver LÁ, e é a primeira coisa que se perde em refatorações de prompt. Trate-a como código crítico, com teste de regressão (módulo 14).
</details>

---

### B4. 💻🍎 Reranker cross-encoder

Adicione o segundo estágio: recupere top-20 com a híbrida e reordene com um cross-encoder multilíngue leve (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` roda em CPU via transformers: entrada = par [pergunta, chunk], saída = score).

Meça: hit@1 antes/depois, e a latência extra por consulta.

<details><summary>Gabarito esperado</summary>

Espere o hit@1 subir alguns pontos (o cross-encoder resolve exatamente os empates que o bi-encoder não discrimina) ao custo de ~20 forwards por consulta (dezenas a centenas de ms em CPU). E teste o bônus do A4: os scores do reranker separam dentro/fora da base melhor que os cossenos? (Tipicamente, muito melhor — é o detector de abstenção que o Lab 7 não tinha.)
</details>

---

### B5. 💻 Reescrita de consulta

Implemente multi-query: para cada pergunta, gere 3 reformulações com o Qwen-0.5B ("reescreva esta pergunta de 3 formas diferentes"), busque com as 4 versões e funda com RRF.

O hit@1 melhora? Em quais perguntas — e o que elas têm em comum?

<details><summary>Gabarito esperado</summary>

Espere ganho concentrado nas perguntas cujo vocabulário diverge do texto-fonte (paráfrases distantes) e nenhum ganho — ou ruído — nas que já acertavam. O custo: 1 chamada de geração + 4 buscas por consulta. É o padrão geral de query rewriting (incluindo HyDE): ajuda na cauda difícil, é desperdício no caso fácil — e o roteamento por dificuldade (módulo 7, A5) reaparece pela terceira vez no curso.
</details>

---

## Desafio — o assistente do SEU domínio

Repita o pipeline completo sobre uma base SUA (documentação interna, notas, contratos — o que você de fato consultaria):

1. Ingestão + chunking por estrutura, com metadados de fonte.
2. Conjunto de avaliação: 20+ perguntas com fonte conhecida, escritas ANTES de otimizar qualquer coisa.
3. BM25 × densa × (híbrida SE a matriz de concordância do A2 justificar).
4. As duas medições de geração do lab_mlx: acerto na base + abstenção fora.
5. Relatório no padrão do módulo 12 — incluindo a seção "o que não funcionou".

Critério de pronto: você respondeu com o sistema, por uma semana, às perguntas que antes respondia lendo os documentos — e registrou quando ele falhou e por quê. RAG que não substitui a leitura manual em NADA ainda é demo.
