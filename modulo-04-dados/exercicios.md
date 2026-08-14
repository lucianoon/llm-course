# Módulo 4 — Exercícios

---

## Parte A — Conceituais

### A1. Sintonizando o LSH

Você quer deduplicar 50 milhões de documentos com limiar de similaridade `J ≥ 0,85`, usando assinaturas MinHash de `k = 128`.

a) Escolha `b` e `r` (com `b·r = 128`) cujo joelho da curva S fique perto de 0,85.
b) Com essa escolha, qual a probabilidade de um par com `J = 0,80` virar candidato? E com `J = 0,90`?
c) Se você errar para o lado de `b` grande demais, o que acontece — e por que isso pode ser aceitável?

<details><summary>Gabarito</summary>

a) O joelho fica em `J ≈ (1/b)^(1/r)`. Testando `b·r = 128`:
- `b=8, r=16` → `(1/8)^(1/16) = 0,878`
- `b=16, r=8` → `(1/16)^(1/8) = 0,707`
- `b=4, r=32` → `(1/4)^(1/32) = 0,957`

**`b=8, r=16`** é a melhor escolha (joelho em 0,878, próximo de 0,85).

b) `P = 1 − (1 − J^r)^b`:
- `J=0,80`: `1 − (1 − 0,80^16)^8 = 1 − (1 − 0,0281)^8 ≈ **20,3%**`
- `J=0,90`: `1 − (1 − 0,90^16)^8 = 1 − (1 − 0,1853)^8 ≈ **80,4%**`

A transição é acentuada — exatamente o que se quer.

c) `b` maior desloca o joelho para a esquerda: mais pares viram candidatos, incluindo muitos com similaridade baixa. Isso aumenta o custo (mais pares para verificar exatamente) mas **não** produz falsos positivos finais, porque a etapa seguinte calcula o Jaccard real e descarta os que não passam do limiar. Errar para o lado permissivo custa compute; errar para o lado restritivo custa **duplicatas não detectadas**, que são irrecuperáveis. Prefira o lado caro.
</details>

---

### A2. O filtro que destrói

Você recebe um corpus multilíngue (inglês, português, alemão, japonês) e aplica os filtros do Gopher com os parâmetros originais. Preveja o que acontece com cada idioma e proponha correções.

<details><summary>Gabarito</summary>

| Idioma | O que quebra | Correção |
|---|---|---|
| Inglês | Nada — os limiares foram calibrados nele | — |
| Português | A lista de stop words elimina ~100% (medido: 99,8%) | Lista de stop words em português |
| Alemão | Comprimento médio de palavra frequentemente > 10 (compostos) | Elevar o limite superior, ou calibrar por idioma |
| Japonês | Sem espaços: `texto.split()` conta o documento inteiro como 1–2 "palavras", falhando no mínimo de 50 e no comprimento médio | Segmentação morfológica (MeCab/SudachiPy) antes de qualquer contagem, e limiares em caracteres, não palavras |

O princípio geral: **detecte o idioma primeiro, depois aplique filtros calibrados por idioma.** É o que FineWeb-2 e CulturaX fazem. Aplicar um único conjunto de regras a um corpus multilíngue não filtra qualidade — filtra idioma.
</details>

---

### A3. A loss que engana

Dois times treinam modelos para a mesma tarefa. O time A reporta loss final de treino de 1,2; o time B, de 2,8. O time A declara vitória.

Que informação está faltando, e por quê?

<details><summary>Gabarito</summary>

Falta a avaliação **num conjunto fixo e comum aos dois**. A loss de treino mede a previsibilidade do corpus de cada time, não a qualidade dos modelos.

O Lab 7 mediu exatamente esse cenário: o modelo treinado no corpus poluído teve loss de treino **3,00** contra 5,34 do corpus limpo — e foi 7,8% **pior** na validação. Corpus repetitivo tem loss baixa porque repetição é fácil de prever.

Cenários que produzem loss de treino baixa e modelo ruim:
- Corpus com muita duplicação (o modelo já viu aquilo)
- Corpus repetitivo ou de baixa entropia (boilerplate, logs, templates)
- Overfitting (muitas épocas em pouco dado)
- Tokenizer diferente (loss por token não é comparável entre vocabulários — módulo 1)

A pergunta certa é sempre: *qual a performance no mesmo conjunto de teste, com o mesmo tokenizer?* — e, idealmente, numa tarefa final, não em perplexidade.
</details>

---

### A4. Auditoria de licença

Você vai treinar um assistente comercial para uma empresa. Sua equipe propõe misturar:

1. 50k exemplos do Alpaca
2. 10k exemplos anotados internamente por funcionários
3. 30k exemplos gerados com a API do GPT-4
4. 20k exemplos do OpenAssistant (Apache 2.0)
5. 15k conversas reais de suporte da empresa

Aponte os problemas de cada fonte antes de aprovar.

<details><summary>Gabarito</summary>

1. **Alpaca — bloqueado.** CC BY-NC 4.0 (não comercial) *e* gerado com `text-davinci-003`, o que atrai a restrição de uso das saídas da OpenAI. Duas barreiras independentes.
2. **Interno — ok**, desde que os funcionários tenham cedido os direitos (normalmente coberto pelo contrato de trabalho) e não haja PII de terceiros nos exemplos.
3. **GPT-4 — bloqueado** para treinar um modelo que compita com a OpenAI. A leitura do que constitui "competir" é ampla e o risco é jurídico, não técnico.
4. **OpenAssistant — ok.** Apache 2.0, dados humanos. É a fonte mais segura da lista.
5. **Suporte real — ok tecnicamente e a melhor fonte de todas** (é a distribuição real!), mas exige: base legal LGPD para uso secundário, anonimização de PII (nomes, CPF, endereços, números de pedido), e verificação de que não há dados sensíveis. Anonimizar mal é pior que não usar — modelos memorizam e regurgitam.

**Recomendação:** construa sobre 2, 4 e 5. Se precisar de volume sintético, gere com um modelo de licença permissiva (Qwen, Llama sob sua licença própria, Mistral Apache 2.0) e documente a proveniência de cada exemplo desde o início. Auditar proveniência retroativamente é quase impossível.
</details>

---

### A5. Quantidade ou qualidade

Você tem 5.000 exemplos coletados às pressas para um assistente de atendimento. Um primeiro fine-tuning produz um modelo que às vezes responde bem, às vezes com o tom errado, às vezes inventa políticas.

Você tem uma semana. Coletar mais 5.000 ou limpar os 5.000 que já tem?

<details><summary>Gabarito</summary>

**Limpar** — e a evidência é o LIMA (1.000 exemplos curados batendo 52.000 sintéticos).

A inconsistência descrita é o sintoma clássico de dataset heterogêneo: o modelo aprendeu a distribuição dos seus dados, e essa distribuição **inclui** o tom errado e as políticas inventadas. Dobrar o volume com a mesma qualidade dobra a exposição ao mesmo problema.

Plano para a semana:
1. **Dia 1** — amostre 100 exemplos ao acaso e leia todos. Sem essa etapa você está adivinhando. Categorize os defeitos.
2. **Dias 2–3** — dedup (exata e near-dup), remova respostas vazias ou truncadas, remova exemplos com o tom errado.
3. **Dia 4** — pontue o resto (heurística ou LLM-as-judge) e fique com o topo.
4. **Dia 5** — corrija à mão os 200 exemplos mais representativos, definindo o padrão-ouro de tom e formato.
5. **Dias 6–7** — retreine com 3 tamanhos (500 melhores, 1.500, todos) e compare numa avaliação fixa.

O passo 5 responde empiricamente a pergunta original, e o passo 1 é o que quase ninguém faz.
</details>

---

## Parte B — Práticas

### B1. Deduplicação completa do Alpaca

Rode MinHash + LSH sobre os 52.002 exemplos (não só a amostra de 4.000). Reporte:

a) quantos near-duplicates com `J ≥ 0,7`;
b) o tempo de execução e como ele escala;
c) o efeito de deduplicar por `instruction` apenas contra `instruction + input`.

<details><summary>Gabarito</summary>

Espere alguns milhares de pares em `J ≥ 0,7` — o Self-Instruct gera variações da mesma tarefa com frequência.

O tempo do MinHash é linear em documentos; o do LSH é aproximadamente linear, mas o número de **pares candidatos** pode explodir se um balde ficar muito grande (um documento genérico que colide com milhares). A proteção padrão é limitar o tamanho do balde.

Deduplicar por `instruction` apenas encontra tarefas repetidas; por `instruction + input` encontra contextos repetidos e é dominado pelo texto longo do `input`. Para SFT, o mais útil costuma ser deduplicar por **instrução**, mantendo variedade de contextos para a mesma tarefa — mas isso depende do seu objetivo, e a decisão deve ser consciente.
</details>

---

### B2. Filtros calibrados para português

Reescreva `filtro_gopher` com limiares apropriados ao português. Considere: stop words, comprimento médio de palavra (o português é mais longo que o inglês), e uma regra nova para excesso de maiúsculas.

Valide contra três corpora: Machado (deve passar), o lixo do Lab 7 (deve reprovar), e textos de redes sociais (o que deveria acontecer?).

<details><summary>Gabarito</summary>

Ajustes principais: stop words em português; comprimento médio de palavra entre 3,5 e 11 (o português tem palavras mais longas que o inglês); tolerância maior a linhas curtas (diálogo literário).

Sobre redes sociais: **não existe resposta certa** — depende do objetivo. Se você quer um modelo formal, esse texto é lixo. Se quer um modelo que entenda linguagem informal brasileira, filtrá-lo destrói exatamente o que você precisa.

É a lição central do módulo: **"qualidade" não é uma propriedade do texto, é uma relação entre o texto e o seu objetivo.** Todo filtro embute uma definição de qualidade, e é melhor que essa definição seja explícita.
</details>

---

### B3. Filtragem por perplexidade

Use o Qwen2.5-0.5B (módulo 1) para calcular a perplexidade de cada documento do corpus poluído do Lab 7. Ordene e inspecione as duas caudas.

a) O que há na cauda de PPL alta? E na de PPL baixa?
b) Compare a eficácia com os filtros heurísticos: qual pega mais lixo? Qual descarta menos texto bom?

<details><summary>Gabarito</summary>

a) **PPL alta:** gibberish, hashes, texto corrompido, idiomas fora do treino do modelo. **PPL baixa:** boilerplate repetido, avisos de cookies, listas de navegação — texto altamente previsível.

É por isso que se descartam as **duas** caudas. Filtrar só a alta deixa passar todo o boilerplate, que é o lixo mais comum na web.

b) A filtragem por perplexidade costuma ser mais precisa (pega lixo semântico que heurísticas não veem) e é muito mais cara: exige um forward por documento. O padrão da indústria é heurísticas primeiro, para eliminar o grosso por quase nada, e perplexidade ou classificador depois, sobre o que sobrou.

Uma ressalva importante: filtrar por perplexidade de um modelo enviesa o corpus **em direção àquele modelo**. Você acaba mantendo o texto que o modelo já acha provável, o que reduz a diversidade e pode reforçar os vieses do modelo filtrador.
</details>

---

### B4. Curva de dados

Usando o corpus limpo do Lab 7, treine o MiniGPT com 10%, 25%, 50% e 100% dos dados, mantendo **os passos constantes** (mesmo compute). Plote loss de validação × fração de dados.

Onde está o joelho? O que acontece com o gap treino-validação?

<details><summary>Gabarito</summary>

Espere ganhos claros até ~50% e rendimentos decrescentes depois. Com 10% dos dados e os mesmos passos, o modelo faz muitas épocas sobre pouco texto: a loss de treino cai bastante e a de validação estaciona alta — o gap explode.

A lição: com compute fixo, mais dados **únicos** sempre ajudam, mas de forma logarítmica. E o gap treino-validação é o indicador mais confiável de que você está no regime de memorização — mais confiável que a loss de treino sozinha, que só melhora.
</details>

---

### B5. Seleção de qualidade, medida

Implemente a lição do LIMA no seu MiniGPT:

1. Pontue todos os documentos do corpus poluído com uma heurística sua.
2. Treine com os 20% melhores, com 20% aleatórios, e com 100%.
3. Compare na mesma validação limpa, com os mesmos passos.

Os 20% melhores batem os 100%?

<details><summary>Gabarito</summary>

Espere: **20% melhores > 100% > 20% aleatórios**.

A comparação com "20% aleatórios" é o controle que torna o experimento válido — sem ele, você não sabe se o ganho veio da *seleção* ou apenas de ter menos lixo por acaso. É o erro metodológico mais comum em experimentos de curadoria: comparar "meus dados selecionados" com "todos os dados" e concluir que a seleção funcionou, quando qualquer subconjunto do mesmo tamanho teria dado o mesmo resultado.

Sempre inclua o controle aleatório de mesmo tamanho.
</details>

---

## Desafio — pipeline completo

Construa um pipeline de curadoria de ponta a ponta para um dataset de SFT em português, partindo do Alpaca traduzido (ou de qualquer dataset de instruções em português que você encontre).

Etapas obrigatórias:

1. Deduplicação exata e near-dup (MinHash + LSH)
2. Filtros heurísticos calibrados para português
3. Detecção e remoção de respostas degeneradas (vazias, truncadas, repetitivas)
4. Pontuação de qualidade e seleção
5. Separação de um split de teste **antes** de tudo, deduplicado contra o treino
6. Formatação final em ChatML com masking e EOS verificados
7. Um relatório com: quantos exemplos entraram e saíram de cada etapa, e por quê

O item 7 é o entregável mais importante. Um pipeline de dados sem relatório de proveniência é irreproduzível — e quando o modelo sair ruim, você não terá como saber qual etapa foi a culpada.

<details><summary>Critérios de avaliação</summary>

Um bom pipeline deve:

- Ser **idempotente** (rodar duas vezes dá o mesmo resultado) e ter seed fixa em qualquer amostragem.
- Registrar contagens em **cada** etapa, não apenas entrada e saída.
- Separar o split de teste **antes** da deduplicação — senão você deduplica o teste contra si mesmo e mantém vazamentos do treino.
- Tratar o EOS e o masking explicitamente, com uma asserção que falha se estiverem errados.
- Guardar exemplos **rejeitados** por amostragem, para auditoria. Você vai querer saber o que jogou fora.
- Não descartar silenciosamente: se uma etapa remove mais de 50%, isso merece um aviso no log.

Se o seu pipeline não permite responder "por que este exemplo específico foi removido?", ele ainda não está pronto.
</details>
