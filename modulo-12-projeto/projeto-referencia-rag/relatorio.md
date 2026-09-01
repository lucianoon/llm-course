# Relatório técnico — assistente de políticas internas fictícias

## 1. Problema e métrica

Usuários precisam localizar regras em um manual curto sem receber respostas inventadas. A
métrica foi definida antes da implementação: uma pergunta respondível só conta como correta
quando a passagem contém os termos de ouro **e** cita o documento esperado; uma pergunta fora
da base só conta quando o sistema se abstém.

## 2. Baseline

A baseline escolhe o documento com maior sobreposição entre a pergunta e o **título**. É uma
solução executável e plausível para um FAQ pequeno, mas não possui abstenção. Baseline e sistema
final recebem exatamente as mesmas perguntas e são avaliados pela mesma função.

## 3. Dados

O corpus tem dez políticas fictícias e licença CC0-1.0. A avaliação tem quinze perguntas,
incluindo quatro fora da base. Nenhum dado pessoal real é usado; `example.org` é um domínio
reservado para documentação. `preparar.py` verifica IDs, licença e separação, e grava SHA-256,
origem e contagens em `dataset-manifest.json`.

O teste não foi usado para treinar. O limiar foi escolhido como valor pedagógico simples e deve
ser calibrado em um conjunto de validação separado antes de qualquer uso real.

## 4. Método

O sistema remove uma lista fixa de stopwords, indexa título e texto com BM25 (`k1=1.5`,
`b=0.75`), recupera duas passagens e usa a primeira quando o score é pelo menos 1.0. A resposta é extrativa, preservando a fonte. SFT,
LoRA e DPO foram descartados porque a falha é conhecimento externo mutável, não comportamento.

## 5. Resultados

Execute `scripts/avaliar.py` para regenerar a tabela estruturada em `resultados.json`. O arquivo
inclui resultados pareados por pergunta e IC bootstrap do delta. Com apenas quinze exemplos, o
intervalo é necessariamente largo; a conclusão está limitada a este corpus.

A inspeção qualitativa deve observar três classes: paráfrases recuperadas, termos ambíguos e
perguntas sem evidência. Os detalhes por ID ficam no JSON para impedir que a média esconda erros.

## 6. Serving

`scripts/servir.py` oferece uma CLI com contrato JSON. `scripts/servidor_http.py` acrescenta
health check, autenticação por chave, limite de payload, validação, request ID e logs JSON sem o
texto da pergunta. `scripts/carga.py` executa 45 requisições com concorrência 8 e regenera
`carga-resultados.json`. Tokens e custo são aproximações claramente rotuladas. O teste continua
local e não inclui rede externa ou tempo de um gerador; não é benchmark de produção.

## 7. O que não funcionou

A baseline por título responde mesmo sem evidência e não oferece um mecanismo defensável de
abstenção. Ela também perde paráfrases que aparecem no corpo, mas não no título. A primeira ideia
de simplesmente usar “algum termo em comum” como limiar foi descartada: stopwords e palavras
genéricas criavam confiança falsa. O score BM25 é melhor para este exemplo, mas ainda não é uma
probabilidade calibrada.

## 8. Limitações e próximos passos

O corpus é artificial, curto e sem controle de acesso. Próximos passos reais seriam: criar split
de validação para calibrar o limiar, ampliar paráfrases, medir MRR/recall do retrieval, adicionar
reranker, testar atualização incremental, verificar citações por sentença, implementar ACL por
documento e executar um servidor HTTP sob carga com observabilidade.
