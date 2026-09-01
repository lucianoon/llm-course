# Model card — `extrativo-bm25-v1`

## Resumo

Sistema sem modelo generativo: BM25 recupera uma política e devolve a passagem original com
citação. Se o score ficar abaixo do limiar, o sistema se abstém.

## Uso pretendido

- ensino de RAG, avaliação e reprodutibilidade;
- perguntas sobre o corpus fictício incluído;
- demonstração local, sem decisões reais sobre pessoas.

## Usos fora do escopo

- políticas reais de RH, jurídico ou segurança;
- resposta sobre documentos não presentes no corpus;
- aconselhamento ou automação de decisões;
- avaliação de qualidade de modelos generativos.

## Dados

Dez políticas sintéticas sob CC0-1.0 e quinze perguntas de avaliação. Checksums e contagens
ficam em `dataset-manifest.json`.

## Métricas

Acurácia exige documento correto, termos esperados e citação; em perguntas fora da base exige
abstenção. O conjunto é pequeno e não sustenta generalização.

## Riscos

- correspondência lexical falha em paráfrases distantes;
- o score BM25 não é probabilidade calibrada;
- uma passagem recuperada pode conter informação desatualizada;
- conteúdo malicioso no corpus não é neutralizado;
- a resposta extrativa pode conter texto além do necessário.

## Mitigações

Limiar de abstenção, citações obrigatórias, corpus versionado, teste fora da base e proibição
explícita de uso decisório.
