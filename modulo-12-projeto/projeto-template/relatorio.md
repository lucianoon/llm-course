# Relatório técnico

Esqueleto seguindo o formato do módulo 12. Substitua cada seção pelo seu conteúdo.

## 1. Problema e métrica

Uma frase: que comportamento muda para quem? Métrica escolhida — e como ela foi
**testada contra um exemplo de ouro** antes de medir qualquer modelo.

## 2. Baselines

- Prompt simples (por extenso).
- Prompt esforçado (por extenso).
- RAG, se couber.

A comparação só é justa com o MESMO teste e a MESMA métrica.

## 3. Dados

Origem, licença, coleta, dedup, filtros, contagens por etapa, splits sem vazamento,
auditoria de confundimentos. Proveniência completa (módulo 4).

## 4. Método

Técnica escolhida **contra** as alternativas. Hiperparâmetros com justificativa —
por que este valor, não outro.

## 5. Resultados

Tabela principal, curvas, exemplos reais (bons E ruins), efeitos colaterais medidos.

## 6. Serving

Formato final, TTFT/TPOT, custo por milhão de tokens, ponto de operação, guardião de
custo e disjuntor (módulo 19). Extrato p50/p95 e taxa de sucesso.

## 7. O que não funcionou — OBRIGATÓRIA

Tentativas descartadas e porquê. É o que separa engenharia de marketing.

## 8. Limitações e próximos passos
