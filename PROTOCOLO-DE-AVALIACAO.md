# Protocolo padrão de avaliação

Use este protocolo no mini-projeto e no projeto final. Ele evita que cada projeto escolha uma
régua diferente depois de ver os resultados.

## Conjunto mínimo

Separe antes de ajustar o sistema:

| Conjunto | Tamanho inicial | Uso |
|---|---:|---|
| Desenvolvimento | 20–40 casos | corrigir prompts e código |
| Teste lacrado | 20–40 casos | resultado principal |
| Segurança/regressão | 10–20 casos | vazamento, abuso, recusa e formato |

O conjunto de teste não deve ser usado para escolher hiperparâmetros. Registre idioma,
subgrupo, dificuldade, origem, licença e critérios de anotação.

## Comparações obrigatórias

1. baseline simples, sem técnica sofisticada;
2. melhor prompt razoável para a baseline;
3. sistema proposto;
4. ablação da parte que você afirma ser responsável pelo ganho.

Use o mesmo conjunto, seed quando aplicável e modo de decoding que será usado em produção.

## Relatório mínimo

- métrica principal e por que representa o usuário;
- intervalo de confiança ou bootstrap quando a métrica for amostral;
- resultados por subgrupo relevante;
- custo, latência e taxa de erro;
- cinco exemplos bons e cinco ruins;
- casos em que o sistema deve se abster;
- efeitos colaterais e limitações;
- decisão: manter, voltar à baseline ou coletar dados melhores.

Para geração livre, não trate “parece melhor” como métrica suficiente. Combine avaliação
automática, amostra manual e um protocolo de juiz auditado quando usar LLM-as-judge.
