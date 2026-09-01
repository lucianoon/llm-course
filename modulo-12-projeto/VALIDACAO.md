# Validação do projeto final

Este roteiro é o protocolo de aceite do módulo 12. Ele está escrito como uma **revisão
técnica de indústria** — o que um engenheiro sênior olharia antes de aprovar uma entrega
— e não como uma defesa de tese. A diferença importa: banca premia quem *defende*;
tech review premia quem *entrega o que outra pessoa consegue reproduzir*.

O diretório do projeto deve conter:

```text
projeto/
  README.md
  config.json
  dataset-manifest.json
  model-card.md
  relatorio.md
  resultados.json
  scripts/
    preparar.py
    treinar.py
    avaliar.py
    servir.py
```

`resultados.json` segue o schema v1 de `tools/reproducao.py`: `metricas` precisa declarar
`baseline`, `modelo_final`, a métrica e o intervalo de confiança quando aplicável; o registro
também inclui `amostra_n`, commit, estado da árvore Git, revisões, seed e hardware. O revisor deve
conseguir regenerá-lo executando os scripts, sem editar caminhos à mão.

## Como um revisor técnico olha para isso

Um engenheiro sênior não pergunta "o que você fez?", pergunta as cinco coisas que decidem
se ele daria o seu OK:

1. **Por que esta solução?** — qual falha ela resolve, e por que prompt/RAG/treino aqui?
2. **Como sabemos que funciona?** — baseline justa, métrica testada em exemplo de ouro,
   leitura manual de erros. Sem isso, o número é anedota.
3. **Quanto custa?** — latência (p50/p95), throughput, custo por milhão de tokens, e a
   camada de produção do módulo 19 (guardião/disjuntor) para não estourar.
4. **Como falha?** — casos conhecidos de erro, comportamento de recusa, e o que NÃO
   funcionou documentado. Golpe baixo é a matriz de uma entrega honesta.
5. **Outra pessoa reproduz?** — ambiente, comandos, manifestos de dados e modelo,
   `resultados.json` regenerável. Se não, é um slide, não um projeto.

É o mesmo código de ética do curso aplicado à entrega: **nada sobe sem um número, um teste
e um jeito fácil de cair.**

## Rubric de revisão técnica (100 pontos)

| Critério | Pontos | Falha eliminatória |
|---|---:|---|
| Problema e métrica definidos antes do treino | 10 | métrica não representa o produto |
| Governança, licença, PII e split | 15 | uso sem autorização ou vazamento |
| Baseline esforçada | 15 | baseline deliberadamente fraca |
| Método e hiperparâmetros | 10 | decisão sem justificativa |
| Reprodutibilidade | 15 | resultado não regenerável |
| Avaliação quantitativa e IC | 10 | conjunto de teste usado para otimizar |
| Inspeção qualitativa e efeitos colaterais | 10 | nenhuma leitura de erros |
| Serving, latência, memória e custo | 10 | só mede qualidade offline |
| Limitações e comunicação | 5 | omite resultado negativo material |

O projeto reprova independentemente da nota se violar uma falha eliminatória.
