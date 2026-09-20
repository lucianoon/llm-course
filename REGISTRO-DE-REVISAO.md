# Registro de revisão de afirmações

O conteúdo de LLM muda rapidamente. Use esta tabela para distinguir uma regra estrutural de
uma recomendação que precisa ser revisada.

| Afirmação | Arquivo/seção | Fonte ou experimento | Revisado em | Escopo/limitação |
|---|---|---|---|---|
| | | | | |

## Quando registrar

Registre toda afirmação que contenha “padrão atual”, “melhor”, “popular”, um número de custo,
memória, latência, qualidade, versão de biblioteca ou recomendação de modelo.

## Como escrever

- prefira uma fonte primária ou um experimento reproduzível;
- fixe modelo, versão, hardware e data;
- troque “sempre” por uma condição observável;
- declare quando o resultado é uma heurística;
- mova números históricos para [`EVIDENCIAS.md`](EVIDENCIAS.md) quando perderem validade.

Uma afirmação sem fonte nem experimento deve ser apresentada como hipótese, não como fato.
