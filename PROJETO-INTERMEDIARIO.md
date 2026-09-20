# Mini-projeto intermediário

Faça este projeto depois dos módulos 4–6 e antes do projeto final. Ele deve caber em **3–5
horas** e serve para ensaiar o ciclo completo sem exigir um modelo grande.

## Desafio

Escolha uma tarefa pequena de classificação ou extração estruturada em português. Exemplos:

- classificar solicitações por categoria;
- extrair campos de um documento curto;
- detectar se uma pergunta deve ser respondida, encaminhada ou recusada.

Use 30–60 exemplos anonimizados ou sintéticos revisados manualmente. Não use dados pessoais
ou material sem licença clara.

## Experimentos obrigatórios

Compare, no mesmo conjunto lacrado:

1. prompt direto;
2. prompt com exemplos;
3. RAG, somente se houver conhecimento externo;
4. SFT/LoRA, somente se a falha for de formato ou comportamento.

Não é obrigatório que fine-tuning vença. A conclusão pode ser que a solução mais simples é
melhor.

## Entrega

- `README.md` com problema, usuário e decisão;
- dataset de exemplo e manifesto de origem;
- script que prepara e avalia;
- tabela com acurácia/F1 ou métrica adequada, custo e tempo;
- cinco acertos e cinco erros comentados;
- limitação principal e próximo experimento.

## Gate

O mini-projeto está pronto quando outra pessoa consegue executar o comando principal, entender
por que a baseline foi escolhida e identificar pelo menos um caso em que o sistema deve recusar.
