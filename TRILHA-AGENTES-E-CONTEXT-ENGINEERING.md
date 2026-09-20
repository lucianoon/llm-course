# Extensão: agentes e context engineering

Esta extensão acompanha os módulos 13, 15 e 19. Ela trata contexto como uma superfície de
engenharia: instruções, dados recuperados, ferramentas, memória, estado e permissões precisam
ser projetados, medidos e limitados.

## Rota de quatro semanas

| Semana | Tema | Entregável |
|---|---|---|
| 1 | Contexto: seleção, ordenação, compressão e orçamento de tokens | mapa de contexto e orçamento por chamada |
| 2 | Ferramentas e contratos: schemas, erros, idempotência e menor privilégio | duas ferramentas com testes de contrato |
| 3 | Agentes: estado, loops, roteamento e fallback determinístico | agente com limite de passos e trilha auditável |
| 4 | Avaliação e segurança: injeção indireta, custo, latência e regressão | conjunto de ataques, métricas e gate de CI |

## Gate de conclusão

O projeto precisa demonstrar:

- que cada trecho do contexto tem origem e finalidade identificáveis;
- que o agente pode recusar, pedir confirmação ou usar um fallback;
- que ferramentas destrutivas exigem autorização explícita;
- que entradas externas são tratadas como dados, nunca como instruções confiáveis;
- que custo, latência, taxa de erro e sucesso da tarefa são medidos separadamente;
- que uma regressão de segurança impede a entrega.

## Relação com o curso

- Módulo 13 fornece recuperação e evidência documental.
- Módulo 15 fornece loops, tool use e modos de falha.
- Módulo 14 fornece a disciplina de avaliação.
- Módulo 19 transforma o protótipo em serviço observável, limitado e reversível.

