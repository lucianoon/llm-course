# Política de segurança

## Versões mantidas

Durante a beta pública, somente a branch `main` recebe correções de segurança.

## Como relatar uma vulnerabilidade

Não abra uma issue pública com credenciais, dados pessoais, detalhes de exploração ou uma
vulnerabilidade ainda não corrigida. Use o recurso **Report a vulnerability** na aba Security do
GitHub. Se o recurso não estiver disponível, abra uma issue sem detalhes sensíveis pedindo um
canal privado de contato.

Inclua, quando possível:

- componente e versão ou commit afetado;
- impacto e pré-condições;
- passos mínimos para reprodução, sem dados reais;
- correção ou mitigação sugerida.

O mantenedor fará uma triagem inicial assim que possível. Este projeto educacional não promete
um SLA formal, mas prioriza vulnerabilidades que possam expor dados, executar código ou gerar
custos externos.

## Escopo

São de especial interesse: downloads de dados e modelos, desserialização, execução de comandos,
tratamento de credenciais, SSRF, logs com PII e exemplos que alunos possam copiar para produção.

Não envie segredos reais em provas de conceito.

## Limitações conhecidas de dependências

O extra opcional `serving` usa vLLM em um ambiente Linux/GPU isolado. Na versão atualmente
resolvida, vLLM exige `setuptools < 81`, enquanto `GHSA-h35f-9h28-mq5c` é corrigida somente em
`setuptools >= 83`. O problema afeta a criação de distribuições-fonte com nomes Unicode em
macOS; este repositório não publica pacotes e o ambiente de vLLM é Linux. Não use esse ambiente
para construir ou publicar pacotes. A restrição deve ser removida assim que vLLM aceitar uma
versão corrigida.
