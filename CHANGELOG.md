# Changelog

Todas as mudanças relevantes deste projeto serão documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e o projeto usa
[Versionamento Semântico](https://semver.org/lang/pt-BR/) para releases públicas.

## [Não publicado]

### Adicionado

- documentação para contribuição, segurança e citação;
- licença Apache-2.0 e aviso de atribuição;
- caminho de início rápido para a beta pública.
- matriz de ambientes com comandos de instalação por rota;
- testes de contrato para a estrutura dos módulos e dos laboratórios.
- validador reutilizável de sintaxe e `--dry-run` para os laboratórios.
- correções de dependências para `accelerate` e `setuptools` vulneráveis.
- setup manual alinhado aos pisos de segurança do lockfile.

### Alterado

- reposicionamento do README para engenharia e customização de LLMs em português;
- separação explícita entre labs disponíveis, executados e reproduzidos.
- onboarding com escolha rápida de trilha e instalação mínima de CPU.
- CI executando os contratos baratos de todos os laboratórios.

## [0.1.0] - 2026-08-31

### Adicionado

- Fase 0, 19 módulos, trilhas de engenharia e pesquisa e infraestrutura compartilhada de labs;
- CI em Linux e macOS para Python 3.11 e 3.12.

[Não publicado]: https://github.com/lucianoon/llm-course/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lucianoon/llm-course/releases/tag/v0.1.0
