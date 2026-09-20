# Certificação e badge

A certificação do curso comprova uma entrega reproduzível; não é apenas a conclusão de
leituras. O artefato principal é um repositório ou diretório de projeto que outra pessoa
consegue executar seguindo o README.

## Rubrica pública

| Critério | Peso | Mínimo |
|---|---:|---:|
| Problema, baseline e conjunto de teste | 20% | baseline executada e teste sem vazamento |
| Dados e governança | 15% | origem, licença, checksum e auditoria de PII |
| Implementação e reprodutibilidade | 20% | comando único, versões fixadas e seed registrada |
| Avaliação e incerteza | 20% | métricas antes/depois, tamanho da amostra e limitações |
| Operação e segurança | 15% | custo/latência, logs sem PII, limites e rollback |
| Comunicação e revisão | 10% | relatório claro e revisão técnica de outro projeto |

Para aprovação, a nota deve ser pelo menos 70/100 e nenhum dos quatro primeiros critérios pode
ficar abaixo de 50% do seu peso. Um projeto sem baseline ou sem teste reproduzível não é
certificável, mesmo que a demo pareça funcionar.

## Pacote de evidências

```text
projeto/
├── README.md              # objetivo e comando de reprodução
├── relatorio.md           # hipótese, método, resultados e limites
├── dados/manifesto.md     # origem, licença, checksum e PII
├── resultados/            # métricas e saídas versionadas
├── avaliacao/              # casos, baseline e análise de erro
└── ambiente/               # Python, lockfile, hardware e seed
```

## Badge verificável

Depois da aprovação, publique um badge apontando para o relatório ou para a revisão do projeto:

```markdown
[![LLM Course — projeto certificado](https://img.shields.io/badge/LLM%20Course-projeto%20certificado-2ea44f)](LINK_PARA_O_RELATORIO)
```

O badge não deve apontar apenas para uma imagem solta. O link precisa permitir conferir o
projeto, a rubrica, a data, a versão do curso e quem fez a revisão.

