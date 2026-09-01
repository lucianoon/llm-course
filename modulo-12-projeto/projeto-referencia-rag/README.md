# Capstone de referência — assistente de políticas internas

Este projeto demonstra, de ponta a ponta, o contrato do módulo 12 com um sistema pequeno,
auditável e executável sem downloads. Ele responde perguntas sobre políticas **fictícias**,
cita o documento usado e se abstém quando a base não sustenta uma resposta.

O objetivo não é demonstrar geração de linguagem. É demonstrar a disciplina do projeto:
problema → dados → baseline → escolha de técnica → avaliação → serving → limitações.

## Resultado que precisa ser demonstrado

> A busca BM25 com abstenção deve superar a baseline de correspondência por título na
> acurácia do conjunto de teste, sem responder às perguntas fora da base.

`resultados.json` é gerado pelo script de avaliação; não é um número digitado no README.

## Reproduzir do zero

Na raiz do repositório, com Python 3.11 ou 3.12:

```bash
python modulo-12-projeto/projeto-referencia-rag/scripts/preparar.py
python modulo-12-projeto/projeto-referencia-rag/scripts/treinar.py
python modulo-12-projeto/projeto-referencia-rag/scripts/avaliar.py
python modulo-12-projeto/projeto-referencia-rag/scripts/servir.py "Qual o prazo para pedir férias?"
python modulo-12-projeto/projeto-referencia-rag/scripts/servir.py "A empresa oferece estacionamento?"
python modulo-12-projeto/projeto-referencia-rag/scripts/carga.py
```

O primeiro comando valida o corpus e grava `dataset-manifest.json`; o segundo registra por
que nenhum treino foi usado; o terceiro regenera `resultados.json`; o quarto mostra uma
resposta citada; o quinto deve se abster; e o último sobe um servidor HTTP efêmero e produz
`carga-resultados.json` com 45 requisições concorrentes.

Para manter o servidor HTTP aberto manualmente:

```bash
CAPSTONE_API_KEY="troque-esta-chave" python \
  modulo-12-projeto/projeto-referencia-rag/scripts/servidor_http.py --porta 8080
```

O endpoint é `POST /v1/responder`, recebe `{"pergunta":"..."}` e exige o header
`X-API-Key`. `GET /health` não exige autenticação.

Para validar junto com o curso:

```bash
python -m pytest
```

## Estrutura

```text
projeto-referencia-rag/
  README.md
  config.json
  dataset-manifest.json       # regenerado por preparar.py
  model-card.md
  relatorio.md
  resultados.json             # regenerado por avaliar.py
  dados/
    corpus.jsonl               # políticas fictícias
    avaliacao.jsonl            # teste mantido separado do corpus
  scripts/
    sistema.py                 # baseline, RAG e contrato de resposta
    preparar.py                # validação, PII e checksums
    treinar.py                 # decisão explícita de não treinar
    avaliar.py                 # métrica, IC pareado, custo e latência
    servir.py                  # CLI de serving
    servidor_http.py           # API com auth, limites e logs sem conteúdo
    carga.py                   # carga concorrente e probe de autenticação
```

## Contrato da resposta

Uma resposta bem-sucedida sempre contém:

- `status="respondido"`;
- texto sustentado por uma passagem recuperada;
- `citacoes` com o ID do documento;
- score de recuperação;
- custo estimado, identificado como aproximação.

Se nenhuma passagem ultrapassar o limiar, o status é `abstencao` e não há resposta inventada.

## Decisão técnica

O problema é falta de conhecimento externo, pequeno e mutável. RAG é apropriado; SFT não
adicionaria fatos de forma auditável e exigiria retreino a cada mudança. Como este exemplo não
usa um modelo gerador, `treinar.py` preserva a decisão “sem treino” em vez de simular uma etapa.

## Limites

- O corpus é pequeno, fictício e lexicalmente simples.
- A resposta é extrativa; não mede qualidade de geração.
- O custo usa contador aproximado e preços ilustrativos, não faturamento real.
- A latência mede processo e HTTP locais, não rede externa ou modelo remoto.
- O intervalo de confiança com poucas perguntas é largo; o projeto o mostra em vez de escondê-lo.

Leia [`relatorio.md`](relatorio.md) para a revisão técnica completa e
[`model-card.md`](model-card.md) para usos e riscos.
