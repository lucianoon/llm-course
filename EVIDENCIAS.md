# Registro de evidências

Este arquivo separa três coisas que não devem ser confundidas:

1. o laboratório existe;
2. ele já foi executado em algum ambiente;
3. o resultado foi reproduzido com versões, dados e saídas preservados.

Um número no curso é uma observação sob condições específicas, não uma propriedade
universal de LLMs. Ao atualizar uma medição, registre commit, data, ambiente, hardware,
seed, revisão do modelo, dados, tamanho da amostra e saída bruta.

## Resultados preliminares aguardando reprodução

Os valores abaixo **não devem ser usados como alegações de divulgação** enquanto o status não
for `Reproduzido`. Eles permanecem registrados porque são hipóteses úteis e porque apagar um
resultado não confirmado esconderia o histórico científico do curso.

| Alegação | Experimento e escopo | Status atual | Limitação principal |
|---|---|---|---|
| Reconstrução bit-exact de uma camada | Módulo 2; Qwen2.5-0.5B, float32/eager | Executado na autoria; artefato bruto ainda não versionado | Outros dtypes e kernels podem exigir tolerância |
| Quantização: +17,4% em literatura PT vs +4,3% em EN | Módulo 6; um modelo e pequenos textos do lab | Executado na autoria; requer nova reprodução registrada | Amostra pequena, um modelo, sem intervalo de confiança |
| CoT: resposta correta 38× mais provável | Módulo 7; modelo e prompts do lab | Executado na autoria; requer nova reprodução registrada | Conjunto pequeno; não generaliza para todo tipo de raciocínio |
| GRPO: 27% → 90% | Módulo 9; tarefa sintética controlada | Executado na autoria; requer nova reprodução registrada | Não mede alinhamento ou RL em domínio aberto |
| Agente com calculadora: 0% → 87% | Módulo 15; 30 multiplicações de três dígitos | Executado na autoria; requer nova reprodução registrada | n=30 e uma única ferramenta/tarefa |
| Steering troca o idioma | Módulo 16; configuração descrita no lab | Executado na autoria; requer nova reprodução registrada | Resultado dependente de modelo, camada e intensidade |
| RAG: comparação BM25/densa/RRF | Módulos 13–14; 25 perguntas | **Métrica corrigida; números históricos retirados** | Reexecutar com gabarito no nível de passagem |

## Formato mínimo de uma reprodução

Crie um arquivo em `resultados/<experimento>/<data>-<commit>.json` com este esquema:

```json
{
  "schema_version": 1,
  "experimento": "modulo-XX/nome-do-lab",
  "commit": "sha completo",
  "executado_em": "AAAA-MM-DDTHH:MM:SSZ",
  "comando": "uv run python modulo-XX/lab_cpu.py",
  "python": "3.12.x",
  "plataforma": "sistema e arquitetura",
  "hardware": "CPU/GPU e memória relevantes",
  "working_tree_dirty": false,
  "seed": 0,
  "modelos": [{"id": "organizacao/modelo", "revision": "commit"}],
  "dados": [{"id": "fonte", "revision_ou_sha256": "valor"}],
  "amostra_n": 0,
  "metricas": {},
  "observacoes": "limitações ou desvios do protocolo"
}
```

O JSON deve guardar métricas estruturadas; stdout, tabelas completas ou gráficos podem
ficar ao lado. Uma alegação só passa a **reproduzida** quando outra execução preservada
confirma o resultado dentro da tolerância declarada.

## Próximos passos

A infraestrutura para registrar reproduções já existe — use o helper
`tools.reproducao.registrar_reproducao`, que grava no esquema acima em
`resultados/<experimento>/<data>-<commit>.json` (veja [`resultados/README.md`](resultados/README.md)).
Falta o trabalho de hardware, que nenhuma função substitui:

- [ ] Executar E REGISTRAR (`resultados/`) a primeira reprodução de cada resultado destacado.
- [x] Fixar `revision` nos modelos do Hugging Face usados pelos labs. O manifesto central está
  em [`MODELOS.json`](MODELOS.json), e `tools/validar_revisoes.py` bloqueia novas chamadas
  Transformers sem `revision` no CI.
- [x] Registrar checksum e licença dos textos baixados. As fontes externas usadas diretamente
  pelos scripts estão em [`DADOS_EXTERNOS.json`](DADOS_EXTERNOS.json); downloads passam por
  `tools/dados_externos.py` e são recusados quando tamanho ou SHA-256 divergem.
- [ ] Executar e registrar os labs MLX em Apple Silicon.
- [x] Fazer o CI validar o esquema dos JSONs preservados. O portão é
  `tools/validar_resultados.py`; vincular automaticamente cada número citado na documentação
  ao seu artefato ainda é uma etapa separada.

## Resultado demonstrativo do capstone

O projeto de referência do módulo 12 preserva dois artefatos no schema v1: avaliação offline e
carga HTTP local. Eles são regenerados pelos testes, mas **não contam como reprodução
independente** enquanto estiverem associados a uma árvore Git suja ou a uma única execução.

## Status da nova rota CUDA

Os labs CUDA têm importação tardia, `--dry-run`, revisão imutável de modelo e registro em
`runs/`. A estrutura foi validada sem GPU; treino, consumo de VRAM, throughput e qualidade
permanecem **não reproduzidos** até uma execução em hardware NVIDIA ser preservada aqui.
