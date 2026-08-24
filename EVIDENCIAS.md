# Registro de evidências

Este arquivo separa três coisas que não devem ser confundidas:

1. o laboratório existe;
2. ele já foi executado em algum ambiente;
3. o resultado foi reproduzido com versões, dados e saídas preservados.

Um número no curso é uma observação sob condições específicas, não uma propriedade
universal de LLMs. Ao atualizar uma medição, registre commit, data, ambiente, hardware,
seed, revisão do modelo, dados, tamanho da amostra e saída bruta.

## Resultados destacados

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
  "experimento": "modulo-XX/nome-do-lab",
  "commit": "sha completo",
  "executado_em": "AAAA-MM-DDTHH:MM:SSZ",
  "comando": "uv run python modulo-XX/lab_cpu.py",
  "python": "3.12.x",
  "plataforma": "sistema, arquitetura e hardware",
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

- [ ] Versionar a primeira execução de referência de cada resultado destacado.
- [ ] Fixar `revision` nos modelos do Hugging Face usados pelos labs.
- [ ] Registrar checksum e licença dos textos baixados.
- [ ] Executar e registrar os labs MLX em Apple Silicon.
- [ ] Fazer o CI validar o esquema dos JSONs e os números citados na documentação.

## Status da nova rota CUDA

Os labs CUDA têm importação tardia, `--dry-run`, revisão imutável de modelo e registro em
`runs/`. A estrutura foi validada sem GPU; treino, consumo de VRAM, throughput e qualidade
permanecem **não reproduzidos** até uma execução em hardware NVIDIA ser preservada aqui.
