# Model card

Ficha técnica do modelo final. Preencha junto com o `config.json` — é o que torna o
resultado verificável, não apenas bonito.

| Campo | Valor |
|---|---|
| Modelo base | `organizacao/modelo` |
| Revision | `sha256` do artefato |
| Técnica | LoRA / QLoRA / SFT / DPO / GRPO |
| Dataset | origem, licença, contagem, splits |
| Hiperparâmetros | r, alpha, dropout, passos, batch, lr |
| Métrica alvo | nome, direção, piso, n |
| Hardware / tempo | o que gerou o número abaixo |
| Efeitos colaterais | forgetting, deriva de PPL, degradação de quantização |

## Limitações

- O que o modelo NÃO faz (casos conhecidos de erro).
- Comportamento de recusa/abstenção.
- Onde não confiar nele (domínios fora de treino, idioma, formato).
