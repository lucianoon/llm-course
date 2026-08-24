# Governança de dados para customização de LLMs

Este documento é um gate: o treino não começa enquanto origem, licença, finalidade,
PII, splits e retenção não estiverem explícitos. “O arquivo estava disponível” não é
base legal nem licença.

## Manifesto obrigatório

Para cada dataset, registre:

- proprietário e origem;
- licença e termos adicionais;
- finalidade autorizada;
- forma de consentimento ou outra base aplicável;
- data de coleta e janela temporal;
- transformações e filtros;
- checksum de cada artefato;
- responsável e pessoas/sistemas com acesso;
- prazo de retenção e procedimento de exclusão;
- presença de PII e método de anonimização;
- regra de split por pessoa, organização, documento ou tempo;
- proibição ou autorização de redistribuição e treinamento de modelos derivados.

O scanner inicial do curso encontra e-mails, telefones e CPFs. Ele é uma defesa de
baixa cobertura, não uma garantia: nomes, endereços, dados de saúde, segredos e quase
identificadores exigem regras do domínio e revisão humana.

```bash
python -m tools.auditar_dataset suporte/train.jsonl suporte/valid.jsonl \
  --nome suporte-v1 \
  --origem "tickets próprios anonimizados" \
  --licenca "uso interno autorizado" \
  --finalidade "SFT de formato" \
  --saida suporte/dataset-manifest.json
```

Exit code `2` significa que o scanner encontrou PII: o manifesto é gravado, mas o gate
falha. Não copie os valores sensíveis para logs, issues ou prompts de depuração.

## Política mínima

1. **Minimização:** remova campos que o objetivo não exige antes de entregar os dados
   ao time de modelagem.
2. **Pseudonimização não é anonimização:** IDs consistentes ainda permitem reidentificar
   pessoas quando combinados com outras tabelas.
3. **Split por entidade:** mensagens da mesma pessoa, ticket ou documento não podem
   aparecer em treino e teste.
4. **Acesso menor possível:** dados brutos e dados de treino têm permissões diferentes.
5. **Retenção:** derive e registre uma data de exclusão; “para sempre” exige justificativa.
6. **Exclusão propagada:** apagar o dado bruto não remove sua cópia de datasets,
   checkpoints, adapters, logs ou caches. Mantenha lineage para localizar derivados.
7. **Dados sintéticos:** registre professor, revisão, prompt, termos de uso e filtros.
8. **Publicação:** model card e dataset card devem dizer o que não pode ser inferido do
   resultado, quais grupos foram avaliados e quais riscos permanecem.

## Gate de projeto

- [ ] Manifesto sem campos vazios.
- [ ] Scanner automático passou e revisão humana foi registrada.
- [ ] Licença permite o uso e o tipo de distribuição pretendidos.
- [ ] Split respeita entidade/tempo e possui teste de vazamento.
- [ ] Dados brutos não entram no Git.
- [ ] Logs não guardam prompts/respostas sensíveis por padrão.
- [ ] Existe responsável por pedidos de exclusão e incidentes.
- [ ] Checkpoints e adapters têm revisão do modelo-base e checksum do dataset.
