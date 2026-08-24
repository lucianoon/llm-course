# Módulo 12 — Projeto final de customização

> **Pergunta central:** você consegue atravessar o pipeline inteiro — problema → dados → treino → alinhamento → serving — tomando cada decisão por medição, e defendê-la?

Não há lab neste módulo. Há um projeto, um contrato de entrega, e os critérios pelos quais você mesmo vai se avaliar. Tudo o que o projeto exige já foi construído nos módulos 1–11; o trabalho aqui é **integração e julgamento**. Antes de concluir, use o protocolo de aceite em [`VALIDACAO.md`](VALIDACAO.md).

---

## 1. O contrato

Escolha **um** problema real — idealmente do seu trabalho — e entregue:

1. **Um modelo customizado** que resolve o problema mensuravelmente melhor que as baselines.
2. **Um repositório reproduzível** — qualquer pessoa com um M4 refaz tudo com `make` ou um script.
3. **Um relatório técnico** (4–8 páginas) no formato da seção 4.

O escopo certo cabe em **duas a três semanas** de trabalho parcial. Errar o escopo para cima é o modo de falha número um: um projeto pequeno completo vale dez vezes um grande pela metade.

### Ideias calibradas para o seu contexto

| Projeto | Módulos que exercita | Risco |
|---|---|---|
| Assistente de suporte com formato e tom da sua empresa, treinado de logs reais anonimizados | 4, 5, 6, 8, 11 | baixo |
| Modelo de raciocínio para um domínio verificável seu (cálculos de negócio, validação de regras) | 7, 9, 10 | médio |
| Destilar um comportamento de um modelo grande para um 0.5B servível a custo mínimo | 7, 10, 11 | médio |
| Um "R1 de bolso": SFT frio + GRPO num domínio verificável estreito | 5, 7, 9 | alto |
| Classificador/extrator estruturado de documentos do seu setor, com garantia de formato | 4, 5, 11 | baixo |

---

## 2. As dez etapas — e o que cada uma cobra dos módulos

```
 1. Definir o problema        → uma frase; que comportamento muda?           (mód. 5 §7)
 2. Definir a métrica ANTES   → e testá-la contra exemplos de ouro           (mód. 5, 7)
 3. As TRÊS baselines         → prompt simples, prompt esforçado, RAG se
                                 couber. Sem isso, nada é interpretável       (mód. 5 §5)
 4. Dados                     → coleta, dedup, filtros, splits SEM vazamento,
                                 auditoria de confundimentos                  (mód. 4, 8)
 5. Escolher a técnica        → a tabela de decisão abaixo                   (mód. 5–10)
 6. Treinar                   → hiperparâmetros JUSTIFICADOS, épocas
                                 calculadas, curvas salvas                    (mód. 3, 6)
 7. Avaliar                   → a métrica do passo 2 + leitura manual
                                 + efeitos colaterais (forgetting, drift)     (mód. 5–9)
 8. Iterar UMA variável       → por vez, com registro                        (todos)
 9. Preparar para servir      → quantizar, medir degradação NO SEU domínio,
                                 TTFT/TPOT, custo/Mtok                        (mód. 6, 11)
10. Relatório                 → seção 4, incluindo o que NÃO funcionou
```

### A tabela de decisão (o curso em oito linhas)

| Diagnóstico | Técnica | Módulo |
|---|---|---|
| Falta conhecimento (volátil ou volumoso) | RAG — e talvez o projeto nem precise de treino | 4 |
| Formato/tom inconsistente com prompt bom | SFT/LoRA, 200–1.000 exemplos | 5, 6 |
| Precisa raciocinar em passos | dados de CoT; destilar de um reasoner | 7, 10 |
| Escolhe mal entre respostas plausíveis | DPO/KTO sobre amostras do próprio modelo | 8 |
| Existe verificador exato e o modelo às vezes acerta | GRPO | 9 |
| Modelo bom demais caro demais | destilar + quantizar | 10, 11 |
| Nada acima com clareza | **volte ao passo 1** — o problema não está definido | — |

---

## 3. Os erros que o curso já viu — a checklist negativa

Cada item abaixo aconteceu de verdade nos labs deste curso. O seu projeto será auditado (por você) contra todos:

- [ ] **Vazamento treino/teste** — o split do módulo 5 vazava por fraseado até a asserção `verificar_vazamento` pegar. O seu tem asserção?
- [ ] **Baseline fraca** — comparou com o modelo base *sem* o melhor prompt? O ganho é ilusão (mód. 5).
- [ ] **Métrica que mede outra coisa** — a extração de resposta que transforma acertos em erros (mód. 7); a loss de treino comparada entre corpora (mód. 4); a PPL entre tokenizers (mód. 1).
- [ ] **Avaliar no modo de decoding errado** — a degeneração era 100% em greedy e 0% com sampling (mód. 8). Você mediu no modo em que vai servir?
- [ ] **Confundimento nos dados de preferência** — o rejected sistematicamente mais longo (mód. 8). Auditou comprimento/idioma/formato?
- [ ] **Curva linda, modelo destruído** — a recompensa subindo enquanto o modelo virava spam de travessões (mód. 9). Você LEU 20 gerações?
- [ ] **Efeitos colaterais não medidos** — catastrophic forgetting (mód. 5), deriva de PPL (mód. 8, 9), degradação de quantização 4× pior no seu idioma (mód. 6).
- [ ] **Números de folclore sem verificar** — o "T=2–4 ótimo" que era T=1 aqui (mód. 10); os TFLOPs com sparsity (mód. 3); o `scale` do MLX ≠ `lora_alpha` (mód. 5).
- [ ] **Épocas não calculadas** — `iters × batch / exemplos` antes de treinar, sempre (mód. 5).
- [ ] **Treinar no que o modelo nunca produziria** — pares off-policy punindo defeito inexistente = deriva pura (mód. 8).

---

## 4. O formato do relatório

```
1. Problema e métrica          (½ pág)  a frase, a métrica, o teste da métrica
2. Baselines                   (½ pág)  as três, com os prompts POR EXTENSO
3. Dados                       (1 pág)  origem, pipeline, contagens por etapa,
                                        splits, auditorias — a proveniência completa
4. Método                      (1 pág)  técnica escolhida CONTRA as alternativas;
                                        hiperparâmetros com justificativa
5. Resultados                  (1-2 pág) tabela principal; curvas; exemplos reais
                                        (bons E ruins); efeitos colaterais medidos
6. Serving                     (½ pág)  o relatório do módulo 11: formato final,
                                        TTFT/TPOT, custo/Mtok, ponto de operação
7. O que não funcionou         (½ pág)  OBRIGATÓRIA. Tentativas descartadas e porquê
8. Limitações e próximos passos (¼ pág)
```

A seção 7 é obrigatória porque é a que separa engenharia de marketing — e porque, nos onze módulos deste curso, os erros documentados (a métrica de degeneração no modo errado, o black-box sem filtro, o T do folclore) ensinaram mais que os acertos.

### Critérios de qualidade

Um projeto está pronto quando você responde **sim** a todas:

1. Outra pessoa reproduz o resultado do zero com o repositório?
2. Cada número da tabela principal tem um script que o regenera?
3. A comparação com a baseline esforçada é justa — mesmo teste, mesma métrica, melhor prompt que você conseguiu?
4. Você sabe dizer *por que* cada hiperparâmetro tem o valor que tem?
5. Você leu, pessoalmente, 30+ saídas do modelo final?
6. A conclusão sobreviveria a um revisor hostil com acesso aos seus dados?
7. Se o fine-tuning não se pagou contra a baseline, o relatório diz isso?

---

## 5. Depois do curso

O que você construiu aqui — a pasta `llm-course` — é um portfólio: onze módulos com implementações do zero verificadas numericamente, e um projeto final reproduzível. Três direções naturais:

- **Aprofundar a teoria:** os papers das leituras de cada módulo, agora com as mãos calejadas. O curso do Karpathy (nanoGPT → nanochat) e o CS336 de Stanford (Language Modeling from Scratch) são as sequências naturais.
- **Escalar a prática:** repetir o pipeline do seu projeto num modelo de 7–14B com GPU alugada (o `00-setup.md` original tem a tabela de custos) — as receitas são as mesmas, os zeros a mais também.
- **Especializar:** interpretabilidade (o transformer-circuits do módulo 2), avaliação (a subárea que mais falta gente), ou inference systems (vLLM/SGLang por dentro).

E o hábito que vale mais que qualquer conteúdo: **desconfie de todo número que você não mediu.** Este curso corrigiu o próprio material uma dúzia de vezes porque a execução desmentiu a teoria escrita de memória. É o método.
