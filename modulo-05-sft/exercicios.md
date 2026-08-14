# Módulo 5 — Exercícios

---

## Parte A — Conceituais

### A1. O cálculo das épocas

Você tem 3.400 exemplos de SFT e roda:

```bash
mlx_lm.lora --model <modelo> --train --data ./dados --iters 2000 --batch-size 8
```

a) Quantas épocas?
b) A loss de treino cai de 1,8 para 0,3 e a de validação para em 1,4 e começa a subir no meio do treino. O que aconteceu e a partir de qual `iters` aproximadamente?
c) Corrija o comando.

<details><summary>Gabarito</summary>

a) `(2000 × 8) / 3400 = **4,7 épocas**` — acima da faixa recomendada de 1–3.

b) Overfitting clássico. A loss de treino em 0,3 indica que o modelo está reproduzindo as respostas quase literalmente. A validação virou provavelmente perto de 2 épocas, isto é, `iters ≈ 3400 × 2 / 8 = 850`.

c) `--iters 600` dá 1,4 épocas; `--iters 850` dá 2,0. Comece com **600** e use `--steps-per-eval 50` para observar onde a validação vira. O melhor checkpoint quase nunca é o último — e é por isso que `save_every` existe.
</details>

---

### A2. A baseline honesta

Um colega apresenta: "fine-tuning melhorou a aderência ao formato de 20% para 95%". Você pergunta como foi medida a baseline e ele responde: "rodei o modelo base com o prompt *'Responda sobre suporte técnico'*".

Por que esse resultado não significa quase nada? Que baseline você exigiria?

<details><summary>Gabarito</summary>

O prompt da baseline não pede o formato. O modelo base foi avaliado numa tarefa que ninguém explicou a ele — os 20% são acidentais.

A baseline correta tem três níveis, e todos deveriam constar:

1. Modelo base com **instrução explícita e detalhada** do formato.
2. Modelo base com instrução **+ 1 a 3 exemplos few-shot** (é o que o Lab 2 chama de "prompt esforçado").
3. Modelo base com o melhor prompt que você conseguir depois de algumas iterações honestas.

Se o nível 2 já entrega 90%, o fine-tuning está brigando por 5 pontos — e talvez ainda valha a pena, porque elimina os tokens do few-shot de **toda** chamada em produção. Mas esse é um argumento de **custo**, não de capacidade, e precisa ser feito explicitamente.

A regra: **todo ganho reportado de fine-tuning precisa vir com o prompt da baseline escrito por extenso no relatório.** Sem isso, o número não é auditável.
</details>

---

### A3. Traduzindo hiperparâmetros

Você encontra um tutorial de CUDA com esta configuração PEFT:

```python
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
           target_modules=["q_proj","k_proj","v_proj","o_proj"])
```

Traduza para o YAML do `mlx_lm.lora`.

<details><summary>Gabarito</summary>

```yaml
lora_parameters:
  keys: ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
  rank: 16
  scale: 2.0        # lora_alpha / r = 32 / 16 = 2.0
  dropout: 0.05
```

O ponto de atenção é o `scale`. No PEFT, o multiplicador efetivo é `lora_alpha / r`; no MLX, `scale` **é** o multiplicador. Copiar `scale: 32` seria 16 vezes mais agressivo que o tutorial.

Note também que o default do MLX, `scale: 20.0` com `rank: 8`, equivale a `lora_alpha = 160` — muito acima do que a maioria dos tutoriais CUDA usa. Se seus treinos parecerem instáveis com os defaults do MLX, esse é o primeiro suspeito.
</details>

---

### A4. Diagnósticos

Diga a causa mais provável e a primeira correção:

1. O modelo responde perfeitamente e depois inventa uma nova pergunta do usuário e responde a ela também.
2. Loss de treino e validação ambas paradas em ~2,5 após 500 iterações.
3. O formato ficou perfeito, mas o modelo passou a errar contas simples que acertava antes.
4. O modelo repete literalmente uma das respostas do treino, mesmo para perguntas diferentes.
5. `Metal out of memory` na iteração 40.

<details><summary>Gabarito</summary>

1. **EOS ausente.** Se você usou o formato `messages`, o MLX cuida disso — então suspeite de dados no formato `text` montados à mão. Verifique um exemplo tokenizado.
2. **Learning rate baixo demais** (o default `1e-5` é de full fine-tune), ou `num_layers` pequeno demais. Suba o LR para `1e-4` primeiro; é a correção de maior efeito.
3. **Catastrophic forgetting.** Reduza `iters`, reduza `num_layers`, ou misture 10–20% de exemplos gerais no dataset. Meça com o Lab 5 antes e depois.
4. **Épocas demais em poucos dados.** Reduza `iters` ou aumente a diversidade do dataset. Também vale checar se seus exemplos são variações do mesmo caso — o experimento B tem 24 problemas com 6 fraseados cada; se todos os fraseados de um problema estiverem no treino, o modelo aprende o problema, não a tarefa.
5. **Memória.** Em ordem de custo-benefício: `--batch-size 1`, `--max-seq-length` menor (meça o p95 real dos seus dados), `--grad-checkpoint`, `--num-layers` menor, e por fim um modelo quantizado. Use `--grad-accumulation-steps` para manter o batch efetivo ao reduzir o batch físico.
</details>

---

### A5. Prompt, RAG ou SFT

Para cada caso, escolha e justifique em uma frase:

1. O modelo precisa responder sempre em JSON com um schema fixo de 8 campos.
2. O modelo precisa saber o catálogo de 12.000 produtos da empresa, atualizado diariamente.
3. O modelo precisa adotar o tom de voz da marca — informal, brasileiro, sem jargão.
4. O modelo precisa citar corretamente as 40 páginas do manual interno de compliance.
5. O modelo precisa resolver equações de segundo grau corretamente.

<details><summary>Gabarito</summary>

1. **Prompt primeiro** (com structured output / grammar constraint, se o runtime suportar), **SFT** se o volume de chamadas tornar o prompt caro. Formato é o caso ideal do SFT, mas restrição de decodificação é ainda mais confiável e não exige treino.
2. **RAG.** Volume grande e mudança diária — fine-tuning ficaria desatualizado no dia seguinte.
3. **SFT.** Tom de voz é comportamento puro; 200–500 exemplos consistentes resolvem, e é exatamente o que o LIMA descreve.
4. **RAG**, com citação da fonte. Quarenta páginas cabem tranquilamente num pipeline de recuperação, e o requisito de *citar corretamente* torna a alucinação inaceitável — SFT aumentaria a confiança sem aumentar a precisão.
5. **Nenhum dos três, sozinho.** É capacidade de raciocínio, não formato nem conhecimento. As opções reais são: um modelo melhor, chain-of-thought (módulo 7), ou — a resposta de engenharia — dar ao modelo uma **ferramenta** (um interpretador Python) e ensiná-lo a usá-la. Fine-tuning em respostas de equações ensina o modelo a produzir respostas com aparência de solução.
</details>

---

## Parte B — Práticas

### B1. Varredura de learning rate

Treine o experimento B com LR `1e-5`, `5e-5`, `1e-4`, `3e-4` e `1e-3`, mantendo `--iters 300`. Registre a loss de validação final e a aderência ao formato.

Onde está o ótimo? O default `1e-5` chegou lá?

<details><summary>Gabarito esperado</summary>

Espere que `1e-5` mal saia do lugar em 300 iterações — a aderência ficará próxima da baseline. O ótimo deve estar entre `1e-4` e `3e-4`. Em `1e-3` a loss provavelmente oscila e a qualidade cai.

A conclusão prática: **o default do `mlx_lm.lora` é conservador para LoRA**, e quem roda o comando da documentação sem pensar tende a concluir que "fine-tuning não funcionou".
</details>

---

### B2. Quantas camadas

Com o melhor LR do B1, varie `--num-layers` em 2, 4, 8, 16 e todas (28 no Qwen2.5-1.5B). Registre aderência, memória de pico (`mx.get_peak_memory()`) e tempo.

Qual o menor `num_layers` que atinge o desempenho máximo?

<details><summary>Gabarito esperado</summary>

Para uma tarefa de **formato**, poucas camadas finais costumam bastar — frequentemente 4 a 8. Formato e estilo são comportamentos codificados tarde na pilha.

Esse é o resultado que importa para os seus 16 GB: se 8 camadas entregam o mesmo que 28, você acabou de liberar memória para dobrar o `batch_size` ou usar um modelo maior. Meça, não suponha.
</details>

---

### B3. A curva de dados

Treine com 1, 2, 4, 8 e 20 problemas (6, 12, 24, 48 e 120 exemplos), sempre testando nos **mesmos 2 problemas** reservados.

a) Quantos problemas bastam para o formato generalizar?
b) O que isso diz sobre a lição do LIMA?

<details><summary>Gabarito esperado</summary>

a) Espere que o formato generalize com **muito poucos** problemas — provavelmente 4 a 8. O modelo não precisa aprender suporte técnico; precisa aprender que respostas neste contexto têm três seções com estes títulos.

b) Confirma o LIMA e refina a formulação: o que importa não é o número de exemplos, é o número de **demonstrações consistentes do padrão que você quer**. 120 exemplos com 6 fraseados de 20 problemas ensinam o padrão 20 vezes, não 120.

Isso tem consequência direta na construção de dataset: aumentar variações superficiais do mesmo caso dá muito menos retorno que adicionar casos genuinamente novos.
</details>

---

### B4. LLM-as-judge com controle de viés

Para o experimento A (Alpaca), implemente uma avaliação por juiz. Use um modelo maior (ex.: `mlx-community/Qwen2.5-7B-Instruct-4bit`) para comparar respostas do modelo base e do fine-tuned.

Requisitos:
1. Cada par avaliado **nas duas ordens** (A-B e B-A), somando os votos.
2. Registre a taxa de inconsistência — quantas vezes o juiz mudou de opinião ao inverter a ordem.
3. Reporte win rate com empates contados separadamente.

O que a taxa de inconsistência diz sobre a confiabilidade do seu resultado?

<details><summary>Gabarito</summary>

Uma inconsistência alta (digamos, acima de 25%) significa que o juiz está decidindo em boa parte por **posição**, não por qualidade. Nesse caso, um win rate de 55% não é evidência de nada — está dentro do ruído do próprio método.

O procedimento correto: descarte ou trate à parte os pares inconsistentes, e reporte o win rate apenas sobre os consistentes, **junto com** a fração descartada. Um resultado de "62% de vitórias sobre os 71% de pares em que o juiz foi consistente" é honesto; "62% de win rate" sozinho esconde a incerteza.

Bônus: meça também o comprimento médio das respostas vencedoras. Se as vencedoras são sistematicamente mais longas, você está medindo verbosidade.
</details>

---

### B5. Multi-turn

Estenda o dataset de suporte para conversas de dois turnos: o usuário relata o problema, o assistente responde no formato, o usuário diz que não funcionou, o assistente oferece uma alternativa.

Treine e avalie: o modelo mantém o formato no segundo turno? Ele usa o contexto do primeiro?

<details><summary>Gabarito</summary>

Pontos de atenção na construção:

- **Um exemplo por conversa**, não um por turno. Quebrar a conversa em pares independentes ensina o modelo a ignorar o histórico.
- A loss é calculada em **todos** os turnos do assistente, o que o formato `messages` já faz.
- No teste, forneça o histórico real e verifique se a segunda resposta é *diferente* da primeira. Se for igual, o modelo não está condicionando ao "não funcionou" — sintoma de que seus exemplos de segundo turno são previsíveis demais.
</details>

---

## Desafio — SFT do seu próprio problema

Escolha um problema real seu — algo que você faria de fato — e execute o pipeline dos 10 passos do README, seção 7.

Entregue um relatório com:

1. **O objetivo**, em uma frase: que comportamento exatamente deve mudar.
2. **A métrica**, definida antes de qualquer dado ser coletado, e o teste dela contra exemplos de ouro.
3. **A baseline**, com o prompt escrito por extenso.
4. **O dataset**: origem, tamanho, como foi curado, e a verificação de vazamento treino/teste.
5. **Os hiperparâmetros** com justificativa — especialmente épocas e learning rate.
6. **O resultado** na mesma métrica da baseline.
7. **O forgetting** medido.
8. **A conclusão honesta**: o fine-tuning se pagou? Se não, diga.

O item 8 é o mais importante. Um relatório que conclui "o prompt few-shot resolvia e o fine-tuning não se justificou" vale mais — profissionalmente — que um que infla um ganho contra uma baseline fraca. A capacidade de chegar a essa conclusão e registrá-la é o que distingue engenharia de demonstração.
