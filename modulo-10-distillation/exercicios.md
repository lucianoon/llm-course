# Módulo 10 — Exercícios

💻 = qualquer máquina | 🍎 = Mac

---

## Parte A — Conceituais

### A1. Qual família?

Para cada cenário, white-box (logits) ou black-box (dados)? E por quê?

1. Destilar o Qwen2.5-7B para o Qwen2.5-0.5B (mesma família).
2. Destilar o Claude para um Llama interno.
3. Destilar o Llama-3-70B para o Qwen2.5-1.5B.
4. Destilar um modelo seu de 3B (que você mesmo treinou) para 0.5B, na mesma base.

<details><summary>Gabarito</summary>

1. **White-box** — mesmo tokenizer, logits acessíveis. É o caso ideal: use a KD de logits (ou mistura), extrai mais sinal por token.
2. **Black-box forçada** — sem acesso aos logits pela API. E **bloqueada juridicamente**: os termos da Anthropic proíbem treinar concorrentes com as saídas (módulo 4).
3. **Black-box forçada por tokenizer** — 128k vs 151k tokens: as distribuições não são comparáveis posição a posição, mesmo com logits disponíveis.
4. **White-box, o melhor caso possível** — você controla tudo. Considere também on-policy KD (GKD), já que amostrar do aluno é barato.
</details>

---

### A2. O vale da forward KL

No Lab 2, a gaussiana ajustada por forward KL assentou em μ≈0 — o vale entre os modos em ±2.

a) Explique pela definição de `KL(p‖q)` por que ela é forçada a isso.
b) Traduza para geração de texto: o que é "assentar no vale" quando o professor domina dois estilos?
c) Por que o problema desaparece se o aluno tem capacidade igual à do professor?

<details><summary>Gabarito</summary>

a) `KL(p‖q) = Σ p·log(p/q)` explode onde `p > 0` e `q → 0`. A gaussiana única não pode dar massa aos dois modos com desvio pequeno — a única forma de não zerar em nenhum modo é centrar entre eles e alargar. O custo de cobrir tudo é estar em lugar nenhum.

b) O professor escreve bem tanto prosa formal quanto diálogo coloquial (dois modos). O aluno forward-KL produz uma mistura mediana dos dois — texto que não é nem formal nem coloquial, com probabilidade espalhada. Em amostragem, isso vira incoerência: o "meio" de dois estilos não é um estilo.

c) Com capacidade suficiente, `q` pode ser bimodal também — o mínimo do KL é `q = p` nas duas direções, e a escolha da direção deixa de importar. A assimetria só morde quando o aluno é menor — ou seja, sempre, em destilação.
</details>

---

### A3. O T² da loss

A loss de KD é `T²·KL(p(T)‖q(T))`. Um colega remove o `T²` "para simplificar" e treina com `α=0,5, T=4`.

O que acontece com o equilíbrio entre `L_KD` e `L_CE`, e por quê?

<details><summary>Gabarito</summary>

Os gradientes da KL através de um softmax com temperatura escalam com `1/T²` — suavizar a distribuição achata as diferenças de logits e encolhe os gradientes. Com T=4, sem a correção, o termo KD contribui ~16× menos gradiente do que aparenta na loss.

Resultado: o `α=0,5` nominal vira um α efetivo de ~0,06 — o treino é dominado pela cross-entropy e o aluno mal recebe a dark knowledge. O sintoma típico: "KD não fez diferença nenhuma" — porque de fato quase não participou. O `T²` restaura a escala para que `α` signifique o que diz.
</details>

---

### A4. A tabela 5 do R1

O paper do R1 reporta: aplicar o mesmo RL diretamente no Qwen-32B rendeu muito menos que destilar o R1 (671B) para o mesmo Qwen-32B.

a) Explique com os conceitos do módulo 9 por que o RL direto no modelo menor rende menos.
b) Isso significa que RL em modelos pequenos é sempre inútil?
c) Qual é a implicação econômica para quem NÃO é um lab de fronteira?

<details><summary>Gabarito</summary>

a) O RL amplifica competência existente (módulo 9, regra de ouro): a vantagem de grupo é zero quando o grupo inteiro erra. O 32B sozinho raramente acerta os problemas mais difíceis — exatamente onde estaria o ganho — então o gradiente nessas regiões é nulo. O 671B tinha massa de acerto para o GRPO amplificar; o 32B, não. A destilação contorna isso: entrega os caminhos prontos por imitação, sem exigir que o aluno os descubra.

b) Não — o próprio lab do módulo 9 elevou 27%→90% num modelo minúsculo, porque a taxa base estava na janela útil. RL em modelo pequeno funciona quando a fronteira da tarefa está ao alcance dele. O que a tabela 5 diz é: para empurrar a fronteira DA CAPACIDADE, o RL precisa do modelo grande; o pequeno herda melhor do que descobre.

c) Que o caminho racional para 99% das equipes é: **deixar quem tem compute descobrir (RL nos gigantes) e destilar o resultado** — via modelos abertos que permitem isso (R1, Qwen). O compute de descoberta é gasto uma vez no mundo; o de disseminação, uma vez por equipe.
</details>

---

### A5. Auditoria de um plano

Um time propõe: "vamos destilar nosso assistente jurídico do GPT-4 usando 50k conversas geradas, treinar um Llama-8B, e servir para clientes."

Liste todos os problemas, em ordem de gravidade.

<details><summary>Gabarito</summary>

1. **Jurídico, fatal:** termos da OpenAI proíbem usar saídas para treinar concorrentes. Para produto comercial, é risco existencial. Alternativa: professor com licença permissiva (Qwen, R1, Mistral) ou dados próprios.
2. **Domínio de alto risco sem verificador:** direito não tem gabarito automático — o rejection sampling do pipeline R1 não se aplica. A filtragem teria que ser humana (cara) ou por juiz LLM (vieses, módulo 5). Alucinações jurídicas herdadas passam direto.
3. **O aluno herda o teto e os erros do professor** em cauda longa — exatamente onde casos jurídicos raros vivem (seção 5: cauda morre primeiro).
4. **50k conversas geradas ≠ 50k conversas úteis:** sem curadoria (módulo 4), diversidade de tarefas e dedup, o volume é ilusório.
5. **Avaliação ausente do plano:** contra o quê? Baseline com RAG + prompt no 8B pode empatar sem nenhum treino (módulos 4 e 5).

Contra-proposta defensável: RAG sobre a base jurídica + SFT de formato/tom com dados próprios anotados + avaliação com revisão de especialista. Destilação, se entrar, entra de um professor licenciado e só para comportamento, não para conhecimento jurídico.
</details>

---

## Parte B — Práticas

### B1. 💻 Reverse KL na destilação real

O Lab 3 usa forward KL (o padrão de Hinton). Implemente a variante reverse — `KL(aluno‖professor)` — e compare PPL e uma inspeção qualitativa de gerações.

Atenção ao detalhe: em `F.kl_div(input, target)`, o `input` é o log da distribuição de quem está *dentro* da divergência à esquerda.

<details><summary>Gabarito</summary>

```python
kd = F.kl_div(F.log_softmax(logits_p / T, -1),      # agora o PROFESSOR é o "input"...
              F.log_softmax(logits_a / T, -1),      # ...e o ALUNO é a referência
              log_target=True, reduction="batchmean") * T * T / x.size(1)
```
(`KL(q‖p) = Σ q·log(q/p)` — quem pondera é o aluno.)

Em PPL de validação, espere o reverse **empatar ou perder** — PPL é forward KL contra os dados, então otimizar forward é otimizar a métrica. A diferença aparece na *geração*: o aluno reverse tende a texto mais coeso (menos massa em regiões medianas). É a lição do módulo: a métrica automática favorece estruturalmente uma das direções; julgue geração gerando.
</details>

---

### B2. 💻 A curva professor-aluno

Com o setup do Lab 3, varie o professor: treine professores com 300, 600 e 1200 passos (PPLs decrescentes) e destile o mesmo aluno de cada um.

O aluno melhora monotonicamente com o professor? Existe saturação — ou até reversão?

<details><summary>Gabarito esperado</summary>

Espere ganho do professor 300→600, e saturação (ou ganho marginal) em 600→1200: o gargalo passa a ser a **capacidade do aluno**, não a qualidade do professor.

A literatura documenta até reversão (professor bom demais prejudica): distribuições muito afiadas de um professor muito confiante viram quase-hard-labels, perdendo a dark knowledge — mitigável com T maior. Se você observar isso, teste T=4 no professor de 1200 passos antes de concluir.
</details>

---

### B3. 💻 Destilar só o comportamento

Destile do professor apenas nas posições de **diálogo** (janelas do corpus contendo `--`): o aluno recebe KD nessas janelas e CE comum nas demais.

Compare com KD uniforme: o aluno fica melhor em diálogo? Pior no resto?

<details><summary>Gabarito esperado</summary>

É a versão em miniatura de "destilação seletiva de capacidade" — transferir um comportamento específico sem pagar o preço inteiro. Espere efeito pequeno mas mensurável (PPL por subconjunto: janelas com diálogo vs sem).

O aprendizado metodológico: a PPL agregada esconde trocas por subconjunto. Toda avaliação de destilação séria segmenta a métrica pelos domínios que importam — exatamente como a degradação de quantização por idioma no módulo 6.
</details>

---

### B4. 🍎 Com ou sem `<think>`

O `limpar_traco` do lab_mlx descarta o bloco `<think>` do professor. Rode o pipeline nas duas versões: traços com o thinking completo vs só a resposta limpa.

Compare: acurácia do aluno, tokens médios por resposta do aluno, e tempo de inferência.

<details><summary>Gabarito esperado</summary>

Com `<think>`: o aluno aprende a "pensar" — espere acurácia maior (é CoT, módulo 7: mais compute por resposta) e custo por resposta 3–10× maior. Sem: aluno rápido e mais fraco.

Não há resposta certa; há a decisão de produto do módulo 7 (A5): o ganho de acurácia paga o custo por chamada? O valor do exercício é obter as DUAS curvas do MESMO pipeline e escolher com números.
</details>

---

### B5. 🍎 O k do rejection sampling

O lab usa K_TENTATIVAS=2. Meça o efeito de k: com os traços já gerados (ou gerando k=4 num subconjunto), plote: taxa de problemas com ≥1 traço aprovado × k, e a acurácia final do aluno × k.

<details><summary>Gabarito esperado</summary>

A cobertura cresce como `1−(1−p)^k` (p = taxa de acerto do professor por tentativa) — rendimentos decrescentes rápidos. O ganho do aluno segue a cobertura: k=2→4 recupera problemas de dificuldade média (o professor acerta às vezes), que são justamente os mais informativos.

Note o custo: k dobra o compute de geração, a parte cara do pipeline. O pass@k do módulo 7 (B3) é exatamente a ferramenta para prever essa curva antes de pagar por ela.
</details>

---

## Desafio — a destilação do seu caso

Feche o ciclo dos módulos 5–10 com um projeto seu:

1. Escolha uma tarefa SUA em que um modelo grande (7B+ local, ou um aberto maior via API licenciada) faz bem e o 0.5B faz mal.
2. Monte o verificador ou o filtro de qualidade (módulo 9, desafio — reutilize).
3. Gere, filtre (registre a taxa e o viés de dificuldade), treine o 0.5B.
4. Avalie as três vias: aluno antes, aluno depois, professor — na mesma métrica.
5. Feche com a conta de engenharia: custo total da destilação vs custo de servir o professor direto. Em quantas inferências o treino se paga?

O item 5 é a pergunta que decide destilação em produção, e quase nunca é feita: se o produto serve 1.000 chamadas/dia, servir o professor quantizado pode ser mais barato que todo o pipeline. Se serve 1M/dia, a destilação se paga em horas. Saiba em qual mundo o seu caso vive.
