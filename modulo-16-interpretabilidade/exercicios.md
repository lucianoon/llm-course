# Módulo 16 — Exercícios

Todos rodam em CPU (interpretabilidade é análise, não treino). 💻 em qualquer máquina.

---

## Parte A — Conceituais

### A1. Correlação vs causa, a lição-mãe

Um pesquisador júnior anuncia: "encontrei a cabeça que faz o modelo responder perguntas de geografia — a cabeça 7 da camada 12 tem atenção altíssima no nome do país".

a) Por que essa evidência é insuficiente?
b) Que experimento transformaria a afirmação em causal?
c) Cite o aviso do módulo 2 que ele ignorou.

<details><summary>Gabarito</summary>

a) Atenção alta é CORRELAÇÃO. A cabeça pode atender ao país sem que isso influencie a saída — o `value` daquele token pode ser quase nulo, ou a informação pode ser sobrescrita camadas adiante. "Onde o modelo olha" ≠ "o que o modelo usa".

b) **Ablação + patching.** Zere a cabeça 7/12 e meça se a resposta de geografia piora (ablação → é necessária?). Depois, patche a ativação dela de um prompt de outro país e veja se a resposta muda de acordo (patching → ela carrega a informação causalmente?). Só a intervenção prova.

c) Módulo 2, Lab 7: "mapas de atenção são péssima evidência causal — atenção mostra para onde se olha, não o que se usa". O módulo 16 inteiro é essa frase virada em método.
</details>

---

### A2. O logit lens honesto

Você aplica o logit lens num modelo e as camadas 1–15 preveem lixo, a 16–24 convergem para a resposta certa.

a) Isso significa que as camadas 1–15 "não fazem nada"?
b) Por que o logit lens cru não consegue ler o que elas fazem?
c) O que o tuned lens muda?

<details><summary>Gabarito</summary>

a) **Não** — elas fazem quase todo o trabalho. Elas movem informação, computam features intermediárias, montam a resposta no residual stream. O que elas NÃO fazem é escrever no formato final que a `lm_head` lê; isso é trabalho das últimas camadas.

b) O logit lens aplica a `lm_head` (treinada para ler a ÚLTIMA camada) a estados intermediários que escrevem num "dialeto" diferente do residual stream. É como tentar ler um rascunho intermediário com o dicionário da versão final — só funciona perto do fim.

c) O tuned lens treina uma projeção afim POR CAMADA (`lens_L`) que traduz o dialeto daquela camada para o da `lm_head`. Aí as camadas intermediárias ficam legíveis, e você vê a resposta se formar de forma muito mais suave e cedo. Custo: é um pequeno treino por camada (a loss do módulo 3, alvo = logits da camada final).
</details>

---

### A3. Desenhando um experimento de patching

Você quer descobrir onde o modelo representa o SUJEITO de uma frase (para entender concordância verbal). Desenhe o experimento de patching completo.

<details><summary>Gabarito</summary>

- **Par limpo/corrompido de estrutura idêntica**, diferindo só no sujeito: "The **keys** to the cabinet **are**..." (plural) vs "The **key** to the cabinet **is**..." (singular). A métrica: logit de "are" menos logit de "is" na posição do verbo.
- **Salve as ativações** do run limpo (plural) em todas as camadas/posições.
- **Rode o corrompido** (singular) transplantando, uma de cada vez, a ativação de cada (camada, posição) do limpo. Meça quando o verbo volta a preferir o plural.
- **O mapa de recuperação** revela em que camada e em que POSIÇÃO (provavelmente na do sujeito "keys", depois movida para a do verbo) a informação de número mora.

É literalmente como se estudou concordância em transformers — e o resultado típico: a informação de número é lida na posição do sujeito por cabeças intermediárias e transportada para a posição do verbo. Bônus: identifique as cabeças que fazem o transporte com ablação.
</details>

---

### A4. Probe vs steering

Você treina um probe que detecta "o modelo está prestes a recusar" com 95% de acurácia na camada 14.

a) O que isso prova e o que não prova?
b) Como você converteria isso num experimento de steering, e o que cada resultado significaria?
c) Que aplicação de segurança isso habilita — e que risco?

<details><summary>Gabarito</summary>

a) Prova que "vou recusar" está linearmente REPRESENTADO na camada 14 — existe uma direção de recusa. NÃO prova que essa direção CAUSA a recusa; poderia ser um epifenômeno (o modelo já decidiu recusar por outro mecanismo, e essa direção é um reflexo).

b) Construa o vetor de recusa (média das ativações em prompts recusados − média nos aceitos) e SOME-o (deveria induzir recusa) ou SUBTRAIA-o (deveria suprimir recusa) na camada 14 durante a geração. Se somar induz recusa e subtrair a suprime → a direção é causal. Se não muda nada → era epifenômeno (representação sem uso).

c) Habilita: detectar em tempo real quando o modelo vai recusar (monitoramento) e ajustar a propensão a recusar sem retreinar (controle). O risco é simétrico e sério: a MESMA técnica que reforça a recusa pode SUPRIMI-LA — subtrair o vetor de recusa é uma forma de jailbreak por ativação. É por isso que interpretabilidade é dual-use e a comunidade de safety a estuda tanto ofensiva quanto defensivamente.
</details>

---

### A5. Superposição

Um colega quer "encontrar o neurônio do sentimento positivo" varrendo os neurônios da camada e vendo qual ativa mais em texto feliz.

Por que a busca provavelmente falha, e qual é a abordagem moderna?

<details><summary>Gabarito</summary>

Falha por causa da **superposição**: o modelo representa muito mais conceitos que neurônios, empacotando-os em direções sobrepostas. Cada neurônio é POLISSEMÂNTICO — ativa para "sentimento positivo" E "menções a comida" E "segunda pessoa do plural", tudo misturado. Não existe "o neurônio do sentimento".

Duas abordagens melhores: (1) **direções, não neurônios** — o sentimento é uma direção no espaço (combinação de neurônios), encontrável por probe/média de diferenças, não em um neurônio isolado; (2) **SAEs** — treinar um sparse autoencoder que reexpressa as ativações numa base maior e esparsa, na qual as features tendem a ser monossemânticas. O conceito "sentimento positivo" seria uma FEATURE do SAE, não um neurônio da rede.

A lição: a unidade de análise da interpretabilidade moderna é a *direção/feature*, não o *neurônio*. O neurônio foi a esperança ingênua que a superposição quebrou.
</details>

---

## Parte B — Práticas

### B1. 💻 O logit lens comparado

Aplique o logit lens (Lab 1) a três tipos de prompt: um fato memorizado ("The capital of France is"), uma operação ("2 + 2 ="), e uma continuação livre ("The weather today is"). Em qual camada a resposta emerge em cada caso?

<details><summary>Gabarito esperado</summary>

Espere: o fato memorizado emerge relativamente cedo (está "guardado", basta recuperar); a operação emerge mais tarde (exige computação ao longo das camadas); a continuação livre nunca converge para um único token (há muitas respostas válidas — a distribuição fica difusa até o fim).

A leitura: a camada de emergência é um proxy grosseiro de "quanta computação a resposta exige". É uma das poucas medidas interpretáveis de "dificuldade interna" de uma previsão — e conecta com o argumento de profundidade do módulo 7 (raciocínio precisa de mais passos = mais camadas/tokens).
</details>

---

### B2. 💻 O mapa de patching completo

Estenda o Lab 2 para patchar TODAS as posições (não só a última) × todas as camadas, e desenhe o heatmap ASCII de recuperação. Onde estão os dois "pontos quentes" clássicos?

<details><summary>Gabarito esperado</summary>

Espere dois focos: (1) nas **camadas iniciais, na posição do token do país** — onde o modelo "lê" o fato; (2) nas **camadas intermediárias/tardias, na posição da última palavra** — onde o fato foi transportado para gerar a resposta. Entre os dois, um "rio" de recuperação mostrando o fluxo da informação da posição do sujeito para a da resposta.

É o padrão canônico do causal tracing (Meng et al./ROME) e um dos resultados mais bonitos da área: você VÊ a informação viajar pela rede. Confirma que fatos são lidos cedo (perto do MLP que os armazena) e usados tarde.
</details>

---

### B3. 💻 Path patching de um circuito

O Lab 3 achou as cabeças importantes por ablação. Confirme que duas delas formam um CIRCUITO com path patching: patche a saída da cabeça A **apenas no caminho que chega à cabeça B** (não na saída geral) e veja se B ainda funciona.

Por que isso é mais forte que ablar A e B separadamente?

<details><summary>Gabarito</summary>

Ablar A e B separadamente mostra que ambas importam, mas não que se COMUNICAM — poderiam contribuir independentemente para a resposta. Path patching isola a conexão: se corromper A→B (mas não A→resto) quebra a tarefa, então B DEPENDE causalmente de A — elas formam um circuito, com A alimentando B.

É a diferença entre "estas duas peças são importantes" e "esta peça alimenta aquela". A segunda afirmação é o que "circuito" significa, e path patching é a ferramenta que a estabelece. É também o método dos papers de circuitos completos (IOI, "Indirect Object Identification", o circuito mais bem mapeado que existe — reproduza-o como projeto).
</details>

---

### B4. 💻 Steering de sentimento

Construa um steering vector de sentimento (média de ativações em frases positivas − negativas) e injete-o com forças crescentes numa geração neutra. Meça: a partir de que força a saída vira positiva? Quando ela degenera em incoerência?

<details><summary>Gabarito esperado</summary>

Espere uma janela útil: força baixa não muda nada, força média empurra o tom para positivo mantendo coerência, força alta degenera (repete palavras positivas, quebra a gramática). A curva "força × (positividade, coerência)" tem um ótimo — exatamente como a temperatura do módulo 1 e o β do módulo 8.

A lição de controle: steering é uma alavanca contínua e cega — ela empurra na direção, sem saber quando parar. Métodos de produção calibram a força por camada e às vezes só aplicam quando um probe detecta que é seguro. Controle por ativação é poderoso e grosseiro.
</details>

---

### B5. 💻 Um SAE mínimo (avançado)

Treine um sparse autoencoder de brinquedo sobre as ativações da camada 12 do Qwen: colete ativações de alguns milhares de tokens, treine `z = ReLU(W_enc·x)`, `x̂ = W_dec·z` com loss `||x − x̂||² + α·||z||₁`, com o dicionário `z` 4× maior que `d`.

Inspecione: as features de topo (dimensões de `z` que mais ativam) são interpretáveis? Que tokens/contextos as disparam?

<details><summary>Gabarito esperado</summary>

Com um SAE pequeno e pouco dado, espere features PARCIALMENTE interpretáveis — algumas dispararão para conceitos legíveis (dígitos, pontuação, um idioma, um tema), outras serão ruído. É o resultado honesto: monossemanticidade completa exige SAEs grandes, muito dado e truques de treino (o paper do Claude 3 usou dezenas de milhões de features).

O valor do exercício é sentir o método: o SAE é treinado com a loss do módulo 3, a esparsidade `α·||z||₁` é o que força cada feature a "escolher" um conceito, e o dicionário maior que `d` é a aposta contra a superposição. Você não vai reproduzir a Anthropic num laptop — mas vai entender exatamente o que eles fizeram, e por quê.
</details>

---

## Desafio — reproduza um circuito publicado

Escolha um circuito documentado e reproduza-o no Qwen (ou num GPT-2, mais estudado):

- **IOI (Indirect Object Identification)** — "When Mary and John went to the store, John gave a drink to ___" → Mary. O circuito mais bem mapeado que existe (Wang et al., 2022), com ~26 cabeças de papéis nomeados (name movers, S-inhibition, etc.).
- **Induction** — estenda o Lab 4 até mapear o circuito de duas cabeças (previous-token head → induction head).
- **Docstring / greater-than** — circuitos menores e bem documentados.

Entregue: o circuito identificado (quais componentes, que papel cada um), a validação por ablação E patching, e a honestidade sobre o que você NÃO conseguiu explicar (todo circuito real tem partes que resistem).

É o rito de passagem da interpretabilidade — e a ponte direta para o módulo 19 (reproduzir papers) e para trabalho de pesquisa real na área. Se você fizer isto bem, tem um projeto de portfólio que abre portas em times de safety.
