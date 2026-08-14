# Glossário — os termos do curso, explicados para humanos

Cada verbete tem três partes: **a ideia em linguagem comum** (muitas vezes uma analogia), a definição precisa, e onde aparece no curso. Está organizado na ordem em que os conceitos surgem nos módulos — dá para ler de ponta a ponta como um "curso em miniatura" — e o índice alfabético abaixo serve para consulta rápida.

**Índice alfabético:**
[Adaptador](#adaptador-lora) · [AdamW](#adamw) · [Alucinação](#alucinação) · [Atenção](#atenção-self-attention) · [Attention sink](#attention-sink) · [Baseline (avaliação)](#baseline-de-avaliação) · [Baseline (RL)](#baseline-e-vantagem-rl) · [Batch](#batch) · [Benchmark](#benchmark) · [bf16/fp16](#precisão-numérica-fp32-fp16-bf16) · [BPE](#bpe) · [Catastrophic forgetting](#catastrophic-forgetting) · [Chain-of-thought](#chain-of-thought-cot) · [Chat template](#chat-template) · [Checkpoint](#checkpoint) · [Chosen/rejected](#chosen--rejected) · [Contaminação](#contaminação-de-benchmark) · [Contexto](#janela-de-contexto) · [Cross-entropy](#loss-e-cross-entropy) · [Dark knowledge](#dark-knowledge) · [Decode](#prefill-e-decode) · [Deduplicação](#deduplicação) · [Distillation](#distillation-destilação) · [DPO](#dpo) · [Draft model](#decodificação-especulativa-e-draft-model) · [Embedding](#embedding) · [EOS](#eos) · [Época](#época) · [Expert](#moe-mixture-of-experts) · [Fine-tuning](#fine-tuning) · [FLOPs](#flops) · [Forward/backward](#forward-e-backward) · [GQA](#gqa) · [Gradiente](#gradiente) · [Gradient accumulation](#gradient-accumulation) · [Gradient clipping](#gradient-clipping) · [Greedy](#estratégias-de-decodificação-greedy-top-k-top-p) · [GRPO](#grpo) · [Hiperparâmetro](#hiperparâmetro) · [Inferência](#inferência) · [KL (divergência)](#divergência-kl) · [KV cache](#kv-cache) · [Learning rate](#learning-rate) · [LLM-as-judge](#llm-as-judge) · [Logit](#logit) · [LoRA](#adaptador-lora) · [Loss](#loss-e-cross-entropy) · [Masking](#masking-de-loss--100) · [Máscara causal](#máscara-causal) · [MLP](#mlp-feed-forward) · [MoE](#moe-mixture-of-experts) · [Modelo base/instruct](#modelo-base-instruct-e-reasoning) · [Overfitting](#overfitting) · [Parâmetros/pesos](#parâmetros-pesos) · [pass@k](#passk) · [Perplexidade](#perplexidade-ppl) · [Política](#política-policy) · [PPO](#ppo) · [Prefill](#prefill-e-decode) · [Pré-treino](#pré-treino) · [Prompt](#prompt) · [Quantização](#quantização) · [Rank](#rank-posto) · [Recompensa](#recompensa-e-recompensa-verificável) · [Reward hacking](#reward-hacking) · [Reward model](#reward-model) · [RLHF](#rlhf) · [RoPE](#rope) · [Roteador](#moe-mixture-of-experts) · [Sampling](#estratégias-de-decodificação-greedy-top-k-top-p) · [Self-consistency](#self-consistency) · [SFT](#sft-supervised-fine-tuning) · [Softmax](#softmax) · [Soft target](#soft-target) · [Split](#split-treinovalidaçãoteste) · [Temperatura](#temperatura) · [Token](#token) · [Tokenizer](#tokenizer) · [Transformer](#transformer) · [TTFT/TPOT](#ttft-e-tpot) · [Vazamento](#vazamento-de-dados) · [Verificador](#recompensa-e-recompensa-verificável) · [Vocabulário](#vocabulário) · [Warmup](#warmup)

---

## Os fundamentos (módulos 1–2)

### Token
**A ideia:** o modelo não lê letras nem palavras — lê "peças de Lego" de texto. `implementação` pode ser uma peça só, ou quebrar em `implement` + `ação`. Cada peça tem um número de identificação.
**Precisão:** a unidade mínima de texto que o modelo processa; um inteiro que indexa o vocabulário. Em português, ~3–3,5 caracteres por token.
**Onde:** módulo 1, Lab 1 — inclusive por que português custa mais tokens que inglês.

### Tokenizer
**A ideia:** o picador que transforma texto nas peças de Lego (e remonta o texto no final). Cada modelo tem o seu, e eles não são intercambiáveis — trocar o tokenizer de um modelo é embaralhar o dicionário dele.
**Precisão:** o algoritmo (e a tabela aprendida) que converte texto ↔ sequência de ids de tokens.
**Onde:** módulo 1.

### BPE
**A ideia:** o método que decide quais pedaços de texto viram peças: começa com letras soltas e vai colando os pares que mais aparecem juntos, até ter (digamos) 50 mil peças. Palavras comuns viram uma peça só; raras se quebram.
**Precisão:** Byte-Pair Encoding — fusão iterativa dos pares de símbolos mais frequentes do corpus até atingir o tamanho de vocabulário alvo.
**Onde:** módulos 1 e 3 (no 3 você treina um do zero).

### Vocabulário
**A ideia:** a caixa completa de peças de Lego do modelo — tipicamente entre 32 mil e 256 mil peças diferentes.
**Precisão:** o conjunto de tokens que o tokenizer conhece; seu tamanho (V) define a dimensão da camada de saída.
**Onde:** módulo 1.

### Embedding
**A ideia:** cada peça de Lego ganha um "perfil numérico" — uma lista de centenas de números que codifica o que ela significa. Peças de significado parecido têm perfis parecidos. É como converter palavras em coordenadas num mapa gigante onde a vizinhança importa.
**Precisão:** o vetor denso (ex.: 896 dimensões) associado a cada token, armazenado numa matriz `[V, d]` aprendida no treino.
**Onde:** módulo 1, seção 3.

### Logit
**A ideia:** antes de decidir a próxima palavra, o modelo dá uma "nota bruta" para cada peça do vocabulário — sem escala definida, pode ser negativa. Ainda não é probabilidade; é o rascunho da opinião.
**Precisão:** a saída da última camada antes do softmax; um número real por token do vocabulário.
**Onde:** módulo 1, Lab 3.

### Softmax
**A ideia:** a máquina que converte as notas brutas em porcentagens que somam 100% — "40% de chance de ser 'telhado', 25% de 'muro'...".
**Precisão:** `p_i = exp(z_i)/Σexp(z_j)` — transforma logits em uma distribuição de probabilidade.
**Onde:** módulos 1 e 2 (na atenção também).

### Temperatura
**A ideia:** o botão de ousadia. Perto de zero, o modelo sempre escolhe a opção mais provável (seguro e repetitivo); alto, ele arrisca opções improváveis (criativo, e depois de um ponto, incoerente).
**Precisão:** divisor aplicado aos logits antes do softmax; T→0 colapsa no argmax, T alto achata a distribuição.
**Onde:** módulo 1, Lab 4.

### Estratégias de decodificação (greedy, top-k, top-p)
**A ideia:** ter as porcentagens não basta — é preciso uma regra para escolher. *Greedy*: sempre a mais provável (robô previsível). *Top-k*: sorteia entre as k melhores. *Top-p*: sorteia entre as melhores que juntas somam p% de chance — o tamanho da lista se adapta à confiança do modelo.
**Precisão:** algoritmos de amostragem sobre a distribuição do próximo token; top-p (nucleus) é o padrão atual.
**Onde:** módulo 1, Labs 5–6 (você os implementa do zero).

### Prompt
**A ideia:** tudo o que você entrega ao modelo antes de ele começar a responder — a pergunta, as instruções, os exemplos. O modelo não "lembra" de nada além do que está no prompt.
**Precisão:** a sequência de tokens de entrada que condiciona a geração.
**Onde:** todos os módulos.

### Loss e cross-entropy
**A ideia:** a nota de erro do modelo durante o treino. Para cada posição do texto, pergunta-se: "quanta probabilidade você deu à palavra que realmente veio?" — deu alta, loss baixa; deu quase zero, loss enorme. Treinar é fazer essa nota cair.
**Precisão:** `L = −média(log p(token correto))`. É a única função otimizada no pré-treino e no SFT.
**Onde:** módulos 1, 3, 5.

### Perplexidade (PPL)
**A ideia:** "entre quantas opções o modelo está em dúvida, em média". PPL 10 = como se hesitasse entre 10 palavras igualmente plausíveis a cada passo. Menor = modelo mais seguro (nos dados medidos!).
**Precisão:** `exp(loss)`. Não comparável entre tokenizers diferentes.
**Onde:** módulo 1, Lab 7 — incluindo os jeitos de medi-la errado.

### Chat template
**A ideia:** o "formulário" invisível que marca quem disse o quê numa conversa (`<|im_start|>user...`). O modelo foi treinado com esse formulário exato; enviar texto sem ele é como entregar uma petição sem os campos — ele responde muito pior, sem avisar.
**Precisão:** a marcação de turnos que o tokenizer aplica via `apply_chat_template`.
**Onde:** módulo 1 (seção 9) e a armadilha nº 1 dos módulos 4–5.

### EOS
**A ideia:** o token "fim da fala". Se os dados de treino não o incluem, o modelo aprende a responder mas nunca a *parar* — responde certo e continua falando sozinho.
**Precisão:** end-of-sequence token; encerra a geração quando emitido.
**Onde:** módulos 4 e 5 (a armadilha nº 1 do SFT).

### Modelo base, instruct e reasoning
**A ideia:** três estágios do mesmo modelo. *Base*: leu a internet inteira e só sabe **continuar texto** (pergunte algo e ele pode responder com mais perguntas). *Instruct*: passou por acabamento para conversar e obedecer. *Reasoning*: treinado para "pensar por escrito" antes de responder.
**Precisão:** base = só pré-treino; instruct = base + SFT + alinhamento; reasoning = instruct + RL sobre tarefas verificáveis.
**Onde:** módulo 1 (seção 9), módulos 5, 7, 9.

### Parâmetros (pesos)
**A ideia:** os "botões de ajuste" internos do modelo — bilhões de números que o treino calibra. "Modelo de 7B" = 7 bilhões de botões. Todo o conhecimento vive neles.
**Precisão:** os tensores treináveis; a contagem define custo de memória e compute.
**Onde:** todos; a anatomia detalhada no módulo 2, Lab 9.

### Inferência
**A ideia:** usar o modelo (gerar respostas), em oposição a treiná-lo. É a fase "de produção" — e tem economia própria.
**Precisão:** o forward do modelo para gerar tokens, sem gradientes.
**Onde:** módulos 1 e 11.

### Prefill e decode
**A ideia:** responder tem duas fases. *Prefill*: ler o prompt inteiro de uma vez (rápido, paralelo — como fotografar uma página). *Decode*: escrever a resposta token por token (lento, sequencial — como datilografar). O caro é o decode, e o gargalo dele não é fazer contas: é **buscar os bilhões de pesos na memória** a cada token.
**Precisão:** prefill é compute-bound e O(n²); decode é memory-bandwidth-bound.
**Onde:** módulos 1 (seção 8) e 11 (toda a economia de servir).

### KV cache
**A ideia:** o "rascunho" que o modelo guarda sobre tudo que já leu, para não reler do zero a cada palavra nova. Ocupa memória de verdade — em conversas longas, mais que o próprio modelo — e é ele que limita quantos usuários cabem numa GPU.
**Precisão:** as matrizes K e V de cada camada, armazenadas por token processado.
**Onde:** módulo 1 (fórmula e medição), módulo 11 (PagedAttention).

### Janela de contexto
**A ideia:** o campo de visão do modelo — quantos tokens ele consegue considerar de uma vez (8k, 128k...). Fora da janela, não existe.
**Precisão:** o comprimento máximo de sequência suportado; limitado por treino (posições vistas) e por memória (KV cache).
**Onde:** módulos 1 e 2 (RoPE e extensão de contexto).

### Alucinação
**A ideia:** o modelo afirmar com confiança algo falso. Não é defeito de fabricação: ele foi treinado para gerar texto *plausível*, e plausível ≠ verdadeiro. Quando não sabe, o texto mais plausível ainda parece uma resposta.
**Precisão:** geração de conteúdo factualmente incorreto com alta fluência; comportamento esperado de um maximizador de verossimilhança fora da distribuição.
**Onde:** módulo 1 (seção 1) e a resposta prática (RAG) nos módulos 4–5.

### FLOPs
**A ideia:** a unidade de "conta feita" — quantas operações aritméticas o treino ou a inferência consome. É como medir uma obra em horas-máquina; converte direto em tempo de GPU e em dólares.
**Precisão:** floating-point operations; treino ≈ `6 × parâmetros × tokens`.
**Onde:** módulo 3, Lab 9 (a calculadora de custo).

### Transformer
**A ideia:** a arquitetura de todos os LLMs modernos. Uma pilha de andares idênticos; em cada andar, as palavras "conversam entre si" (atenção) e depois cada uma é processada individualmente (MLP).
**Precisão:** blocos de self-attention + MLP com conexões residuais e normalização, empilhados.
**Onde:** módulo 2 inteiro — reconstruído do zero e validado bit a bit.

### Atenção (self-attention)
**A ideia:** o mecanismo pelo qual cada palavra olha para as anteriores e decide quais importam para ela. Em "o gato que vi ontem *fugiu*", o verbo precisa achar o sujeito — a atenção é a busca que o encontra.
**Precisão:** `softmax(QKᵀ/√d)V` — médias ponderadas dos values, com pesos dados pela afinidade query–key.
**Onde:** módulo 2, seções 2–3 e Labs 1–4.

### Q, K, V (query, key, value)
**A ideia:** três papéis que cada palavra exerce na busca. *Query*: "o que eu procuro". *Key*: "como me encontram". *Value*: "o que eu entrego se me acharem". Separar o critério de busca do conteúdo entregue é o truque — como um fichário onde a etiqueta da ficha é diferente do conteúdo dela.
**Precisão:** três projeções lineares aprendidas do mesmo vetor de entrada.
**Onde:** módulo 2, seção 2.

### Máscara causal
**A ideia:** a venda que impede cada palavra de espiar as palavras futuras durante o treino. Sem ela, prever "a próxima palavra" seria trapaça (a resposta está ali do lado) e o modelo não aprenderia nada útil.
**Precisão:** matriz triangular de −∞ somada aos scores de atenção antes do softmax.
**Onde:** módulo 2, Lab 3 (com prova de que o futuro vaza sem ela).

### GQA
**A ideia:** um truque de economia: várias "buscas" (queries) compartilham o mesmo fichário (keys/values). Corta o rascunho de memória (KV cache) em 4–7× quase sem perda de qualidade.
**Precisão:** Grouped-Query Attention — grupos de query heads compartilham cada KV head.
**Onde:** módulos 1 (a conta de memória) e 2 (Lab 5).

### RoPE
**A ideia:** o jeito moderno de informar ao modelo *a ordem* das palavras (sem isso, "o cão mordeu o homem" e "o homem mordeu o cão" seriam idênticos). Em vez de numerar as posições, ele **gira** os vetores num ângulo proporcional à posição — como ponteiros de relógio: a *diferença* entre dois ponteiros diz a distância entre as palavras.
**Precisão:** Rotary Position Embedding — rotação de pares de componentes de q e k por ângulo m·θᵢ; o produto interno resulta função só da posição relativa.
**Onde:** módulo 2, seção 6 e Lab 6 (a identidade verificada numericamente).

### MLP (feed-forward)
**A ideia:** a "estação de processamento individual" de cada andar: depois de conversar com as outras via atenção, cada palavra passa sozinha por uma mini-rede que expande, transforma e comprime. É onde mora a maior parte dos botões do modelo (~88%!) e, ao que tudo indica, a maior parte dos fatos memorizados.
**Precisão:** duas ou três projeções lineares com não-linearidade (SwiGLU nos modelos atuais); 85–90% dos parâmetros de um bloco.
**Onde:** módulo 2, seções 7–8.

### Attention sink
**A ideia:** um vício curioso: quase toda a atenção do modelo escorre para o **primeiro token** do texto, mesmo quando ele é irrelevante — porque o softmax obriga as porcentagens a somarem 100%, e quando uma cabeça de atenção "não precisa de nada", despeja tudo no ralo mais próximo. Medimos 92% da atenção indo para o token 0 em algumas camadas.
**Precisão:** concentração de massa de atenção no(s) primeiro(s) token(s); origem dos outliers de ativação que atrapalham quantização.
**Onde:** módulo 2 (Lab 7) e módulo 11 (por que quantizar é difícil).

---

## Treino e dados (módulos 3–4)

### Pré-treino
**A ideia:** a fase cara: o modelo lê trilhões de palavras da internet e aprende, só de tentar prever a próxima palavra, a língua, os fatos e os padrões do mundo. Custa milhões de dólares e é >98% de todo o esforço — tudo que vem depois é acabamento.
**Precisão:** treino auto-supervisionado de next-token prediction em corpus massivo.
**Onde:** módulo 3 (você faz um em miniatura, em 2 minutos).

### Gradiente
**A ideia:** a seta que diz, para cada um dos bilhões de botões, "gire um pouquinho para cá que o erro diminui". O treino inteiro é: medir o erro, calcular as setas, girar os botões, repetir.
**Precisão:** o vetor de derivadas parciais da loss em relação aos parâmetros.
**Onde:** módulo 3.

### Forward e backward
**A ideia:** os dois movimentos de cada passo de treino. *Forward*: a entrada atravessa o modelo e produz a previsão (e o erro). *Backward*: o erro volta pelo caminho inverso calculando a seta de ajuste de cada botão.
**Precisão:** propagação da ativação e retropropagação do gradiente (backpropagation).
**Onde:** módulo 3.

### AdamW
**A ideia:** o "piloto automático" que decide o tamanho do giro de cada botão. Ele guarda, por botão, uma média da direção recente e da intensidade típica — botões de gradiente barulhento recebem passos menores. É o otimizador padrão de todos os LLMs. Detalhe: guarda 12 bytes de anotações por botão — é por isso que treinar é tão mais caro em memória do que usar.
**Precisão:** Adam com weight decay desacoplado; estados m e v por parâmetro; os 16 bytes/parâmetro do full fine-tuning.
**Onde:** módulo 3, Lab 4 (implementado do zero e conferido).

### Learning rate
**A ideia:** o tamanho do passo. Grande demais: o treino explode ou oscila. Pequeno demais: nunca chega lá. É o hiperparâmetro mais importante de todos, e o primeiro suspeito de qualquer treino que "não funcionou".
**Precisão:** o multiplicador escalar do passo do otimizador; em LLMs, sempre com agenda (warmup + decay).
**Onde:** módulos 3 e 5 (LoRA usa 10× o LR de full fine-tune!).

### Warmup
**A ideia:** começar o treino em câmera lenta. Nos primeiros passos, o piloto automático (AdamW) ainda não calibrou suas médias — um passo grande com estatísticas ruins pode arruinar o modelo logo de saída. Sobe-se o learning rate gradualmente do zero.
**Precisão:** rampa linear do LR nos primeiros 0,1–2% dos passos.
**Onde:** módulo 3, seção 7.

### Batch
**A ideia:** o "lote" de exemplos processados de uma vez antes de cada ajuste de botões. Lotes maiores dão uma direção de ajuste mais confiável (menos ruído), como pesquisar com mais entrevistados.
**Precisão:** o conjunto de sequências por passo; em pré-treino, medido em milhões de tokens.
**Onde:** módulo 3.

### Gradient accumulation
**A ideia:** truque para simular um lote gigante quando a memória só comporta um pequeno: processa vários mini-lotes anotando as setas, e só ajusta os botões no final da rodada. Matematicamente idêntico ao lote grande — desde que se lembre de dividir (o lab prova, e mostra o estrago de esquecer).
**Precisão:** somar gradientes de micro-batches antes do optimizer.step(), dividindo a loss pelo número de acumulações.
**Onde:** módulo 3, Lab 5.

### Gradient clipping
**A ideia:** o disjuntor. Se um lote podre gera uma seta absurdamente grande (que destruiria semanas de treino num passo), o clipping corta o excesso. Quase nunca atua — e nas poucas vezes, salva o treino.
**Precisão:** reescalar o gradiente global quando sua norma excede um limiar (universalmente 1,0).
**Onde:** módulo 3.

### Época
**A ideia:** uma volta completa pelos dados de treino. Pré-treino faz ~1 volta (dados demais); fine-tuning faz 1–3 (mais que isso, o modelo decora em vez de aprender). O curso insiste: **calcule antes** — `épocas = passos × batch ÷ exemplos`.
**Precisão:** uma passagem completa pelo dataset.
**Onde:** módulos 3 e 5.

### Precisão numérica (fp32, fp16, bf16)
**A ideia:** com quantas "casas decimais" os números do modelo são guardados. Menos casas = metade da memória e mais velocidade, mas estoura mais fácil. O formato bf16 venceu porque aceita números enormes (mesmo alcance do fp32) sacrificando casas decimais — e as casas que faltam, o otimizador compensa.
**Precisão:** formatos de ponto flutuante de 32/16 bits; bf16 mantém os 8 bits de expoente do fp32.
**Onde:** módulo 3, Lab 6 (onde cada um quebra, medido).

### Checkpoint
**A ideia:** o "save game" do treino. Salva-se o estado completo periodicamente para poder voltar se algo der errado (e algo dá).
**Precisão:** snapshot de pesos + estados do otimizador + posição nos dados.
**Onde:** módulo 3.

### Overfitting
**A ideia:** decorar em vez de aprender. O aluno que memoriza o gabarito tira 10 na lista de exercícios e zero na prova nova. Sintoma inconfundível: erro caindo nos dados de treino e subindo nos de validação.
**Precisão:** gap crescente entre loss de treino e de validação; memorização da amostra em detrimento da generalização.
**Onde:** módulos 3, 4, 5.

### Split (treino/validação/teste)
**A ideia:** dividir os dados em três gavetas: uma para treinar, uma para acompanhar o progresso durante o treino, e uma **lacrada** que só se abre no final para a nota honesta. Misturar as gavetas invalida tudo.
**Precisão:** partição do dataset; o teste jamais influencia decisões de treino.
**Onde:** módulos 4 e 5 (onde um vazamento sutil foi pego por asserção).

### Vazamento de dados
**A ideia:** quando a prova final "vaza" para o material de estudo — o modelo é testado em algo que ele viu (ou quase viu) no treino, e a nota vira ilusão. É o erro mais comum e mais caro de avaliação.
**Precisão:** sobreposição entre treino e teste, direta ou por quase-duplicatas.
**Onde:** módulo 4 (detecção) e 5 (o caso real do split por fraseado).

### Deduplicação
**A ideia:** a faxina que remove textos repetidos ou quase-repetidos do corpus. A internet repete tudo; sem faxina, o modelo decora em vez de generalizar. É a etapa de preparação de dados de maior impacto comprovado.
**Precisão:** remoção de duplicatas exatas (hash) e aproximadas (MinHash + LSH).
**Onde:** módulo 4, Lab 2 (implementada do zero).

### Contaminação de benchmark
**A ideia:** quando as questões da prova oficial (benchmark) estavam no material de treino. O modelo "gabarita" por memória, e a nota não mede capacidade nenhuma.
**Precisão:** presença de itens de avaliação no corpus de treino; detectada por sobreposição de n-gramas.
**Onde:** módulo 4, Lab 5.

### Benchmark
**A ideia:** uma prova padronizada (MMLU, GSM8K...) que permite comparar modelos. Útil e perigosa: vira alvo, sofre contaminação, e mede só o que mede.
**Precisão:** conjunto fixo de tarefas com métrica definida.
**Onde:** módulos 4 e 7.

### LLM-as-judge
**A ideia:** usar um modelo grande como corretor de redação: ele compara duas respostas e diz qual é melhor. Escala barato, mas tem manias conhecidas — prefere respostas longas, prefere a primeira opção apresentada, prefere o próprio estilo. Usável, com controles.
**Precisão:** avaliação por modelo com prompt de julgamento; vieses de posição/comprimento mitigáveis por permutação e rubrica.
**Onde:** módulos 4, 5 (vieses e mitigação), 8.

---

## Customização (módulos 5–6)

### Fine-tuning
**A ideia:** o guarda-chuva de todas as técnicas que **ajustam um modelo pronto** para o seu caso, em vez de treinar do zero. A diferença de custo é brutal: o zero custa milhões; o ajuste, de graça a alguns dólares.
**Precisão:** continuar o treino de um modelo pré-treinado em dados específicos.
**Onde:** módulos 5–10, cada um uma variante.

### SFT (supervised fine-tuning)
**A ideia:** ensinar por exemplos resolvidos: mostra-se ao modelo milhares de pares pergunta→resposta-ideal e ele aprende a imitar. É como estagiário aprendendo pelo manual de casos. Ensina **comportamento** (formato, tom) — não instala conhecimento novo.
**Precisão:** cross-entropy sobre pares instrução→resposta, com loss geralmente só nos tokens da resposta.
**Onde:** módulo 5 inteiro.

### Masking de loss (−100)
**A ideia:** durante o SFT, marcar quais palavras "contam ponto" no erro. Marca-se a pergunta do usuário com −100 ("não conte"), para o modelo aprender a *responder* — não a redigir perguntas.
**Precisão:** posições com label −100 são ignoradas pela cross-entropy do PyTorch.
**Onde:** módulos 4 (Lab 6) e 5. Exceção importante no módulo 7: raciocínio NÃO se mascara.

### Catastrophic forgetting
**A ideia:** o efeito colateral de estudar só uma matéria: o modelo fica ótimo no seu domínio e **esquece o resto** (piora em conhecimentos gerais). Não é raro — é o comportamento padrão, e se mede, não se supõe.
**Precisão:** degradação em tarefas fora do domínio de fine-tuning; medida por perplexidade antes/depois em textos diversos.
**Onde:** módulo 5 (Lab 5) e 6 (LoRA esquece 4× menos, medido).

### Baseline de avaliação
**A ideia:** o "concorrente honesto" contra o qual seu modelo treinado deve ser comparado. Não é o modelo cru sem instruções (fácil demais de vencer) — é o modelo cru **com o melhor prompt que você conseguir escrever**. Se um bom prompt resolve, o treino não provou nada.
**Precisão:** referência de comparação; no curso, sempre em três níveis (prompt simples, prompt esforçado com exemplos, RAG se aplicável).
**Onde:** módulo 5, Lab 2 — e o módulo 12 a exige.

### Hiperparâmetro
**A ideia:** os ajustes que VOCÊ escolhe antes do treino (learning rate, épocas, batch...) — em oposição aos parâmetros, que o treino ajusta sozinho. São o painel de controle do experimento.
**Precisão:** configurações do processo de treino, não aprendidas.
**Onde:** todos os módulos de treino; o 5 exige justificá-los.

### Adaptador (LoRA)
**A ideia:** em vez de reformar o prédio inteiro (ajustar bilhões de botões), instala-se um **módulo pequeno e removível** ao lado de cada parede importante — só ele é ajustado; o prédio original fica intacto. Resultado: treinar custa ~1% da memória, o arquivo final tem megabytes (não gigabytes), e dá para remover ou trocar o módulo a qualquer momento.
**Precisão:** Low-Rank Adaptation — a atualização ΔW é fatorada em duas matrizes finas B·A de posto r; só elas treinam; `W + (α/r)·B·A` pode ser fundida sem custo de latência.
**Onde:** módulo 6 inteiro (implementado do zero, com as três propriedades provadas).

### Rank (posto)
**A ideia:** a "largura" do módulo removível do LoRA — quanta capacidade de mudança ele tem. Rank 8 basta para mudar estilo e formato; tarefas maiores pedem 32–64. Mais rank = mais capacidade e mais custo, com retorno decrescente (medido no curso: 32× mais parâmetros compraram 2 pontos).
**Precisão:** a dimensão interna r da fatoração B·A.
**Onde:** módulo 6.

### QLoRA
**A ideia:** LoRA sobre um modelo **comprimido**: o prédio congela E é guardado em formato compacto (4 bits); só os módulos removíveis ficam em precisão cheia. É o que permite treinar um modelo de 8 bilhões de parâmetros num Mac de 16 GB.
**Precisão:** base quantizada em NF4 congelada + adaptadores LoRA em bf16.
**Onde:** módulo 6 (a quantização NF4 implementada do zero).

### Quantização
**A ideia:** guardar cada botão do modelo com menos "casas decimais" — como comprimir uma foto: 4 bits por número em vez de 16 corta a memória a ¼, com perda pequena (e medível — e maior em português, o curso mediu). É a diferença entre "não cabe" e "cabe" no seu hardware.
**Precisão:** mapear pesos para níveis discretos de baixa precisão (int4/NF4/etc.), por blocos com constante de escala.
**Onde:** módulos 6 (fundamentos, NF4 do zero) e 11 (o cardápio de servir: GPTQ, AWQ, GGUF).

### VRAM / memória unificada
**A ideia:** a memória de trabalho onde o modelo precisa caber para rodar. Nas GPUs é a VRAM (fixa, cara); nos Macs, a memória unificada (a GPU usa a RAM toda). É o recurso que decide o que você consegue rodar e treinar.
**Precisão:** memória acessível ao acelerador; no M4 do curso, 16 GB unificados ≈ 10 úteis.
**Onde:** `00-setup-mac.md` e módulo 6 (a tabela do que cabe).

---

## Raciocínio e alinhamento (módulos 7–9)

### Chain-of-thought (CoT)
**A ideia:** pedir que o modelo **mostre o rascunho** antes da resposta. Não é cosmético: cada palavra escrita é um passo extra de computação, e o rascunho funciona como memória de trabalho — contas intermediárias ficam anotadas onde o modelo consegue relê-las. Medido no curso: a resposta certa fica 38× mais provável com o rascunho no contexto.
**Precisão:** geração de passos intermediários antes da resposta final; converte profundidade fixa em computação proporcional ao comprimento.
**Onde:** módulo 7 inteiro.

### Self-consistency
**A ideia:** resolver a mesma questão 5 vezes (com variação) e ficar com a resposta mais votada. Funciona porque os erros se espalham (cada tentativa erra diferente) e os acertos coincidem. Pré-requisito que o curso mediu na prática: o modelo precisa acertar com frequência razoável — votar entre erros só formaliza o ruído.
**Precisão:** amostrar k cadeias e votar na resposta extraída; amplifica competência existente.
**Onde:** módulo 7, Lab 4.

### pass@k
**A ideia:** "em k tentativas, pelo menos uma acerta?" — a métrica de quando você tem como *verificar* cada tentativa (testes de código). pass@1 é a nota honesta de uso único.
**Precisão:** probabilidade de ≥1 sucesso em k amostras; estimador sem viés via combinatória.
**Onde:** módulo 7 (exercício B3).

### RLHF
**A ideia:** o processo que transformou "modelos que continuam texto" em assistentes: humanos comparam respostas, um modelo-juiz aprende o gosto deles, e o modelo principal é treinado para agradar esse juiz sem se descaracterizar. Caro e complexo — o DPO existe para atalhar isso.
**Precisão:** Reinforcement Learning from Human Feedback: SFT → reward model → PPO com penalidade KL.
**Onde:** módulo 8, seção 1.

### Reward model
**A ideia:** o "juiz artificial": um modelo treinado para dar nota a respostas, imitando as preferências humanas coletadas. O RL depois maximiza essa nota — com o risco conhecido de o aluno aprender a enganar o juiz.
**Precisão:** modelo que mapeia (prompt, resposta) → escalar de qualidade.
**Onde:** módulos 8 e 9 (por que recompensas por regra são mais seguras).

### Chosen / rejected
**A ideia:** o formato dos dados de preferência: para a mesma pergunta, a resposta **preferida** e a **preterida**. O modelo aprende com o *contraste* — coisa que exemplos soltos não ensinam.
**Precisão:** os pares (y_w, y_l) do treino de preferências.
**Onde:** módulo 8.

### DPO
**A ideia:** o atalho que dispensou o juiz e o RL: uma conta direta sobre os pares preferida/preterida que produz o mesmo efeito do RLHF. O truque matemático central: a parte incomputável da equação aparece dos dois lados da comparação e **se cancela**. Virou classificação binária.
**Precisão:** Direct Preference Optimization — `−log σ(β·[Δlogprob_chosen − Δlogprob_rejected])`; a política ótima do objetivo KL-regularizado em forma fechada.
**Onde:** módulo 8 (derivação completa + implementação + patologias medidas).

### Divergência KL
**A ideia:** a régua de "quão diferente uma distribuição de probabilidade é de outra". No alinhamento, funciona como **elástico**: prende o modelo novo ao original, impedindo que ele se descaracterize enquanto persegue a recompensa. Solto o elástico (o curso mediu), o modelo colapsa em degeneração.
**Precisão:** `KL(p‖q) = Σ p·log(p/q)`; assimétrica — a direção importa (módulo 10).
**Onde:** módulos 8 (o β do DPO), 9 (a penalidade do GRPO), 10 (forward vs reverse).

### On-policy / off-policy
**A ideia:** treinar com material *que o próprio modelo produz agora* (on-policy) versus material de outra origem (off-policy). A diferença importa: corrigir erros que o modelo nunca cometeria é malhar em ferro frio — paga-se o custo do treino sem mover o comportamento real.
**Precisão:** se os dados de treino vêm da distribuição da política atual ou não.
**Onde:** módulos 8 (a limitação do DPO) e 9 (a vantagem do RL).

### Política (policy)
**A ideia:** no vocabulário de RL, "política" é simplesmente **o modelo sendo treinado** — a estratégia atual de gerar respostas. "Melhorar a política" = melhorar o modelo.
**Precisão:** a distribuição π(resposta|prompt) parametrizada pelo modelo.
**Onde:** módulos 8–9.

### Recompensa e recompensa verificável
**A ideia:** a "pontuação" que diz ao RL o que é sucesso. A revolução recente: usar recompensas **verificáveis por regra** — a resposta confere com o gabarito? o teste passa? — em vez de opiniões de juiz. Regra não tem gosto para ser enganado.
**Precisão:** função R(resposta) → escalar; verificável = computada por verificador exato (RLVR).
**Onde:** módulo 9 (a receita do R1).

### PPO
**A ideia:** o algoritmo clássico de RL para LLMs: melhora o modelo aos poucos, com uma trava que o proíbe de mudar demais de uma vez (mudanças bruscas em RL costumam ser catastróficas). Poderoso e pesado: exige quatro modelos na memória ao mesmo tempo.
**Precisão:** Proximal Policy Optimization — clipped surrogate objective + value model como baseline.
**Onde:** módulo 9, seção 3.

### Baseline e vantagem (RL)
**A ideia:** para aprender, não basta saber a nota — importa saber se foi **acima ou abaixo do esperado**. Tirar 7 vale muito se a média é 5, e pouco se é 9. A "vantagem" é exatamente isso: nota menos expectativa. Subtrair a expectativa não muda a direção do aprendizado, mas corta drasticamente o ruído (53× no experimento do curso).
**Precisão:** A = R − b; subtrair baseline não enviesa o policy gradient e reduz variância.
**Onde:** módulo 9, Lab 1.

### GRPO
**A ideia:** a simplificação que viabilizou o DeepSeek-R1: em vez de um modelo extra para estimar a expectativa (o value model do PPO), gera-se um **grupo** de 4–16 respostas para a mesma pergunta e usa-se a média do grupo como régua. Quem ficou acima da média do próprio grupo é reforçado. Elegante, e de graça: elimina metade da memória do PPO.
**Precisão:** Group Relative Policy Optimization — vantagem = z-score da recompensa dentro do grupo; clipped ratio + KL contra referência.
**Onde:** módulo 9 (implementado do zero: 27% → 90% de sucesso medido).

### Reward hacking
**A ideia:** o gênio da lâmpada maligno: o RL realiza **o pedido literal**, não a intenção. Recompense "conter travessões" por contagem e o modelo devolve `----------------------` — pontuação máxima, texto destruído. O curso produziu esse desastre de propósito, e a curva de treino ficava *linda* enquanto isso.
**Precisão:** exploração de discrepâncias entre a recompensa escrita e a intenção; lei de Goodhart aplicada.
**Onde:** módulo 9, Lab 4 — e a checklist de defesas (recompensa binária, KL, ler as gerações).

### Entropia
**A ideia:** o termômetro de "quanta variedade/incerteza" há numa distribuição. Alta = o modelo considera muitas opções; baixa = decidido (ou colapsado — em RL, entropia despencando é o alarme antecipado de que o modelo virou um robô de um truque só).
**Precisão:** `H = −Σ p·log p`; em nats.
**Onde:** módulos 1, 7, 9 (exercício B4), 10.

---

## Compressão e produção (módulos 10–11)

### Distillation (destilação)
**A ideia:** o professor particular: um modelo grande e caro "ensina" um pequeno e barato — pelo texto que gera (o pequeno imita) ou pelas suas probabilidades internas (que carregam mais informação). É como o R1 gigante virou versões de bolso que rodam num laptop. Regra de ouro: transfere bem, raramente supera o professor.
**Precisão:** treinar um aluno para imitar as saídas (black-box) ou as distribuições (white-box) de um professor.
**Onde:** módulo 10 inteiro (+16,3% medido sobre treinar do zero).

### Soft target
**A ideia:** a diferença entre corrigir com gabarito ("a resposta é B") e corrigir como um bom professor ("é B, mas C era quase — e D é absurda"). A distribuição completa do professor ensina as *relações* entre as opções, não só o acerto.
**Precisão:** a distribuição de probabilidade completa do professor por posição, usada como alvo no lugar do rótulo one-hot.
**Onde:** módulo 10, Labs 1 e 3.

### Dark knowledge
**A ideia:** o conhecimento "escondido" nas probabilidades pequenas do professor — quais erros são quase-acertos, quais são absurdos. Invisível no gabarito, valiosíssimo para o aluno.
**Precisão:** a estrutura de similaridade codificada na cauda da distribuição do professor.
**Onde:** módulo 10, Lab 1 (medida: 469 alternativas efetivas num contexto real).

### Forward KL vs reverse KL (mode-covering / mode-seeking)
**A ideia:** duas filosofias de imitação para um aluno com menos capacidade que o professor. *Forward* ("cubra tudo"): o aluno tenta abraçar todos os estilos do professor e vira uma média sem graça — literalmente assenta **no vale entre os picos** (o curso visualizou). *Reverse* ("escolha e faça bem"): o aluno domina um estilo e ignora os outros. Para geração de texto, escolher costuma ser melhor que abraçar.
**Precisão:** KL(p‖q) força cobertura dos modos de p; KL(q‖p) proíbe massa onde p≈0 e seleciona modos.
**Onde:** módulo 10, Lab 2 — incluindo a armadilha do mínimo local.

### Rejection sampling (filtragem por rejeição)
**A ideia:** o controle de qualidade do professor: gerar muitas respostas, **conferir com o gabarito** e treinar o aluno só nas corretas. De graça quando a tarefa é verificável — e obrigatório: o curso mediu o desastre de pular essa etapa (aluno 16× pior).
**Precisão:** filtrar amostras do professor por um verificador antes do SFT do aluno.
**Onde:** módulos 7 (desafio), 10 (pipeline R1-Distill).

### MoE (Mixture of Experts)
**A ideia:** em vez de um generalista gigante que lê *todos* os seus botões para cada palavra, uma equipe de especialistas + um recepcionista (roteador) que encaminha cada palavra para 1–2 deles. Você paga memória pela equipe toda, mas cada palavra só aciona uma fração — capacidade de modelo grande com conta de modelo pequeno.
**Precisão:** o MLP de cada bloco substituído por E experts com roteamento top-k; parâmetros ativos ≪ totais.
**Onde:** módulo 11 (implementado do zero, com o colapso de roteamento medido).

### Colapso de roteamento
**A ideia:** o vício da equipe de especialistas: se um começa ligeiramente melhor, recebe mais trabalho, treina mais, fica melhor ainda — e os outros atrofiam. Sem contramedida (uma penalidade por desequilíbrio), o time de 4 vira 1 fazendo tudo, com 3 salários pagos à toa.
**Precisão:** realimentação winner-take-all do roteador; mitigada pela loss auxiliar de balanceamento do Switch.
**Onde:** módulo 11, Lab 2.

### Decodificação especulativa e draft model
**A ideia:** o estagiário rápido e o revisor sênior: um modelo pequeno (draft) **rascunha** 4–5 palavras baratas; o grande **revisa todas de uma vez** (revisar em lote custa quase o mesmo que escrever uma). Rascunho bom → aceita e avança; ruim → corrige e segue. O resultado matematicamente garantido: **texto idêntico ao que o grande escreveria sozinho**, 2–3× mais rápido. E o melhor estagiário é um aluno destilado do próprio revisor.
**Precisão:** speculative decoding — draft propõe k tokens; alvo verifica em um forward; aceitação via rejection sampling preserva exatamente a distribuição do alvo.
**Onde:** módulo 11, Lab 4 (implementada do zero com prova empírica de equivalência).

### TTFT e TPOT
**A ideia:** as duas latências que o usuário sente. *TTFT*: quanto demora a **primeira** palavra aparecer (a espera inicial). *TPOT*: o ritmo das palavras seguintes (a fluência da "digitação"). Um serviço pode ser bom numa e ruim na outra — e otimizá-las puxa em direções opostas ao custo.
**Precisão:** time-to-first-token (dominado pelo prefill) e time-per-output-token (dominado pelo decode).
**Onde:** módulo 11, seção 4 e Lab 4 do MLX.

### Throughput e batching
**A ideia:** o dilema do restaurante: servir muitas mesas ao mesmo tempo (throughput alto = custo baixo por prato) deixa cada mesa esperando mais (latência pior). O "batch" é quantas conversas a GPU processa simultaneamente; escolher o tamanho é escolher um ponto entre custo e experiência.
**Precisão:** tokens/s agregados; batch maior amortiza a leitura dos pesos entre mais usuários, degradando o TPOT individual.
**Onde:** módulo 11, seções 4–5.

---

## Conhecimento externo (módulo 13)

### RAG
**A ideia:** em vez de esperar que o modelo "saiba", entrega-se a ele uma cola: um buscador encontra os trechos relevantes de uma base de documentos e os coloca no prompt junto com a pergunta. O modelo responde lendo, não lembrando. *Fine-tuning muda o que o modelo É; RAG muda o que ele VÊ.*
**Precisão:** Retrieval-Augmented Generation — recuperação de chunks relevantes de um índice + geração condicionada a eles.
**Onde:** módulo 13 (construído sobre o próprio curso).

### Chunk / chunking
**A ideia:** os documentos são picados em pedaços (chunks) antes de indexar — porque a busca encontra pedaços, não livros. O tamanho é um dilema: pedaços grandes carregam contexto mas borram a busca; pequenos são precisos mas chegam órfãos. A primeira boa estratégia: cortar onde o AUTOR já cortou (títulos e seções).
**Precisão:** segmentação do corpus em unidades de recuperação (~100–300 palavras), idealmente por estrutura semântica, com sobreposição de 10–20%.
**Onde:** módulo 13, Lab 1.

### BM25
**A ideia:** o buscador clássico (anos 90) que ainda briga com redes neurais: pontua documentos pelas palavras em comum com a pergunta, valorizando palavras **raras** (achar "quantização" vale muito; achar "modelo", quase nada) e ignorando repetição excessiva. Imbatível para termos exatos: códigos, flags, nomes.
**Precisão:** função de ranking com IDF e saturação de frequência de termo; implementada do zero no lab em ~25 linhas.
**Onde:** módulo 13, Lab 2.

### Embedding de busca (bi-encoder)
**A ideia:** um modelo que converte perguntas e documentos em coordenadas no mesmo mapa — textos que *significam* o mesmo ficam perto, mesmo com palavras diferentes ("impedir o modelo de esquecer" encontra "catastrophic forgetting"). A busca vira geometria: os vizinhos mais próximos da pergunta.
**Precisão:** encoder treinado contrastivamente que produz vetores normalizados; relevância = cosseno. O "banco vetorial" é, na essência, uma matriz e um produto.
**Onde:** módulo 13, Lab 3 (multilingual-e5-small).

### Reranking (cross-encoder)
**A ideia:** o segundo revisor, mais caro e mais atento: em vez de comparar resumos (embeddings), ele lê pergunta e documento **juntos**, palavra por palavra, e dá a nota final. Caro demais para a base inteira; perfeito para reordenar os 30 finalistas do buscador barato.
**Precisão:** modelo que pontua o par (consulta, documento) com atenção cruzada completa; segundo estágio do pipeline de recuperação.
**Onde:** módulo 13 (exercício B4).

### hit@k e MRR
**A ideia:** as notas da prova do buscador. *hit@k*: "a resposta certa estava entre os k primeiros resultados?" *MRR*: "em que posição, em média, o primeiro acerto apareceu?" Avaliadas com gabarito de fonte — SEM precisar de juiz nem de geração.
**Precisão:** métricas de recuperação; hit@k = fração de consultas com fonte correta no top-k; MRR = média de 1/posição do primeiro acerto.
**Onde:** módulo 13, Lab 5.

### Grounding
**A ideia:** amarrar o modelo ao material: "responda APENAS com base no contexto; se não estiver nele, diga que não sabe". A instrução que separa um assistente confiável de um gerador de alucinações com citações — e que custa zero.
**Precisão:** condicionamento explícito da geração ao contexto recuperado, com abstenção instruída.
**Onde:** módulo 13, lab_mlx.

## Estatística de avaliação (módulo 14)

### Intervalo de confiança (IC)
**A ideia:** a margem de erro de qualquer medição feita em amostra. "65% de acurácia em 100 perguntas" é como uma pesquisa eleitoral com 100 entrevistados: o valor real está numa FAIXA (±9 pontos!), não no número. Medido no curso: com 25 exemplos, um modelo 3 pontos pior VENCE a avaliação 36% das vezes.
**Precisão:** faixa que contém o valor verdadeiro com probabilidade nominal (95%); para proporções, EP = √(p(1−p)/n).
**Onde:** módulo 14, Lab 1.

### Bootstrap
**A ideia:** o truque universal para pôr margem de erro em qualquer métrica: reembaralhe suas próprias perguntas (sorteando com repetição) milhares de vezes, recalcule a métrica em cada versão, e veja o quanto ela balança. O balanço É a incerteza.
**Precisão:** reamostragem com reposição; os percentis 2,5/97,5 das reamostras formam o IC95.
**Onde:** módulo 14, Lab 3.

### Teste pareado / McNemar
**A ideia:** ao comparar dois modelos nas MESMAS perguntas, olhe só onde eles DISCORDAM — as perguntas em que ambos acertam ou ambos erram são plateia. Como modelos parecidos concordam quase sempre, esse desconto multiplica o poder: no curso, 300× (detectar 3pp: 88% pareado vs 0,2% não pareado).
**Precisão:** McNemar: sob H0, as discordâncias a favor de A seguem Binomial(b+c, ½). O ganho de poder cresce com a correlação entre os sistemas.
**Onde:** módulo 14, Labs 2–3 (incluindo a auditoria que rebaixou uma conclusão do módulo 13).

### Comparações múltiplas (o melhor-de-k)
**A ideia:** teste 20 variantes idênticas e "a melhor" parecerá 6 pontos acima da verdade — por puro sorteio. Toda escolha feita OLHANDO a avaliação (melhor checkpoint, melhor prompt, melhor seed) fabrica um pouco de melhora. A defesa: escolher num conjunto de desenvolvimento, reportar num teste lacrado aberto uma vez.
**Precisão:** inflação do máximo de k estimativas ruidosas; medida no lab: +6,2pp (k=20, n=200).
**Onde:** módulo 14, Lab 5.

### Calibração e ECE
**A ideia:** o modelo sabe quando não sabe? Calibrado = quando diz "80% de certeza", acerta ~80% das vezes — e aí a confiança vira informação útil (rotear, abster, priorizar revisão). O Qwen-0.5B medido: 68% de confiança média com 33% de acurácia — superconfiante em 35 pontos.
**Precisão:** ECE = desvio médio |confiança − acurácia| por faixa de confiança; RLHF tende a descalibrar para cima.
**Onde:** módulo 14, Lab 6 — e o espelho humano no METODO-DE-ESTUDO (a SUA calibração nos exercícios).

## Agentes (módulo 15)

### Agente
**A ideia:** um LLM com ferramentas e um loop em volta. Em vez de só responder, ele pode PEDIR uma ação (calcular, buscar, rodar código), ver o resultado, e continuar — até decidir a resposta final. De preditor de palavras a sistema que age no mundo.
**Precisão:** LLM + conjunto de ferramentas + orquestrador que executa os pedidos e realimenta as observações (padrão ReAct).
**Onde:** módulo 15.

### Tool calling (function calling)
**A ideia:** o modelo, vendo um catálogo de ferramentas, gera um pedido estruturado (`{"name": "calculadora", "arguments": {...}}`) em vez de responder. Ponto crucial: ele **pede**, não executa — quem executa é o seu código. O modelo é o cérebro; o loop são as mãos.
**Precisão:** geração de chamada estruturada condicionada ao campo `tools` do chat template; capacidade treinada, emergente com escala.
**Onde:** módulo 15, Lab 1 (funciona até no 0.5B).

### ReAct
**A ideia:** o loop que alterna pensar (Reasoning) e agir (Acting): o modelo raciocina sobre o que fazer, chama uma ferramenta, lê o resultado, e usa isso para decidir o próximo passo. É o que dá composição (tarefas de vários passos) e recuperação de erro.
**Precisão:** Reasoning + Acting em loop, com cada observação realimentando a próxima decisão.
**Onde:** módulo 15, Lab 2.

### Prompt injection (indireta)
**A ideia:** o buraco de segurança dos agentes. Como o modelo não separa "instrução" de "dado" (tudo é texto), um documento, e-mail ou página que ele LÊ pode conter uma ordem escondida ("ignore o resto e faça X") — e o modelo obedece. Pior: modelos mais capazes obedecem melhor.
**Precisão:** injeção de instruções via conteúdo que entra pelo contexto (saída de ferramenta, RAG, entrada de usuário); defesa robusta é arquitetural (menor privilégio + confirmação humana), não de prompt.
**Onde:** módulo 15, Lab 6.

### Menor privilégio
**A ideia:** a defesa que funciona quando as outras falham: dar a cada ferramenta só o poder que ela precisa. A calculadora não deveria poder deletar arquivos; o agente de e-mail não deveria poder encaminhar para fora sem confirmação. Assim, mesmo um agente sequestrado não causa dano grave.
**Precisão:** princípio de segurança (least privilege) aplicado às permissões de cada ferramenta; desenhar para o modelo comprometido, não para o confiável.
**Onde:** módulo 15, seção 6.

## Interpretabilidade (módulo 16)

### Residual stream
**A ideia:** imagine um barramento — um cabo grosso de dados — atravessando o modelo do começo ao fim. Cada camada não reescreve o cabo; ela LÊ um pedaço, calcula algo e SOMA o resultado de volta. A resposta final é a soma de todas essas pequenas edições. Entender o modelo é entender o que cada camada adiciona ao cabo.
**Precisão:** o vetor de dimensão d que percorre a rede via conexões residuais; cada componente lê e escreve em subespaços dele.
**Onde:** módulo 16 (e a base do módulo 2).

### Logit lens
**A ideia:** um "raio-X" do pensamento em formação: pega o estado do modelo numa camada intermediária e pergunta "se você tivesse que responder AGORA, o que diria?". Vê-se a resposta emergir camada a camada — com a ressalva de que, em modelos pequenos, o meio é ilegível e a resposta só aparece no fim.
**Precisão:** projeção de estados intermediários pela lm_head; o tuned lens treina uma correção por camada.
**Onde:** módulo 16, Lab 1.

### Activation patching
**A ideia:** o experimento que prova CAUSA, não correlação. Rode duas frases quase iguais (Paris vs Moscou), e depois "transplante" um pedacinho do cérebro da primeira para a segunda. Se a resposta muda para "Paris", aquele pedacinho é onde a informação mora — você provou por intervenção, como um neurocientista estimulando uma região.
**Precisão:** substituir a ativação de (camada, posição) de um run corrompido pela do run limpo, medindo a recuperação da resposta.
**Onde:** módulo 16, Lab 2 (o padrão-ouro causal).

### Induction head
**A ideia:** a cabeça de atenção que completa padrões — "isso já apareceu antes; o que veio depois?". É o mecanismo interno que permite ao modelo aprender com os exemplos do próprio prompt (few-shot), sem treinar. Uma das poucas capacidades emergentes rastreadas até um circuito concreto.
**Precisão:** cabeça que, dado [A][B]...[A], atende a [B] e aumenta sua probabilidade; base do in-context learning.
**Onde:** módulo 16, Lab 4 (achada no Qwen: 14× a média).

### Probe / steering vector
**A ideia:** duas faces da mesma moeda. Um *probe* é um detector: treina-se um classificador simples nas ativações para perguntar "o conceito X está aqui dentro?". Um *steering vector* é a alavanca: pega-se a direção do conceito e SOMA-se às ativações para empurrar o comportamento — mudar o idioma, o tom ou a recusa do modelo sem treinar nada. Probe mostra que a direção existe; steering prova que ela controla.
**Precisão:** probe = classificador linear sobre ativações (representação); steering = adição de uma direção às ativações em inferência (intervenção).
**Onde:** módulo 16, Labs 5–6 (steering trocou a geração para português somando uma direção).

## Sistemas de treino (módulo 17)

### Autograd (diferenciação automática)
**A ideia:** o mecanismo que calcula, sozinho, como ajustar cada botão do modelo. Ele grava cada operação do cálculo num "grafo" e depois o percorre de trás para frente aplicando a regra da cadeia do cálculo — é o que `.backward()` faz. Cabe em 40 linhas; todo framework é essa ideia escalada.
**Precisão:** construção de um grafo computacional + retropropagação por ordenação topológica; cada operação define seu forward e seu backward local.
**Onde:** módulo 17, Lab 1 (implementado do zero, bate com o PyTorch).

### Gradient checkpointing
**A ideia:** para caber na memória, o modelo "esquece" os cálculos intermediários e os REFAZ quando precisa deles, em vez de guardá-los todos. Troca tempo de processamento (barato) por memória (escassa) — a alavanca que faz um modelo grande caber num M4.
**Precisão:** não armazenar ativações intermediárias no forward, recomputá-las no backward; memória de O(n) para O(√n) camadas, ~+30-50% de tempo.
**Onde:** módulo 17, Lab 3 (é o `--grad-checkpoint` do módulo 6).

### FSDP / ZeRO
**A ideia:** como treinar um modelo que não cabe numa GPU: em vez de cada GPU ter uma cópia inteira (impossível), elas DIVIDEM o modelo e seus estados entre si, cada uma guardando um pedaço e juntando quando precisa. É o que transforma 8 GPUs de 80 GB num espaço de treino de 640 GB.
**Precisão:** Fully Sharded Data Parallel / Zero Redundancy Optimizer — particionamento de otimizador, gradientes e parâmetros entre GPUs, com all-gather sob demanda.
**Onde:** módulo 17, Lab 4 (um 7B: 112 GB/GPU em DDP → 14 GB/GPU em FSDP).

### A bolha do pipeline
**A ideia:** quando as camadas do modelo são divididas entre GPUs em sequência, cada GPU passa tempo ESPERANDO a anterior terminar — como uma linha de montagem que ainda está enchendo. Essa ociosidade é a bolha, e é por que 4 GPUs em fila não rendem 4×.
**Precisão:** ociosidade das GPUs no enchimento/esvaziamento do pipeline; eficiência = micro-batches / (micro-batches + estágios − 1), diluída com mais micro-batches.
**Onde:** módulo 17, Lab 6.

## Fronteira de arquiteturas (módulo 18)

### SSM / Mamba
**A ideia:** em vez de guardar TODO o histórico da conversa (como o transformer faz, pagando cada vez mais memória), o modelo mantém um "resumo" de tamanho fixo que atualiza a cada palavra — como uma pessoa que lembra a essência de um livro sem decorá-lo palavra por palavra. Isso permite contextos de milhões de tokens com memória constante. O preço: lembra o gist, não o detalhe exato.
**Precisão:** State Space Model; estado recorrente `h_t = A·h_{t-1} + B·x_t` de dimensão fixa; Mamba acrescenta seletividade (parâmetros dependentes da entrada) e parallel scan.
**Onde:** módulo 18, Lab 2.

### Recall vs eficiência
**A ideia:** o dilema que organiza toda a fronteira. A atenção do transformer é cara (O(L²)) mas recupera um detalhe exato do passado com precisão cirúrgica; as alternativas eficientes (Mamba, atenção linear) são baratas mas "borram" — lembram o geral, não o específico. Medido: atenção 100% de recall, linear 20%.
**Precisão:** o softmax exponencial permite seleção precisa de uma chave; mecanismos O(L) sem softmax fazem médias suaves.
**Onde:** módulo 18, Lab 5.

### Arquitetura híbrida
**A ideia:** já que atenção e Mamba têm forças opostas, os modelos de ponta usam os DOIS — muitas camadas baratas de Mamba com poucas camadas caras de atenção salpicadas onde o recall importa. ~1 a cada 8 basta. É o "melhor dos dois mundos" que a indústria adotou (Jamba, Nemotron-H).
**Precisão:** intercalação de camadas de atenção e SSM; ~12% de atenção captura quase todo o recall com uma fração do custo.
**Onde:** módulo 18, Lab 6.

### Multimodalidade
**A ideia:** o mesmo modelo que processa texto processa imagem, áudio e vídeo — porque um encoder converte cada modalidade em "tokens" no mesmo espaço, e a atenção mistura tudo sem precisar de mecanismo novo. É por isso que "LLM" virou "modelo de fundação".
**Precisão:** encoders (ViT para imagem) projetam a entrada no espaço dos embeddings de texto; a self-attention faz a fusão cross-modal nativamente.
**Onde:** módulo 18, seção 6.

## Termos de ferramentas (transversais)

### PyTorch / MLX / CUDA / MPS
**A ideia:** as "oficinas" onde os modelos rodam. **PyTorch**: a biblioteca padrão da área (usada nos módulos 1–4 e nos labs de CPU). **CUDA**: a plataforma das GPUs NVIDIA — o padrão da indústria. **MLX**: a oficina da Apple para os chips M1–M4 (a trilha dos módulos 5+ no seu Mac). **MPS**: o backend Apple do PyTorch (alternativa menos otimizada que MLX no Mac).
**Onde:** `00-setup.md` e `00-setup-mac.md`.

### HuggingFace
**A ideia:** o "GitHub dos modelos": o repositório público de onde os labs baixam modelos, tokenizers e datasets (`Qwen/Qwen2.5-0.5B-Instruct` é um endereço de lá), e a biblioteca `transformers` que os carrega.
**Onde:** todos os labs.

### Tensor
**A ideia:** a estrutura de dados universal do deep learning: uma tabela de números de qualquer número de dimensões (lista = 1D, planilha = 2D, pilha de planilhas = 3D...). Tudo — pesos, textos tokenizados, gradientes — vive em tensores.
**Precisão:** array n-dimensional com suporte a operações vetorizadas e diferenciação automática.
**Onde:** todos os labs.

### Seed (semente aleatória)
**A ideia:** o número que "congela o acaso": com a mesma seed, o sorteio sai igual toda vez — e o experimento fica reproduzível. Todo lab do curso fixa a seed por isso.
**Precisão:** estado inicial do gerador pseudoaleatório.
**Onde:** todos os labs (`torch.manual_seed(...)`).
