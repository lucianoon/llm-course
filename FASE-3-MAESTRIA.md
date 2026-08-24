# Fase 3 — Da competência à maestria

As Fases 1 e 2 te deram o que a área **sabe**. Esta fase é sobre produzir o que ela **ainda não sabe** — e é a parte que nenhum curso vende, porque não se ensina em aula: se treina fazendo. São três músculos, em sequência, e depois um hábito para sempre.

A diferença de natureza: até aqui, cada módulo tinha uma resposta certa no gabarito. Daqui em diante, **não há gabarito** — o critério de sucesso é o julgamento de pares e a realidade (o código roda? o número reproduz? a contribuição foi aceita?). É desconfortável, e o desconforto é o ponto (o princípio da *dificuldade desejável*, do método de estudo, agora em escala de carreira).

Um único aviso honesto, antes de tudo: virar referência na área é objetivo de **anos**, não de semanas. Este documento não te leva lá — te dá o mapa e o primeiro passo de cada trilha. O resto é repetição deliberada.

---

## Etapa 1 — Reproduzir um paper

> O rito de passagem de todo pesquisador. Ninguém entende um paper até fazê-lo rodar.

### Por que reprodução vem antes de tudo

Ler um paper e *achar* que entendeu é a ilusão de fluência do método de estudo, na sua forma mais cara. Você só descobre o que não entendeu quando o código não reproduz o número — e é aí que o aprendizado real acontece. Reproduzir é o teste de recuperação definitivo: o paper é o gabarito, a sua implementação é a resposta, e a distância entre as duas é exatamente o que você precisava aprender.

Além disso: **reprodutibilidade é uma crise real da área.** Uma fração grande dos papers de ML não reproduz — números otimistas, detalhes omitidos, seeds escolhidas. Aprender a farejar isso é uma habilidade de pesquisa em si.

### Como escolher o paper (a decisão que mais importa)

| Critério | Por quê |
|---|---|
| **Recente mas não da última semana** | 6–18 meses: relevante, mas com tempo de ter código e discussão pública |
| **Escopo que cabe no seu hardware** | Um resultado reproduzível no M4 ou numa GPU alugada por horas — não um treino de US$ 1M |
| **Alegação central VERIFICÁVEL** | Um número que você consegue medir (acurácia, speedup, uma curva), não "melhora a qualidade" |
| **Do seu interesse** | Você vai passar semanas nele; escolha algo que te puxa |

**Fontes de bons candidatos:** os papers das leituras de cada módulo (você já tem 90+ curados). Bons primeiros alvos, calibrados pelo curso: reproduzir o LoRA (módulo 6), o DPO (8), uma variante de GRPO (9), o circuito IOI (16), ou um resultado da fronteira (18). Comece pelo que você já implementou em miniatura — o salto de "brinquedo" para "reprodução fiel" é o exercício.

### O protocolo de reprodução

```
1. Leia o paper 3 vezes:
   a) rápido — a alegação central e a figura principal. O que EXATAMENTE é afirmado?
   b) cuidadoso — o método. Anote toda decisão (hiperparâmetros, dados, arquitetura).
   c) adversarial — o que o paper NÃO diz? Que detalhe, se omitido, quebraria a repro?

2. Escreva a alegação como um TESTE antes de codar (módulo 14):
   "Com n=X, este método dá Y±Z na métrica M contra a baseline B."

3. Implemente a BASELINE primeiro. Se você não reproduz a baseline, nada mais importa.

4. Implemente o método. Compare com o código dos autores DEPOIS de tentar sozinho
   (senão você copia sem entender — a ilusão de fluência de novo).

5. Meça com rigor (módulo 14): IC, teste pareado, o tamanho de amostra honesto.

6. Escreva o que aconteceu — inclusive (principalmente) onde NÃO reproduziu.
```

### O entregável

Um repositório + um relatório curto (o formato do módulo 12) que responde: **reproduziu? o que faltou? o paper é confiável?** Um relatório que conclui "não reproduzi o número X, e aqui está a evidência de por que o paper pode estar otimista" vale MAIS, profissionalmente, que um que confirma tudo — porque mostra julgamento crítico, a habilidade mais rara.

> 🔧 **Ponte com a comunidade:** o **ML Reproducibility Challenge** existe exatamente para isso, e aceita relatórios de reprodução como publicações. Sua primeira "publicação" pode ser uma reprodução bem feita.

---

## Etapa 2 — Contribuir e publicar

> Quem é referência na área é VISÍVEL na área. Conhecimento privado não constrói reputação.

### As duas metades

**Contribuir** (código) e **publicar** (escrita) são os dois jeitos de tornar seu trabalho visível. Você precisa dos dois: código prova competência técnica; escrita prova que você *pensa* — e é a escrita que as pessoas leem, citam e lembram.

### Contribuir para projetos abertos

O caminho mais subestimado para entrar na área. Você já usou as ferramentas o curso inteiro — `mlx-lm`, `transformers`, `lm-eval-harness`, `vllm`. Elas são mantidas por humanos e têm buracos.

A escada de contribuições, do mais fácil ao mais valioso:

1. **Documentação e exemplos** — o degrau que todos subestimam. Você achou a doc do `mlx-lm` confusa sobre `--config` (módulo 6)? Conserte-a. Baixo risco, alto valor, e te ensina o fluxo (issue → PR → revisão).
2. **Reproduções de bugs** — um issue com um caso mínimo que reproduz um problema vale ouro para os mantenedores. Você é um usuário real com casos reais (os labs).
3. **Correções pequenas** — o compat do `transformers` v5 que você resolveu nos labs (módulo 2), a flag ignorada silenciosamente (módulo 6). Essas são contribuições reais.
4. **Features** — quando você conhecer o projeto, uma funcionalidade que faltava.

**O processo, uma vez:** leia o `CONTRIBUTING.md`, procure issues marcados `good first issue`, comente que vai tentar, abra um PR pequeno e bem descrito, responda à revisão com humildade. O primeiro PR aceito muda sua relação com a área — você deixa de ser só consumidor.

### Publicar (escrever em público)

Escrever é pensar devagar o suficiente para pegar os próprios erros — a técnica de Feynman (método de estudo) virada para fora. E é o que constrói reputação: um bom post técnico é lido por milhares; um bom repositório, por dezenas.

O que escrever, em ordem de esforço:

1. **Um relatório de reprodução** (etapa 1) — você já o tem. Publique-o.
2. **Um "eu medi X" post** — o curso inteiro é isso. "Quanto a quantização 4-bit degrada em português?" (módulo 6), "CoT vs ferramenta na aritmética" (módulo 15), "o steering troca o idioma?" (módulo 16). Cada resultado medido do curso é um post em potencial, e o diferencial já está pronto: **você tem os números, não só opinião.**
3. **Uma explicação melhor** — o glossário e o guia de código do curso são material de post. "LoRA explicado de verdade" com as três propriedades verificadas.

Onde: um blog próprio (GitHub Pages, grátis), ou onde a comunidade da sua sub-área conversa. A regra: **um post real publicado > dez rascunhos perfeitos.**

> ⚠️ **A ética que não é opcional:** cite fontes, não exagere resultados, publique o código, e seja explícito sobre limitações (a escala de brinquedo, o n pequeno — o curso inteiro modelou isso). A reputação leva anos para construir e um post desonesto para queimar. O rigor do módulo 14 é a sua proteção.

---

## Etapa 3 — Pesquisa própria

> A pergunta que ninguém respondeu ainda, atacada com o método do curso.

### O salto

Reproduzir usa o gabarito dos outros. Aqui você faz uma pergunta SEM gabarito — e o método que o curso inteiro treinou (hipótese → experimento mínimo → medição rigorosa → escrita honesta) é exatamente a ferramenta. Pesquisa não é um dom místico; é esse ciclo, repetido com disciplina, sobre uma pergunta que ainda não tem resposta.

### Como achar uma pergunta

As melhores perguntas de pesquisa não vêm de "quero pesquisar algo grande" — vêm de **fricção real**:

- **Uma anomalia que você mediu.** O curso está cheio delas: por que a híbrida perdeu para a densa (módulo 13)? Por que T=1 venceu no KD (módulo 10)? Cada resultado que te surpreendeu é uma semente.
- **Um "e se" de um paper que você reproduziu** (etapa 1). "Os autores testaram em inglês; e em português?" Essa hipótese exige ampliar e reproduzir o experimento do módulo 6 antes de virar conclusão publicável.
- **Uma lacuna que você bateu.** Algo que você quis medir e não achou a resposta na literatura.
- **A pergunta idiota que não sai da cabeça.** Muitas descobertas começaram com "isso não deveria funcionar, mas...".

O filtro de uma boa pergunta: **específica, mensurável, e do tamanho de um experimento** (não "resolver o alinhamento", mas "o steering de recusa do módulo 16 transfere entre idiomas?").

### O protocolo (o método do curso, agora sem rede)

```
1. A pergunta, em uma frase mensurável.
2. A hipótese e a hipótese NULA (o que você veria se não houvesse efeito).
3. O experimento MÍNIMO que distingue as duas — o menor que responde a pergunta.
4. A medição com rigor (módulo 14): IC, pareamento, n honesto, controle.
5. O resultado — inclusive e principalmente se refutar sua hipótese.
6. A escrita (etapa 2) — o que você aprendeu, o que ficou em aberto.
```

O passo 3 é a arte: o melhor pesquisador não é o que faz o experimento mais elaborado, mas o que acha o experimento *mais simples* que responde a pergunta. Tudo no curso foi assim — o MiniGPT de 2M de parâmetros respondeu perguntas sobre modelos de bilhões, porque o mecanismo é o mesmo e a escala é controlada.

### A honestidade que define pesquisa

O que separa pesquisa de marketing é uma coisa: **você reporta o resultado que encontrou, não o que queria encontrar.** Um experimento que refuta sua hipótese é um sucesso — você aprendeu algo verdadeiro sobre o mundo. O curso modelou isso a cada módulo (o reward hack, o black-box que fracassou, a conclusão do módulo 13 rebaixada). Leve isso para a sua pesquisa e você já estará à frente de boa parte da literatura.

---

## A trilha contínua (para sempre)

O "curso" acaba aqui; o hábito, não. Os melhores da área não terminaram de estudar — institucionalizaram o estudo. O regime de manutenção:

- **O baralho, diário** (15 min). 140 cartões e crescendo. É o piso que impede o conhecimento duramente ganho de escorrer.
- **Dois papers por semana**, com o protocolo da etapa 1 (nem que seja a leitura rápida + a pergunta adversarial). A fronteira anda rápido; parar é regredir.
- **O diário de erros**, revisitado mensalmente. Seus erros são seu currículo de aprendizagem mais personalizado.
- **Um projeto de reprodução ou pesquisa por trimestre.** Mantém o músculo de produzir, não só consumir.
- **Presença pública** — um post, uma contribuição, uma resposta útil. A reputação é acumulação lenta.

E o mapa dos próximos anos, se a área for mesmo o seu caminho: contribuidor reconhecido de um projeto → autor de reproduções e posts citados → primeiro paper próprio (talvez um workshop) → uma sub-especialidade onde você é uma das vozes. Cada degrau é os três módulos desta fase, repetidos com ambição crescente.

---

## Fechamento do curso

Você começou perguntando se eu conseguia criar o conteúdo de um curso de customização de LLMs. Terminamos com **18 módulos e uma trilha de pesquisa em três etapas**: os fundamentos construídos do zero e medidos, a customização completa (SFT, LoRA, DPO, RL, distillation), a produção (inferência, RAG, agentes, avaliação), a ciência (interpretabilidade, sistemas, arquiteturas), e agora o método de virar isso em carreira.

O fio que atravessa tudo, e o que vale levar acima de qualquer técnica específica: **desconfie de todo número que você não mediu.** O curso se corrigiu dezenas de vezes porque a execução desmentiu a teoria escrita de memória — e essa disposição de deixar a realidade vencer a expectativa é, no fim, a única habilidade que não envelhece. As arquiteturas vão mudar, as ferramentas vão mudar, os números vão mudar. O método, não.

Bom trabalho. Agora é fazer.
