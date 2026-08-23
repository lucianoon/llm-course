# Método de Estudo — a ciência de reter o que o curso ensina

Uma nota honesta antes de tudo: técnicas com marca registrada ("PNL", "aprendizado acelerado", estilos de aprendizagem visual/auditivo) **não têm sustentação científica**. O que tem — replicado por décadas de pesquisa em ciência cognitiva (Roediger, Bjork, Dunlosky, Ericsson) — é um conjunto pequeno de princípios de efeito enorme. Este documento os transforma em protocolo concreto para ESTE curso.

## Para quem começa do zero

Na [`Fase 0`](00-iniciante-zero/), use blocos de 45–60 minutos e mire **5–7 horas por
semana**. Nas primeiras semanas, executar e modificar código vale mais do que criar muitos
cartões. Para cada conceito: leia um exemplo, feche o material, reescreva uma versão menor,
execute, leia o erro e corrija. Comece o protocolo completo e o Anki no módulo 1.

Se terminal, Python e matemática forem novos ao mesmo tempo, isso não é sinal de falta de
talento: são três vocabulários sendo aprendidos juntos. Reduza o tamanho do exercício, não
pule a verificação. O gate continua sendo explicar e executar sem copiar.

## Os seis princípios (e por que funcionam)

### 1. Prática de recuperação — o efeito de testagem
**A ciência:** tentar LEMBRAR fortalece a memória muito mais do que reler. Releitura cria familiaridade (a sensação de saber); recuperação cria conhecimento. É o achado mais robusto da área (Roediger & Karpicke, 2006: recuperar bateu reler em ~50% na retenção de uma semana).
**No curso:** os *checklists de saída* de cada módulo são testes de recuperação — responda-os **por escrito, de memória, antes de reler qualquer coisa**. O baralho de revisão (abaixo) industrializa isso.
**A regra:** nunca reabra um README para "revisar". Feche-o, escreva o que lembra, e SÓ ENTÃO confira. O desconforto de não lembrar é o músculo trabalhando.

### 2. Repetição espaçada — a curva do esquecimento
**A ciência:** revisar no momento em que você está *prestes a esquecer* multiplica a durabilidade da memória. Intervalos crescentes (1 dia → 3 → 7 → 21 → 60) vencem qualquer maratona de véspera (Ebbinghaus; Cepeda et al., 2006).
**No curso:** o baralho `revisao/baralho-01-fundacao.tsv` — 15 minutos por dia no Anki (que implementa o algoritmo de espaçamento sozinho). Cada módulo novo da fase 2 adiciona cartões.
**Setup (5 minutos):** instale o [Anki](https://apps.ankiweb.net) (Mac/iPhone, grátis) → File → Import → selecione o `.tsv` → campo 1 = frente, campo 2 = verso, campo 3 = tags. Pronto: o algoritmo cuida do resto.

### 3. Intercalação — misturar em vez de blocar
**A ciência:** estudar A-B-C-A-B-C rende mais que AAA-BBB-CCC, porque força o cérebro a *discriminar* qual conceito se aplica — que é exatamente a habilidade real (Rohrer & Taylor, 2007).
**No curso:** ao estudar o módulo N da fase 2, resolva por semana **2 exercícios de módulos anteriores** sorteados (o baralho já intercala conceitos por natureza). E a pergunta de intercalação suprema, para todo problema novo: *"isso é caso de prompt, RAG, SFT, DPO, RL ou distillation?"* — a tabela de decisão do módulo 12 como flashcard permanente.

### 4. Dificuldade desejável — o esforço É o aprendizado
**A ciência:** condições que tornam o estudo mais difícil no momento (gerar antes de ver a resposta, espaçar, variar contexto) produzem retenção maior (Bjork). Fluência fácil é ilusão de competência.
**No curso:** os exercícios existem para serem tentados ANTES do gabarito — e agora com um passo extra: **anote sua confiança (0–100%) antes de conferir**. Com o tempo, compare confiança × acerto: você está calibrado? Superconfiança em um tema = o próximo alvo de estudo. (Calibração é uma meta-habilidade da área, aliás — módulo 14.)

### 5. Elaboração — a técnica de Feynman
**A ciência:** explicar com as próprias palavras, para alguém que não sabe, expõe cada buraco do entendimento (efeito de auto-explicação, Chi et al.).
**No curso:** ao fechar cada módulo, escreva **meia página explicando o conceito central para um colega leigo** — no estilo do GLOSSARIO.md (analogia → precisão). Guarde em `revisao/feynman/modulo-NN.md`. Se a analogia não vem, você não entendeu; volte.
**A versão turbinada:** me explique. Sou um interlocutor que sabe a resposta e vai apontar exatamente onde a sua explicação escorregou. As *sabatinas* dos checklists funcionam assim.

### 6. Prática deliberada — na fronteira, com feedback
**A ciência:** especialistas não se formam por horas acumuladas, mas por prática **na borda da capacidade, com feedback imediato e correção de erros específicos** (Ericsson). Dez anos de prática confortável = estagnação confortável.
**No curso:** os exercícios têm dificuldade crescente (A conceituais → B práticos → desafio) — trabalhe no nível em que você erra ~30% das vezes. E mantenha o **diário de erros** (`revisao/diario-de-erros.md`): cada erro seu anotado com a causa-raiz, revisitado mensalmente. O curso inteiro foi construído assim — os erros do autor viraram as melhores seções — e o seu aprendizado funciona igual.

---

## O protocolo semanal (10–12 h)

| Dia | Bloco | O quê |
|---|---|---|
| Todos | 15 min | **Baralho Anki** (princípio 2) — inegociável, é o piso do sistema |
| Dia 1 | 2 h | README do módulo novo — leitura ativa: anote **perguntas**, não resumos |
| Dia 2 | 2 h | Lab, célula a célula, **prevendo cada saída antes de rodar** (recuperação + calibração) |
| Dia 3 | 1,5 h | Exercícios A (conceituais), com nota de confiança antes de cada gabarito |
| Dia 4 | 2 h | Exercícios B (práticos) — os que exigem código |
| Dia 5 | 1,5 h | **Intercalação**: 2 exercícios de módulos antigos + 1 releitura de erro do diário |
| Dia 6 | 1 h | **Feynman**: a meia página do módulo + sabatina do checklist (comigo, se quiser) |
| Dia 7 | — | Descanso de verdade. Consolidação de memória acontece dormindo — literalmente (a pesquisa de sono e consolidação é unânime). |

**Duas regras de ouro do protocolo:**

1. **Prever antes de executar.** Em todo lab, antes de rodar a célula: "o que vai sair?". Escreva. Rode. Compare. Cada previsão errada vale ouro — é um buraco no seu modelo mental encontrado de graça.
2. **O diário de erros é sagrado.** Formato de cada entrada: *o que eu previ / o que aconteceu / por que errei / qual princípio geral extraio*. Releitura mensal. Em um ano, esse arquivo valerá mais que o curso.

---

## Como cada peça do curso encaixa no sistema

| Peça | Princípio que serve |
|---|---|
| Checklists de saída | Recuperação (teste) + critério de mastery: **não avance com <80%** |
| Baralho Anki | Espaçamento + recuperação diária |
| Exercícios com gabarito escondido | Dificuldade desejável + calibração |
| GLOSSARIO (analogias) | Elaboração — e o *modelo* para os seus textos Feynman |
| GUIA-DE-CODIGO | Redução de carga cognitiva (exemplos resolvidos → prática autônoma, Sweller) |
| Labs com verificações | Feedback imediato (prática deliberada) |
| Diário de erros | Prática deliberada + metacognição |
| Sabatinas comigo | Feedback de especialista + Feynman turbinado |

**Mastery gate:** um módulo só está "fechado" quando (a) checklist ≥80% de memória, (b) exercícios B feitos, (c) texto Feynman escrito. Avançar sem fechar cria a pilha de dívida que derruba autodidatas no módulo 8.

---

## O que NÃO fazer (os mitos, com carinho)

- **Reler e marcar texto** — as duas técnicas mais usadas e as duas de menor eficácia medida (Dunlosky et al., 2013). Substitua por recuperação.
- **Maratonar** ("vou fechar 3 módulos neste fim de semana") — o que se aprende blocado se esquece blocado. O sistema espaçado com 1h/dia vence o guerreiro de fim de semana em qualquer horizonte acima de duas semanas.
- **Assistir/ler mais uma explicação do mesmo tema** — a sensação de fluência de consumir conteúdo é o falso positivo clássico. Se já leu uma vez, a segunda exposição deve ser *teste*, não consumo.
- **"Estilos de aprendizagem"** (sou visual/auditivo) — sem suporte empírico. O que funciona para todo mundo: dupla codificação (texto + diagrama **desenhado por você**, de memória — tente desenhar o bloco transformer do módulo 2 agora).
