# Trilha essencial — do zero ao primeiro nível profissional

Esta é a rota principal do curso. Ela contém os fundamentos que se repetem em projetos
reais e evita exigir temas de pesquisa antes de você saber construir um sistema confiável.

> “Profissional” aqui significa: consegue receber um problema limitado, construir uma
> solução de LLM reproduzível, medir qualidade e custo, explicar decisões e entregar o
> projeto para outra pessoa executar. Não significa dominar toda a pesquisa da área.

## Visão geral

| Etapa | Conteúdo | Resultado observável | Tempo sugerido |
|---|---|---|---:|
| 0 | Computação, Python e matemática mínima | Programa testado e documentado | 20–30 h |
| 1 | Como LLMs representam e aprendem | MiniGPT treinado e explicado | 60–80 h |
| 2 | Dados e customização eficiente | Dataset + LoRA reproduzível | 60–80 h |
| 3 | Inferência, RAG, avaliação, agentes e produção | Assistente medido e servido com ferramentas | 60–80 h |
| 4 | Projeto profissional | Repositório executável e auditável | 60–100 h |

Em um ritmo de 8–10 horas por semana, espere aproximadamente **6–9 meses**. Tempo é uma
estimativa, não promessa: o critério para avançar são as entregas e os testes.

---

## Etapa 0 — Alfabetização técnica

Faça [`00-iniciante-zero/`](00-iniciante-zero/) se qualquer item abaixo for novo:

- terminal e caminhos;
- variáveis, coleções, condições, laços e funções;
- exceções, `assert` e testes;
- JSONL;
- vetores, produto escalar, tensores, loss e gradiente;
- Git básico.

**Entrega:** classificador por regras com testes, medição e README.

**Gate:** pelo menos 80% do checklist da Fase 0 sem consultar.

---

## Etapa 1 — Entender o motor

Estude em ordem:

1. [`modulo-01-fundamentos/`](modulo-01-fundamentos/) — tokens, logits, geração e custo;
2. [`modulo-02-attention/`](modulo-02-attention/) — embeddings, attention e transformer;
3. [`modulo-03-treino/`](modulo-03-treino/) — loss, gradiente, otimização e escala;
4. [`modulo-04-dados/`](modulo-04-dados/) — qualidade, formato e vazamento de dados.

Princípios que importam:

- texto entra como tokens, não como significado pronto;
- o modelo prevê distribuições de probabilidade;
- arquitetura define quais transformações são possíveis;
- treino ajusta parâmetros para reduzir uma métrica;
- dados e avaliação limitam a conclusão que podemos tirar.

**Entrega:** treinar o MiniGPT, desenhar o fluxo completo e explicar cada tensor principal.

**Gate:** alterar um hiperparâmetro, prever o efeito e explicar a medição observada.

---

## Etapa 2 — Customizar sem desperdiçar recursos

Estude:

1. [`modulo-05-sft/`](modulo-05-sft/) — formato, chat template e supervised fine-tuning;
2. [`modulo-06-lora/`](modulo-06-lora/) — adaptação eficiente e esquecimento;
3. [`modulo-11-inferencia/`](modulo-11-inferencia/) — memória, quantização e orçamento.

Princípios que importam:

- escolha a técnica pela falha observada, não pela moda;
- preserve uma baseline antes de treinar;
- treino e serviço precisam usar o mesmo formato;
- qualidade, latência, memória e custo formam um único problema;
- artefatos e versões precisam ser reproduzíveis.

**Entrega:** dataset versionado por script, treino LoRA, baseline, avaliação e relatório de custo.

**Gate:** outra pessoa consegue reproduzir o treino seguindo somente o README.

---

## Etapa 3 — Construir sistemas úteis

Estude:

1. [`modulo-13-rag/`](modulo-13-rag/) — recuperação e conhecimento externo;
2. [`modulo-14-avaliacao/`](modulo-14-avaliacao/) — amostragem, incerteza e comparação;
3. [`modulo-15-agentes/`](modulo-15-agentes/) — ferramentas, loops e limites de segurança;
4. [`modulo-19-producao/`](modulo-19-producao/) — servir, medir, orçar e proteger o sistema.

Princípios que importam:

- conhecimento mutável normalmente pertence fora dos pesos;
- toda métrica é uma estimativa sobre uma amostra;
- uma ferramenta precisa de contrato, validação e limite;
- falhar explicitamente é melhor que produzir resultado silenciosamente errado;
- logs e casos de erro são parte da avaliação;
- **custo, latência e falha são propriedades do sistema, não do modelo** — e se medem e protegem.

**Entrega:** assistente RAG com citação, abstenção, conjunto de avaliação e uma ferramenta
segura, servido com **medição de p50/p95 e custo** e protegido por guardião/disjuntor.
Registre casos em que ele funciona e em que deve recusar.

**Gate:** comparação pareada contra uma baseline simples, análise manual dos erros e um
extrato de tráfego (latência, throughput, custo, sucesso) que outra pessoa consegue ler.

---

## Etapa 4 — Projeto profissional

Use [`modulo-12-projeto/`](modulo-12-projeto/) depois dos módulos 13–15. A numeração foi
mantida por compatibilidade, mas a trilha essencial posiciona o projeto no final.

O projeto precisa conter:

- problema e usuário definidos em poucas frases;
- baseline que não usa a solução sofisticada;
- dados ou documentos com origem explicada;
- pipeline executável a partir de ambiente limpo;
- testes unitários para parsing, ferramentas e regras críticas;
- avaliação com exemplos nunca usados no ajuste;
- custo, latência e consumo de memória;
- análise de erros e limitações;
- decisões registradas no README;
- histórico Git com commits pequenos e compreensíveis.

### Critério de conclusão

Você está no primeiro nível profissional quando consegue defender:

1. **Por que esta solução?** — qual falha ela resolve;
2. **Como sabemos que funciona?** — baseline, dados e métrica;
3. **Quanto custa?** — tempo, memória e dinheiro;
4. **Como falha?** — casos conhecidos e comportamento seguro;
5. **Outra pessoa reproduz?** — ambiente, comandos, testes e artefatos.

---

## Especializações opcionais

Faça somente quando um projeto real exigir ou quando quiser aprofundar pesquisa:

| Tema | Módulos | Quando estudar |
|---|---|---|
| Reasoning | 7 | Precisa medir raciocínio ou dados com traços |
| Preferências e RL | 8–9 | Possui feedback/preferência ou recompensa verificável |
| Distillation | 10 | Precisa transferir capacidade para modelo menor |
| Interpretabilidade | 16 | Precisa investigar mecanismos internos |
| Treino distribuído | 17 | Trabalha com modelos que não cabem em uma máquina |
| Novas arquiteturas | 18 | Pesquisa alternativas ao transformer |
| Pesquisa própria | 19–21 | Já reproduz resultados e quer produzir conhecimento |

O profissional não é quem estudou todas as técnicas. É quem reconhece qual técnica não
precisa usar.

## Portfólio mínimo

Ao final, publique três projetos pequenos em vez de um projeto gigantesco:

1. **Fundamento:** MiniGPT ou tokenizer explicado e medido;
2. **Customização:** LoRA com baseline, dataset e relatório;
3. **Sistema:** RAG/agente com avaliação, testes e limitações.

Cada projeto deve caber em uma conversa técnica de 10 minutos: problema, decisão,
experimento, resultado, erro encontrado e próximo passo.
