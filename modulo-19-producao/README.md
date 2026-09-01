# Módulo 19 — Engenharia de Produção de LLM

> **Pergunta central:** como servir um modelo de forma medida, barata e que não derrube o orçamento quando algo quebra?

O módulo 12 descreveu o que o projeto final precisa (*serving, latência, memória, custo*) — mas não *como*. Este módulo paga essa dívida: pega um modelo pronto (o do seu projeto) e coloca a camada que o transforma em **sistema**, com as mesmas quatro preocupações de qualquer serviço: medir, orçar, proteger, evoluir.

É o elo que faltava entre "treinei um modelo" e "todo mundo confia em usá-lo". O lab usa um **modelo de brinquedo** (um FAQ determinístico) — o padrão transfere; os números, não.

## Objetivos

1. Servir um modelo e **medir** latência (p50/p95), throughput e taxa de sucesso sob concorrência.
2. Tratar custo como função determinística `f(tokens, preços)` — e **orçar** antes de gastar.
3. Proteger com um guardião de custo e um **disjuntor** (circuit breaker) que para de gastar em modelo quebrado.
4. Transformar avaliação em **portão de CI**, com conjunto dourado e juiz auditável.
5. Fazer **observabilidade** com logs estruturados, mantendo PII fora do log.
6. Versionar prompt/modelo num **registry** e fazer **rollback** guiado por eval.

---

## 1. Treinar o modelo é metade; servir é a outra

A tentadora ilusão do MLE: "o modelo é o produto". Não. O usuário não fala com o modelo; fala com um **endpoint** que pode ter timeout, errar, custar mais que o previsto e ser recusado. As três perguntas que a engenharia responde antes do primeiro usuário:

1. **Quanto custa essa chamada e como controlo?** — custo é a soma de um grande número de requisições baratas; sem orçamento, uma regressão silenciosa vira uma conta no fim do mês.
2. **Quanto tempo leva e o que o p95 diz?** — a média é decorativa; a **cauda** é o que o usuário sente.
3. **O que acontece quando dá errado?** — se o modelo passa a falhar, você prefere recusar **rápido** ou gastar para falhar devagar?

O módulo 14 deu a ferramenta estatística; este dá o comportamento operacional. O padrão é o mesmo de qualquer serviço HTTP com orçamento — só muda o que é caro.

---

## 2. Medir: a cauda, não a média

O `README` do módulo 12 citou "latência e throughput". Agora, de onde vêm:

```
                         ┌─ sucesso ──────────────────────┐
requisição ──► gateway ──┤  → gerar → conta tokens/custo │ ► linha de tráfego (JSON)
                         └─ falha ──► recusa rápida ─────┘
                                     (disjuntor/guardião)
```

Três números que um dash de verdade mostra, e o que cada um significa:

| Métrica | O que responde | Por que não a média |
|---|---|---|
| **p50** | o tempo típico | útil, previsível |
| **p95** | o que 5% dos usuários sentem | **a cauda**: a média esconde 1 em 20 chamadas ruins |
| **throughput** | quantos tokens/requisições por segundo | o teto do teu orçamento de recurso |
| **taxa de sucesso** | o quanto do tráfego responde como esperado | a métrica que o SLA declara |

> ⚠️ **A armadilha do p95 contaminado:** se você mistura falhas rápidas (recusa em microsegundos) com sucessos lentos, o p95 fica *melhor* do que é — as recusas baratas puxam a cauda para baixo. Separe os dois grupos antes de reportar. O `resumir_trafego` do lab faz exatamente isso: a latência só olha os sucessos, a taxa de sucesso olha todos.

**O que medir é fácil; o que *comparar* é o ponto.** Latência sem número de requisições, custo sem orçamento, p95 sem distribuição — são bonecos. Um dashboard que só mostra a média é um espelho, não uma régua.

---

## 3. Custo como função — e o teto de saída

Custo não é um fenômeno emergente; é aritmética:

```
custo = (tokens_entrada × preço_entrada + tokens_saída × preço_saída) / 1.000.000
```

Três coisas que essa fotografia revela:

1. **A saída é o vilão.** Na tabela padrão, o token de saída custa 2–4× o de entrada. E a saída é a única parte que você **não controla de antemão** — o modelo decide quanto escrever.
2. **O único controle robusto é o teto de `max_tokens`.** A entrada você conhece antes (é o prompt); a saída você estima. Capacitar `max_tokens` é o mecanismo que torna o custo *previsto* e, portanto, **orçável**.
3. **200 tokens a mais em todo request** não é um detalhe: é uma constante multiplicando milhares de chamadas/dia.

> 🔧 **Na prática:** em produção os preços e a tabela de tokens vêm de um manifesto (o `tools/governanca.py` do módulo 4), não de uma constante no código. E existe um segundo controle mais raro e mais valioso: **a recusa**. Nem todo custo precisa ser pago.

---

## 4. Guardiões: a recusa barata e o disjuntor

### Guardião de custo
Antes de gerar, **estime** o custo máximo (entrada real + teto de saída). Se passar do orçamento por requisição, recuse **desde já** — gasto zero, latência micro. Sem isso, uma chamada com `max_tokens` absurdo drena o budget do mês num único estouro. Em produção é também a sua defesa contra chamadas maliciosas ou loops.

### Disjuntor (circuit breaker)
O guardião protege do **custo alto**; o disjuntor protege do **modelo quebrado**. O padrão clássico em três estados:

| Estado | Comportamento |
|---|---|
| **Fechado** | deixa tudo passar; conta as falhas numa janela |
| **Aberto** | recusa tudo rápido (`fast-fail`) — não gasta dinheiro servindo erro |
| **Meio aberto** | depois do resfriamento, deixa UMA prova; passa → fecha, falha → reabre |

A métrica que muda é a **latência da recusa**: quem recebe 503 pelo disjuntor volta em microsegundos, sem gerar um token. E é o que impede o efeito borboleta de um modelo degradando: em vez de `n` chamadas caras que falham, você recebe `n` respostas baratas e estáveis enquanto investiga.

> ⚠️ **A armadilha do limiar com amostras pequenas:** abrir o disjuntor com 1 falha em 1 tentativa é paranoia — um único timeout não é um incidente. Exija um mínimo de **amostras** na janela antes de abrir; sem isso, o disjuntor abre à toa e vira um serviço que recusa tudo. O lab mostra por que o teste unitário do `Disjuntor` exige `amostras_minimas`.

---

## 5. Avaliação como portão de CI

O módulo 14 proibiu concluir sem intervalo de confiança. A engenharia vai além: transforma a avaliação em **portão**. Um **conjunto dourado** (`(pergunta, resposta esperada)`) executa a cada *push*; se a acurácia cair abaixo do piso, o CI reprova e nada vai para produção.

- **Juiz determinístico** (gabarito exato, resposta estruturada): comparação normalizada. Simples e reprodutível — é o que o lab usa.
- **LLM-as-judge** (critério subjetivo): um modelo julga qual resposta é melhor. **Traiçoeiro** — tem viés de posição, está sujeito a auto-preferência e não é calibrado. Precisou ser **auditado contra gabarito** no módulo 14 (lab 4) e, até lá, só reporta com o protocolo das duas ordens.

O hack pedagógico do lab: **deixar um caso falhando de propósito.** "Quanto é 12 × 3?" o bot de FAQ responde "não sei" e o gabarito espera "36". O eval não diz "modelo ruim" — diz "**capacidade ausente**", que em produção se resolve com **ferramenta** (módulo 15), não com retreino. Ver um eval falhar e saber por quê é mais útil que um número que passa por acaso.

> 🔧 **Na prática:** o portão de CI vale tanto para a qualidade quanto para o custo. O lab 5 tem só o eixo da qualidade; o eixo do custo mora no mesmo `ci.yml` — "a mudança de prompt não pode subir o custo médio por requisição em mais de X%".

---

## 6. Observabilidade: logs estruturados, PII fora do log

Produção não tem o teu console. **Log linear é inútil** — ninguém grepza um monte de texto para achar uma requisição. **Log estruturado** (JSONL) permite filtrar, correlacionar por `trace_id`, e apontar quem gastou e quanto.

O mínimo de um registro de telemetria: `trace_id`, `modelo`, `status`, `latência`, `tokens.in/out`, `custo`, `motivo`. O que **não** entra é o texto do usuário com PII.

> ⚠️ **A armadilha de "detectar PII" que vira vazamento:** a governança do módulo 4 `anonimizar_texto` — mas se você **guarda o dado que anonimizou** num campo, deixou de detectar PII e passou a **armazenar** PII num campo que todo log shippa. O padrão seguro: registre a *presença* (`pii_detectada: true`) e o *tamanho*, nunca o valor. O lab prova que o e-mail não aparece no JSON final.

---

## 7. Registry e rollback

Prompt e modelo são **código de configuração** do sistema: mudam a qualidade tanto quanto um retreino, e são a causa nº 1 de regressão "sem motivo". Um **registry** versiona os dois:

```json
{ "config_id": "3", "modelo": "seu-modelo", "sistema": "...", "max_tokens": 200 }
```

O fluxo de **rollback** com o registro:

1. pinar `config_id=3` (a mudança);
2. rodar o **conjunto dourado** (lab 5) contra ela;
3. se reprovar, voltar para `config_id=2` — **sem retreino, em segundos**.

O objetivo não é nunca errar; é **errar barato e desfazer rápido**. O lab demonstra o caso clássico: uma versão "que economiza" (max_tokens maior) que não melhora a qualidade — o eval é o que impede de ser enganado por uma otimização de custo que só corta qualidade.

---

## 8. O que a camada de produção REAL tem (e este módulo não construiu)

Honestidade de fronteira. O lab constrói o **padrão** em CPU com um brinquedo. Um serviço LLM de verdade troca as peças, mas o desenho continua o mesmo:

| O lab mostra (padrão) | A produção real usa |
|---|---|
| `time.sleep` para simular geração | `vllm` / `mlx-lm` com tokens reais, `--served-model-name` |
| função `solicitar` | API HTTP + contrato versionado; FastAPI/uvicorn quando o projeto exigir |
| `ThreadPoolExecutor` para concorrência | uvicorn + workers, balanceador, rate limiting |
| extrato em dict | Prometheus/Grafana, tracing distribuído (OpenTelemetry), dashboards |
| registry em dict | registry de prompts (registro imutável, hash), HuggingFace/vllm registry |
| eval determinístico | suite de eval em CI (lm-eval-harness / LLM-as-judge) |
| "rollback" manual | canary, blue-green, pin de versão + artefato |

A linha que fica: **saber o nome da ferramenta não é engenharia; saber onde ela entra no desenho, é.** O lab te dá o desenho; a ferramenta você troca quando o projeto real pedir.

O [`capstone de referência`](../modulo-12-projeto/projeto-referencia-rag/) fecha parte dessa
lacuna com biblioteca padrão: endpoint HTTP, health check, API key, limite de payload, logs
estruturados sem a pergunta e um teste de carga concorrente. Ainda não substitui deploy, proxy,
TLS, rate limiting distribuído ou observabilidade externa.

---

## 9. Leituras

1. **Martin Fowler — *Circuit Breaker*** ([martinfowler.com](https://martinfowler.com/bliki/CircuitBreaker.html)). O padrão, em três estados, sem hype.
2. **Amazon — *Fault Isolation / Backpressure*** (AWS Well-Architected). O porquê de recusar rápido sob estresse.
3. **HuggingFace — *Inference* / vLLM docs**. O contrato de serving que o lab simula.
4. **OpenTelemetry (docs)**. O `trace_id` e a correlação de logs que o lab 6 esboça.
5. **LMSYS — *Chatbot Arena* / Evalita** (módulo 14). Por que LLM-as-judge precisa de auditoria — a mesma razão de o lab usar juiz determinístico aqui.

---

## 10. Checklist de saída

- [ ] Por que a média de latência é enganosa, e o que reportar no lugar (p50, p95, taxa)?
- [ ] Por que o custo é dominado pelos tokens de SAÍDA — e qual é o controle robusto?
- [ ] Qual a diferença entre guardião de custo e disjuntor? O que cada um protege?
- [ ] Por que o disjuntor recusa RÁPIDO, e o que isso faz com a latência reportada?
- [ ] O que é um conjunto dourado e como ele vira um portão de CI?
- [ ] Quando usar juiz determinístico vs LLM-as-judge — e por que o segundo precisa de auditoria?
- [ ] Qual o mínimo de um log estruturado de telemetria — e por que PII fica de fora?
- [ ] Como funciona o fluxo de rollback baseado em registry e eval?
- [ ] O que (e por que) o padrão do lab preserva quando você troca o brinquedo pela produção?

Depois: `lab_cpu.py` (executado — o portão de produção do CI roda ele). E os cartões novos em `revisao/baralho-02-expansao.tsv`. Se quiser ir além, generalize o lab apontando `ModeloFAQ` ao modelo real do projeto do módulo 12.

---

*Construído para ser medido, não decorado. Cada número aqui nasceu de um lab que você pode reexecutar.*
