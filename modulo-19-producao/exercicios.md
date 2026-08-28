# Módulo 19 — Exercícios

Faça sem consultar o `lab_cpu.py`. O gabarito está no fim de cada exercício, escondido — abra só depois de tentar.

---

## Parte A — Conceituais

### A1. A cauda que a média esconde

Seu serviço responde em p50 = 30 ms, mas o p95 = 280 ms, para 1000 req/s.

a) Qual é a implicação de você otimizar só o p50?
b) O que o usuário, na prática, "sente"? E o que o p95 te diz que a média esconde?

<details><summary>Gabarito</summary>

a) Otimizar o p50 é mexer na experiência típica sem tocar na **cauda**: 50 req/s continuam sofrendo 280 ms (5% de 1000). Trabalho perdido se o problema do usuário é justamente a cauda.

b) A maioria dos usuários sente o *pico*, não o típico — é a latência do percentil alto que marca a memória de "o serviço é lento". O p95 diz: 5 em 100 requisições estouram — e, como a média é puxada para baixo pelos usuários rápidos, ela superestima sistematicamente a qualidade percebida.
</details>

---

### A2. Custo: onde está o dinheiro?

Você tem preços de entrada = US$ 0,50 / token de entrada, saída = US$ 2,50 / token de saída (por milhão).

a) Qual é a razão saída/entrada?
b) Um colega sugere "reduzir o prompt" para economizar. O raciocínio está completo? O que ele está ignorando?

<details><summary>Gabarito</summary>

a) 2,50 / 0,50 = **5×**. (No lab, 2× em escala menor — a proporção em produção é maior.)

b) Reduzir o prompt economiza na entrada, que é a parte **barata** — e a que você já conhece. O custo dominante é a **saída**, que você não controla de antemão (o modelo decide o tamanho). A economia de verdade está no **teto de `max_tokens`** e na **recusa** de saídas caras, não em enxugar a entrada.
</details>

---

### A3. Guardião de custo vs disjuntor

Explique a diferença entre os dois, e diga qual protege você contra: (i) um `max_tokens` absurdo numa chamada acidental; (ii) um modelo que começou a dar timeout em cascata.

<details><summary>Gabarito</summary>

O **guardião de custo** estima o custo *antes* de gerar e recusa a requisição — protege contra custo alto por chamada (i). O **disjuntor** observa as falhas ao longo de uma janela e passa a recusar *rapidamente* enquanto o modelo está degradando — protege contra desperdiçar dinheiro e p95 num sistema quebrado (ii). São ortogonais: um controla o *preço* do que você sabe que vai pedir; o outro controla o que acontece *depois* de o sistema já estar falhando.
</details>

---

### A4. Por que recusar rápido?

Quando o disjuntor abre, as requisições voltam em microssegundos, sem gerar nada. Por que isso é a decisão certa? O que aconteceria se você continuasse deixando passar?

<details><summary>Gabarito</summary>

Recusar rápido transforma "n chamadas caras que falham devagar" em "n respostas baratas e estáveis". Continuar deixando passar sob um modelo degradado é queimar dinheiro (tokens gerados para algo que vai cair) e degradar o p95 para todos, enquanto o problema se espalha — e sem a interrupção, o gateway do cliente fica lotado (backpressure) e outros usuários sofrem. A recusa rápida é também o mecanismo de *backpressure*: sinaliza cedo que algo está errado, em vez de envenenar a cauda de latência.
</details>

---

## Parte B — Práticos

### B1. Monte um guardião de custo

Escreva uma função `guardiar(tokens_entrada, max_tokens_saida, orcamento)` que decide se permite uma requisição, usando preços de entrada 0,30 e saída 0,60 por milhão. Devolva `(bool, custo_estimado)`.

<details><summary>Gabarito</summary>

```python
def guardiar(tokens_entrada, max_tokens_saida, orcamento,
             p_in=0.30, p_saida=0.60):
    custo = (tokens_entrada * p_in + max_tokens_saida * p_saida) / 1_000_000
    return custo <= orcamento, custo
```

A chave é que o orçamento usa o **teto** de saída, não a saída real (desconhecida antes) — assim você recusa antes de gastar. No lab, `guardiao_custo` faz exatamente isso (e o `tools/producao.calcular_custo` encapsula a conta).
</details>

---

### B2. Por que a "otimização" do lab 7 é uma armadilha?

No lab, a config `3` tinha `max_tokens=200` (o dobro da qualidade? não — só o dobro do custo) e o eval não melhorou. O que isso ensina sobre otimização de custo por `max_tokens`?

<details><summary>Gabarito</summary>

Isso ensina que **aumentar o teto não aumenta a qualidade quando a resposta já cabe no teto menor**. Você compra *right de escrever mais*, não escrita melhor — e paga por ele. O ponto prático: o custo deve ser otimizado **com o eval do lado**; uma "economia" que não é medida contra a qualidade é só um corte cego. Qualquer mudança de prompt/modelo/config precisa passar pelo portão de CI antes de subir.
</details>

---

### B3. Desenhe o log de telemetria mínimo

Sem olhar o lab, escreva o JSON de um log de requisição que você gostaria de ver num dash de produção — incluindo a decisão de como tratar PII.

<details><summary>Gabarito</summary>

```json
{
  "trace_id": "req-0001",
  "modelo": "meu-modelo-v2",
  "status": 200,
  "motivo": "",
  "latencia_s": 0.234,
  "tokens": {"entrada": 150, "saida": 60},
  "custo_usd": 0.000081,
  "seguranca": {"pii_detectada": false}
}
```

O que importa: um **id de correlação** (`trace_id`), o custo, a latência e a **presença de PII** — nunca o valor do dado. Guardar o e-mail que você mascarou é pior que não detectar: vira armazenamento de PII num campo que se shippa para todo lugar. (Compare com o `linha_de_log` do lab 6.)
</details>

---

## Desafio

Aponte o `ModeloFAQ` do lab a um modelo real do seu projeto do módulo 12 e refaça os labs 1, 4 e 5. Em seguida responda, por escrito: **o que muda e o que NÃO muda?** A resposta esperada (e a que separa engenharia de uso de biblioteca): o desenho do padrão não muda — os números e o domínio dos teus guardiões, sim. Se algo no *núcleo* precisou mudar, você descobriu onde o modelo de brinquedo estava enganando você.
