# %% [markdown]
# # Módulo 19 — Laboratório: Engenharia de Produção de LLM
#
# **Roda em CPU em segundos.** Um modelo treinado é metade do problema; a outra metade é
# servi-lo sob controle de custo, latência e falha. Este lab monta, do zero e **com um
# modelo de brinquedo avisado como brinquedo**, a camada de produção que o módulo 12 só
# descreve: servir e medir, orçar o custo, proteger com disjuntor, avaliar no CI,
# observar com logs estruturados e versionar prompt/modelo.
#
# > **Aviso de escala:** o "modelo" é um bot de FAQ determinístico, não um LLM. A latência
# > e o custo são **simulados** para serem mensuráveis e determinísticos. O que transfere
# > para produção é o *padrão* (as funções de `tools/producao.py` não sabem o que é um
# > modelo); o que não transfere é o número. Troque `modelo_faq` pelo seu modelo e refaça
# > as medições de verdade.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Servir e medir: p50/p95, throughput e sucesso sob concorrência |
# | 2 | Contabilidade: tokens e custo por requisição e por lote |
# | 3 | Guardião de custo: a recusa barata |
# | 4 | Disjuntor: parar de gastar em modelo quebrado |
# | 5 | Evals como CI: o portão que rejeita regressão |
# | 6 | Observabilidade: logs estruturados e PII fora do log |
# | 7 | Registry e rollback: versionar prompt e modelo |

# %%
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "tools"))

from governanca import anonimizar_texto  # PII fora do log
from producao import (  # as peças reutilizáveis
    DEFAULT_PRECOS,
    Disjuntor,
    LinhaDeTrafego,
    calcular_custo,
    contar_tokens,
    resumir_trafego,
)
from respostas import extrair_numero

# %% [markdown]
# ## Lab 1 — Servir e medir
#
# Todo número que decide algo precisa nascer de uma medição. Primeiro o **modelo de
# brinquedo**: um FAQ com pouquíssimas respostas e "não sei" fora da base. A latência
# simulada cresce com o tamanho da resposta, como num LLM de verdade — para que a
# medição seja uma função do comportamento, não um acaso.

# %%
BASE_CONHECIMENTO = {
    "o que é um llm?": "Um modelo de linguagem prevê a próxima palavra dado um contexto, "
                      "e aprendeu isso de grandes quantidades de texto.",
    "quantos parâmetros tem esse modelo?": "A conta depende do modelo; a ordem de grandeza "
                                           "vai de milhões (brinquedo) a dezenas de trilhões (fronteira).",
    "como funciona o tokenizer?": "Ele quebra texto em unidades do vocabulário do modelo, "
                                  "por subpalavra — em português, com mais pedaços que em inglês.",
    "o que é quantização?": "Reduzir a precisão dos pesos para caber em menos memória, "
                            "ao custo de alguma qualidade.",
}
FATOR_LATENCIA_S = 0.0005   # segundos por token de saída (brinquedo truncado para rodar rápido)


class ModeloFAQ:
    """Um 'modelo' determinístico: responde da base ou se abstém.

    A latência é proporcional aos tokens de saída, para que o throughput e os percentis
    não sejam constantes mágicas. `degradado` faz a geração *falhar* — é o que o
    disjuntor do lab 4 vai caçar.
    """

    def __init__(self, degradado: bool = False, falhas_restantes: int = 3):
        self.degradado = degradado
        self.falhas_restantes = falhas_restantes

    def gerar(self, pergunta: str, max_tokens: int = 40) -> str:
        if self.degradado and self.falhas_restantes > 0:
            self.falhas_restantes -= 1
            raise TimeoutError("timeout simulado de geração")
        chave = " ".join(pergunta.strip().lower().split())
        resposta = BASE_CONHECIMENTO.get(chave, "Não sei — foge da minha base de conhecimento.")
        texto = resposta[:max_tokens * 4]
        time.sleep(contar_tokens(texto) * FATOR_LATENCIA_S)
        return texto


# %% [markdown]
# **Dispara 30 requisições concorrentes** e consolida no extrato que um dash mostraria.
# O `timeout` simulado do deque vira um 200 ou um 503; o `resumir_trafego` olha só os
# sucessos para a latência — a recusa rápida não deve puxar o p95 de quem respondeu.

# %%
def solicitar(modelo, pergunta, req_id, max_tokens=40):
    """Uma requisição HTTP fictícia: gera, conta, orça e devolve a linha de tráfego."""
    inicio = time.perf_counter()
    try:
        texto = modelo.gerar(pergunta, max_tokens=max_tokens)
        latencia = time.perf_counter() - inicio
        tokens_in, tokens_out = contar_tokens(pergunta), contar_tokens(texto)
        custo = calcular_custo(tokens_in, tokens_out)
        return LinhaDeTrafego(req_id, True, latencia, tokens_in, tokens_out, custo,
                              "faq-toy-v1", 200)
    except TimeoutError:
        latencia = time.perf_counter() - inicio
        return LinhaDeTrafego(req_id, False, latencia, contar_tokens(pergunta), 0, 0.0,
                              "faq-toy-v1", 503, "timeout_simulado")


modelo_sau = ModeloFAQ()
perguntas = list(BASE_CONHECIMENTO) + [p for p in BASE_CONHECIMENTO for _ in range(5)] + ["pergunta fora?"]
with ThreadPoolExecutor(max_workers=8) as pool:
    linhas = list(pool.map(
        lambda idx: solicitar(modelo_sau, perguntas[idx % len(perguntas)], req_id=idx),
        range(30),
        chunksize=1,
    ))
resumo = resumir_trafego(linhas)
print(f"{'requisições':>13} {resumo['requisicoes']}")
print(f"{'sucesso':>13} {resumo['sucesso']:.0%}")
print(f"{'latência p50':>13} {resumo['latencia_p50_s']*1000:.1f} ms")
print(f"{'latência p95':>13} {resumo['latencia_p95_s']*1000:.1f} ms")
print(f"{'throughput':>13} {resumo['throughput_tokens_s']:.0f} tokens/s")
print(f"{'custo total':>13} ${resumo['custo_total']:.6f}")

# %% [markdown]
# **Leia a assimetria:** com um gateway na frente, o que importa no dash não é a média,
# é o **p95** (a cauda que o usuário sente) e a **taxa de sucesso**. E o custo total, que
# em produção é a primeira linha do orçamento. Mudar `max_tokens` muda o custo — é o que
# os próximos labs vão controlar.
#
# ## Lab 2 — Contabilidade: tokens e custo
#
# Custo não é um acaso do mercado: é uma função determinística `f(tokens, preços)`.
# Entender isso é o que permite **orçar** a requisição antes de gastar. O ponto cego que
# o guardião do lab 3 explora: o custo é dominado pelos SAÍDA, e a saída é decidida
# durante a geração — então o único controle robusto é o **teto de tokens de saída**.

# %%
print(f"{'tokens_saída':>13} {'custo/req':>12}")
for out in (100, 500, 1000, 4000):
    custo = calcular_custo(1500, out)
    print(f"{out:>13} ${custo:>10.6f}")
print("\n(1500 tokens de entrada fixos; preços padrão por milhão).")
print(f"o custo é dominado pela saída: {DEFAULT_PRECOS['saida']/DEFAULT_PRECOS['entrada']:.0f}× "
      "mais caro por token.")

# %% [markdown]
# ## Lab 3 — Guardião de custo: a recusa barata
#
# Uma requisição com um `max_tokens` absurdo pode custar 10× mais. O **guardião de
# orçamento** estima o custo *antes* de gerar e recusa cedo — gasto zero, latência
# mínima. Em produção isso é o que impede uma chamada acidental (ou maliciosa) de drenar
# o budget do mês.

# %%
def guardiao_custo(pergunta, max_tokens, orcamento_usd, precos=None):
    """Estima o custo com o teto de saída antes de gerar. Devolve (permitido, motivo)."""
    tokens_in = contar_tokens(pergunta)
    custo_max = calcular_custo(tokens_in, max_tokens, precos)
    if custo_max > orcamento_usd:
        return False, f"custo_estimado_{custo_max:.6f}_>_orcamento_{orcamento_usd:.6f}"
    return True, ""


pergunta_longa = " ".join(["explique"] * 60)   # uns 60 tokens de entrada no brinquedo
print(f"{'max_tokens':>12} {'custo_max':>14} {'passa?':>8}")
for max_tokens, orcamento in [(50, 0.0005), (500, 0.0005), (4000, 0.0005)]:
    permitido, motivo = guardiao_custo(pergunta_longa, max_tokens, orcamento)
    custo_max = calcular_custo(contar_tokens(pergunta_longa), max_tokens)
    print(f"{max_tokens:>12} ${custo_max:>13.6f} {permitido!s:>8}  {motivo if not permitido else ''}")

print("\nO guardião recusa a requisição de 4000 tokens de saída ANTES de um único "
      "token ser gerado: 0 ms de geração, 0 tokens, 0 custo.")

# %% [markdown]
# ## Lab 4 — Disjuntor: parar de gastar em modelo quebrado
#
# Economia de custo e honra de SLA. Quando o modelo começa a falhar (timeout, erro),
# continuar chamando é queimar dinheiro e p95. O disjuntor abre depois de uma janela de
# falhas e passa a **recusar rápido** (`fast-fail`); depois do resfriamento, uma prova
# decide entre reabrir ou fechar.

# %%
def modelo_quebrado():
    """Um 'modelo' que começa falhando muito — para o disjuntor abrir no lab."""
    return ModeloFAQ(degradado=True, falhas_restantes=100)


disjuntor = Disjuntor(limiar_falha=0.3, janela_s=5.0, resfriamento_s=0.5, amostras_minimas=4)
modelo_ruim = modelo_quebrado()
base = 0.0
resultados = []
for i in range(6):
    # o relógio avança a cada requisição para caber no resfriamento do teste
    t = base + i * 0.1
    aberto_antes = disjuntor._aberto
    if disjuntor.permitir(t):
        try:
            modelo_ruim.gerar("o que é um llm?")
            disjuntor.registrar_sucesso(t)
            status, motivo = "200", "sucesso"
        except TimeoutError:
            disjuntor.registrar_falha(t)
            status, motivo = "503", "timeout"
    else:
        status, motivo = "503", "disjuntor_recusa"
    resultados.append((i, status, motivo, aberto_antes, disjuntor.taxa_falha(t)))

print(f"{'#':>3} {'status':>8} {'motivo':>18} {'fechado?':>9} {'taxa':>7}")
for i, status, motivo, aberto_antes, taxa in resultados:
    print(f"{i:>3} {status:>8} {motivo:>18} {not aberto_antes!s:>9} {taxa:>7.1%}")

# após o resfriamento, a prova decide
print(f"\ndisjuntor aberto após 4 falhas: {disjuntor._aberto}")
t_prova = base + 2.0
print(f"  após cooldown, primeira chamada (prova): permitir = {disjuntor.permitir(t_prova)}")
disjuntor.registrar_sucesso(t_prova)   # a prova passou: fecha
print(f"  prova OK -> disjuntor fechado: {not disjuntor._aberto}")

# %% [markdown]
# **A métrica que muda é a latência da recusa:** quem recebe 503 pelo disjuntor volta em
# microssegundos, sem gerar nada. O `resumir_trafego` separa os sucessos dos recusados —
# sem isso, o p95 pareceria melhor do que é, porque as recusas rápidas entram na conta.
#
# ## Lab 5 — Evals como CI
#
# O módulo 14 ensinou que toda métrica é uma estimativa. Em engenharia, ela vira um
# **portão**: um conjunto dourado de (pergunta, resposta esperada) que é executado a cada
# *push*. Se a acurácia cai abaixo do piso, o CI reprova e nada vai para produção.
#
# A extensão honesta: um juiz determinístico checa a resposta normalizada. Quando o
# critério é subjetivo, entra o **LLM-as-judge** — que tem viés e por isso precisa do
# protocolo das duas ordens e de auditoria (módulo 14, lab 4). Aqui o gabarito é exato,
# então o juiz pode ser determinístico.

# %%
CONJUNTO_DOURADO = [
    ("o que é um llm?", BASE_CONHECIMENTO["o que é um llm?"]),
    ("quantos parâmetros tem esse modelo?", BASE_CONHECIMENTO["quantos parâmetros tem esse modelo?"]),
    ("como funciona o tokenizer?", BASE_CONHECIMENTO["como funciona o tokenizer?"]),
    ("qual a capital da frança?", "Não sei — foge da minha base de conhecimento."),  # abstenção correta
    ("quanto é 12 * 3?", "36"),  # capacidade ausente (aritmética): o eval marca a regressão
]


def normalizar_resposta(texto: str) -> str:
    """Pequena normalização para comparação exata (sem juiz neural)."""
    texto = texto.lower()
    for tonica, simples in [("ã", "a"), ("á", "a"), ("â", "a"), ("à", "a"), ("ê", "e"),
                            ("é", "e"), ("í", "i"), ("ó", "o"), ("ô", "o"), ("õ", "o"),
                            ("ú", "u"), ("ü", "u"), ("ç", "c")]:
        texto = texto.replace(tonica, simples)
    return " ".join("".join(c for c in texto if c.isalnum() or c.isspace()).split())


def juiz_deterministico(esperada: str, gerada: str) -> bool:
    """Compara a resposta gerada ao gabarito, com a extração numérica quando aplicável."""
    numero_esp = extrair_numero(esperada)
    numero_ger = extrair_numero(gerada)
    if numero_esp is not None and numero_ger is not None:
        return numero_esp == numero_ger
    return normalizar_resposta(esperada) == normalizar_resposta(gerada)


def avaliar_conjunto_dourado(modelo, max_tokens=40):
    """Roda o conjunto dourado e devolve o relatório que o CI consome."""
    linhas = []
    for pergunta, esperada in CONJUNTO_DOURADO:
        gerada = modelo.gerar(pergunta, max_tokens=max_tokens)
        linhas.append({
            "pergunta": pergunta,
            "esperada": esperada,
            "gerada": gerada,
            "passa": juiz_deterministico(esperada, gerada),
        })
    acertos = sum(1 for linha in linhas if linha["passa"])
    return {
        "n": len(linhas),
        "acertos": acertos,
        "acuracia": round(acertos / len(linhas), 3),
        "casos": linhas,
    }


relatorio = avaliar_conjunto_dourado(ModeloFAQ())
print(f"acurácia: {relatorio['acertos']}/{relatorio['n']} = {relatorio['acuracia']:.0%}")
for caso in relatorio["casos"]:
    marcador = "✓" if caso["passa"] else "✗"
    print(f"  {marcador} {caso['pergunta'][:36]:<36} -> {caso['gerada'][:44]}")

piso = 0.6
print(f"\nportão CI (piso {piso:.0%}): {'PASSOU' if relatorio['acuracia'] >= piso else 'REPROVOU'}")

# %% [markdown]
# **O caso que falha é a aula:** "quanto é 12 × 3?" o bot responde "não sei" e o gabarito
# espera "36". O eval não decreta um modelo ruim — decreta uma **capacidade ausente**
# (aritmética), que em produção se resolve com ferramenta (módulo 15), não com retreino.
# Ver o eval falhar e saber POR QUÊ é mais valioso que um número que passa por acaso.
#
# ## Lab 6 — Observabilidade: logs estruturados e PII fora do log
#
# Produção não tem o teu console. Log linear é inútil; **log estruturado** (JSONL) permite
# filtrar, correlacionar e apontar quem gastou e quanto. E, antes de enviar, **PII não
# entra no log** — a governança do módulo 4 aplicada à telemetria.

# %%
def linha_de_log(req_id, pergunta, texto, latencia_s, status, motivo="", modelo="faq-toy-v1"):
    """O registro de telemetria: caminho que o gestor de logs shippa."""
    return {
        "trace_id": f"req-{req_id:04d}",
        "modelo": modelo,
        "status": status,
        "motivo": motivo,
        "latencia_s": round(latencia_s, 4),
        "tokens": {
            "entrada": contar_tokens(anonimizar_texto(pergunta) if status != 200 else pergunta),
            "saida": contar_tokens(texto),
        },
        "custo_usd": round(calcular_custo(
            contar_tokens(pergunta), contar_tokens(texto)), 6) if status == 200 else 0.0,
        # PII: o texto NUNCA entra cru — só o tamanho e indicadores de presença.
        "seguranca": {"pii_detectada": bool(anonimizar_texto(pergunta) != pergunta)},
    }


inicio = time.perf_counter()
texto = ModeloFAQ().gerar("o que é um llm?")
latencia = time.perf_counter() - inicio
print(json.dumps(linha_de_log(1, "o que é um llm?", texto, latencia, 200),
                 ensure_ascii=False, indent=2))

com_pii = linha_de_log(2, "meu email é ana@exemplo.com", "ok", 0.01, 200)
print("\nPII detectada e REVELADA com máscara:", "pii_detectada =", com_pii["seguranca"]["pii_detectada"])
print("texto original no log? ", "ana@exemplo.com" in json.dumps(com_pii, ensure_ascii=False))

# %% [markdown]
# **As duas linhas que importam:** `pii_detectada` (a presença, não o dado) e o fato de o
# e-mail NÃO aparecer no JSON. Guardar o dado que você detectou é o erro de segurança
# clássico — para de detectar PII e passa a *vazar* PII num campo que todo log shippa.
#
# ## Lab 7 — Registry e rollback
#
# Prompt e modelo são **código de configuração** e mudam a qualidade tanto quanto o
# retreino. Um registry versiona os dois; o eval do lab 5 é quem valida a subida. O
# padrão de rollback: pinar `config_id=3`, rodar o eval, e se reprovar, voltar para `2` —
# sem retreino, em segundos.

# %%
REGISTRY = {
    "1": {"modelo": "faq-toy-v1", "sistema": "Você é um assistente técnico.", "max_tokens": 40},
    "2": {"modelo": "faq-toy-v1", "sistema": "Responda apenas com base no contexto.", "max_tokens": 40},
    "3": {"modelo": "faq-toy-v1", "sistema": "", "max_tokens": 200},   # versão "que economiza" — na real, dobra o custo
}

print(f"{'config_id':>10} {'max_tokens':>12} {'avaliada':>10}")
melhor = None
for config_id, config in REGISTRY.items():
    with ThreadPoolExecutor(max_workers=1) as pool:
        respostas = [pool.submit(ModeloFAQ().gerar, p, config["max_tokens"]) for p, _ in CONJUNTO_DOURADO]
        geradas = [f.result() for f in respostas]
    acertos = sum(1 for (p, e), g in zip(CONJUNTO_DOURADO, geradas) if juiz_deterministico(e, g))
    print(f"{config_id:>10} {config['max_tokens']:>12} {acertos}/{len(CONJUNTO_DOURADO)}")
    if melhor is None or acertos > melhor[1]:
        melhor = (config_id, acertos)

print(f"\nbest: config_id={melhor[0]} (a mudança de max_tokens não mudou a qualidade — "
      f"o eval é o trunfo para não ser enganado por 'otimização' de custo que só corta qualidade).")

# %% [markdown]
# ---
#
# ## Encerramento
#
# O curso terminou o pipeline do modelo (módulos 1–12) e os sistemas do entorno
# (13–15). Este módulo fecha o que o sistema 12 descrevia sem receita: **colocar a
# coisa em pé, de forma medida e protegida**.
#
# - **Medir** — p50/p95/throughput/sucesso sob concorrência (o dash do projeto real);
# - **Orçar** — o custo como função determinística e a recusa barata do guardião;
# - **Proteger** — disjuntor que para de gastar em modelo quebrado e recusa no micro;
# - **Avaliar** — o conjunto dourado como portão de CI, com gabarito e juiz auditável;
# - **Observar** — logs estruturados, com PII fora do log;
# - **Versionar** — registry de prompt/modelo e rollback guiado por eval.
#
# A regra que o módulo 14 declara e este módulo operacionaliza: **nada sobe para produção
# sem um número e sem um jeito de cair.** Ganhar tempo no deploy para perder a confiança
# é o pior negócio da engenharia.

# %%

# %% [markdown]
# ## Desafio — generalize onde importa
#
# Remova o brinquedo: aponte `ModeloFAQ` para o modelo real do seu projeto (módulo 12) e
# refaça labs 1, 4 e 5. O que muda? O que NÃO muda? A resposta certa é: o padrão não muda;
# os números e o domínio do seu guardião, sim. É o mesmo teste que separa "usei a
# biblioteca" de "entendi a engenharia".

# %%

