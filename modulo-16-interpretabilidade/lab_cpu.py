# %% [markdown]
# # Módulo 16 — Laboratório: abrindo a caixa-preta (tudo em CPU)
#
# **Roda em CPU (Windows ou Mac), ~12 minutos.** Interpretabilidade é análise de
# ativações, não treino — o Qwen2.5-0.5B basta, e o laptop também. Este é o módulo em
# que paramos de perguntar *o que o modelo faz* e começamos a perguntar *como*.
#
# | Lab | Assunto |
# |---|---|
# | 1 | Logit lens: lendo a "resposta em formação" camada a camada |
# | 2 | **Activation patching: o método causal padrão-ouro** |
# | 3 | Ablação de cabeças: quais importam para uma tarefa? |
# | 4 | Induction heads: o circuito mais famoso, caçado no Qwen |
# | 5 | Linear probing: onde mora um conceito? |
# | 6 | Steering vectors: mudar o comportamento SEM treinar |

# %%
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
torch.manual_seed(0)

V5 = int(transformers.__version__.split(".")[0]) >= 5
DTYPE_KW = {"dtype": torch.float32} if V5 else {"torch_dtype": torch.float32}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# eager é obrigatório para output_attentions (Lab 4)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                             attn_implementation="eager", **DTYPE_KW)
model.eval()
N_CAMADAS = model.config.num_hidden_layers
print(f"Qwen2.5-0.5B: {N_CAMADAS} camadas, d={model.config.hidden_size}")

# %% [markdown]
# ## Lab 1 — Logit lens
#
# A ideia (nostalgebraist, 2020): o *residual stream* (módulo 2) é um barramento que
# cada camada edita. Se projetarmos o estado intermediário de QUALQUER camada pela
# `lm_head`, vemos "o que o modelo preveria se parasse aqui" — a resposta se formando.

# %%
def logit_lens(texto, posicao=-1):
    ids = tok(texto, return_tensors="pt")["input_ids"]
    hs = model(ids, output_hidden_states=True).hidden_states   # 25 estados (embeddings + 24)
    linhas = []
    for L, h in enumerate(hs):
        logits = model.lm_head(model.model.norm(h[0, posicao]))
        p = F.softmax(logits, dim=-1)
        top = torch.topk(p, 1)
        linhas.append((L, tok.decode(int(top.indices[0])), float(top.values[0])))
    return linhas

print("Prompt: 'The capital of France is'  (previsão final esperada: ' Paris')\n")
print(f"{'camada':>7} {'top-1 previsto':>18} {'confiança':>10}")
print("-" * 40)
for L, token, conf in logit_lens("The capital of France is"):
    marca = "  ← resposta emerge" if token.strip() == "Paris" else ""
    print(f"{L:>7} {repr(token):>18} {conf:>10.1%}{marca}")

# %% [markdown]
# **A honestidade do logit lens** (e a primeira lição de interpretabilidade): nas
# camadas intermediárias de um modelo pequeno, o top-1 é RUÍDO — tokens de código,
# fragmentos aleatórios. A resposta "Paris" só emerge nas últimas camadas. Isso NÃO é
# bug: o logit lens cru assume que todas as camadas escrevem no mesmo "dialeto" da
# `lm_head`, o que só é verdade perto do fim. O *tuned lens* (Belrose et al.) treina uma
# projeção por camada para corrigir isso — e é por que ele existe.
#
# Onde o logit lens brilha: ver a resposta *ganhar confiança* nas últimas camadas, e
# comparar prompts. Um fato conhecido emerge cedo; um que exige "computação" emerge tarde.
#
# ## Lab 2 — Activation patching
#
# O método causal padrão-ouro (o módulo 2 avisou: mapas de atenção são correlação, não
# causa). A pergunta: QUAL parte da rede carrega uma informação específica?
#
# A receita: rode um prompt LIMPO e um CORROMPIDO; depois rode o corrompido de novo, mas
# **transplante** a ativação de UMA posição/camada do run limpo. Se a resposta correta
# voltar, aquela ativação CARREGA a informação. É intervenção, não observação.

# %%
LIMPO = "The capital of France is"
CORROMPIDO = "The capital of Russia is"       # mesma estrutura, país trocado

ids_limpo = tok(LIMPO, return_tensors="pt")["input_ids"]
ids_corr = tok(CORROMPIDO, return_tensors="pt")["input_ids"]
assert ids_limpo.shape == ids_corr.shape, "prompts precisam do mesmo nº de tokens"

id_paris = tok(" Paris", add_special_tokens=False)["input_ids"][0]
id_moscou = tok(" Moscow", add_special_tokens=False)["input_ids"][0]

# captura as ativações do run LIMPO em cada camada
hs_limpo = model(ids_limpo, output_hidden_states=True).hidden_states

def logit_de(ids, token_id, patch=None):
    """Roda ids; se patch=(camada, hidden), substitui a saída daquela camada."""
    handles = []
    if patch is not None:
        camada_alvo, novo_h = patch
        def hook(mod, inp, out):
            saida = out[0] if isinstance(out, tuple) else out
            saida[:, -1, :] = novo_h          # transplanta só a última posição
            return out
        handles.append(model.model.layers[camada_alvo].register_forward_hook(hook))
    logits = model(ids).logits[0, -1]
    for h in handles:
        h.remove()
    return float(logits[token_id])

base_moscou = logit_de(ids_corr, id_paris)     # quanto de "Paris" o corrompido dá (baixo)
alvo_limpo = logit_de(ids_limpo, id_paris)     # quanto o limpo dá (alto)
print(f"logit de ' Paris' — corrompido: {base_moscou:.2f} | limpo: {alvo_limpo:.2f}\n")

print("efeito de transplantar a última posição de cada camada limpa no run corrompido:")
print(f"{'camada':>7} {'logit Paris':>12} {'recuperação':>13}")
print("-" * 36)
for L in range(1, N_CAMADAS + 1):
    h_limpo = hs_limpo[L][0, -1]               # saída da camada L no run limpo
    logit = logit_de(ids_corr, id_paris, patch=(L - 1, h_limpo))
    rec = (logit - base_moscou) / (alvo_limpo - base_moscou + 1e-9)
    barra = "█" * max(0, int(rec * 30))
    print(f"{L:>7} {logit:>12.2f} {rec:>12.0%} {barra}")

# %% [markdown]
# **Leia a coluna de recuperação:** as camadas onde transplantar a ativação limpa faz
# "Paris" voltar são as que CARREGAM a informação do país. Tipicamente há um salto numa
# faixa de camadas — é ali que o modelo "move" o fato do país para a posição da resposta.
# Isso é uma afirmação CAUSAL ("esta ativação causa a resposta"), não correlacional — a
# diferença que o módulo 2 martelou.
#
# ## Lab 3 — Ablação de cabeças
#
# Zerar uma cabeça de atenção e medir o dano à tarefa. As cabeças cuja ablação mais
# machuca são as importantes para AQUELA tarefa. É o mesmo princípio do patching, aplicado
# a componentes.

# %%
H = model.config.num_attention_heads
HEAD_DIM = model.config.hidden_size // H

def ablar_cabeca(ids, token_id, camada, cabeca):
    """Zera a saída de UMA cabeça (via hook no o_proj input)."""
    def hook(mod, inp):
        x = inp[0].clone()                     # [batch, seq, hidden] = concat das cabeças
        x[:, :, cabeca * HEAD_DIM:(cabeca + 1) * HEAD_DIM] = 0
        return (x,)
    handle = model.model.layers[camada].self_attn.o_proj.register_forward_pre_hook(hook)
    logits = model(ids).logits[0, -1]
    handle.remove()
    return float(logits[token_id])

base = logit_de(ids_limpo, id_paris)
print(f"logit de ' Paris' sem ablação: {base:.2f}\n")
print("cabeças cuja ablação mais DERRUBA a resposta (top-8 de 24×14):")
danos = []
for camada in range(N_CAMADAS):
    for cabeca in range(H):
        logit = ablar_cabeca(ids_limpo, id_paris, camada, cabeca)
        danos.append((base - logit, camada, cabeca))
danos.sort(reverse=True)
print(f"{'dano':>8} {'camada':>7} {'cabeça':>7}")
for dano, c, h in danos[:8]:
    print(f"{dano:>8.2f} {c:>7} {h:>7}")
print(f"\n(dano positivo = ablar essa cabeça reduz o logit de Paris = ela ajudava)")

# %% [markdown]
# > 🔧 Isto é *ablação de componente* — uma versão barata do patching. As cabeças no topo
# > da lista são candidatas a fazer parte do "circuito" que responde a pergunta. Para
# > confirmar que formam um circuito (e não só coincidem), o método completo é *path
# > patching* — traçar por onde a informação flui entre elas (exercício B3).
#
# ## Lab 4 — Induction heads
#
# O circuito mais famoso da interpretabilidade (Olsson et al., 2022): cabeças que
# COMPLETAM PADRÕES. Dado `[A][B] ... [A]`, elas atendem a `[B]` e preveem `[B]`. É a base
# do in-context learning. Vamos caçá-las com um padrão repetido explícito.

# %%
# Sequência com repetição: tokens aleatórios repetidos duas vezes.
torch.manual_seed(3)
metade = torch.randint(1000, 5000, (20,))
seq = torch.cat([metade, metade]).unsqueeze(0)     # [A B C ... A B C ...]

atencoes = model(seq, output_attentions=True).attentions   # tupla [1,H,s,s] por camada
n = 20

# Uma induction head na 2ª metade, na posição i, atende à posição i-n+1 (o token que
# SEGUIU a ocorrência anterior do token atual). Medimos esse "score de indução".
print("score de indução por cabeça (quanto atende ao token+1 da repetição anterior):")
print(f"{'camada':>7} {'cabeça':>7} {'score':>8}")
scores = []
for L in range(N_CAMADAS):
    a = atencoes[L][0]                          # [H, s, s]
    for h in range(H):
        # para posições na 2ª metade, atenção à posição (i - n + 1)
        vals = [float(a[h, n + i, i + 1]) for i in range(n - 1)]
        scores.append((sum(vals) / len(vals), L, h))
scores.sort(reverse=True)
for s, L, h in scores[:6]:
    print(f"{L:>7} {h:>7} {s:>8.2f}")

media_geral = sum(s for s, _, _ in scores) / len(scores)
print(f"\nscore médio de todas as cabeças: {media_geral:.3f}")
print(f"a melhor cabeça é {scores[0][0]/media_geral:.0f}x a média — é uma induction head.")

# %% [markdown]
# **Se a melhor cabeça tem score muito acima da média, você encontrou uma induction
# head** — o circuito que permite ao modelo aprender padrões DENTRO do contexto, sem
# treinar. É a base mecanicista do few-shot learning (módulo 1): o modelo "copia" a
# estrutura dos exemplos do prompt via essas cabeças.
#
# ## Lab 5 — Linear probing
#
# Onde mora um conceito? Um *probe* é um classificador linear treinado sobre as ativações
# de uma camada. Se ele consegue ler o conceito, o conceito está LINEARMENTE representado
# ali. Testamos: o modelo representa "idioma" (português vs inglês)?

# %%
FRASES_PT = ["O gato dormiu no sofá.", "Hoje o dia está bonito.", "Ela comprou pão fresco.",
             "A cidade é muito grande.", "Nós fomos ao mercado.", "O livro está na mesa.",
             "Amanhã vai chover forte.", "Ele gosta de música.", "A comida estava deliciosa.",
             "Meu carro é azul."]
FRASES_EN = ["The cat slept on the couch.", "Today the weather is nice.", "She bought fresh bread.",
             "The city is very large.", "We went to the market.", "The book is on the table.",
             "Tomorrow it will rain hard.", "He likes music.", "The food was delicious.",
             "My car is blue."]

def ativacao_media(texto, camada):
    ids = tok(texto, return_tensors="pt")["input_ids"]
    h = model(ids, output_hidden_states=True).hidden_states[camada]
    return h[0].mean(0)                          # média sobre tokens -> [d]

def testar_probe(camada):
    X = torch.stack([ativacao_media(f, camada) for f in FRASES_PT + FRASES_EN])
    y = torch.tensor([0] * len(FRASES_PT) + [1] * len(FRASES_EN)).float()
    # probe linear treinado com leave-one-out (honesto: testa no que não viu)
    acertos = 0
    for i in range(len(X)):
        mask = torch.ones(len(X), dtype=torch.bool); mask[i] = False
        w = torch.linalg.lstsq(X[mask], y[mask].unsqueeze(1)).solution
        pred = float(X[i] @ w) > 0.5
        acertos += (pred == bool(y[i]))
    return acertos / len(X)

torch.set_grad_enabled(True)   # lstsq não precisa, mas garante
print(f"{'camada':>7} {'acurácia do probe (idioma PT/EN)':>34}")
print("-" * 44)
for L in [0, 6, 12, 18, 24]:
    print(f"{L:>7} {testar_probe(L):>34.0%}")
torch.set_grad_enabled(False)

# %% [markdown]
# **Se o probe atinge alta acurácia, o idioma é uma direção LINEAR no espaço de
# ativações** — o modelo tem um "eixo de português vs inglês". A camada onde a acurácia
# sobe diz ONDE o conceito fica disponível. Probes são a ferramenta de "o modelo
# representa X?" — usados para achar direções de verdade/mentira, sentimento, toxicidade,
# e até planejamento.
#
# > ⚠️ Probe alto prova REPRESENTAÇÃO, não USO. O modelo pode codificar o idioma sem que
# > isso influencie a saída — a mesma armadilha correlação-vs-causa do módulo 2. Para
# > provar uso, é preciso INTERVIR — o próximo lab.
#
# ## Lab 6 — Steering vectors: mudar o comportamento sem treinar
#
# A carga: se um conceito é uma direção, SOMAR essa direção às ativações deve empurrar o
# comportamento naquela direção — controle sem fine-tuning. Construímos um vetor "idioma"
# e o injetamos.

# %%
CAMADA_STEER = 12

# direção = média das ativações PT menos média das ativações EN (na última posição)
def ativacao_ultima(texto, camada):
    ids = tok(texto, return_tensors="pt")["input_ids"]
    return model(ids, output_hidden_states=True).hidden_states[camada][0, -1]

vec_pt = torch.stack([ativacao_ultima(f, CAMADA_STEER) for f in FRASES_PT]).mean(0)
vec_en = torch.stack([ativacao_ultima(f, CAMADA_STEER) for f in FRASES_EN]).mean(0)
direcao = vec_pt - vec_en
direcao = direcao / direcao.norm()

def gerar_com_steer(prompt, forca=0.0, n_novos=40):
    handle = None
    if forca != 0.0:
        def hook(mod, inp, out):
            saida = out[0] if isinstance(out, tuple) else out
            saida[:, -1, :] = saida[:, -1, :] + forca * direcao
            return out
        handle = model.model.layers[CAMADA_STEER].register_forward_hook(hook)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    for _ in range(n_novos):
        logits = model(ids).logits[0, -1]
        prox = int(logits.argmax())
        ids = torch.cat([ids, torch.tensor([[prox]])], dim=1)
        if prox == tok.eos_token_id:
            break
    if handle:
        handle.remove()
    return tok.decode(ids[0, -n_novos:], skip_special_tokens=True)

prompt = "My favorite thing about the weekend is"
print(f"prompt: {prompt!r}\n")
for forca in [0.0, 6.0, 12.0]:
    saida = gerar_com_steer(prompt, forca=forca)
    print(f"força {forca:>5}: {saida[:120]!r}")

# %% [markdown]
# **Se aumentar a força empurra a geração para o português** (ou degrada de forma
# coerente com a direção), você acabou de CONTROLAR o modelo por intervenção direta nas
# ativações — sem tocar num peso. Isto é *activation steering* (Turner et al., 2023), e é
# a base de uma linha inteira de controle e segurança: direções de recusa, de veracidade,
# de sicofância podem ser somadas ou subtraídas em tempo de inferência.
#
# > 🔧 O steering é o teste causal que o probe (Lab 5) não é: probe mostra que a direção
# > EXISTE; steering prova que ela CAUSA comportamento. Representação + intervenção =
# > a evidência completa da interpretabilidade mecanicista.
#
# ---
#
# ## Encerramento
#
# Você abriu a caixa-preta com as ferramentas reais da área:
#
# - **logit lens** — a resposta se formando no residual stream (e por que o cru falha);
# - **activation patching** — a afirmação CAUSAL de onde a informação mora;
# - **ablação de cabeças** — os componentes que importam para uma tarefa;
# - **induction heads** — o circuito do in-context learning, medido;
# - **linear probing** — conceitos como direções (representação);
# - **steering vectors** — direções como controle (intervenção sem treino).
#
# O fio condutor é o do módulo 2, agora com método: **correlação (mapas, probes) sugere;
# só a intervenção (patching, ablação, steering) prova.** É a diferença entre olhar e
# entender.
#
# O módulo 17 volta à engenharia de escala — o autograd e o paralelismo que treinam os
# modelos que aqui dissecamos.

# %%
