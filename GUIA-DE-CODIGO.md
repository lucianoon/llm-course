# Guia de código — como ler os labs

Os labs reutilizam um punhado de padrões. Este guia explica cada um em linguagem simples — **o que faz, por que é assim, e onde tropeça quem lê pela primeira vez**. Depois de entender estes blocos, qualquer lab do curso vira leitura fluida: são sempre os mesmos tijolos em combinações novas.

Convenção dos títulos: `arquivo → o que procurar`.

---

## 1. O cabeçalho de todo lab

```python
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

import torch
sys.path.insert(0, str(AQUI.parent / "tools"))   # habilita: import minigpt
import minigpt

torch.manual_seed(0)          # congela o acaso: mesmo sorteio toda vez
torch.set_grad_enabled(False) # (só em labs de inferência) desliga a contabilidade de treino
```

**O que faz:** encontra a pasta do módulo, prepara o ambiente e torna o experimento reproduzível.
**Por que funciona também no notebook:** `tools/build_notebooks.py` acrescenta à primeira
célula um pequeno prólogo que procura o `pyproject.toml` e define `__file__` como o caminho
do `lab.py` original. Assim, o mesmo cabeçalho funciona com o kernel iniciado na raiz ou
na pasta do módulo; o notebook não depende do diretório acidental do processo.
**Por que a seed:** `manual_seed` faz o "aleatório" sair igual em toda execução — sem isso,
você nunca sabe se uma diferença veio da sua mudança ou do sorteio.
**Tropeço comum:** esquecer que a seed fixa *a sequência* de sorteios — inserir uma chamada
aleatória no meio do código desloca todos os sorteios seguintes.

---

## 2. Treinar um tokenizer BPE (módulos 3+)

```python
tk = Tokenizer(models.BPE(unk_token=None))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator([texto], trainers.BpeTrainer(vocab_size=2048, ...))
ids = torch.tensor(tk.encode(texto).ids)
```

**O que faz:** aprende, a partir do SEU corpus, quais pedaços de texto viram as "peças de Lego" (tokens) — e converte o corpus inteiro numa lista de números.
**Por quê `ByteLevel`:** operando sobre bytes, *qualquer* texto é representável — não existe caractere desconhecido. É por isso que `ção` aparece como `Ã§Ã£o` nas tabelas internas: são os bytes do UTF-8 vistos um a um (módulo 1 explica).
**Por quê `vocab_size=2048`:** o MiniGPT é minúsculo; um vocabulário pequeno deixa a matriz de embeddings proporcional. Modelos reais usam 32k–256k.
**Tropeço:** o tokenizer é um retrato do corpus de treino — o nosso, treinado em Machado, pica palavras técnicas em pedacinhos (e vice-versa).

---

## 3. `pegar_batch` — o alimentador do treino

```python
def pegar_batch(fonte, batch_size, bloco):
    inicio = torch.randint(len(fonte) - bloco, (batch_size,))
    x = torch.stack([fonte[i : i + bloco] for i in inicio])         # a entrada
    y = torch.stack([fonte[i + 1 : i + 1 + bloco] for i in inicio]) # o alvo: x deslocado 1
    return x, y
```

**O que faz:** sorteia `batch_size` janelas de texto do corpus e monta os pares (entrada, alvo).
**A linha que importa:** `y` é `x` **deslocado uma posição** — porque a tarefa do modelo é, em cada posição, prever o token *seguinte*. Esse deslocamento ("shift") é o coração de todo treino de LLM, e errá-lo é o bug clássico (módulo 1, Lab 7: o modelo roda, treina e fica inútil, sem nenhuma mensagem de erro).
**Tropeço:** o batch é sorteado *com reposição* de posições aleatórias — não há noção de época exata aqui; épocas são estimadas por `passos × batch × bloco ÷ tamanho_do_corpus`.

---

## 4. O loop de treino canônico (módulo 3 — e todos os seguintes)

```python
for passo in range(passos):
    lr = agenda_lr(passo, passos)              # 1. o passo de hoje: warmup + decay
    for g in otim.param_groups: g["lr"] = lr

    otim.zero_grad(set_to_none=True)           # 2. zera as anotações do passo anterior
    x, y = pegar_batch(...)                    # 3. pega um lote
    _, perda = modelo(x, y)                    # 4. forward: mede o erro
    perda.backward()                           # 5. backward: calcula as setas de ajuste
    torch.nn.utils.clip_grad_norm_(..., 1.0)   # 6. disjuntor: corta setas absurdas
    otim.step()                                # 7. gira os botões
```

**O que faz:** o ciclo mede-erro → calcula-correção → aplica-correção, repetido milhares de vezes. **Todo treino de rede neural é este loop** — SFT, LoRA, DPO e GRPO só mudam o que acontece no passo 4 (como o erro é calculado).
**Por que zerar (passo 2):** o PyTorch **soma** gradientes por padrão (é o que permite gradient accumulation); sem zerar, cada passo carregaria o lixo do anterior.
**Por que o clipping (passo 6):** um lote podre gera uma correção gigante que pode destruir o modelo. O disjuntor quase nunca atua — e quando atua, salva o treino (módulo 3 mediu: norma 1,67 no passo 0, ~0,8 depois).
**Tropeço:** a ordem importa. `backward` antes de `zero_grad` = gradientes duplicados; `step` antes de `clip` = disjuntor inútil.

---

## 5. `tools/minigpt.py` — a anatomia do modelo

O arquivo compartilhado por todos os labs. Cada classe em uma frase:

| Classe/função | O que é, em linguagem simples |
|---|---|
| `Config` | A ficha técnica: quantas camadas, qual largura, quantas cabeças de atenção. Mudar aqui muda o tamanho do modelo. |
| `RMSNorm` | O "estabilizador de voltagem" aplicado antes de cada operação: reescala os números para uma faixa saudável. (Note o `.float()`: a conta é feita em precisão alta de propósito — módulo 2.) |
| `rope_cache` / `rotate_half` | A maquinaria do relógio de posições (RoPE): pré-calcula os ângulos de rotação que informam ao modelo a ordem das palavras. |
| `Atencao` | Onde as palavras conversam: projeta Q, K, V, aplica o RoPE em Q e K (**nunca em V** — módulo 2), e chama `scaled_dot_product_attention` com `is_causal=True` (a venda que impede espiar o futuro). |
| `MLP` | A estação de processamento individual (SwiGLU): `abaixo(silu(portao(x)) * acima(x))` — o portão decide quanto de cada canal passa. |
| `Bloco` | Um andar completo: estabiliza → conversa (atenção) → soma de volta; estabiliza → processa (MLP) → soma de volta. As somas ("residuais") são o corrimão que deixa o gradiente subir a pilha inteira. |
| `MiniGPT` | O prédio: embedding → N andares → estabilizador final → projeção de volta ao vocabulário. `self.cabeca.weight = self.emb.weight` é o *weight tying*: a mesma tabela serve de entrada e de saída. |
| `MiniGPT.forward` | Recebe tokens, devolve logits — e, se receber os alvos, calcula a loss. **Atenção:** os alvos já chegam deslocados pelo `pegar_batch`, então NÃO há shift aqui (fazê-lo de novo = deslocar duas vezes). |
| `MiniGPT.gerar` | O loop de escrita: prevê → sorteia (temperatura + top-k) → anexa → repete. É o `generate()` do módulo 1 em miniatura. |
| `treinar` | O loop canônico da seção 4, empacotado: devolve `(modelo, histórico)`. |
| `avaliar` | Mede a loss média em lotes do conjunto de validação — o termômetro honesto do progresso. |

---

## 6. `logprob_resposta` — o canivete dos módulos 7–10

```python
def logprob_resposta(modelo, prompts, respostas):
    seq = torch.cat([prompts, respostas], dim=1)      # cola prompt + resposta
    logits, _ = modelo(seq)
    lp = F.log_softmax(logits[:, PROMPT_LEN - 1 : -1], dim=-1)  # ← o shift!
    return lp.gather(2, respostas.unsqueeze(-1)).squeeze(-1).sum(dim=1)
```

**O que faz:** responde à pergunta "**quão provável o modelo acha esta resposta, dada esta pergunta?**" — em log (números negativos; mais perto de zero = mais provável).
**Como:** roda o modelo sobre prompt+resposta colados, pega a previsão de cada posição, e soma o log da probabilidade que o modelo deu a cada token *que de fato veio*.
**A fatia `[PROMPT_LEN - 1 : -1]`:** é o shift de sempre — os logits da posição `t` preveem o token `t+1`, então a previsão do primeiro token da resposta está na *última posição do prompt*. Errar esse índice por 1 é o bug mais fácil de cometer no curso inteiro.
**Onde aparece:** módulo 7 (o efeito do raciocínio na probabilidade), 8 (as quatro log-probs do DPO), 9 (as razões de política do GRPO), 10 (avaliação).

---

## 7. `LoRALinear` — o módulo removível (módulo 6)

```python
class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        self.base = base                      # a camada original...
        for p in self.base.parameters():
            p.requires_grad = False           # ...CONGELADA (não treina)
        self.lora_A = nn.Parameter(...)       # matriz fina 1 (aleatória)
        self.lora_B = nn.Parameter(zeros)     # matriz fina 2 (ZEROS!)

    def forward(self, x):
        return self.base(x) + (alpha/r) * B(A(x))   # original + correção pequena
```

**O que faz:** envelopa uma camada congelada e soma a ela uma "correção" treinável e minúscula (duas matrizes finas).
**Por que `B` começa em zeros:** zeros × qualquer coisa = zero → no primeiro passo, a correção é nula e o modelo é EXATAMENTE o original. O treino parte de casa, sem solavanco. (E por que não as duas em zero? Aí o gradiente também seria zero e nada aprenderia — exercício A2 do módulo 6.)
**`requires_grad = False`:** a linha que economiza 90%+ da memória — sem gradiente, a camada congelada não precisa dos 12 bytes/parâmetro de anotações do otimizador.
**`merge()`:** no final, a correção pode ser somada aos pesos originais (`W + (α/r)·B·A`) — o módulo removível é absorvido pela parede, e servir não custa nada a mais.

---

## 8. O loop de DPO (módulo 8)

```python
# fora do loop: a REFERÊNCIA — uma cópia congelada do modelo original
referencia = copy.deepcopy(base); referencia.eval()
for p in referencia.parameters(): p.requires_grad = False

# dentro do loop:
with torch.no_grad():                       # a referência nunca treina
    lc_ref = logprob_resposta(referencia, p, chosen)
    lr_ref = logprob_resposta(referencia, p, rejected)
lc_pol = logprob_resposta(politica, p, chosen)      # a política treina
lr_pol = logprob_resposta(politica, p, rejected)

loss = -F.logsigmoid(beta * ((lc_pol - lc_ref) - (lr_pol - lr_ref)))
```

**O que faz:** quatro medições de probabilidade (resposta boa e ruim, no modelo atual e no original) e uma conta que diz: *aumente a preferida em relação ao que você era, diminua a preterida*.
**O papel da referência:** é o "elástico" — as diferenças `pol − ref` medem o quanto o modelo *se afastou de si mesmo*, e o β regula o custo desse afastamento. Sem ela, o modelo se descaracteriza perseguindo a margem.
**O teste de sanidade grátis:** no passo 0, política = referência → todas as diferenças são zero → `loss = ln 2 ≈ 0,693`. Se a sua loss inicial não for essa, há bug (é o análogo do `ln(V)` do pré-treino).
**Tropeço de leitura:** a loss só pede que a *diferença* cresça — e o caminho preguiçoso é derrubar as DUAS probabilidades (a ruim mais rápido). Por isso se monitora chosen e rejected separadas, nunca só a loss (o lab mediu: chosen caiu 44 nats num treino "bem-sucedido").

---

## 9. O loop de GRPO (módulo 9)

```python
# 1. GERAR: para cada pergunta, G tentativas do próprio modelo
seqs = politica.gerar(prompts.repeat_interleave(G, dim=0), ...)   # temperatura > 0!

# 2. PONTUAR: o verificador dá a nota de cada tentativa
rs = torch.tensor([recompensa(decodificar(s)) for s in seqs])

# 3. COMPARAR DENTRO DO GRUPO: nota − média do grupo, normalizada
vantagens = (rs - rs.mean_do_grupo) / rs.std_do_grupo

# 4. REFORÇAR: sobe a probabilidade de quem ficou acima da média do próprio grupo,
#    com trava (clip) e elástico (KL contra a referência)
ratio = torch.exp(lp_novo - lp_old)
loss = -(min(ratio*A, clip(ratio)*A) - beta_kl * kl).mean()
```

**O que faz:** o ciclo tenta → confere → reforça. O modelo gera as próprias tentativas, um verificador as pontua, e as acima da média do grupo ficam mais prováveis.
**Por que "grupo":** a média das G tentativas da MESMA pergunta é a régua de "o que era esperado" — sem precisar de um modelo extra para estimá-la (a simplificação do GRPO sobre o PPO).
**Por que temperatura > 0 na geração:** com T=0 as G tentativas sairiam idênticas → todas iguais à média → vantagem zero → nada a aprender. A diversidade do grupo É o mecanismo de exploração.
**As duas travas do passo 4:** o `clip` impede um passo grande demais de uma vez; o `kl` impede o afastamento acumulado da linguagem original. O lab mediu o que acontece sem o KL: 100% de sucesso no verificador e o texto destruído.

---

## 10. Os padrões de avaliação

### Perplexidade (o termômetro de linguagem)
```python
loss = F.cross_entropy(shift_logits, shift_labels)
ppl = math.exp(loss)
```
"Entre quantas opções o modelo hesita." Serve para comparar **o mesmo modelo** antes/depois, ou modelos com o **mesmo tokenizer**. Regras de uso: milhares de tokens (30 tokens = ruído, módulo 6 provou), domínios variados, e nunca entre tokenizers diferentes.

### Taxa de sucesso verificável (o padrão dos módulos 7 e 9)
```python
acertou = extrair_resposta(texto_gerado) == gabarito
```
Gera, extrai, compara. A parte traiçoeira é a **extração** — uma regex ingênua transforma acertos em erros e você conclui que o modelo é pior do que é. Por isso a função de extração tem os próprios testes de unidade (módulo 7, Lab 1) e uma regra de ouro: inspecione manualmente uma amostra dos "erros" antes de acreditar em qualquer número.

### O teste de sanidade universal
Todo experimento do curso começa conferindo um valor conhecido a priori:
- loss inicial de treino ≈ `ln(vocabulário)` (modelo aleatório);
- loss inicial de DPO ≈ `ln 2` (política = referência);
- LoRA no passo 0 = idêntico à base (B = zeros);
- métrica testada contra o exemplo de ouro antes de avaliar qualquer modelo.

**Se o valor de largada não bate com o teórico, pare** — o bug está no pipeline, não no modelo. É o hábito mais barato e mais rentável do curso.

---

## 11. Os padrões dos labs MLX (módulos 5+, no Mac)

```python
from execucao import executar_modulo

resultado = executar_modulo("mlx_lm", "lora", "--train", ..., mostrar=2000)
```

**Por que subprocess em vez de importar:** os treinos do MLX são ferramentas de linha de comando (`mlx_lm.lora`, `mlx_lm.generate`) — o lab as chama como você chamaria no terminal, capturando a saída. O módulo compartilhado `tools/execucao.py` concentra tempo, log e código de saída; por padrão, qualquer falha interrompe o lab em vez de deixar as células seguintes usarem artefatos incompletos.
**O padrão defensivo `try/except (ImportError, TypeError)`:** a API do `mlx-lm` muda entre versões (o sampler, por exemplo); o lab tenta a forma nova e cai para a antiga. Se ambas falharem, a mensagem de erro é a informação útil — reporte-a.
**Configuração por YAML:** `rank`, `scale` e afins **não têm flags de linha de comando** no `mlx_lm.lora` — só via `--config arquivo.yaml`. O lab gera o YAML explicitamente e o **imprime antes de treinar**: flags silenciosamente ignoradas são o pior tipo de bug.
**`mx.clear_cache()` entre modelos:** o Mac tem 16 GB para tudo; descarregar um modelo antes de carregar o próximo (`del model; mx.clear_cache()`) é o que evita o estouro de memória no meio do lab.

---

## 12. O padrão de produção (módulo 19)

`tools/producao.py` tem as peças que o lab de engenharia de produção reutiliza:

- `calcular_custo(tokens_entrada, tokens_saida, precos)` — o custo como função determinística;
- `contar_tokens(texto)` — contagem de **brinquedo** (aviso no docstring: troque pelo tokenizer real);
- `LinhaDeTrafego` + `resumir_trafego(...)` — o extrato p50/p95/throughput de sucesso/custo;
- `Disjuntor(...)` — o circuit breaker em três estados.

O desenho que eles servem:

```python
if not disjuntor.permitir():
    return 503_fast_fail              # não gasta um token
custo_max = calcular_custo(tok_entrada, max_tokens)
if custo_max > orcamento:
    return 403_orcamento              # recusa ANTES de gerar
resposta = modelo.gerar(...)          # o trabalho caro, só aqui
```

**O que importa:** o desenho não muda quando o brinquedo vira produção — muda o número.
É o mesmo princípio do curso: **escalar o modelo, não a lógica**. E o aprendizado que vale:
o custo está na **saída** e na **recusa** — não na entrada e não no retreino.

> ⚠️ **A armadilha do brinquedo:** `contar_tokens` e o `time.sleep` simulam o comportamento para
> o lab ser mensurável. Usar a função de brinquedo para *cobrar* ou para uma conta que decide
> algo é o erro que o módulo chama de "escala de brinquedo com aviso de escala".

---

## Como usar este guia

Na primeira leitura de cada lab, mantenha este arquivo aberto ao lado. Quando um bloco de código parecer opaco, procure o padrão aqui (são sempre estes 11). Na segunda leitura, você não vai mais precisar dele — e essa é a medida de que o curso está funcionando.

Termos desconhecidos: consulte o [GLOSSARIO.md](GLOSSARIO.md).
