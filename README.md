# 🧠 Engenharia e Customização de LLMs em Português

![Engenharia de LLMs em Português — do zero à produção, com evidência](.github/social-preview.png)

[![CI](https://github.com/lucianoon/llm-course/actions/workflows/ci.yml/badge.svg)](https://github.com/lucianoon/llm-course/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/Python-3.11%E2%80%933.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Licença Apache--2.0](https://img.shields.io/badge/licen%C3%A7a-Apache--2.0-blue.svg)](LICENSE)
[![Status: beta pública](https://img.shields.io/badge/status-beta%20p%C3%BAblica-orange.svg)](#status-do-projeto)

Um curso aberto para entender, customizar, avaliar e servir modelos de linguagem.
É escrito em português, construído de baixo para cima, com laboratórios que tornam as hipóteses mensuráveis.

Há uma porta para quem nunca programou e uma rota até o primeiro nível profissional. O restante do repositório é especialização: dá para trabalhar bem sem concluir os 19 módulos.

> **Fase 0 + 19 módulos · 36 scripts de laboratório**
>
> A rota inteira roda em **CPU**. Mac (MLX) e GPU NVIDIA (CUDA) repetem o treino num modelo grande.

## Experimente em 10 minutos

O primeiro laboratório não baixa modelos. Ele percorre o ciclo do curso: entrada, transformação, previsão e avaliação.

```bash
git clone https://github.com/lucianoon/llm-course.git
cd llm-course
uv sync --extra cpu --extra dev --extra test --locked
uv run python 00-iniciante-zero/lab.py
uv run python -m pytest
```

O próximo arquivo é um só:

- Nunca programou, ou prefere ler antes de instalar → [Fase 0](00-iniciante-zero/)
- Já programa → [trilha essencial](TRILHA-ESSENCIAL.md)

## O que você vai construir

- as peças de um transformer, inclusive a atenção, escritas do zero;
- um conjunto de dados e um ajuste supervisionado (SFT), com adaptação LoRA numa máquina pequena;
- uma avaliação com baseline, intervalo de confiança e leitura das saídas;
- um sistema que consulta documentos, chama ferramentas e recusa quando o custo passa do limite;
- um [projeto final](modulo-12-projeto/) que outra pessoa executa a partir do README.

## A ordem de leitura

`Fase 0 → módulos 1–6 → 11 → 13–15 → 19 → projeto (módulo 12)`

O passo da tabela abaixo é a ordem. O número no nome do módulo é o da pasta: o 12 é o projeto e entra por último, depois de servir o sistema. Os módulos 7–10 e 16–18 ficam de fora até um projeto pedir.

| Passo | Módulo | Pergunta | Onde roda |
|---|---|---|---|
| 1 | [Fase 0 — iniciante](00-iniciante-zero/) | Consigo alterar e testar um programa pequeno? | CPU |
| 2 | [1 · Fundamentos](modulo-01-fundamentos/) | O que um LLM calcula? | CPU |
| 3 | [2 · Atenção](modulo-02-attention/) | Como o contexto vira representação? | CPU |
| 4 | [3 · Treino](modulo-03-treino/) | Como os pesos chegam a escrever? | CPU |
| 5 | [4 · Dados](modulo-04-dados/) | Por que o dado limita a conclusão? | CPU |
| 6 | [5 · Supervised Fine-Tuning](modulo-05-sft/) | Como se ensina um formato? | CPU |
| 7 | [6 · LoRA](modulo-06-lora/) | Como adaptar um modelo grande numa máquina pequena? | CPU |
| 8 | [11 · Inferência](modulo-11-inferencia/) | Como a resposta cabe no orçamento? | CPU |
| 9 | [13 · RAG](modulo-13-rag/) | Como informar o modelo sem treinar de novo? | CPU |
| 10 | [14 · Avaliação](modulo-14-avaliacao/) | Quantas amostras bastam para dizer que A é melhor que B? | CPU |
| 11 | [15 · Agentes](modulo-15-agentes/) | O que muda quando o modelo age? | CPU |
| 12 | [19 · Produção](modulo-19-producao/) | Como servir sem estourar o custo nem esconder a falha? | CPU |
| 13 | [12 · Projeto](modulo-12-projeto/) | Outra pessoa reproduz a conclusão? | conforme o projeto |

A partir do módulo 5, a **CPU fecha o ciclo**: o laboratório treina e mede num modelo pequeno. Mac ou GPU repetem a mesma pergunta num modelo grande. O número medido descreve o modelo pequeno que o laboratório rodou.

### Especializações

| Módulo | Pergunta | Quando abrir |
|---|---|---|
| [7 · Reasoning](modulo-07-reasoning/) | O que muda quando o modelo explica o raciocínio? | A tarefa pede uma cadeia de passos verificável |
| [8 · Preferência (DPO)](modulo-08-dpo/) | Como ensinar uma escolha entre duas respostas boas? | Há pares de respostas preferidas |
| [9 · Reforço (RL)](modulo-09-rl/) | Quando a recompensa pode ser checada por código? | Existe um verificador automático |
| [10 · Destilação](modulo-10-distillation/) | Como passar capacidade para um modelo menor? | O modelo grande custa caro demais para servir |
| [16 · Interpretabilidade](modulo-16-interpretabilidade/) | O que acontece dentro do modelo? | A pergunta é de mecanismo, não de produto |
| [17 · Escala](modulo-17-sistemas/) | Como treinar o que não cabe numa GPU? | O treino passa de uma máquina |
| [18 · Arquiteturas](modulo-18-arquiteturas/) | O que vem depois do transformer? | A pesquisa compara alternativas |
| [Pesquisa](FASE-3-MAESTRIA.md) | Como reproduzir, contribuir e publicar? | A rota essencial já está concluída |

O [mapa de cargos](MAPA-CARGOS.md) liga cada módulo ao que um profissional faz no trabalho. Tempo, pré-requisito e evidência de cada passo estão no [mapa de módulos](MAPA-MODULOS.md).

---

## Uma aula de cada vez

Quando o estudo começa, deixe abertos três arquivos: o `README.md` do módulo, o laboratório e o `exercicios.md`.

| Se acontecer isto | Abra isto |
|---|---|
| O módulo está denso | a caixa **Como ler** no topo do README do módulo |
| Não sei se posso avançar | o checklist no fim desse mesmo README |
| Travei num pré-requisito | a linha daquele módulo no [mapa](MAPA-MODULOS.md) |
| Quero reter na semana seguinte | o [método de estudo](METODO-DE-ESTUDO.md) |
| Quero situar o currículo | o [plano mestre](PLANO-MESTRE.md) |

A primeira passagem da caixa **Como ler** basta para o módulo seguinte. A segunda passagem espera o momento em que a caixa indicar, em geral antes de uma decisão de treino ou de custo. O critério de saída é 80% do checklist, de memória. A partir do módulo 1, 15 minutos de [Anki](https://apps.ankiweb.net) por dia com os baralhos em [`revisao/`](revisao/).

---

## Por que este curso é diferente

Cada alegação experimental aponta para o laboratório que a produz e declara ambiente, amostra e limite. O curso separa três estados:

| Estado | O que significa |
|---|---|
| Lab disponível | O código e o protocolo existem |
| Executado na autoria | Houve uma execução, e o artefato pode não estar preservado |
| Reproduzido | Versões, dados, ambiente e saída foram registrados no repositório |

Os números preliminares, inclusive medições que desmentiram o próprio texto, ficam no [registro de evidências](EVIDENCIAS.md). A prioridade da beta pública é guardar essas reproduções em [`resultados/`](resultados/).

O [comparativo](COMPARATIVO-CURSOS.md) situa o curso diante de Stanford, Hugging Face, Full Stack Deep Learning, DeepLearning.AI e fast.ai. Quem já estudou em outra formação usa a [trilha de equivalência](TRILHA-EQUIVALENCIA.md).

## Certificação

O projeto final é avaliado por uma [rubrica pública com badge verificável](CERTIFICACAO.md): baseline, dados, reprodução, avaliação, operação e revisão técnica. A [trilha de 12 semanas](TRILHA-CERTIFICACAO-12-SEMANAS.md) é o mesmo caminho essencial, com uma evidência por encontro.

---

## Como cada módulo é organizado

Cada pasta `modulo-NN-*/` tem a estrutura abaixo. A Fase 0 usa a mesma convenção com
`README.md`, `lab.py`, `lab.ipynb` gerado e `exercicios.md`. O módulo 12 é a exceção
intencional: substitui o laboratório e os exercícios por um projeto, um contrato de entrega
e um protocolo de validação.

| Arquivo | O que é |
|---|---|
| **`README.md`** | A aula: teoria, matemática, armadilhas e escopo das medições. Leia primeiro. |
| **`lab_cpu.py`** (ou `lab.py`) | O algoritmo do zero e as verificações numéricas. O status está em `EVIDENCIAS.md`. |
| **`lab_mlx.py`** | A mesma receita no Mac (MLX), com modelo real. |
| **`exercicios.md`** | Exercícios com gabarito escondido. Faça antes de olhar o lab. |
| **`dados.py`** | Baixa os datasets, de forma idempotente, quando o módulo usa dados externos. |

Os labs usam células marcadas com `# %%`: o arquivo é um script e vira notebook com `python tools/build_notebooks.py`.

### Referência

Estes arquivos sustentam o curso. Durante a aula, ficam fechados.

| Arquivo | Para quê |
|---|---|
| 📖 [**GLOSSARIO.md**](GLOSSARIO.md) | Termos em linguagem comum, depois a definição, depois o módulo. |
| 🔧 [**GUIA-DE-CODIGO.md**](GUIA-DE-CODIGO.md) | Os padrões que se repetem nos labs: o que fazem, por que existem e onde tropeçam. |
| 🧭 [**PLANO-MESTRE.md**](PLANO-MESTRE.md) | O currículo comparado a outras formações. |
| 🎯 [**METODO-DE-ESTUDO.md**](METODO-DE-ESTUDO.md) | Protocolo semanal para reter. |
| 🗂️ [**revisao/**](revisao/) | Baralhos e diário de erros. |
| 🧾 [**EVIDENCIAS.md**](EVIDENCIAS.md) | Alegações, reproduções e limites. |
| 🔐 [**GOVERNANCA-DE-DADOS.md**](GOVERNANCA-DE-DADOS.md) | Proveniência, licença, checksum e PII antes do treino. |
| 🧭 [**MAPA-MODULOS.md**](MAPA-MODULOS.md) | Pré-requisito, tempo, dificuldade e evidência de saída. |
| 🗺️ [**MAPA-CARGOS.md**](MAPA-CARGOS.md) | Módulo ligado ao trabalho de cada cargo. |
| ✅ [**GATES-DE-APRENDIZAGEM.md**](GATES-DE-APRENDIZAGEM.md) | O que demonstrar antes de avançar. |
| 🧪 [**PROJETO-INTERMEDIARIO.md**](PROJETO-INTERMEDIARIO.md) | Ensaio curto antes do projeto final. |
| 📏 [**PROTOCOLO-DE-AVALIACAO.md**](PROTOCOLO-DE-AVALIACAO.md) | Amostra, baseline, métrica e análise de erro. |
| 📝 [**REGISTRO-DE-REVISAO.md**](REGISTRO-DE-REVISAO.md) | Fontes e datas das afirmações que envelhecem. |
| 🎓 [**Certificação em 12 semanas**](TRILHA-CERTIFICACAO-12-SEMANAS.md) | A rota essencial com entrega semanal. |
| 🔁 [**Equivalência**](TRILHA-EQUIVALENCIA.md) | Por onde entrar se outra formação já cobriu a base. |
| 🤖 [**Agentes**](TRILHA-AGENTES-E-CONTEXT-ENGINEERING.md) | Atalho para a especialização de agentes. |
| 📊 [**Comparativo**](COMPARATIVO-CURSOS.md) | Este curso ao lado de outras formações. |

---

## Instalação

O bloco [Experimente em 10 minutos](#experimente-em-10-minutos) já instala a rota de CPU. Outro ambiente só entra para repetir a customização num modelo grande:

| Se você vai | Abra |
|---|---|
| ficar na CPU, inclusive o SFT do módulo 5 | [`00-setup.md`](00-setup.md) |
| repetir num Mac Apple Silicon | [`00-setup-mac.md`](00-setup-mac.md) |
| repetir numa GPU NVIDIA | [`00-setup-gpu.md`](00-setup-gpu.md) |
| comparar as rotas antes de instalar | [`AMBIENTES.md`](AMBIENTES.md) |

Os `dados.py` baixam os datasets na primeira execução de cada módulo.

## Convenções do material

- **⚠️ Armadilha** — erro comum na primeira tentativa.
- **📐 Matemática** — derivação. Na primeira passagem, leia o resultado e a armadilha. Na segunda, leia o porquê do hiperparâmetro.
- **🔧 Na prática** — o que muda quando isso vai para produção.
- Número concreto (memória, tokens, custo) ao lado do conceito.

## Nota de honestidade

Os labs de **CPU** foram executados durante a escrita. Uma execução passada não garante a próxima: dependências, modelos e dados mudam. Os labs de **MLX** e **CUDA** seguem as APIs documentadas, têm validação estática e modo `--dry-run`, e só contam como reproduzidos depois de uma execução registrada no hardware correspondente. O registro está em [EVIDENCIAS.md](EVIDENCIAS.md).

O fio do curso: **desconfie de todo número que você não mediu.**

---

## Status do projeto

Este repositório está em **beta pública**. O CI cobre a infraestrutura compartilhada, os testes sem download de modelos e labs de CPU selecionados. As execuções aceleradas e os artefatos indicados em [EVIDENCIAS.md](EVIDENCIAS.md) seguem em aberto. Relatos de instalação, erros conceituais e reproduções independentes são bem-vindos.

Veja [CONTRIBUTING.md](CONTRIBUTING.md) para contribuir, [SECURITY.md](SECURITY.md) para relatar vulnerabilidades e [CITATION.cff](CITATION.cff) para citar o projeto.

Distribuído sob a [licença Apache-2.0](LICENSE).

*Feito para ser trabalhado, questionado e medido — não só lido.*
