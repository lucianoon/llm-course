# Mapa competência → cargo

Uma dúvida legítima de quem estuda sozinho: **"isso serve para o mercado?"**. Para que ela não fique no ar, cada módulo do curso foi mapeado para o que um profissional realmente faz no trabalho. Este documento é a ponte entre o conteúdo e a proposta de contratação.

Não é uma promessa de que estes módulos *soltos* bastam para o cargo — é o mapa do que cada um cobre, para você saber onde investir tempo conforme o trabalho (ou a vaga) pede.

## As duas portas de entrada

| Trilha | Meta | Para quem |
|---|---|---|
| **Engenharia (mercado)** | Colocar sistemas de LLM de pé, medidos e protegidos — em qualquer stack | Quem quer o primeiro nível profissional com entrega real |
| **Pesquisa (academia)** | Reproduzir, questionar e estender o estado da arte | Quem quer produzir conhecimento, publicar e disputar a fronteira |

O mesmo repositório serve às duas. O que muda é a ordem e o que você *omite*.

---

## Mapas por cargo

A convenção: 🟢 essencial · 🟡 importante · ⚪ quando o projeto real pedir.

### Engenheiro de LLM aplicado (a maioria das vagas de "AI Engineer" / "LLM")

| Habilidade no trabalho | Onde no curso |
|---|---|
| Entender o que o modelo calcula (tokens, logits, cache) | Módulo 1 🟢 |
| Implementar/usar um transform com atenção | Módulo 2 🟢 |
| Escolher entre prompt, RAG, SFT, DPO, RL | Módulo 12 (tabela de decisão) 🟢 |
| Fazer SFT e fine-tuning eficiente (LoRA/QLoRA) | Módulos 5–6 🟢 |
| Girar inferência dentro do orçamento (quantização, memória) | Módulo 11 🟢 |
| Construir RAG com busca e citação | Módulo 13 🟢 |
| Avaliar a qualidade e comparar com rigor | Módulo 14 🟢 |
| Agentes e tool use com limites de segurança | Módulo 15 🟢 |
| **Colocar em produção e medir** (custo, latência, disjuntor) | **Módulo 19** 🟢 |
| **Portfólio e projeto com baseline + custo** | Módulo 12 🟢 |

### Engenheiro de Fine-tuning / Treinamento

| Habilidade | Onde |
|---|---|
| Pré-treino e o efeito de escala | Módulo 3 🟢 |
| Curadoria e governança de dados | Módulo 4 🟢 |
| SFT e chat template | Módulo 5 🟢 |
| LoRA/QLoRA, memória e esquecimento | Módulo 6 🟢 |
| Reasoning e dados com traços | Módulo 7 🟡 |
| DPO/ORPO (preferência sem RL) | Módulo 8 🟡 |
| RL com recompensa verificável (GRPO) | Módulo 9 🟡 |
| Destilação para modelos menores | Módulo 10 🟡 |

### Engenheiro de Plataforma / Serving / Infra de LLM

| Habilidade | Onde |
|---|---|
| Inferência, MoE, quantização, QAT | Módulo 11 🟢 |
| Ferramenta de produção: servir, medir, orçar | **Módulo 19** 🟢 |
| Sistemas de treino em escala (DDP/FSDP) | Módulo 17 🟡 |
| Orquestração de agentes e tool use | Módulo 15 🟡 |
| Funcionamento do cache e custos de RV | Módulos 1, 11 🟢 |

### Pesquisador / Engenheiro de research (ML Research Eng)

| Habilidade | Onde |
|---|---|
| Implementar do zero para entender | Módulos 1–3 🟢 |
| Interpretabilidade mecanicista e intervenção | Módulo 16 🟢 |
| Novas arquiteturas (SSM, MLA, híbridos) | Módulo 18 🟢 |
| Fundamentos de sistemas de treino distribuído | Módulo 17 🟢 |
| **Trilha de pesquisa** (reproduzir, contribuir, publicar) | FASE-3-MAESTRIA 🟢 |

---

## A régua da rota de mercado

Para a **trilha de engenharia**, a ordem que entrega:

```
Fase 0 → módulos 1–6 → 11 → 13–15 → 19 → projeto (12)
```

E o entendimento central para não cair no "quadrado da técnica": o profissional não é quem estudou todas as técnicas; é quem **reconhece qual técnica não precisa usar** — e consegue provar com um baseline e um custo o porquê. Ver [TRILHA-ESSENCIAL.md](TRILHA-ESSENCIAL.md).

Se você está estudando para uma vaga específica, abra a aba correspondente, marque o que o anúncio pede e siga só os módulos 🟢 até **fechar com o projeto do módulo 12**. Os 🟡/⚪ são aprofundamento — volte quando o cargo cobrar.
