# %% [markdown]
# # Módulo 7 — Laboratório C: process supervision vs outcome supervision
#
# **Roda em CPU, ~1 minuto.** A resposta final pode estar certa pelo caminho errado.
# Outcome supervision (ORM) enxerga só o final; process supervision rotula cada passo.
# Este lab constrói o formato de dados, um verificador por etapa e a comparação que
# mostra exatamente quando os dois sinais discordam.

# %%
from __future__ import annotations

import random
from dataclasses import dataclass

random.seed(7)


@dataclass(frozen=True)
class Passo:
    texto: str
    valor_declarado: int
    valor_esperado: int

    @property
    def correto(self) -> bool:
        return self.valor_declarado == self.valor_esperado


@dataclass(frozen=True)
class Traco:
    pergunta: str
    resposta_esperada: int
    passos: tuple[Passo, ...]

    @property
    def resposta_final(self) -> int:
        return self.passos[-1].valor_declarado


def construir_traco(a: int, b: int, c: int, tipo: str) -> Traco:
    """Cria caminho correto, resposta sortuda ou resposta final errada."""
    produto = b * c
    final = a + produto
    if tipo == "correto":
        declarado_produto, declarado_final = produto, final
    elif tipo == "sortudo":
        declarado_produto, declarado_final = produto + 1, final
    elif tipo == "errado":
        declarado_produto, declarado_final = produto + 1, final + 1
    else:
        raise ValueError(f"tipo desconhecido: {tipo}")
    return Traco(
        pergunta=f"Quanto é {a} + {b} × {c}?",
        resposta_esperada=final,
        passos=(
            Passo(f"{b} × {c} = {declarado_produto}", declarado_produto, produto),
            Passo(
                f"{a} + {declarado_produto} = {declarado_final}",
                declarado_final,
                a + declarado_produto,
            ),
        ),
    )


def recompensa_outcome(traco: Traco) -> float:
    return float(traco.resposta_final == traco.resposta_esperada)


def recompensa_processo(traco: Traco) -> float:
    return sum(passo.correto for passo in traco.passos) / len(traco.passos)


# %% [markdown]
# ## Lab 1 — O caso que o gabarito final não detecta

# %%
a, b, c = 11, 7, 8
for tipo in ("correto", "sortudo", "errado"):
    traco = construir_traco(a, b, c, tipo)
    print(f"\n{tipo.upper()}: {traco.pergunta}")
    for passo in traco.passos:
        print(f"  {'✓' if passo.correto else '✗'} {passo.texto}")
    print(f"  reward outcome={recompensa_outcome(traco):.1f} "
          f"| processo={recompensa_processo(traco):.1f}")

assert recompensa_outcome(construir_traco(a, b, c, "correto")) == 1
assert recompensa_outcome(construir_traco(a, b, c, "sortudo")) == 1
assert recompensa_processo(construir_traco(a, b, c, "sortudo")) < 1

# %% [markdown]
# O ORM empata CORRETO e SORTUDO. O sinal por processo desfaz o empate — mas só porque
# existe um verificador confiável por passo. Em domínio aberto, criar esses rótulos é a
# parte cara: anotadores, um Process Reward Model (PRM) ou verificadores formais.
#
# ## Lab 2 — O dataset de processo
#
# Um dataset de PRM guarda o prefixo até cada passo e o rótulo daquele passo. Não se
# deve dar ao avaliador acesso a passos futuros: isso vazaria a resposta.

# %%
dataset_prm = []
for _ in range(100):
    a, b, c = (random.randint(2, 20) for _ in range(3))
    for tipo in ("correto", "sortudo", "errado"):
        traco = construir_traco(a, b, c, tipo)
        prefixo = traco.pergunta
        for indice, passo in enumerate(traco.passos):
            prefixo += f"\nPasso {indice + 1}: {passo.texto}"
            dataset_prm.append(
                {
                    "prefixo": prefixo,
                    "passo": indice + 1,
                    "label": int(passo.correto),
                    "resposta_final_correta": int(
                        traco.resposta_final == traco.resposta_esperada
                    ),
                }
            )

positivos = sum(item["label"] for item in dataset_prm)
sortudos = sum(
    item["resposta_final_correta"] == 1 and item["label"] == 0 for item in dataset_prm
)
print(f"{len(dataset_prm)} prefixos rotulados | {positivos/len(dataset_prm):.0%} positivos")
print(f"{sortudos} passos ruins escondidos por uma resposta final correta")

# %% [markdown]
# ## Lab 3 — Regras de integridade antes de treinar um PRM

# %%
assert all(item["passo"] in (1, 2) for item in dataset_prm)
assert all("Passo 2:" not in item["prefixo"] for item in dataset_prm if item["passo"] == 1)
assert 0 < positivos < len(dataset_prm), "dataset sem contraste não ensina um PRM"

print("checks: sem passos futuros, classes não degeneradas, rótulos por prefixo ✓")

# %% [markdown]
# **Limite:** o verificador deste lab é exato porque a tarefa é aritmética. Um PRM
# aprendido pode premiar passos convincentes e errados. Em produção, audite-o como um
# LLM-as-judge (módulo 14): gabarito, inversão, calibração e avaliação fora do domínio.
