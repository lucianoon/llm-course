# %% [markdown]
# # Módulo 10 — Laboratório C: feature, multi-teacher e progressive distillation
#
# **Roda em CPU, ~2 minutos.** O lab principal destila logits e texto. Aqui comparamos
# três extensões: alinhar representações internas, combinar professores e comprimir em
# dois estágios.

# %%
import copy

import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(10)


def criar_dados(n=3000):
    x = torch.randn(n, 8)
    regras = torch.stack(
        [x[:, 0] + x[:, 1] * x[:, 2], x[:, 3] - x[:, 4] ** 2, x[:, 5] * x[:, 6] + x[:, 7]],
        dim=1,
    )
    return x, regras.argmax(1)


class Rede(nn.Module):
    def __init__(self, oculto):
        super().__init__()
        self.projecao = nn.Linear(8, oculto)
        self.saida = nn.Linear(oculto, 3)

    def forward(self, x, devolver_feature=False):
        feature = torch.tanh(self.projecao(x))
        logits = self.saida(feature)
        return (logits, feature) if devolver_feature else logits


def treinar_supervisionado(modelo, x, y, passos=300):
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-3)
    for _ in range(passos):
        loss = F.cross_entropy(modelo(x), y)
        otimizador.zero_grad()
        loss.backward()
        otimizador.step()
    return modelo


def treinar_kd(aluno, professores, x, y, passos=250, feature=False):
    projecoes = nn.ModuleList()
    if feature:
        dimensao_aluno = aluno.projecao.out_features
        projecoes = nn.ModuleList(
            nn.Linear(dimensao_aluno, professor.projecao.out_features)
            for professor in professores
        )
    parametros = list(aluno.parameters()) + list(projecoes.parameters())
    otimizador = torch.optim.AdamW(parametros, lr=2e-3)
    temperatura = 2.0
    for _ in range(passos):
        logits_a, feature_a = aluno(x, devolver_feature=True)
        with torch.no_grad():
            saidas_p = [professor(x, devolver_feature=True) for professor in professores]
            logits_p = torch.stack([saida[0] for saida in saidas_p]).mean(0)
        loss_logits = F.kl_div(
            F.log_softmax(logits_a / temperatura, dim=-1),
            F.softmax(logits_p / temperatura, dim=-1),
            reduction="batchmean",
        ) * temperatura**2
        loss = 0.25 * F.cross_entropy(logits_a, y) + 0.75 * loss_logits
        if feature:
            loss_features = torch.stack(
                [F.mse_loss(projecao(feature_a), saida[1])
                 for projecao, saida in zip(projecoes, saidas_p)]
            ).mean()
            loss = loss + 0.2 * loss_features
        otimizador.zero_grad()
        loss.backward()
        otimizador.step()
    return aluno


def acuracia(modelo, x, y):
    with torch.no_grad():
        return float((modelo(x).argmax(-1) == y).float().mean())


x_treino, y_treino = criar_dados()
x_teste, y_teste = criar_dados(1500)

professor_a = treinar_supervisionado(Rede(48), x_treino, y_treino)
torch.manual_seed(11)
professor_b = treinar_supervisionado(Rede(48), x_treino, y_treino)

# %% [markdown]
# ## Quatro alunos com a mesma capacidade

# %%
base = Rede(6)
hard = treinar_supervisionado(copy.deepcopy(base), x_treino, y_treino, passos=250)
logit = treinar_kd(copy.deepcopy(base), [professor_a], x_treino, y_treino)
feature = treinar_kd(copy.deepcopy(base), [professor_a], x_treino, y_treino, feature=True)
multi = treinar_kd(copy.deepcopy(base), [professor_a, professor_b], x_treino, y_treino)

# Progressive: professor 48 → assistente 16 → aluno 6.
assistente = treinar_kd(Rede(16), [professor_a], x_treino, y_treino)
progressivo = treinar_kd(copy.deepcopy(base), [assistente], x_treino, y_treino)

modelos = {
    "teacher A": professor_a,
    "hard labels": hard,
    "logit KD": logit,
    "feature KD": feature,
    "multi-teacher": multi,
    "progressive": progressivo,
}
for nome, modelo in modelos.items():
    print(f"{nome:<16} {acuracia(modelo, x_teste, y_teste):.1%}")

# %% [markdown]
# Não existe vencedor universal. Feature KD exige escolher camadas e projeções
# compatíveis; multi-teacher pode reduzir variância ou misturar erros; progressive KD
# ajuda quando o salto de capacidade é grande, mas paga um treino intermediário. O
# protocolo correto mantém aluno, dados, passos e métrica fixos e varia uma decisão.
