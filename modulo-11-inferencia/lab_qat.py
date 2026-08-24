# %% [markdown]
# # Módulo 11 — Laboratório C: PTQ vs QAT do zero
#
# **Roda em CPU, ~1 minuto.** PTQ quantiza depois do treino. QAT insere fake quantization
# durante o treino para que os pesos aprendam a sobreviver à grade discreta.

# %%
import torch
from torch import nn

torch.manual_seed(11)


def quantizar_simetrico(x: torch.Tensor, bits=4, eixo=None) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    if eixo is None:
        escala = x.detach().abs().max().clamp_min(1e-8) / qmax
    else:
        escala = x.detach().abs().amax(dim=eixo, keepdim=True).clamp_min(1e-8) / qmax
    return (x / escala).round().clamp(-qmax, qmax) * escala


class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, eixo):
        return quantizar_simetrico(x, bits=bits, eixo=eixo)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


class Classificador(nn.Module):
    def __init__(self, qat=False):
        super().__init__()
        self.qat = qat
        self.camadas = nn.ModuleList([nn.Linear(2, 32), nn.Linear(32, 2)])

    def forward(self, x):
        w0, w1 = (camada.weight for camada in self.camadas)
        if self.qat:
            w0 = FakeQuantSTE.apply(w0, 4, 1)  # escala por canal de saída
            w1 = FakeQuantSTE.apply(w1, 4, 1)
        x = torch.relu(nn.functional.linear(x, w0, self.camadas[0].bias))
        return nn.functional.linear(x, w1, self.camadas[1].bias)


def dados(n):
    x = torch.randn(n, 2)
    y = ((x[:, 0] * x[:, 1] + 0.25 * x[:, 0]) > 0).long()
    return x, y


def treinar(modelo, x, y, passos=300, lr=3e-3):
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr)
    for _ in range(passos):
        loss = nn.functional.cross_entropy(modelo(x), y)
        otimizador.zero_grad()
        loss.backward()
        otimizador.step()
    return modelo


def acuracia(modelo, x, y):
    with torch.no_grad():
        return float((modelo(x).argmax(-1) == y).float().mean())


x_treino, y_treino = dados(3000)
x_teste, y_teste = dados(3000)

# %% [markdown]
# ## Lab 1 — Modelo float e PTQ

# %%
float_model = treinar(Classificador(), x_treino, y_treino)
acc_float = acuracia(float_model, x_teste, y_teste)

ptq_tensor = Classificador()
ptq_tensor.load_state_dict(float_model.state_dict())
with torch.no_grad():
    for camada in ptq_tensor.camadas:
        camada.weight.copy_(quantizar_simetrico(camada.weight, bits=4))

ptq_canal = Classificador()
ptq_canal.load_state_dict(float_model.state_dict())
with torch.no_grad():
    for camada in ptq_canal.camadas:
        camada.weight.copy_(quantizar_simetrico(camada.weight, bits=4, eixo=1))

# %% [markdown]
# ## Lab 2 — QAT com straight-through estimator

# %%
qat_model = Classificador(qat=True)
qat_model.load_state_dict(float_model.state_dict())
qat_model = treinar(qat_model, x_treino, y_treino, passos=150, lr=5e-4)

resultados = {
    "float": acc_float,
    "PTQ por tensor": acuracia(ptq_tensor, x_teste, y_teste),
    "PTQ por canal": acuracia(ptq_canal, x_teste, y_teste),
    "QAT por canal": acuracia(qat_model, x_teste, y_teste),
}
for nome, valor in resultados.items():
    print(f"{nome:<18} {valor:.1%}")

assert resultados["QAT por canal"] > 0.8

# %% [markdown]
# **Como ler:** por canal dá uma escala a cada neurônio de saída; por tensor força toda
# a matriz a compartilhar uma escala. QAT não garante superar PTQ em todo problema — ele
# compra uma chance de adaptar os pesos à grade. Em LLMs, o custo é retreinar e o risco
# é otimizar para uma implementação de kernel diferente daquela usada no serving.
