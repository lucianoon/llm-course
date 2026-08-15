"""MiniGPT reutilizável — extraído do Módulo 3.

O Módulo 3 define esta arquitetura passo a passo dentro do próprio lab, que é onde ela
deve ser lida. Aqui ela vira uma peça importável para os módulos seguintes usarem em
experimentos controlados, sem reescrever tudo.

Arquitetura: pre-norm RMSNorm + atenção causal multi-head com RoPE + MLP SwiGLU +
weight tying. As mesmas peças dos módulos 2 e 3, em miniatura.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Config:
    def __init__(self, vocab=2048, d=192, n_camadas=4, n_heads=6, bloco=128, d_ff=512, theta=10_000.0):
        self.vocab, self.d, self.n_camadas = vocab, d, n_camadas
        self.n_heads, self.bloco, self.d_ff, self.theta = n_heads, bloco, d_ff, theta

    @property
    def head_dim(self):
        return self.d // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.peso = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (self.peso * (x.float() * torch.rsqrt(var + self.eps))).type_as(x)


def rope_cache(bloco, head_dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(bloco).float(), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    m = x.shape[-1] // 2
    return torch.cat([-x[..., m:], x[..., :m]], dim=-1)


class Atencao(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.saida = nn.Linear(cfg.d, cfg.d, bias=False)

    def forward(self, x, cos, sin):
        b, s, d = x.shape
        h, hd = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(b, s, h, hd).transpose(1, 2)
        k = k.view(b, s, h, hd).transpose(1, 2)
        v = v.view(b, s, h, hd).transpose(1, 2)
        cos, sin = cos[:s], sin[:s]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.saida(y.transpose(1, 2).contiguous().view(b, s, d))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.portao = nn.Linear(cfg.d, cfg.d_ff, bias=False)
        self.acima = nn.Linear(cfg.d, cfg.d_ff, bias=False)
        self.abaixo = nn.Linear(cfg.d_ff, cfg.d, bias=False)

    def forward(self, x):
        return self.abaixo(F.silu(self.portao(x)) * self.acima(x))


class Bloco(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.d), RMSNorm(cfg.d)
        self.attn, self.mlp = Atencao(cfg), MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocos = nn.ModuleList(Bloco(cfg) for _ in range(cfg.n_camadas))
        self.norm_final = RMSNorm(cfg.d)
        self.cabeca = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self.cabeca.weight = self.emb.weight
        cos, sin = rope_cache(cfg.bloco, cfg.head_dim, cfg.theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, alvos=None):
        x = self.emb(idx)
        for bloco in self.blocos:
            x = bloco(x, self.cos, self.sin)
        logits = self.cabeca(self.norm_final(x))
        if alvos is None:
            return logits, None
        perda = F.cross_entropy(logits.view(-1, logits.size(-1)), alvos.reshape(-1))
        return logits, perda

    @torch.no_grad()
    def gerar(self, idx, n_novos, temperatura=0.8, top_k=40):
        for _ in range(n_novos):
            logits, _ = self(idx[:, -self.cfg.bloco:])
            logits = logits[:, -1, :] / temperatura
            if top_k:
                corte = torch.topk(logits, top_k).values[:, -1:]
                logits = logits.masked_fill(logits < corte, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx


# ---------------------------------------------------------------- treino

def pegar_batch(fonte, batch_size, bloco, generator=None):
    """Amostra um batch; ``generator`` permite isolar treino e avaliação."""
    if len(fonte) <= bloco + 1:
        raise ValueError(f"fonte precisa ter mais de {bloco + 1} tokens")
    inicio = torch.randint(len(fonte) - bloco - 1, (batch_size,), generator=generator)
    x = torch.stack([fonte[i: i + bloco] for i in inicio])
    y = torch.stack([fonte[i + 1: i + 1 + bloco] for i in inicio])
    return x, y


def agenda_lr(passo, total, pico=1e-3, warmup=None, minimo_frac=0.1):
    warmup = warmup if warmup is not None else max(1, total // 20)
    if passo < warmup:
        return pico * (passo + 1) / warmup
    progresso = (passo - warmup) / max(1, total - warmup)
    return pico * (minimo_frac + (1 - minimo_frac) * 0.5 * (1 + math.cos(math.pi * progresso)))


def grupos_de_parametros(modelo, weight_decay=0.1):
    return [
        {"params": [p for p in modelo.parameters() if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in modelo.parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]


@torch.no_grad()
def avaliar(modelo, fonte, n=20, batch=8, seed=0):
    """Avalia em batches determinísticos sem consumir o RNG do treino."""
    estava_treinando = modelo.training
    generator = torch.Generator().manual_seed(seed)
    modelo.eval()
    perdas = [
        modelo(*pegar_batch(fonte, batch, modelo.cfg.bloco, generator=generator))[1].item()
        for _ in range(n)
    ]
    modelo.train(estava_treinando)
    return sum(perdas) / len(perdas)


def treinar(cfg, dados_treino, dados_val=None, passos=300, batch=16, lr=1e-3, seed=1337, verboso=False):
    """Treina um MiniGPT do zero. Devolve (modelo, historico)."""
    torch.manual_seed(seed)
    modelo = MiniGPT(cfg)
    otim = torch.optim.AdamW(grupos_de_parametros(modelo), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    historico = {"passo": [], "treino": [], "val": []}

    for passo in range(passos):
        taxa = agenda_lr(passo, passos, pico=lr)
        for g in otim.param_groups:
            g["lr"] = taxa

        otim.zero_grad(set_to_none=True)
        x, y = pegar_batch(dados_treino, batch, cfg.bloco)
        _, perda = modelo(x, y)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        otim.step()

        if passo % 50 == 0 or passo == passos - 1:
            historico["passo"].append(passo)
            historico["treino"].append(perda.item())
            historico["val"].append(avaliar(modelo, dados_val) if dados_val is not None else None)
            if verboso:
                print(f"    passo {passo:>4} treino {perda.item():.4f} val {historico['val'][-1]}")

    return modelo, historico
