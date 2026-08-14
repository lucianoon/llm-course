"""Infraestrutura reutilizável de recuperação do módulo 13.

Ao contrário de ``modulo-13-rag/lab_cpu.py``, importar este módulo não executa
avaliações nem carrega um modelo gerador. O encoder E5 só é carregado quando
``IndiceRAG`` é construído.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

PERGUNTAS = [
    ("Quantos bytes por parâmetro custa o full fine-tune com AdamW?", "16 bytes", {"modulo-01-fundamentos", "modulo-03-treino", "modulo-06-lora"}),
    ("Que fração dos parâmetros de um bloco transformer está no MLP?", "87,7%", {"modulo-02-attention", "modulo-06-lora", "modulo-11-inferencia"}),
    ("Qual é a loss inicial esperada de um treino de DPO?", "ln 2", {"modulo-08-dpo"}),
    ("Quantos KV heads tem o Qwen2.5-0.5B?", "2", {"modulo-02-attention"}),
    ("Qual algoritmo de RL treinou o DeepSeek-R1?", "GRPO", {"modulo-09-rl", "modulo-07-reasoning"}),
    ("O que significa NF4?", "4-bit NormalFloat", {"modulo-06-lora"}),
    ("Qual a fórmula de FLOPs de treino de um LLM?", "6ND", {"modulo-03-treino"}),
    ("Quantos tokens de treino por parâmetro recomenda o Chinchilla?", "20", {"modulo-01-fundamentos", "modulo-03-treino"}),
    ("Qual flag do mlx_lm.lora calcula a loss só na resposta?", "--mask-prompt", {"modulo-05-sft"}),
    ("Qual o valor de beta2 do AdamW usado em LLMs?", "0,95", {"modulo-03-treino"}),
    ("Que técnica reduz o KV cache compartilhando keys e values entre query heads?", "GQA", {"modulo-01-fundamentos", "modulo-02-attention"}),
    ("Por que a matriz B do LoRA é inicializada em zeros?", "identidade no passo 0", {"modulo-06-lora"}),
    ("Qual foi a degradação da quantização 4-bit em literatura portuguesa?", "17,4%", {"modulo-06-lora"}),
    ("O que é o attention sink?", "atenção concentrada no primeiro token", {"modulo-02-attention", "modulo-11-inferencia"}),
    ("Quantos exemplos curados usou o LIMA?", "1.000", {"modulo-04-dados", "modulo-05-sft"}),
    ("Qual estimador de KL o GRPO usa?", "k3", {"modulo-09-rl"}),
    ("Que loss auxiliar previne o colapso de roteamento em MoE?", "balanceamento do Switch", {"modulo-11-inferencia"}),
    ("Qual a taxa de aceitação medida na decodificação especulativa do curso?", "59%", {"modulo-11-inferencia"}),
    ("O que o masking com -100 faz no SFT?", "ignora os tokens do prompt na loss", {"modulo-04-dados", "modulo-05-sft"}),
    ("Qual dataset de matemática com gabarito verificável o curso usa?", "GSM8K", {"modulo-07-reasoning", "modulo-09-rl"}),
    ("Por que bf16 dispensa loss scaling?", "mesmo expoente do fp32", {"modulo-03-treino"}),
    ("Qual método detecta contaminação de benchmark?", "sobreposição de 13-gramas", {"modulo-04-dados"}),
    ("O que é catastrophic forgetting?", "perda de capacidades gerais", {"modulo-05-sft", "modulo-06-lora"}),
    ("Qual arquitetura substitui o MLP por especialistas roteados?", "MoE", {"modulo-11-inferencia"}),
    ("O que o rejection sampling filtra no pipeline de distillation?", "traços com resposta errada", {"modulo-10-distillation", "modulo-07-reasoning"}),
]

FORA_DA_BASE = [
    "Qual a capital da Austrália?",
    "Como fazer um bolo de cenoura?",
    "Quem ganhou a Copa do Mundo de 2022?",
    "Qual o melhor framework de frontend em 2026?",
    "Como investir em renda fixa?",
]


def extrair_chunks(md_path: Path, alvo_palavras: int = 220, overlap: int = 40) -> list[dict]:
    """Corta um README por seções e subdivide seções longas com sobreposição."""
    if not 0 <= overlap < alvo_palavras:
        raise ValueError("overlap deve ser menor que alvo_palavras")
    texto = md_path.read_text(encoding="utf-8")
    modulo = md_path.parent.name
    chunks: list[dict] = []
    titulo_atual = modulo
    for parte in re.split(r"(?m)^(#{1,2} .+)$", texto):
        if re.match(r"^#{1,2} ", parte or ""):
            titulo_atual = parte.lstrip("# ").strip()
            continue
        palavras = (parte or "").split()
        if len(palavras) < 30:
            continue
        for i in range(0, len(palavras), alvo_palavras - overlap):
            pedaco = " ".join(palavras[i : i + alvo_palavras])
            if len(pedaco.split()) >= 30:
                chunks.append({"modulo": modulo, "titulo": titulo_atual, "texto": pedaco})
            if i + alvo_palavras >= len(palavras):
                break
    return chunks


def carregar_chunks(raiz: Path) -> list[dict]:
    chunks: list[dict] = []
    for md_path in sorted(raiz.glob("modulo-*/README.md")):
        chunks.extend(extrair_chunks(md_path))
    if not chunks:
        raise ValueError(f"nenhum README de módulo encontrado em {raiz}")
    return chunks


def tokenizar_busca(texto: str) -> list[str]:
    return re.findall(r"[a-záàâãéêíóôõúüç0-9@#\-]+", texto.lower())


class BM25:
    """Okapi BM25 sem dependências externas."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise ValueError("BM25 requer ao menos um documento")
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenizar_busca(doc) for doc in docs]
        self.n = len(docs)
        self.tamanho_medio = sum(map(len, self.docs_tokens)) / self.n
        df = Counter(token for tokens in self.docs_tokens for token in set(tokens))
        self.idf = {
            token: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1)
            for token, freq in df.items()
        }
        self.tf = [Counter(tokens) for tokens in self.docs_tokens]

    def buscar(self, consulta: str, k: int = 10) -> tuple[list[int], list[float]]:
        scores: list[float] = []
        for i, tokens in enumerate(self.docs_tokens):
            tamanho = len(tokens)
            score = 0.0
            for termo in tokenizar_busca(consulta):
                frequencia = self.tf[i].get(termo, 0)
                if not frequencia:
                    continue
                score += self.idf.get(termo, 0.0) * frequencia * (self.k1 + 1) / (
                    frequencia
                    + self.k1 * (1 - self.b + self.b * tamanho / self.tamanho_medio)
                )
            scores.append(score)
        ordem = sorted(range(self.n), key=lambda indice: -scores[indice])[:k]
        return ordem, [scores[indice] for indice in ordem]


class IndiceRAG:
    """Índice híbrido BM25 + E5, sem carregar um LLM gerador."""

    def __init__(self, raiz: Path, modelo_embeddings: str = "intfloat/multilingual-e5-small"):
        import torch
        from torch.nn import functional
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._functional = functional
        self.chunks = carregar_chunks(raiz)
        self.bm25 = BM25([chunk["texto"] for chunk in self.chunks])
        self.tokenizer = AutoTokenizer.from_pretrained(modelo_embeddings)
        self.encoder = AutoModel.from_pretrained(modelo_embeddings)
        self.encoder.eval()
        self.embeddings = self._embed(
            [f"passage: {chunk['texto']}" for chunk in self.chunks]
        )

    def _embed(self, textos: list[str], batch: int = 16):
        saidas = []
        with self._torch.no_grad():
            for i in range(0, len(textos), batch):
                encoded = self.tokenizer(
                    textos[i : i + batch],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                hidden = self.encoder(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                saidas.append(self._functional.normalize(pooled, dim=-1))
        return self._torch.cat(saidas)

    def buscar_densa(self, consulta: str, k: int = 10) -> tuple[list[int], list[float]]:
        query = self._embed([f"query: {consulta}"])
        similaridades = (query @ self.embeddings.T)[0]
        ordem = similaridades.argsort(descending=True)[:k]
        return ordem.tolist(), similaridades[ordem].tolist()

    def buscar_hibrida(self, consulta: str, k: int = 10, c: int = 60):
        ordem_bm25, _ = self.bm25.buscar(consulta, k=30)
        ordem_densa, _ = self.buscar_densa(consulta, k=30)
        pontos: Counter = Counter()
        for rank, indice in enumerate(ordem_bm25):
            pontos[indice] += 1 / (c + rank)
        for rank, indice in enumerate(ordem_densa):
            pontos[indice] += 1 / (c + rank)
        topo = [indice for indice, _ in pontos.most_common(k)]
        return topo, [pontos[indice] for indice in topo]
