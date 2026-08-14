"""Baixa e limpa o corpus de treino do Módulo 3.

Dois romances de Machado de Assis em domínio público (Projeto Gutenberg), ~800 KB de
português literário — pequeno o bastante para treinar em CPU e grande o bastante para o
modelo aprender morfologia e sintaxe de verdade.

Uso:
    python dados.py            # baixa para ./data/corpus.txt (idempotente)
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
CORPUS = DATA / "corpus.txt"

OBRAS = {
    "Dom Casmurro": "https://www.gutenberg.org/cache/epub/55752/pg55752.txt",
    "Memórias Póstumas de Brás Cubas": "https://www.gutenberg.org/cache/epub/54829/pg54829.txt",
}


def limpar_gutenberg(texto: str) -> str:
    """Remove cabeçalho e rodapé legais, mantendo só a obra."""
    inicio = re.search(r"\*\*\* ?START OF TH[EIS]+ PROJECT GUTENBERG EBOOK.*?\*\*\*", texto)
    fim = re.search(r"\*\*\* ?END OF TH[EIS]+ PROJECT GUTENBERG EBOOK.*?\*\*\*", texto)
    if inicio:
        texto = texto[inicio.end():]
    if fim:
        texto = texto[: fim.start()]
    texto = texto.replace("\r\n", "\n")
    texto = re.sub(r"\n{3,}", "\n\n", texto)          # normaliza parágrafos
    return texto.strip()


def carregar() -> str:
    """Devolve o corpus, baixando na primeira chamada."""
    if CORPUS.exists():
        return CORPUS.read_text(encoding="utf-8")

    DATA.mkdir(exist_ok=True)
    partes = []
    for titulo, url in OBRAS.items():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        bruto = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
        limpo = limpar_gutenberg(bruto)
        print(f"  {titulo}: {len(bruto):,} -> {len(limpo):,} caracteres após limpeza")
        partes.append(limpo)

    texto = "\n\n".join(partes)
    CORPUS.write_text(texto, encoding="utf-8")
    print(f"  corpus salvo em {CORPUS} ({len(texto):,} caracteres)")
    return texto


if __name__ == "__main__":
    t = carregar()
    print(f"\n{len(t):,} caracteres | {len(t.split()):,} palavras")
    print(f"\nprimeiros 300 caracteres:\n{t[:300]}")
