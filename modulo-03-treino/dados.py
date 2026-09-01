"""Baixa e limpa o corpus de treino do Módulo 3.

Dois romances de Machado de Assis em domínio público (Projeto Gutenberg), ~800 KB de
português literário — pequeno o bastante para treinar em CPU e grande o bastante para o
modelo aprender morfologia e sintaxe de verdade.

Uso:
    python dados.py            # baixa para ./data/corpus.txt (idempotente)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"
CORPUS = DATA / "corpus.txt"
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "tools"))

from dados_externos import baixar_bytes_verificados, validar_arquivo

OBRAS = {
    "Dom Casmurro": "gutenberg-dom-casmurro",
    "Memórias Póstumas de Brás Cubas": "gutenberg-memorias-postumas",
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
        validar_arquivo("machado-corpus-limpo", CORPUS)
        return CORPUS.read_text(encoding="utf-8")

    DATA.mkdir(exist_ok=True)
    partes = []
    for titulo, artefato_id in OBRAS.items():
        bruto = baixar_bytes_verificados(artefato_id).decode("utf-8")
        limpo = limpar_gutenberg(bruto)
        print(f"  {titulo}: {len(bruto):,} -> {len(limpo):,} caracteres após limpeza")
        partes.append(limpo)

    texto = "\n\n".join(partes)
    CORPUS.write_text(texto, encoding="utf-8")
    validar_arquivo("machado-corpus-limpo", CORPUS)
    print(f"  corpus salvo em {CORPUS} ({len(texto):,} caracteres)")
    return texto


if __name__ == "__main__":
    t = carregar()
    print(f"\n{len(t):,} caracteres | {len(t.split()):,} palavras")
    print(f"\nprimeiros 300 caracteres:\n{t[:300]}")
