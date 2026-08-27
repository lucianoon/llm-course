"""Datasets do Módulo 4, baixados sem dependências além da stdlib.

- Alpaca (52.002 exemplos de instrução, gerados por self-instruct com text-davinci-003).
  Usado para os labs de SFT. Licença CC BY-NC 4.0 + restrição dos termos da OpenAI —
  ver seção 6 do README antes de usar comercialmente.
- Corpus literário do Módulo 3, reaproveitado nos labs de pré-treino.

Uso:
    python dados.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI.parent / "tools"))

# Import depois do sys.path acima, como nos demais labs do curso.
from modulos import importar_por_caminho

DATA = AQUI / "data"
ALPACA = DATA / "alpaca.json"
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

CORPUS_M3 = AQUI.parent / "modulo-03-treino" / "data" / "corpus.txt"


def carregar_alpaca() -> list[dict]:
    if not ALPACA.exists():
        DATA.mkdir(exist_ok=True)
        req = urllib.request.Request(ALPACA_URL, headers={"User-Agent": "Mozilla/5.0"})
        bruto = urllib.request.urlopen(req, timeout=120).read()
        ALPACA.write_bytes(bruto)
        print(f"  alpaca baixado: {len(bruto):,} bytes -> {ALPACA}")
    return json.loads(ALPACA.read_text(encoding="utf-8"))


def carregar_corpus() -> str:
    """Corpus literário do Módulo 3 (baixa-o se ainda não existir)."""
    if not CORPUS_M3.exists():
        # Por caminho, não por nome: quem chega aqui via lab.py já importou o
        # `dados` DESTE módulo, e um `import dados` devolveria ele mesmo.
        dados_m3 = importar_por_caminho(CORPUS_M3.parent.parent / "dados.py", "dados_modulo_03")
        return dados_m3.carregar()
    return CORPUS_M3.read_text(encoding="utf-8")


if __name__ == "__main__":
    a = carregar_alpaca()
    print(f"alpaca : {len(a):,} exemplos | campos: {list(a[0].keys())}")
    c = carregar_corpus()
    print(f"corpus : {len(c):,} caracteres")
