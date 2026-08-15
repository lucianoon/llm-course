from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.build_notebooks import parse_percent
from tools.calculadora import calcular
from tools.rag import BM25, PERGUNTAS, extrair_chunks
from tools.respostas import extrair_numero


class TestBuildNotebooks(unittest.TestCase):
    def test_parseia_markdown_e_codigo(self):
        cells = parse_percent("# %% [markdown]\n# Título\n# %%\nx = 1\n")
        self.assertEqual([cell["cell_type"] for cell in cells], ["markdown", "code"])
        self.assertEqual(cells[0]["source"], ["Título"])


class TestCalculadora(unittest.TestCase):
    def test_calcula_operacoes_permitidas(self):
        self.assertEqual(calcular("(847 * 293) + 2"), "248173")
        self.assertEqual(calcular("10 / 4"), "2.5")

    def test_rejeita_codigo_e_expoentes(self):
        with self.assertRaises(ValueError):
            calcular("__import__('os').system('id')")
        with self.assertRaises(ValueError):
            calcular("10 ** 100000000")


class TestRespostas(unittest.TestCase):
    def test_prioriza_marcador_final(self):
        self.assertEqual(extrair_numero("Usei 20 itens. Resposta final: 42."), "42")

    def test_usa_ultimo_numero_como_fallback(self):
        self.assertEqual(extrair_numero("primeiro 10; depois $1,234.00"), "1234")
        self.assertIsNone(extrair_numero("sem números"))


class TestRAG(unittest.TestCase):
    def test_banco_de_perguntas_e_unico_e_estavel(self):
        enunciados = [pergunta for pergunta, _, _ in PERGUNTAS]
        self.assertEqual(len(enunciados), 25)
        self.assertEqual(len(set(enunciados)), len(enunciados))

    def test_bm25_prioriza_termo_raro(self):
        indice = BM25(["gato comum", "transformer attention qkv", "gato doméstico"])
        ordem, _ = indice.buscar("attention qkv", k=1)
        self.assertEqual(ordem, [1])

    def test_extrai_chunks_e_valida_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            modulo = Path(temp) / "modulo-01-teste"
            modulo.mkdir()
            palavras = " ".join(f"palavra{i}" for i in range(80))
            readme = modulo / "README.md"
            readme.write_text(f"# Curso\n\n## Seção\n\n{palavras}\n", encoding="utf-8")
            chunks = extrair_chunks(readme, alvo_palavras=40, overlap=10)
            self.assertGreaterEqual(len(chunks), 2)
            self.assertEqual(chunks[0]["modulo"], "modulo-01-teste")
        with self.assertRaises(ValueError):
            extrair_chunks(Path("inexistente"), alvo_palavras=10, overlap=10)


try:
    import torch

    from tools import minigpt
except ModuleNotFoundError:
    torch = None
    minigpt = None


@unittest.skipUnless(torch is not None, "PyTorch não instalado")
class TestMiniGPT(unittest.TestCase):
    def test_avaliacao_preserva_rng_e_modo(self):
        cfg = minigpt.Config(vocab=16, d=8, n_camadas=1, n_heads=2, bloco=4, d_ff=16)
        modelo = minigpt.MiniGPT(cfg)
        modelo.eval()
        fonte = torch.arange(80) % cfg.vocab
        estado = torch.random.get_rng_state().clone()
        minigpt.avaliar(modelo, fonte, n=2, batch=2, seed=7)
        self.assertTrue(torch.equal(estado, torch.random.get_rng_state()))
        self.assertFalse(modelo.training)


if __name__ == "__main__":
    unittest.main()
