from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.adapters import (
    ManifestoAdapter,
    carregar_manifesto,
    salvar_manifesto,
    verificar_compatibilidade,
)
from tools.build_notebooks import ROOT, descobrir_labs, notebook_bootstrap, parse_percent
from tools.calculadora import calcular
from tools.dados_externos import validar_bytes
from tools.execucao import executar_modulo
from tools.experimentos import RegistroExperimento
from tools.governanca import anonimizar_texto, auditar_pii, criar_manifesto_dataset
from tools.jsonl import preparar_jsonl_retomavel
from tools.modulos import importar_por_caminho
from tools.producao import Disjuntor, LinhaDeTrafego, calcular_custo, contar_tokens, resumir_trafego
from tools.rag import (
    BM25,
    GABARITO_PASSAGENS,
    PERGUNTAS,
    Chunk,
    carregar_chunks,
    extrair_chunks,
    passagem_relevante,
)
from tools.reproducao import registrar_reproducao
from tools.respostas import extrair_numero
from tools.serving import AmostraServing, percentil, resumir_carga
from tools.smoke_labs import LABS_OFFLINE, LABS_OFFLINE_LONGOS, executar_lab
from tools.validar_dados_externos import validar_downloads_diretos
from tools.validar_dados_externos import validar_repositorio as validar_dados_repositorio
from tools.validar_execucao_labs import validar_manifesto as validar_execucao_labs
from tools.validar_resultados import validar_repositorio as validar_resultados_repositorio
from tools.validar_resultados import validar_resultado
from tools.validar_revisoes import carregar_manifesto as carregar_manifesto_modelos
from tools.validar_revisoes import validar_arquivo, validar_repositorio

# Os labs imprimem acentos e setas (→). No Windows o filho herda um stdout em
# cp1252 e falha ao *emitir* esses caracteres, então UTF-8 é forçado nas duas
# pontas: PYTHONIOENCODING no filho e encoding= na decodificação aqui.
_ENV_UTF8 = {**os.environ, "PYTHONIOENCODING": "utf-8"}
_PROJETO_REFERENCIA = ROOT / "modulo-12-projeto" / "projeto-referencia-rag"


class TestDocumentacao(unittest.TestCase):
    def test_links_markdown_locais_apontam_para_artefatos_existentes(self):
        links_quebrados = []

        for documento in ROOT.rglob("*.md"):
            relativo = documento.relative_to(ROOT)
            if any(parte.startswith(".") for parte in relativo.parts):
                continue

            texto = documento.read_text(encoding="utf-8")
            for destino in re.findall(r"!?\[[^]]*\]\(([^)\s]+)", texto):
                if destino.startswith(("http://", "https://", "mailto:", "#")):
                    continue

                caminho = destino.split("#", maxsplit=1)[0]
                if caminho and not (documento.parent / caminho).exists():
                    links_quebrados.append(f"{relativo}: {destino}")

        self.assertEqual(links_quebrados, [], "Links locais quebrados:\n" + "\n".join(links_quebrados))

    def test_total_anunciado_de_modulos_corresponde_as_pastas(self):
        modulos = [pasta for pasta in ROOT.glob("modulo-[0-9][0-9]-*") if pasta.is_dir()]
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn(f"Fase 0 + {len(modulos)} módulos", readme)


class TestRevisoesImutaveis(unittest.TestCase):
    def test_repositorio_nao_tem_download_de_modelo_sem_revision(self):
        self.assertEqual(validar_repositorio(), [])

    def test_detector_rejeita_from_pretrained_sem_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "lab.py"
            caminho.write_text(
                'AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")\n',
                encoding="utf-8",
            )
            violacoes = validar_arquivo(caminho, carregar_manifesto_modelos())
        self.assertEqual(len(violacoes), 1)
        self.assertIn("sem revision", violacoes[0].mensagem)

    def test_detector_rejeita_modelo_literal_fora_do_manifesto(self):
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "lab.py"
            caminho.write_text(
                'AutoModel.from_pretrained("org/novo-modelo", revision="' + "a" * 40 + '")\n',
                encoding="utf-8",
            )
            violacoes = validar_arquivo(caminho, carregar_manifesto_modelos())
        self.assertEqual(len(violacoes), 1)
        self.assertIn("ausente de MODELOS.json", violacoes[0].mensagem)


class TestDadosExternos(unittest.TestCase):
    def test_manifesto_e_downloads_do_repositorio_sao_auditaveis(self):
        self.assertEqual(validar_dados_repositorio(), [])

    def test_checksum_rejeita_conteudo_alterado(self):
        with self.assertRaisesRegex(ValueError, "checksum divergente"):
            validar_bytes("stanford-alpaca", b"conteudo alterado")

    def test_detector_rejeita_urlopen_fora_do_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "dados.py"
            caminho.write_text(
                'urllib.request.urlopen("https://example.org/dados.json")\n',
                encoding="utf-8",
            )
            violacoes = validar_downloads_diretos(Path(tmp))
        self.assertEqual(len(violacoes), 1)
        self.assertIn("urlopen direto", violacoes[0].mensagem)


class TestSchemaResultados(unittest.TestCase):
    def test_resultados_preservados_seguem_schema(self):
        self.assertEqual(validar_resultados_repositorio(), [])

    def test_detector_rejeita_resultado_incompleto(self):
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "resultado.json"
            caminho.write_text('{"experimento":"modulo-01/lab"}\n', encoding="utf-8")
            violacoes = validar_resultado(caminho)
        self.assertTrue(any("campo ausente" in item.mensagem for item in violacoes))


class TestSmokeLabs(unittest.TestCase):
    def test_todos_os_labs_tem_perfil_de_execucao(self):
        self.assertEqual(validar_execucao_labs(), [])

    def test_catalogos_sao_disjuntos_e_apontam_para_labs_existentes(self):
        self.assertTrue(set(LABS_OFFLINE).isdisjoint(LABS_OFFLINE_LONGOS))
        for caminho in LABS_OFFLINE + LABS_OFFLINE_LONGOS:
            self.assertTrue((ROOT / caminho).is_file(), caminho)

    def test_runner_distingue_sucesso_falha_e_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            (raiz / "sucesso.py").write_text('print("ok")\n', encoding="utf-8")
            (raiz / "falha.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
            (raiz / "lento.py").write_text(
                "import time\ntime.sleep(1)\n",
                encoding="utf-8",
            )

            sucesso = executar_lab("sucesso.py", raiz=raiz, timeout_s=1)
            falha = executar_lab("falha.py", raiz=raiz, timeout_s=1)
            timeout = executar_lab("lento.py", raiz=raiz, timeout_s=0.01)

        self.assertEqual(sucesso.status, "passou")
        self.assertEqual(falha.status, "falhou")
        self.assertEqual(timeout.status, "timeout")


class TestProjetoReferencia(unittest.TestCase):
    def executar(self, script: str, *argumentos: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(_PROJETO_REFERENCIA / "scripts" / script), *argumentos],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=_ENV_UTF8,
            check=False,
        )

    def test_pipeline_regenera_manifestos_e_supera_baseline(self):
        for script in ("preparar.py", "treinar.py", "avaliar.py"):
            resultado = self.executar(script)
            self.assertEqual(resultado.returncode, 0, resultado.stderr)

        manifesto = json.loads((_PROJETO_REFERENCIA / "dataset-manifest.json").read_text())
        resultados = json.loads((_PROJETO_REFERENCIA / "resultados.json").read_text())
        self.assertTrue(all(len(arquivo["sha256"]) == 64 for arquivo in manifesto["arquivos"]))
        self.assertEqual(resultados["metricas"]["modelo_final"], 1.0)
        self.assertGreater(
            resultados["metricas"]["modelo_final"], resultados["metricas"]["baseline"]
        )
        self.assertGreater(resultados["metricas"]["delta_ic95_bootstrap_pareado"][0], 0)

    def test_serving_cita_e_se_abstem_sem_evidencia(self):
        respondida = self.executar("servir.py", "Qual o prazo para pedir férias?")
        abstencao = self.executar("servir.py", "A empresa oferece vale-estacionamento?")
        self.assertEqual(respondida.returncode, 0, respondida.stderr)
        self.assertEqual(abstencao.returncode, 0, abstencao.stderr)

        resposta = json.loads(respondida.stdout)
        sem_evidencia = json.loads(abstencao.stdout)
        self.assertEqual(resposta["citacoes"], ["POL-FERIAS"])
        self.assertEqual(sem_evidencia["status"], "abstencao")
        self.assertEqual(sem_evidencia["citacoes"], [])

    def test_servidor_http_suporta_carga_e_exige_autenticacao(self):
        resultado = self.executar("carga.py")
        if resultado.returncode != 0 and "Operation not permitted" in resultado.stderr:
            self.skipTest("ambiente não permite abrir socket de loopback")
        self.assertEqual(resultado.returncode, 0, resultado.stderr)
        carga = json.loads((_PROJETO_REFERENCIA / "carga-resultados.json").read_text())
        self.assertEqual(carga["metricas"]["probe_sem_api_key_status"], 401)
        self.assertEqual(carga["metricas"]["requisicoes"], 45)
        self.assertEqual(carga["metricas"]["sucesso"], 1.0)
        self.assertGreater(carga["metricas"]["requisicoes_s"], 0)



class TestBuildNotebooks(unittest.TestCase):
    def test_descobre_fase_zero_e_modulos(self):
        relativos = {caminho.relative_to(ROOT).as_posix() for caminho in descobrir_labs()}
        self.assertIn("00-iniciante-zero/lab.py", relativos)
        self.assertIn("modulo-01-fundamentos/lab.py", relativos)

    def test_parseia_markdown_e_codigo(self):
        cells = parse_percent("# %% [markdown]\n# Título\n# %%\nx = 1\n")
        self.assertEqual([cell["cell_type"] for cell in cells], ["markdown", "code"])
        self.assertEqual(cells[0]["source"], ["Título"])

    def test_bootstrap_encontra_lab_com_kernel_na_raiz_ou_no_modulo(self):
        with tempfile.TemporaryDirectory() as temp:
            raiz = Path(temp)
            modulo = raiz / "modulo-01-teste"
            modulo.mkdir()
            (raiz / "pyproject.toml").touch()
            fonte = "".join(notebook_bootstrap(Path("modulo-01-teste/lab.py")))

            for cwd in (raiz, modulo):
                namespace: dict = {}
                with patch("pathlib.Path.cwd", return_value=cwd):
                    exec(fonte, namespace)  # noqa: S102 - valida o prólogo Python gerado
                self.assertEqual(Path(namespace["__file__"]).resolve(), (modulo / "lab.py").resolve())

    def test_lab_da_fase_zero_executa_por_inteiro(self):
        resultado = subprocess.run(
            [sys.executable, "00-iniciante-zero/lab.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=_ENV_UTF8,
            check=False,
        )
        self.assertEqual(resultado.returncode, 0, resultado.stderr)
        self.assertIn("acurácia: 2/3", resultado.stdout)


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
        self.assertEqual(set(enunciados), set(GABARITO_PASSAGENS))

    def test_toda_pergunta_tem_ao_menos_uma_passagem_relevante(self):
        chunks = carregar_chunks(ROOT, ate_modulo=12)
        for pergunta, _, _ in PERGUNTAS:
            relevantes = [
                chunk
                for chunk in chunks
                if passagem_relevante(pergunta, chunk.modulo, chunk.texto)
            ]
            self.assertTrue(relevantes, f"sem passagem relevante para: {pergunta}")

    def test_modulo_correto_sem_evidencia_nao_conta_como_acerto(self):
        pergunta, _, modulos = PERGUNTAS[0]
        self.assertIn("modulo-03-treino", modulos)
        self.assertFalse(
            passagem_relevante(pergunta, "modulo-03-treino", "Este trecho fala de warmup."),
        )

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
            self.assertIsInstance(chunks[0], Chunk)
            self.assertEqual(chunks[0].modulo, "modulo-01-teste")
        with self.assertRaises(ValueError):
            extrair_chunks(Path("inexistente"), alvo_palavras=10, overlap=10)

    def test_carregamento_pode_excluir_modulo(self):
        with tempfile.TemporaryDirectory() as temp:
            raiz = Path(temp)
            for nome in ("modulo-01-a", "modulo-02-b"):
                pasta = raiz / nome
                pasta.mkdir()
                (pasta / "README.md").write_text("palavra " * 40, encoding="utf-8")
            chunks = carregar_chunks(raiz, excluir_modulos={"modulo-02-b"})
            self.assertEqual({chunk.modulo for chunk in chunks}, {"modulo-01-a"})

    def test_carregamento_pode_limitar_modulos_para_evitar_vazamento(self):
        with tempfile.TemporaryDirectory() as temp:
            raiz = Path(temp)
            for nome in ("modulo-12-base", "modulo-13-avaliacao"):
                pasta = raiz / nome
                pasta.mkdir()
                (pasta / "README.md").write_text("palavra " * 40, encoding="utf-8")
            chunks = carregar_chunks(raiz, ate_modulo=12)
            self.assertEqual({chunk.modulo for chunk in chunks}, {"modulo-12-base"})


class TestJSONL(unittest.TestCase):
    def test_remove_ultima_linha_truncada(self):
        with tempfile.TemporaryDirectory() as temp:
            caminho = Path(temp) / "tracos.jsonl.part"
            registro = json.dumps({"indice": 1}).encode() + b"\n"
            caminho.write_bytes(registro + b'{"indice":')
            self.assertEqual(preparar_jsonl_retomavel(caminho), 1)
            self.assertEqual(caminho.read_bytes(), registro)

    def test_rejeita_corrupcao_no_meio(self):
        with tempfile.TemporaryDirectory() as temp:
            caminho = Path(temp) / "tracos.jsonl.part"
            caminho.write_text('{"ok": 1}\nquebrado\n{"ok": 2}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                preparar_jsonl_retomavel(caminho)


class TestExperimentos(unittest.TestCase):
    def test_registra_configuracao_metricas_e_resultado(self):
        with tempfile.TemporaryDirectory() as temp:
            raiz = Path(temp)
            registro = RegistroExperimento("teste", {"seed": 7}, raiz / "runs", ROOT)
            registro.registrar(0, loss=1.2)
            resultado = registro.concluir(acuracia=0.8)
            self.assertTrue((registro.pasta / "metadados.json").exists())
            self.assertEqual(json.loads(registro.metricas.read_text(encoding="utf-8"))["loss"], 1.2)
            self.assertEqual(json.loads(resultado.read_text(encoding="utf-8"))["acuracia"], 0.8)

    def test_rejeita_nome_que_escapa_do_destino(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            RegistroExperimento("../fora", {}, Path(temp), ROOT)


class TestGovernanca(unittest.TestCase):
    def test_detecta_e_anonimiza_pii_sem_expor_valores(self):
        texto = "Contato: pessoa@example.com, CPF 123.456.789-00, tel (11) 98765-4321"
        with tempfile.TemporaryDirectory() as temp:
            caminho = Path(temp) / "dados.txt"
            caminho.write_text(texto, encoding="utf-8")
            tipos = {achado.tipo for achado in auditar_pii(caminho)}
            self.assertEqual(tipos, {"email", "cpf", "telefone"})
        anonimizado = anonimizar_texto(texto)
        self.assertNotIn("pessoa@example.com", anonimizado)
        self.assertIn("<EMAIL_REMOVIDO>", anonimizado)

    def test_manifesto_guarda_checksum_licenca_e_auditoria(self):
        with tempfile.TemporaryDirectory() as temp:
            raiz = Path(temp)
            dados = raiz / "dados.jsonl"
            dados.write_text('{"texto": "sem pii"}\n', encoding="utf-8")
            destino = criar_manifesto_dataset(
                [dados],
                raiz / "dataset-manifest.json",
                nome="teste",
                origem="dados próprios",
                licenca="proprietária/autorizada",
                finalidade="teste unitário",
            )
            manifesto = json.loads(destino.read_text(encoding="utf-8"))
            self.assertEqual(manifesto["licenca"], "proprietária/autorizada")
            self.assertEqual(len(manifesto["arquivos"][0]["sha256"]), 64)


class TestAdapters(unittest.TestCase):
    def test_manifesto_impede_adapter_em_revisao_errada(self):
        manifesto = ManifestoAdapter(
            nome="suporte-v1",
            modelo_base="org/modelo",
            revisao_base="abc123",
            metodo="LoRA",
            tarefa="suporte",
            dataset_sha256="0" * 64,
            metricas={"aderencia": 0.9},
        )
        with tempfile.TemporaryDirectory() as temp:
            caminho = salvar_manifesto(manifesto, Path(temp) / "adapter-manifest.json")
            carregado = carregar_manifesto(caminho)
            verificar_compatibilidade(carregado, "org/modelo", "abc123")
            with self.assertRaises(ValueError):
                verificar_compatibilidade(carregado, "org/modelo", "outra")


class TestServing(unittest.TestCase):
    def test_resume_percentis_throughput_e_erros(self):
        amostras = [
            AmostraServing(0.1, 10, True),
            AmostraServing(0.2, 20, True),
            AmostraServing(9.0, 0, False),
        ]
        resumo = resumir_carga(amostras, duracao_s=1.0)
        self.assertEqual(resumo["sucesso"], 2 / 3)
        self.assertEqual(resumo["latencia_p95_s"], 0.2)
        self.assertEqual(resumo["throughput_tokens_s"], 30)
        self.assertEqual(percentil([1, 2, 3, 4], 0.5), 2)

    def test_rejeita_carga_sem_sucesso(self):
        with self.assertRaises(ValueError):
            resumir_carga([AmostraServing(1.0, 0, False)], 1.0)


class TestLabsExternos(unittest.TestCase):
    def test_dry_run_nao_importa_stack_gpu(self):
        scripts = (
            "modulo-05-sft/lab_cuda.py",
            "modulo-06-lora/lab_adapters.py",
            "modulo-09-rl/lab_cuda.py",
            "modulo-10-distillation/lab_cuda.py",
            "modulo-11-inferencia/lab_moe_cuda.py",
            "modulo-11-inferencia/benchmark_vllm.py",
        )
        for script in scripts:
            with self.subTest(script=script):
                resultado = subprocess.run(
                    [sys.executable, script, "--dry-run"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env=_ENV_UTF8,
                    check=False,
                )
                self.assertEqual(resultado.returncode, 0, resultado.stderr)
                self.assertIn("{", resultado.stdout)


class TestExecucao(unittest.TestCase):
    @patch("tools.execucao.subprocess.run")
    def test_retorna_saida_e_falha_rapido(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, stdout="ok", stderr="")
        resultado = executar_modulo("modulo.fake", "--teste")
        self.assertTrue(resultado.ok)
        self.assertEqual(resultado.saida, "ok")

        run.return_value = subprocess.CompletedProcess([], 2, stdout="", stderr="erro")
        with self.assertRaises(RuntimeError):
            executar_modulo("modulo.fake", "--teste")


try:
    import torch

    from tools import minigpt
except ModuleNotFoundError:
    torch = None
    minigpt = None


@unittest.skipUnless(torch is not None, "PyTorch não instalado")
class TestMiniGPT(unittest.TestCase):
    def test_batch_aceita_uma_unica_janela_valida(self):
        fonte = torch.arange(5)
        x, y = minigpt.pegar_batch(fonte, batch_size=1, bloco=4)
        self.assertTrue(torch.equal(x[0], torch.tensor([0, 1, 2, 3])))
        self.assertTrue(torch.equal(y[0], torch.tensor([1, 2, 3, 4])))

    def test_avaliacao_preserva_rng_e_modo(self):
        cfg = minigpt.Config(vocab=16, d=8, n_camadas=1, n_heads=2, bloco=4, d_ff=16)
        modelo = minigpt.MiniGPT(cfg)
        modelo.eval()
        fonte = torch.arange(80) % cfg.vocab
        estado = torch.random.get_rng_state().clone()
        minigpt.avaliar(modelo, fonte, n=2, batch=2, seed=7)
        self.assertTrue(torch.equal(estado, torch.random.get_rng_state()))
        self.assertFalse(modelo.training)


class TestImportarPorCaminho(unittest.TestCase):
    def test_ignora_modulo_homonimo_ja_em_sys_modules(self):
        """O bug real: `import dados` devolvia o `dados.py` errado pelo cache."""
        with tempfile.TemporaryDirectory() as tmp:
            outro = Path(tmp) / "dados.py"
            outro.write_text("QUEM = 'impostor'\n", encoding="utf-8")
            alvo = Path(tmp) / "verdadeiro.py"
            alvo.write_text("QUEM = 'alvo'\n", encoding="utf-8")

            impostor = importar_por_caminho(outro, "dados")
            sys.modules["dados"] = impostor
            self.addCleanup(sys.modules.pop, "dados", None)

            carregado = importar_por_caminho(alvo, "dados")
            self.assertEqual(carregado.QUEM, "alvo")

    def test_nao_registra_o_modulo_carregado_em_sys_modules(self):
        importar_por_caminho(ROOT / "modulo-03-treino" / "dados.py", "dados_modulo_03")
        self.assertNotIn("dados_modulo_03", sys.modules)

    def test_modulo_04_alcanca_o_carregador_de_corpus_do_modulo_03(self):
        """Sem isto, `modulo-04-dados/lab.py` morre em AttributeError num clone limpo."""
        dados_m4 = importar_por_caminho(ROOT / "modulo-04-dados" / "dados.py", "dados")
        self.addCleanup(sys.modules.pop, "dados", None)
        sys.modules["dados"] = dados_m4  # o estado que lab.py cria ao fazer `import dados`

        dados_m3 = importar_por_caminho(
            dados_m4.CORPUS_M3.parent.parent / "dados.py", "dados_modulo_03"
        )
        self.assertTrue(hasattr(dados_m3, "carregar"))


class TestProducao(unittest.TestCase):
    def test_contar_tokens_e_deterministico(self):
        self.assertEqual(contar_tokens(""), 0)
        self.assertGreater(contar_tokens("mais palavras aqui"), contar_tokens("."))
    def test_calcular_custo_usando_padrao(self):
        # 1 milhão de tokens de entrada + 1 milhão de saída na tabela padrão.
        self.assertAlmostEqual(calcular_custo(1_000_000, 1_000_000), 0.30 + 0.60)

    def test_custo_rejeita_contagem_negativa(self):
        with self.assertRaises(ValueError):
            calcular_custo(-1, 0)

    def test_resumo_de_trafego_so_considera_sucesso_na_latencia(self):
        linhas = [
            LinhaDeTrafego(1, True, 0.10, 5, 20, 0.0001, "gpt-x", 200),
            LinhaDeTrafego(2, True, 0.20, 5, 20, 0.0001, "gpt-x", 200),
            LinhaDeTrafego(3, False, 0.01, 0, 0, 0.0, "gpt-x", 503, "orcamento_excedido"),
        ]
        resumo = resumir_trafego(linhas)
        self.assertAlmostEqual(resumo["sucesso"], (2 / 3), places=2)
        self.assertEqual(resumo["motivo_falha_comum"], "orcamento_excedido")
        self.assertAlmostEqual(resumo["custo_total"], 0.0002)

    def test_disjuntor_abre_fecha_e_nao_deixa_prova_dobrada(self):
        d = Disjuntor(limiar_falha=0.5, janela_s=60.0, resfriamento_s=30.0, amostras_minimas=5)
        base = 100.0
        for i in range(5):
            self.assertTrue(d.permitir(base + i))
            d.registrar_falha(base + i)
        self.assertTrue(d._aberto)
        self.assertFalse(d.permitir(base + 5))  # aberto: recusa até o resfriamento
        # Após o resfriamento entra em meio aberto e deixa UMA prova.
        self.assertTrue(d.permitir(base + 40))
        self.assertFalse(d.permitir(base + 41))  # a segunda é recusada
        d.registrar_sucesso(base + 41)
        self.assertFalse(d._aberto)
        self.assertFalse(d._meio_aberto)


class TestEvalCi(unittest.TestCase):
    def test_portao_de_producao_passa(self):
        from tools.eval_ci import main

        self.assertEqual(main(), 0)


class TestReproducao(unittest.TestCase):
    def test_grava_no_esquema_e_retorna_caminho(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            arquivo = registrar_reproducao(
                raiz,
                experimento="modulo-01/lab",
                comando="uv run python modulo-01/lab.py",
                metricas={"bit_exact": True},
            )
            self.assertEqual(arquivo.parent, raiz / "resultados" / "modulo-01" / "lab")
            self.assertEqual(arquivo.parent.parent.parent, raiz / "resultados")
            registro = json.loads(arquivo.read_text(encoding="utf-8"))
            self.assertEqual(registro["schema_version"], 1)
            self.assertEqual(registro["experimento"], "modulo-01/lab")
            self.assertIn("executado_em", registro)
            self.assertIn("hardware", registro)
            self.assertIn("working_tree_dirty", registro)
            self.assertEqual(registro["metricas"], {"bit_exact": True})

    def test_rejeita_nome_de_experimento_invalido(self):
        with self.assertRaises(ValueError):
            registrar_reproducao(Path("."), experimento="simples", comando="x")


if __name__ == "__main__":
    unittest.main()
