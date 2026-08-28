"""Importação de um arquivo .py pelo caminho, sem passar por ``sys.path``.

Vários módulos do curso têm um ``dados.py`` próprio. O padrão de inserir a pasta
do vizinho em ``sys.path`` e escrever ``import dados`` só funciona enquanto ninguém
tiver importado *outro* ``dados`` antes: a partir daí o cache de ``sys.modules``
vence a busca por caminho e devolve o módulo errado, calado. É o tipo de erro que
aparece como ``AttributeError`` numa função que nada tem a ver com imports.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def importar_por_caminho(caminho: Path, nome: str) -> ModuleType:
    """Carrega ``caminho`` como um módulo novo, ignorando o cache de ``sys.modules``.

    O módulo carregado NÃO é registrado em ``sys.modules``: cada chamada devolve um
    objeto novo, e nenhum ``import`` posterior é contaminado por este nome.
    """
    caminho = Path(caminho)
    especificacao = importlib.util.spec_from_file_location(nome, caminho)
    if especificacao is None or especificacao.loader is None:
        raise ImportError(f"não foi possível carregar um módulo de {caminho}")
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)
    return modulo
