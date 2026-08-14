"""Avaliador aritmético limitado para os laboratórios de agentes."""

from __future__ import annotations

import ast
import operator

_BINARIOS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARIOS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_LIMITE_ABSOLUTO = 10**100


def calcular(expressao: str) -> str:
    """Calcula +, -, *, / e parênteses com limites contra abuso de recursos."""
    if not expressao or len(expressao) > 200:
        raise ValueError("expressão vazia ou longa demais")
    arvore = ast.parse(expressao, mode="eval")

    def visitar(no: ast.AST, profundidade: int = 0):
        if profundidade > 20:
            raise ValueError("expressão profunda demais")
        if isinstance(no, ast.Expression):
            return visitar(no.body, profundidade + 1)
        if isinstance(no, ast.Constant) and type(no.value) in (int, float):
            valor = no.value
        elif isinstance(no, ast.BinOp) and type(no.op) in _BINARIOS:
            valor = _BINARIOS[type(no.op)](
                visitar(no.left, profundidade + 1), visitar(no.right, profundidade + 1)
            )
        elif isinstance(no, ast.UnaryOp) and type(no.op) in _UNARIOS:
            valor = _UNARIOS[type(no.op)](visitar(no.operand, profundidade + 1))
        else:
            raise ValueError("operação não permitida")
        if not isinstance(valor, (int, float)) or abs(valor) > _LIMITE_ABSOLUTO:
            raise ValueError("resultado fora do limite")
        return valor

    return str(visitar(arvore))
