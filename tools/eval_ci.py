"""Portão de qualidade para o CI: valida que o curso continua íntegro.

Não mede desempenho de modelo (isso cabe ao projeto). Mede que, após uma mudança no
material, os laboratórios que sustentam as afirmações do curso ainda produzem um
resultado são: o conjunto dourado do módulo 19 continua passando do piso, o serviço
simulado não degrada taxa de sucesso e o disjuntor continua abrindo e recusando rápido.

Isto é a *avaliação como portão* do módulo 19 aplicada ao próprio repositório.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.modulos import importar_por_caminho

PISO_GATE = 0.6


def main() -> int:
    # Carrega o lab do módulo 19 como módulo: o topo executa a demonstração inteira
    # e expõe `resumo`, `resultados` e `relatorio` para o portão inspecionar.
    lab = importar_por_caminho(ROOT / "modulo-19-producao" / "lab_cpu.py", "lab_producao")

    falhas: list[str] = []

    relatorio = lab.relatorio
    if relatorio["acuracia"] < PISO_GATE:
        falhas.append(
            f"conjunto dourado {relatorio['acuracia']:.0%} < piso {PISO_GATE:.0%}"
        )

    resumo = lab.resumo
    if resumo["sucesso"] < 1.0:
        falhas.append(f"serviço simulado não atingiu sucesso 100%: {resumo['sucesso']:.0%}")

    # O disjuntor precisa ter aberto (recusa rápida) em ao menos um passo do lab 4.
    recusas = sum(1 for _, _, motivo, _, _ in lab.resultados if motivo == "disjuntor_recusa")
    if recusas == 0:
        falhas.append("disjuntor não recusou requisição — o circuit breaker não abriu")

    if falhas:
        print("FALHA no portão do módulo 19:")
        for falha in falhas:
            print(f"  - {falha}")
        return 1

    print(f"OK: conjunto dourado {relatorio['acertos']}/{relatorio['n']} "
          f"({relatorio['acuracia']:.0%}) · sucesso {resumo['sucesso']:.0%} · "
          f"disjuntor recusou {recusas}×")
    return 0


if __name__ == "__main__":
    sys.exit(main())
