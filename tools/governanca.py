"""Auditoria mínima de PII e proveniência para datasets dos laboratórios."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PADROES_PII = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "cpf": re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}[-.]?\d{2}(?!\d)"),
    "telefone": re.compile(r"(?<!\d)(?:\+?55\s*)?\(?\d{2}\)?\s*9?\d{4}[-\s]?\d{4}(?!\d)"),
}


@dataclass(frozen=True)
class AchadoPII:
    tipo: str
    linha: int
    ocorrencias: int


def sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def auditar_pii(caminho: Path) -> list[AchadoPII]:
    """Retorna contagens sem copiar o dado sensível para logs ou relatórios."""
    achados: list[AchadoPII] = []
    for numero, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1):
        for tipo, padrao in PADROES_PII.items():
            quantidade = len(padrao.findall(linha))
            if quantidade:
                achados.append(AchadoPII(tipo=tipo, linha=numero, ocorrencias=quantidade))
    return achados


def anonimizar_texto(texto: str) -> str:
    for tipo, padrao in PADROES_PII.items():
        texto = padrao.sub(f"<{tipo.upper()}_REMOVIDO>", texto)
    return texto


def criar_manifesto_dataset(
    arquivos: list[Path],
    destino: Path,
    *,
    nome: str,
    origem: str,
    licenca: str,
    finalidade: str,
) -> Path:
    if not arquivos:
        raise ValueError("o manifesto requer ao menos um arquivo")
    if not licenca.strip():
        raise ValueError("a licença não pode ficar implícita")
    registros = []
    for arquivo in arquivos:
        registros.append(
            {
                "arquivo": arquivo.name,
                "bytes": arquivo.stat().st_size,
                "sha256": sha256_arquivo(arquivo),
                "pii": [asdict(achado) for achado in auditar_pii(arquivo)],
            }
        )
    manifesto = {
        "nome": nome,
        "origem": origem,
        "licenca": licenca,
        "finalidade": finalidade,
        "arquivos": registros,
    }
    destino.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destino
