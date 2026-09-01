"""Download e validação dos datasets externos declarados pelo curso."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTO = ROOT / "DADOS_EXTERNOS.json"


def carregar_manifesto_dados() -> dict[str, dict]:
    return json.loads(MANIFESTO.read_text(encoding="utf-8"))


def validar_bytes(artefato_id: str, conteudo: bytes) -> None:
    metadados = carregar_manifesto_dados().get(artefato_id)
    if metadados is None:
        raise KeyError(f"artefato externo não declarado: {artefato_id}")
    observado = hashlib.sha256(conteudo).hexdigest()
    if observado != metadados["sha256"]:
        raise ValueError(
            f"checksum divergente para {artefato_id}: esperado {metadados['sha256']}, "
            f"observado {observado}. Não use os dados até revisar a origem."
        )
    if len(conteudo) != metadados["bytes"]:
        raise ValueError(
            f"tamanho divergente para {artefato_id}: esperado {metadados['bytes']}, "
            f"observado {len(conteudo)}"
        )


def validar_arquivo(artefato_id: str, caminho: Path) -> bytes:
    conteudo = caminho.read_bytes()
    validar_bytes(artefato_id, conteudo)
    return conteudo


def baixar_bytes_verificados(artefato_id: str) -> bytes:
    metadados = carregar_manifesto_dados().get(artefato_id)
    if metadados is None or "url" not in metadados:
        raise KeyError(f"artefato remoto não declarado: {artefato_id}")
    requisicao = urllib.request.Request(
        metadados["url"], headers={"User-Agent": "llm-course-reproducibility/1.0"}
    )
    conteudo = urllib.request.urlopen(requisicao, timeout=120).read()
    validar_bytes(artefato_id, conteudo)
    return conteudo


def carregar_ou_baixar(artefato_id: str, destino: Path) -> bytes:
    if destino.exists():
        return validar_arquivo(artefato_id, destino)
    conteudo = baixar_bytes_verificados(artefato_id)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(conteudo)
    return conteudo
