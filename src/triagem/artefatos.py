"""Versionamento de artefatos de modelo e quality gate de promoção (champion vs. challenger)."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

ARQUIVO_JOBLIB = "modelo.joblib"
ARQUIVO_ONNX = "modelo.onnx"
ARQUIVO_METADATA = "metadata.json"
ARQUIVO_METRICAS = "metricas.json"
DIR_PRODUCAO = "producao"
BIBLIOTECAS = ("scikit-learn", "skl2onnx", "onnxruntime", "numpy")


def nova_versao(agora: datetime | None = None) -> str:
    return (agora or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def sha256_arquivo(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def salvar_json(caminho: Path, dados: dict) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")
    return caminho


def ler_json(caminho: Path) -> dict:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def versoes_bibliotecas() -> dict[str, str | None]:
    versoes: dict[str, str | None] = {}
    for nome in BIBLIOTECAS:
        try:
            versoes[nome] = metadata.version(nome)
        except metadata.PackageNotFoundError:
            versoes[nome] = None
    return versoes


@dataclass(frozen=True)
class DecisaoPromocao:
    promover: bool
    motivo: str


def decidir_promocao(
    metricas: dict,
    metricas_campeao: dict | None,
    f1_minimo: float = 0.80,
    recall_urgente_minimo: float = 0.85,
    tolerancia: float = 0.01,
) -> DecisaoPromocao:
    f1 = metricas["f1_macro"]
    recall = metricas["recall_urgente"]
    if f1 < f1_minimo:
        return DecisaoPromocao(False, f"F1-macro {f1:.4f} abaixo do mínimo {f1_minimo}")
    if recall < recall_urgente_minimo:
        return DecisaoPromocao(
            False, f"recall de urgente {recall:.4f} abaixo do mínimo {recall_urgente_minimo}"
        )
    if metricas_campeao is None:
        return DecisaoPromocao(True, "sem campeão em produção; limiares atendidos")
    f1_campeao = metricas_campeao["f1_macro"]
    if f1 < f1_campeao - tolerancia:
        return DecisaoPromocao(
            False, f"F1-macro {f1:.4f} pior que o campeão ({f1_campeao:.4f}) além da tolerância"
        )
    return DecisaoPromocao(True, f"F1-macro {f1:.4f} ≥ campeão {f1_campeao:.4f} − {tolerancia}")


def promover(diretorio_versao: Path, models_dir: Path) -> Path:
    """Copia a versão para `models_dir/producao`, substituindo o conteúdo anterior."""
    models_dir = Path(models_dir)
    destino = models_dir / DIR_PRODUCAO
    temporario = models_dir / f".{DIR_PRODUCAO}_tmp"
    if temporario.exists():
        shutil.rmtree(temporario)
    shutil.copytree(diretorio_versao, temporario)
    if destino.exists():
        shutil.rmtree(destino)
    temporario.rename(destino)
    return destino
