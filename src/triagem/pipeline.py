"""Etapas do pipeline de treino, reutilizadas pela DAG do Airflow e pelo CLI.

Cada etapa recebe e devolve apenas caminhos e dicionários JSON-serializáveis (XCom leve).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib

from triagem.artefatos import (
    ARQUIVO_JOBLIB,
    ARQUIVO_METADATA,
    ARQUIVO_METRICAS,
    ARQUIVO_ONNX,
    DIR_PRODUCAO,
    decidir_promocao,
    ler_json,
    nova_versao,
    promover,
    salvar_json,
    sha256_arquivo,
    versoes_bibliotecas,
)
from triagem.dados import carregar_dataset, dividir, salvar_dataset, validar_dataset
from triagem.modelo import avaliar, treinar
from triagem.onnx_export import exportar_onnx

logger = logging.getLogger(__name__)


def ingerir(caminho_csv: Path, destino: Path, seed: int = 42) -> dict[str, str]:
    df = carregar_dataset(caminho_csv)
    validar_dataset(df)
    treino, teste = dividir(df, seed=seed)
    destino = Path(destino)
    caminho_treino = salvar_dataset(treino, destino / "treino.csv")
    caminho_teste = salvar_dataset(teste, destino / "teste.csv")
    logger.info("Dados ingeridos: %d treino, %d teste", len(treino), len(teste))
    return {
        "treino": str(caminho_treino),
        "teste": str(caminho_teste),
        "sha256_dados": sha256_arquivo(caminho_csv),
    }


def treinar_versao(
    caminho_treino: Path,
    models_dir: Path,
    versao: str | None = None,
    n_estimators: int = 200,
) -> str:
    df = carregar_dataset(caminho_treino)
    pipeline = treinar(df["texto"], df["classe"], n_estimators=n_estimators)
    # nova_versao tem resolução de segundo: sufixo evita colisão em execuções seguidas.
    base = Path(models_dir) / (versao or nova_versao())
    diretorio, sufixo = base, 2
    while diretorio.exists():
        diretorio, sufixo = base.with_name(f"{base.name}-{sufixo}"), sufixo + 1
    diretorio.mkdir(parents=True)
    joblib.dump(pipeline, diretorio / ARQUIVO_JOBLIB)
    logger.info("Modelo treinado em %s", diretorio)
    return str(diretorio)


def avaliar_versao(diretorio_versao: Path, caminho_teste: Path) -> dict:
    diretorio = Path(diretorio_versao)
    pipeline = joblib.load(diretorio / ARQUIVO_JOBLIB)
    df = carregar_dataset(caminho_teste)
    metricas = avaliar(pipeline, df["texto"], df["classe"])
    salvar_json(diretorio / ARQUIVO_METRICAS, metricas)
    logger.info(
        "Métricas: F1-macro=%.4f recall_urgente=%.4f",
        metricas["f1_macro"],
        metricas["recall_urgente"],
    )
    return metricas


def exportar_versao(
    diretorio_versao: Path, caminho_teste: Path, metricas: dict, sha256_dados: str
) -> dict:
    diretorio = Path(diretorio_versao)
    pipeline = joblib.load(diretorio / ARQUIVO_JOBLIB)
    df = carregar_dataset(caminho_teste)
    paridade = exportar_onnx(pipeline, diretorio / ARQUIVO_ONNX, df["texto"])
    metadata = {
        "versao": diretorio.name,
        "criado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "classes": [str(c) for c in pipeline.classes_],
        "sha256_dados": sha256_dados,
        "metricas": metricas,
        "paridade_onnx": paridade,
        "bibliotecas": versoes_bibliotecas(),
    }
    salvar_json(diretorio / ARQUIVO_METADATA, metadata)
    return metadata


def promover_se_aprovado(diretorio_versao: Path, models_dir: Path) -> dict:
    diretorio = Path(diretorio_versao)
    metadata = ler_json(diretorio / ARQUIVO_METADATA)
    campeao_path = Path(models_dir) / DIR_PRODUCAO / ARQUIVO_METADATA
    campeao = ler_json(campeao_path)["metricas"] if campeao_path.is_file() else None
    decisao = decidir_promocao(metadata["metricas"], campeao)
    if decisao.promover:
        promover(diretorio, models_dir)
    logger.info("Promoção de %s: %s (%s)", metadata["versao"], decisao.promover, decisao.motivo)
    return {"promovido": decisao.promover, "motivo": decisao.motivo, "versao": metadata["versao"]}


def executar_pipeline(
    caminho_csv: Path, models_dir: Path, trabalho_dir: Path, n_estimators: int = 200
) -> dict:
    split = ingerir(caminho_csv, trabalho_dir)
    diretorio = treinar_versao(split["treino"], models_dir, n_estimators=n_estimators)
    metricas = avaliar_versao(diretorio, split["teste"])
    metadata = exportar_versao(diretorio, split["teste"], metricas, split["sha256_dados"])
    promocao = promover_se_aprovado(diretorio, models_dir)
    return {
        "versao": metadata["versao"],
        "metricas": metricas,
        "paridade_onnx": metadata["paridade_onnx"],
        "promocao": promocao,
    }
