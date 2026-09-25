"""Retreino semanal do classificador de urgência de laudos.

ingerir_dados → treinar_modelo → avaliar_modelo → exportar_onnx → promover_modelo

Entre as tasks trafegam só caminhos e métricas (XCom leve); a lógica vive em `triagem.pipeline`.
A promoção aplica o quality gate champion vs. challenger (F1-macro e recall de urgente).
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import dag, get_current_context, task

BASE = Path(os.environ.get("TRIAGEM_BASE_DIR", "/opt/airflow"))
DADOS = BASE / "data" / "raw" / "laudos_sinteticos.csv"
PROCESSADOS = BASE / "data" / "processed"
MODELOS = BASE / "models"


@dag(
    dag_id="triagem_retreino",
    schedule="@weekly",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["triagem", "mlops"],
    doc_md=__doc__,
)
def triagem_retreino():
    @task
    def ingerir_dados() -> dict[str, str]:
        from triagem.pipeline import ingerir

        run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", get_current_context()["run_id"])
        return ingerir(DADOS, PROCESSADOS / run_id)

    @task
    def treinar_modelo(split: dict[str, str]) -> str:
        from triagem.pipeline import treinar_versao

        return treinar_versao(Path(split["treino"]), MODELOS)

    @task
    def avaliar_modelo(diretorio_versao: str, split: dict[str, str]) -> dict:
        from triagem.pipeline import avaliar_versao

        return avaliar_versao(Path(diretorio_versao), Path(split["teste"]))

    @task
    def exportar_onnx(diretorio_versao: str, split: dict[str, str], metricas: dict) -> dict:
        from triagem.pipeline import exportar_versao

        metadata = exportar_versao(
            Path(diretorio_versao), Path(split["teste"]), metricas, split["sha256_dados"]
        )
        return {"versao": metadata["versao"], "paridade_onnx": metadata["paridade_onnx"]}

    @task
    def promover_modelo(diretorio_versao: str, exportado: dict) -> dict:
        from triagem.pipeline import promover_se_aprovado

        return promover_se_aprovado(Path(diretorio_versao), MODELOS)

    split = ingerir_dados()
    versao = treinar_modelo(split)
    metricas = avaliar_modelo(versao, split)
    exportado = exportar_onnx(versao, split, metricas)
    promover_modelo(versao, exportado)


triagem_retreino()
