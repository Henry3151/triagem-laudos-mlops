"""Valida que a DAG carrega sem erros e tem as tasks esperadas (usado no CI)."""

import sys

from airflow.dag_processing.dagbag import DagBag

ESPERADAS = {
    "ingerir_dados",
    "treinar_modelo",
    "avaliar_modelo",
    "exportar_onnx",
    "promover_modelo",
}

bag = DagBag("/opt/airflow/dags")
if bag.import_errors:
    print("Erros de importação:", bag.import_errors)
    sys.exit(1)
dag = bag.dags.get("triagem_retreino")
if dag is None:
    print("DAG triagem_retreino não encontrada")
    sys.exit(1)
tarefas = {t.task_id for t in dag.tasks}
if tarefas != ESPERADAS:
    print("Tasks divergentes:", sorted(tarefas))
    sys.exit(1)
print("DAG OK:", sorted(tarefas))
