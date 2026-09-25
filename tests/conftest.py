import pytest

from triagem.dados import dividir, gerar_dataset
from triagem.modelo import treinar


@pytest.fixture(scope="session")
def split_pequeno():
    return dividir(gerar_dataset(n=800, seed=7), proporcao_teste=0.25, seed=42)


@pytest.fixture(scope="session")
def pipeline_treinado(split_pequeno):
    treino, _ = split_pequeno
    return treinar(treino["texto"], treino["classe"], n_estimators=40)


@pytest.fixture(scope="session")
def modelo_onnx(pipeline_treinado) -> bytes:
    from triagem.onnx_export import converter_para_onnx

    return converter_para_onnx(pipeline_treinado)


@pytest.fixture(scope="session")
def diretorio_modelo(tmp_path_factory, pipeline_treinado, modelo_onnx):
    import joblib

    from triagem.artefatos import ARQUIVO_JOBLIB, ARQUIVO_METADATA, ARQUIVO_ONNX, salvar_json

    diretorio = tmp_path_factory.mktemp("modelo")
    joblib.dump(pipeline_treinado, diretorio / ARQUIVO_JOBLIB)
    (diretorio / ARQUIVO_ONNX).write_bytes(modelo_onnx)
    salvar_json(
        diretorio / ARQUIVO_METADATA,
        {"versao": "teste-v1", "classes": [str(c) for c in pipeline_treinado.classes_]},
    )
    return diretorio
