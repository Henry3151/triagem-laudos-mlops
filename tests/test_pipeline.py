import json

import pytest

from triagem.artefatos import (
    ARQUIVO_JOBLIB,
    ARQUIVO_METADATA,
    ARQUIVO_METRICAS,
    ARQUIVO_ONNX,
    DIR_PRODUCAO,
    ler_json,
)
from triagem.dados import DatasetInvalidoError, gerar_dataset, salvar_dataset
from triagem.pipeline import (
    avaliar_versao,
    executar_pipeline,
    exportar_versao,
    ingerir,
    promover_se_aprovado,
    treinar_versao,
)
from triagem.treinar import main


@pytest.fixture(scope="module")
def csv_dados(tmp_path_factory):
    return salvar_dataset(gerar_dataset(n=2000, seed=11), tmp_path_factory.mktemp("d") / "d.csv")


def test_etapas_encadeadas(csv_dados, tmp_path):
    split = ingerir(csv_dados, tmp_path / "proc")
    assert set(split) == {"treino", "teste", "sha256_dados"}
    diretorio = treinar_versao(split["treino"], tmp_path / "models", versao="v1", n_estimators=30)
    assert (tmp_path / "models" / "v1" / ARQUIVO_JOBLIB).is_file()
    metricas = avaliar_versao(diretorio, split["teste"])
    assert ler_json(tmp_path / "models" / "v1" / ARQUIVO_METRICAS) == metricas
    metadata = exportar_versao(diretorio, split["teste"], metricas, split["sha256_dados"])
    assert metadata["versao"] == "v1"
    assert metadata["classes"] == ["atencao", "normal", "urgente"]
    assert metadata["paridade_onnx"]["concordancia"] >= 0.99
    assert (tmp_path / "models" / "v1" / ARQUIVO_ONNX).is_file()
    decisao = promover_se_aprovado(diretorio, tmp_path / "models")
    assert decisao == {"promovido": True, "motivo": decisao["motivo"], "versao": "v1"}
    assert (tmp_path / "models" / DIR_PRODUCAO / ARQUIVO_METADATA).is_file()
    json.dumps([split, metricas, metadata, decisao])  # tudo serializável para XCom


def test_reexecucao_com_mesmo_modelo_mantem_producao(csv_dados, tmp_path):
    models = tmp_path / "models"
    primeiro = executar_pipeline(csv_dados, models, tmp_path / "t1", n_estimators=30)
    segundo = executar_pipeline(csv_dados, models, tmp_path / "t2", n_estimators=30)
    assert primeiro["promocao"]["promovido"]
    assert segundo["promocao"]["promovido"]  # empate com o campeão fica dentro da tolerância
    assert segundo["versao"] != primeiro["versao"]
    assert ler_json(models / DIR_PRODUCAO / ARQUIVO_METADATA)["versao"] == segundo["versao"]


def test_rejeita_challenger_pior(csv_dados, tmp_path):
    models = tmp_path / "models"
    executar_pipeline(csv_dados, models, tmp_path / "t1", n_estimators=30)
    campeao = ler_json(models / DIR_PRODUCAO / ARQUIVO_METADATA)
    campeao["metricas"]["f1_macro"] = 0.9999
    (models / DIR_PRODUCAO / ARQUIVO_METADATA).write_text(json.dumps(campeao), encoding="utf-8")
    resultado = executar_pipeline(csv_dados, models, tmp_path / "t2", n_estimators=30)
    assert not resultado["promocao"]["promovido"]
    assert ler_json(models / DIR_PRODUCAO / ARQUIVO_METADATA)["versao"] == campeao["versao"]


def test_ingerir_falha_com_dataset_invalido(tmp_path):
    csv = salvar_dataset(gerar_dataset(n=500, seed=1), tmp_path / "pequeno.csv")
    with pytest.raises(DatasetInvalidoError):
        ingerir(csv, tmp_path / "proc")


def test_cli(csv_dados, tmp_path, capsys):
    codigo = main(
        [
            "--dados",
            str(csv_dados),
            "--models-dir",
            str(tmp_path / "models"),
            "--trabalho",
            str(tmp_path / "trab"),
            "--n-estimators",
            "20",
        ]
    )
    assert codigo == 0
    saida = json.loads(capsys.readouterr().out)
    assert saida["promocao"]["promovido"] is True
