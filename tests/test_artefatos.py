from datetime import UTC, datetime

from triagem.artefatos import (
    ARQUIVO_METADATA,
    DIR_PRODUCAO,
    decidir_promocao,
    ler_json,
    nova_versao,
    promover,
    salvar_json,
    sha256_arquivo,
    versoes_bibliotecas,
)

BOAS = {"f1_macro": 0.92, "recall_urgente": 0.95}


def test_nova_versao_formato():
    assert nova_versao(datetime(2026, 9, 25, 14, 3, 9, tzinfo=UTC)) == "20260925T140309Z"


def test_sha256_estavel(tmp_path):
    arquivo = tmp_path / "a.txt"
    arquivo.write_bytes(b"abc")
    assert sha256_arquivo(arquivo) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_json_ida_e_volta(tmp_path):
    caminho = salvar_json(tmp_path / "x" / "m.json", {"a": 1, "texto": "ação"})
    assert ler_json(caminho) == {"a": 1, "texto": "ação"}


def test_versoes_bibliotecas_tem_sklearn():
    assert versoes_bibliotecas()["scikit-learn"] == "1.9.1"


def test_primeiro_modelo_promove_se_passa_limiares():
    decisao = decidir_promocao(BOAS, None)
    assert decisao.promover
    assert "sem campeão" in decisao.motivo


def test_rejeita_f1_abaixo_do_minimo():
    decisao = decidir_promocao({"f1_macro": 0.70, "recall_urgente": 0.95}, None)
    assert not decisao.promover
    assert "F1-macro" in decisao.motivo


def test_rejeita_recall_urgente_baixo():
    decisao = decidir_promocao({"f1_macro": 0.90, "recall_urgente": 0.80}, None)
    assert not decisao.promover
    assert "recall" in decisao.motivo


def test_rejeita_quando_campeao_e_melhor():
    decisao = decidir_promocao(BOAS, {"f1_macro": 0.95, "recall_urgente": 0.97})
    assert not decisao.promover
    assert "campeão" in decisao.motivo


def test_empate_com_campeao_promove_dentro_da_tolerancia():
    assert decidir_promocao(BOAS, dict(BOAS)).promover


def test_promover_substitui_producao(tmp_path):
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / ARQUIVO_METADATA).write_text('{"versao": "v1"}')
    (v1 / "sobra.txt").write_text("x")
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / ARQUIVO_METADATA).write_text('{"versao": "v2"}')

    promover(v1, tmp_path)
    destino = promover(v2, tmp_path)

    assert destino == tmp_path / DIR_PRODUCAO
    assert ler_json(destino / ARQUIVO_METADATA) == {"versao": "v2"}
    assert not (destino / "sobra.txt").exists()
