import numpy as np
import pytest

from triagem.onnx_export import ParidadeOnnxError, exportar_onnx, verificar_paridade
from triagem.onnx_runtime import criar_sessao_onnx, prever_probabilidades_onnx
from triagem.preprocessamento import normalizar_texto


def test_paridade_com_sklearn(pipeline_treinado, modelo_onnx, split_pequeno):
    _, teste = split_pequeno
    paridade = verificar_paridade(pipeline_treinado, modelo_onnx, teste["texto"])
    assert paridade["concordancia"] >= 0.99
    assert paridade["max_diff_prob"] <= 1e-2


def test_grafo_sem_string_normalizer(modelo_onnx):
    # StringNormalizer exige locale en_US.UTF-8, ausente em imagens slim; a caixa já é
    # tratada por normalizar_texto.
    import onnx

    operadores = {n.op_type for n in onnx.load_from_string(modelo_onnx).graph.node}
    assert "StringNormalizer" not in operadores


def test_probabilidades_somam_um(modelo_onnx, split_pequeno):
    _, teste = split_pequeno
    sessao = criar_sessao_onnx(modelo_onnx)
    probs = prever_probabilidades_onnx(sessao, [normalizar_texto(t) for t in teste["texto"][:20]])
    assert probs.shape == (20, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_paridade_violada_lanca_erro(pipeline_treinado, modelo_onnx, split_pequeno):
    _, teste = split_pequeno
    with pytest.raises(ParidadeOnnxError):
        verificar_paridade(pipeline_treinado, modelo_onnx, teste["texto"], min_concordancia=1.01)


def test_exportar_grava_arquivo(pipeline_treinado, split_pequeno, tmp_path):
    _, teste = split_pequeno
    destino = tmp_path / "modelo.onnx"
    paridade = exportar_onnx(pipeline_treinado, destino, teste["texto"])
    assert destino.stat().st_size > 0
    assert paridade["concordancia"] >= 0.99


def test_exportar_nao_grava_se_paridade_falha(
    pipeline_treinado, split_pequeno, tmp_path, monkeypatch
):
    import triagem.onnx_export as modulo

    def falha(*_args, **_kwargs):
        raise ParidadeOnnxError("forçado")

    monkeypatch.setattr(modulo, "verificar_paridade", falha)
    destino = tmp_path / "modelo.onnx"
    _, teste = split_pequeno
    with pytest.raises(ParidadeOnnxError):
        exportar_onnx(pipeline_treinado, destino, teste["texto"])
    assert not destino.exists()
