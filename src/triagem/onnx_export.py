"""Conversão do pipeline scikit-learn para ONNX, com verificação de paridade bloqueante."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

from triagem.modelo import TOKEN_PATTERN
from triagem.onnx_runtime import NOME_ENTRADA, criar_sessao_onnx, prever_probabilidades_onnx
from triagem.preprocessamento import normalizar_texto

TARGET_OPSET = 18


class ParidadeOnnxError(RuntimeError):
    """O modelo ONNX diverge do scikit-learn além da tolerância."""


def converter_para_onnx(pipeline: Pipeline) -> bytes:
    opcoes = {
        id(pipeline.named_steps["tfidf"]): {"tokenexp": TOKEN_PATTERN},
        id(pipeline.named_steps["rf"]): {"zipmap": False},
    }
    modelo = to_onnx(
        pipeline,
        initial_types=[(NOME_ENTRADA, StringTensorType([None, 1]))],
        options=opcoes,
        target_opset=TARGET_OPSET,
    )
    return modelo.SerializeToString()


def verificar_paridade(
    pipeline: Pipeline,
    modelo_onnx: bytes,
    textos: Iterable[str],
    min_concordancia: float = 0.99,
    max_diff_prob: float = 1e-2,
) -> dict:
    normalizados = [normalizar_texto(t) for t in textos]
    probs_sklearn = pipeline.predict_proba(normalizados)
    probs_onnx = prever_probabilidades_onnx(criar_sessao_onnx(modelo_onnx), normalizados)
    concordancia = float(np.mean(probs_sklearn.argmax(axis=1) == probs_onnx.argmax(axis=1)))
    diff = float(np.abs(probs_sklearn - probs_onnx).max())
    if concordancia < min_concordancia or diff > max_diff_prob:
        raise ParidadeOnnxError(
            f"Paridade ONNX violada: concordância {concordancia:.4f} (mín. {min_concordancia}), "
            f"diferença máxima {diff:.5f} (máx. {max_diff_prob})"
        )
    return {"concordancia": concordancia, "max_diff_prob": diff}


def exportar_onnx(pipeline: Pipeline, destino: Path, textos_validacao: Iterable[str]) -> dict:
    modelo = converter_para_onnx(pipeline)
    paridade = verificar_paridade(pipeline, modelo, textos_validacao)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(modelo)
    return paridade
