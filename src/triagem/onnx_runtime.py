"""Inferência com ONNX Runtime (sem dependências de treino)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort

NOME_ENTRADA = "texto"


def criar_sessao_onnx(modelo: bytes | str | Path, threads: int = 1) -> ort.InferenceSession:
    opcoes = ort.SessionOptions()
    opcoes.intra_op_num_threads = threads
    opcoes.inter_op_num_threads = threads
    fonte = modelo if isinstance(modelo, bytes) else str(modelo)
    return ort.InferenceSession(fonte, opcoes, providers=["CPUExecutionProvider"])


def prever_probabilidades_onnx(
    sessao: ort.InferenceSession, textos_normalizados: Sequence[str]
) -> np.ndarray:
    entrada = np.array(list(textos_normalizados), dtype=object).reshape(-1, 1)
    _rotulos, probabilidades = sessao.run(None, {NOME_ENTRADA: entrada})
    return np.asarray(probabilidades, dtype=np.float64)
