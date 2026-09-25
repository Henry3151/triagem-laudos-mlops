"""Backends de inferência usados pela API: scikit-learn (original) e ONNX Runtime (otimizado)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

from triagem.artefatos import ARQUIVO_JOBLIB, ARQUIVO_METADATA, ARQUIVO_ONNX, ler_json
from triagem.onnx_runtime import criar_sessao_onnx, prever_probabilidades_onnx
from triagem.preprocessamento import normalizar_texto

BACKENDS = ("onnx", "sklearn")


@dataclass(frozen=True)
class Predicao:
    classe: str
    probabilidades: dict[str, float]


class Classificador(Protocol):
    backend: str
    versao: str

    def prever(self, texto: str) -> Predicao: ...


def _montar_predicao(classes: Sequence[str], probabilidades: np.ndarray) -> Predicao:
    mapa = {c: float(p) for c, p in zip(classes, probabilidades, strict=True)}
    return Predicao(classe=max(mapa, key=mapa.__getitem__), probabilidades=mapa)


class SklearnClassificador:
    backend = "sklearn"

    def __init__(self, pipeline, versao: str):
        self._pipeline = pipeline
        self._classes = [str(c) for c in pipeline.classes_]
        self.versao = versao

    def prever(self, texto: str) -> Predicao:
        probs = self._pipeline.predict_proba([normalizar_texto(texto)])[0]
        return _montar_predicao(self._classes, probs)


class OnnxClassificador:
    backend = "onnx"

    def __init__(self, sessao, classes: Sequence[str], versao: str):
        self._sessao = sessao
        self._classes = list(classes)
        self.versao = versao

    def prever(self, texto: str) -> Predicao:
        probs = prever_probabilidades_onnx(self._sessao, [normalizar_texto(texto)])[0]
        return _montar_predicao(self._classes, probs)


def _exigir(caminho: Path) -> Path:
    if not caminho.is_file():
        raise FileNotFoundError(f"Artefato de modelo não encontrado: {caminho}")
    return caminho


def carregar_classificador(diretorio: Path, backend: str = "onnx") -> Classificador:
    if backend not in BACKENDS:
        raise ValueError(f"Backend '{backend}' inválido; use um de {BACKENDS}")
    diretorio = Path(diretorio)
    metadata = ler_json(_exigir(diretorio / ARQUIVO_METADATA))
    if backend == "sklearn":
        pipeline = joblib.load(_exigir(diretorio / ARQUIVO_JOBLIB))
        return SklearnClassificador(pipeline, metadata["versao"])
    sessao = criar_sessao_onnx(_exigir(diretorio / ARQUIVO_ONNX))
    return OnnxClassificador(sessao, metadata["classes"], metadata["versao"])
