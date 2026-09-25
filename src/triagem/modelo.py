"""Construção, treino e avaliação do classificador TF-IDF + Random Forest."""

from __future__ import annotations

from collections.abc import Iterable

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline

from triagem.dados import CLASSES
from triagem.preprocessamento import normalizar_texto

# Mesmo padrão usado como `tokenexp` na conversão ONNX (paridade de tokenização).
TOKEN_PATTERN = r"\b\w\w+\b"


def construir_pipeline(n_estimators: int = 200, seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                    # sublinear_tf fica desligado: o skl2onnx não o reproduz (quebra a paridade).
                    sublinear_tf=False,
                    token_pattern=TOKEN_PATTERN,
                    # Caixa já tratada em normalizar_texto; evita o StringNormalizer no ONNX,
                    # que exige o locale en_US.UTF-8 (ausente em imagens slim).
                    lowercase=False,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def treinar(
    textos: Iterable[str], rotulos: Iterable[str], n_estimators: int = 200, seed: int = 42
) -> Pipeline:
    pipeline = construir_pipeline(n_estimators=n_estimators, seed=seed)
    pipeline.fit([normalizar_texto(t) for t in textos], list(rotulos))
    return pipeline


def avaliar(pipeline: Pipeline, textos: Iterable[str], rotulos: Iterable[str]) -> dict:
    verdadeiros = list(rotulos)
    previstos = pipeline.predict([normalizar_texto(t) for t in textos])
    return {
        "acuracia": float(accuracy_score(verdadeiros, previstos)),
        "f1_macro": float(f1_score(verdadeiros, previstos, average="macro")),
        "recall_urgente": float(
            recall_score(verdadeiros, previstos, labels=["urgente"], average="macro")
        ),
        "matriz_confusao": {
            "rotulos": list(CLASSES),
            "valores": confusion_matrix(verdadeiros, previstos, labels=list(CLASSES)).tolist(),
        },
        "n_amostras": len(verdadeiros),
    }
