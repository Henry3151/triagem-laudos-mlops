"""Compara candidatos de modelo no mesmo split e com as mesmas seeds do pipeline de produção.

Comparação feita a posteriori, para documentar a escolha: o modelo em produção
(TF-IDF + Random Forest) foi definido antes, e este script não altera o pipeline.

Uso: uv run python scripts/comparar_candidatos.py  →  reports/candidatos.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from triagem.dados import carregar_dataset, dividir
from triagem.modelo import avaliar, construir_pipeline
from triagem.preprocessamento import normalizar_texto

SEED = 42


def candidatos() -> dict[str, Pipeline]:
    producao = construir_pipeline(seed=SEED)
    tfidf = producao.named_steps["tfidf"]

    def com(classificador) -> Pipeline:
        return Pipeline([("tfidf", clone(tfidf)), ("clf", classificador)])

    sublinear = construir_pipeline(seed=SEED).set_params(tfidf__sublinear_tf=True)
    return {
        "Random Forest (produção)": producao,
        "Random Forest + sublinear_tf": sublinear,
        "Regressão Logística": com(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        ),
        "LinearSVC": com(LinearSVC(class_weight="balanced", random_state=SEED)),
        "Naive Bayes Multinomial": com(MultinomialNB()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=Path("data/raw/laudos_sinteticos.csv"))
    parser.add_argument("--saida", type=Path, default=Path("reports/candidatos.json"))
    args = parser.parse_args(argv)

    treino, teste = dividir(carregar_dataset(args.dados), seed=SEED)
    textos_treino = [normalizar_texto(t) for t in treino["texto"]]
    resultados = []
    for nome, pipeline in candidatos().items():
        inicio = time.perf_counter()
        pipeline.fit(textos_treino, list(treino["classe"]))
        metricas = avaliar(pipeline, teste["texto"], teste["classe"])
        metricas["treino_s"] = time.perf_counter() - inicio
        resultados.append({"candidato": nome, **metricas})
        print(
            f"{nome:32s} F1-macro={metricas['f1_macro']:.4f} "
            f"recall_urgente={metricas['recall_urgente']:.4f}"
        )

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    relatorio = {
        "descricao": "Comparação a posteriori no split de produção (80/20 estratificado, seed 42)",
        "n_treino": len(treino),
        "n_teste": len(teste),
        "candidatos": resultados,
    }
    args.saida.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
