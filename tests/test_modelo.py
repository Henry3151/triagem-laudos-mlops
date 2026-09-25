from triagem.modelo import avaliar, construir_pipeline


def test_pipeline_tem_tfidf_e_random_forest():
    pipe = construir_pipeline()
    assert list(pipe.named_steps) == ["tfidf", "rf"]
    assert pipe.named_steps["tfidf"].ngram_range == (1, 2)
    assert pipe.named_steps["rf"].n_jobs == 1


def test_modelo_treinado_atinge_limiar(pipeline_treinado, split_pequeno):
    _, teste = split_pequeno
    metricas = avaliar(pipeline_treinado, teste["texto"], teste["classe"])
    assert metricas["f1_macro"] >= 0.80
    assert metricas["recall_urgente"] >= 0.85


def test_metricas_tem_formato_esperado(pipeline_treinado, split_pequeno):
    _, teste = split_pequeno
    metricas = avaliar(pipeline_treinado, teste["texto"], teste["classe"])
    assert set(metricas) == {
        "acuracia",
        "f1_macro",
        "recall_urgente",
        "matriz_confusao",
        "n_amostras",
    }
    assert metricas["n_amostras"] == len(teste)
    assert metricas["matriz_confusao"]["rotulos"] == ["normal", "atencao", "urgente"]
    assert sum(map(sum, metricas["matriz_confusao"]["valores"])) == len(teste)


def test_classes_do_modelo(pipeline_treinado):
    assert sorted(pipeline_treinado.classes_) == ["atencao", "normal", "urgente"]
