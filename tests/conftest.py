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
