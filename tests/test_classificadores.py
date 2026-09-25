import math

import pytest

from triagem.classificadores import carregar_classificador

URGENTE = "Tomografia de crânio. Hemorragia intraparenquimatosa extensa com desvio da linha média."
NORMAL = (
    "Radiografia de tórax. Parênquima pulmonar com transparência preservada. "
    "Seios costofrênicos livres."
)


@pytest.mark.parametrize("backend", ["sklearn", "onnx"])
def test_carrega_e_prediz(diretorio_modelo, backend):
    clf = carregar_classificador(diretorio_modelo, backend)
    assert clf.backend == backend
    assert clf.versao == "teste-v1"
    predicao = clf.prever(URGENTE)
    assert predicao.classe == "urgente"
    assert set(predicao.probabilidades) == {"atencao", "normal", "urgente"}
    assert math.isclose(sum(predicao.probabilidades.values()), 1.0, abs_tol=1e-5)


def test_backends_concordam(diretorio_modelo, split_pequeno):
    _, teste = split_pequeno
    sk = carregar_classificador(diretorio_modelo, "sklearn")
    ox = carregar_classificador(diretorio_modelo, "onnx")
    textos = list(teste["texto"][:100])
    iguais = sum(sk.prever(t).classe == ox.prever(t).classe for t in textos)
    assert iguais / len(textos) >= 0.99


@pytest.mark.parametrize("backend", ["sklearn", "onnx"])
def test_texto_com_acentos_equivale_a_normalizado(diretorio_modelo, backend):
    clf = carregar_classificador(diretorio_modelo, backend)
    original = clf.prever(
        "  RADIOGRAFIA de Tórax.   Parênquima pulmonar com transparência PRESERVADA. "
    )
    normalizado = clf.prever(
        "radiografia de torax. parenquima pulmonar com transparencia preservada."
    )
    assert original == normalizado


def test_backend_invalido(diretorio_modelo):
    with pytest.raises(ValueError, match="tensorflow"):
        carregar_classificador(diretorio_modelo, "tensorflow")


def test_diretorio_sem_modelo(tmp_path):
    with pytest.raises(FileNotFoundError, match=str(tmp_path.name)):
        carregar_classificador(tmp_path, "onnx")


def test_normal_classificado_como_normal(diretorio_modelo):
    assert carregar_classificador(diretorio_modelo, "onnx").prever(NORMAL).classe == "normal"
