import pytest
from fastapi.testclient import TestClient

from triagem.api.app import criar_app
from triagem.classificadores import Predicao, carregar_classificador

URGENTE = "Tomografia de crânio. Hemorragia intraparenquimatosa extensa com desvio da linha média."


class ClassificadorQuebrado:
    backend = "fake"
    versao = "quebrado"

    def prever(self, texto: str) -> Predicao:
        raise RuntimeError("falha simulada")


@pytest.fixture
def cliente(diretorio_modelo):
    app = criar_app(carregar_classificador(diretorio_modelo, "onnx"))
    with TestClient(app) as c:
        yield c


def test_predict_ok(cliente):
    resposta = cliente.post("/predict", json={"texto": URGENTE})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["classe"] == "urgente"
    assert corpo["backend"] == "onnx"
    assert corpo["versao_modelo"] == "teste-v1"
    assert set(corpo["probabilidades"]) == {"atencao", "normal", "urgente"}


@pytest.mark.parametrize(
    "payload",
    [
        {"texto": ""},
        {"texto": "     "},
        {"texto": "a" * 5001},
        {"texto": 123},
        {},
        {"outro": "campo"},
    ],
)
def test_payload_invalido_retorna_422(cliente, payload):
    assert cliente.post("/predict", json=payload).status_code == 422


def test_corpo_nao_json_retorna_422(cliente):
    resposta = cliente.post(
        "/predict", content=b"nao e json", headers={"Content-Type": "application/json"}
    )
    assert resposta.status_code == 422


def test_texto_no_limite_aceito(cliente):
    assert cliente.post("/predict", json={"texto": "a" * 5000}).status_code == 200


def test_health(cliente):
    assert cliente.get("/health").json() == {
        "status": "ok",
        "versao_modelo": "teste-v1",
        "backend": "onnx",
    }


def test_metricas_contam_requisicoes_e_predicoes(cliente):
    cliente.post("/predict", json={"texto": URGENTE})
    cliente.post("/predict", json={"texto": ""})
    texto = cliente.get("/metrics").text
    assert 'triagem_http_requests_total{method="POST",rota="/predict",status="200"} 1.0' in texto
    assert 'triagem_http_requests_total{method="POST",rota="/predict",status="422"} 1.0' in texto
    assert 'triagem_predicoes_total{classe="urgente"} 1.0' in texto
    assert 'triagem_modelo_info{backend="onnx",versao="teste-v1"} 1.0' in texto
    assert "triagem_inferencia_duration_seconds_bucket" in texto
    assert 'rota="/metrics"' not in texto


def test_rota_inexistente_nao_explode_cardinalidade(cliente):
    cliente.get("/qualquer/coisa/123")
    texto = cliente.get("/metrics").text
    assert 'rota="desconhecida",status="404"' in texto
    assert "/qualquer/coisa/123" not in texto


def test_falha_de_inferencia_retorna_500_generico():
    app = criar_app(ClassificadorQuebrado())
    with TestClient(app) as c:
        resposta = c.post("/predict", json={"texto": URGENTE})
        assert resposta.status_code == 500
        assert resposta.json() == {"detail": "Erro interno ao classificar o laudo"}
        assert 'status="500"' in c.get("/metrics").text


def test_carrega_modelo_do_ambiente(diretorio_modelo, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(diretorio_modelo))
    monkeypatch.setenv("MODEL_BACKEND", "sklearn")
    with TestClient(criar_app()) as c:
        assert c.get("/health").json()["backend"] == "sklearn"


def test_startup_falha_sem_modelo(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError), TestClient(criar_app()):
        pass
