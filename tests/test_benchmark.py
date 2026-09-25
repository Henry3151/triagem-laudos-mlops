import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from triagem.benchmark import (
    benchmark_http,
    benchmark_inprocess,
    gerar_relatorio_md,
    main,
    medir,
    percentis,
)
from triagem.dados import gerar_dataset, salvar_dataset


class _Stub(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (nome exigido pelo http.server)
        self.rfile.read(int(self.headers["Content-Length"]))
        corpo = json.dumps({"classe": "normal"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def log_message(self, *_args):
        pass


@pytest.fixture
def url_stub():
    servidor = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{servidor.server_port}"
    servidor.shutdown()
    servidor.server_close()


def test_benchmark_http_sem_dependencias_externas(url_stub, monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)  # roda na imagem runtime, sem httpx
    r = benchmark_http({"sklearn": url_stub, "onnx": url_stub}, ["a", "b"], n=5, warmup=1)
    assert r["sklearn"]["latencia_ms"]["n"] == 5
    assert r["speedup_p50"] > 0


def test_main_inprocess_sem_pandas(diretorio_modelo, tmp_path, monkeypatch):
    csv = salvar_dataset(gerar_dataset(n=50, seed=1), tmp_path / "d.csv")
    monkeypatch.setitem(sys.modules, "pandas", None)  # a imagem runtime não tem pandas
    codigo = main(
        [
            "--modo",
            "inprocess",
            "--model-dir",
            str(diretorio_modelo),
            "--dados",
            str(csv),
            "--n",
            "5",
            "--warmup",
            "1",
            "--saida",
            str(tmp_path / "rel"),
        ]
    )
    assert codigo == 0
    assert (tmp_path / "rel" / "latencia.md").is_file()
    assert json.loads((tmp_path / "rel" / "latencia.json").read_text("utf-8"))["inprocess"]


def test_percentis():
    r = percentis(list(range(1, 101)))
    assert r["n"] == 100
    assert r["p50"] == 50.5
    assert 95 <= r["p95"] <= 96
    assert 99 <= r["p99"] <= 100
    assert r["media"] == 50.5


def test_medir_conta_chamadas_com_warmup():
    chamadas = []
    resultado = medir(chamadas.append, ["a", "b"], n=10, warmup=3)
    assert len(chamadas) == 13
    assert resultado["n"] == 10


def test_benchmark_inprocess_estrutura(diretorio_modelo, split_pequeno):
    _, teste = split_pequeno
    r = benchmark_inprocess(diretorio_modelo, list(teste["texto"]), n=20, warmup=2)
    for backend in ("sklearn", "onnx"):
        assert set(r[backend]) == {"latencia_ms", "carga_s", "tamanho_bytes"}
        assert r[backend]["latencia_ms"]["n"] == 20
    assert r["speedup_p50"] > 0


def test_relatorio_markdown(diretorio_modelo, split_pequeno):
    _, teste = split_pequeno
    resultado = {
        "ambiente": {"python": "3.12", "plataforma": "x", "cpus": 4, "bibliotecas": {}},
        "config": {"n": 20, "warmup": 2},
        "inprocess": benchmark_inprocess(diretorio_modelo, list(teste["texto"]), n=20, warmup=2),
    }
    md = gerar_relatorio_md(resultado)
    assert "| sklearn |" in md and "| onnx |" in md
    assert "P95" in md
