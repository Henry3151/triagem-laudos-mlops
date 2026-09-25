from triagem.benchmark import benchmark_inprocess, gerar_relatorio_md, medir, percentis


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
