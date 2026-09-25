"""Benchmark de latência: modelo original (scikit-learn) vs. otimizado (ONNX Runtime).

Mede percentis (P50/P95/P99), não só a média, com batch 1 e warmup.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Callable, Sequence
from itertools import cycle, islice
from pathlib import Path

import numpy as np

from triagem.artefatos import ARQUIVO_JOBLIB, ARQUIVO_ONNX, versoes_bibliotecas
from triagem.classificadores import carregar_classificador

ARQUIVOS = {"sklearn": ARQUIVO_JOBLIB, "onnx": ARQUIVO_ONNX}


def percentis(amostras_ms: Sequence[float]) -> dict:
    dados = np.asarray(amostras_ms, dtype=float)
    p50, p95, p99 = np.percentile(dados, [50, 95, 99])
    return {
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "media": float(dados.mean()),
        "n": int(dados.size),
    }


def medir(
    funcao: Callable[[str], object], textos: Sequence[str], n: int = 1000, warmup: int = 50
) -> dict:
    for texto in islice(cycle(textos), warmup):
        funcao(texto)
    amostras = []
    for texto in islice(cycle(textos), n):
        inicio = time.perf_counter()
        funcao(texto)
        amostras.append((time.perf_counter() - inicio) * 1000)
    return percentis(amostras)


def _speedups(resultado: dict) -> dict:
    sk, ox = resultado["sklearn"]["latencia_ms"], resultado["onnx"]["latencia_ms"]
    return {
        "speedup_p50": sk["p50"] / ox["p50"],
        "speedup_p95": sk["p95"] / ox["p95"],
    }


def benchmark_inprocess(
    diretorio_modelo: Path, textos: Sequence[str], n: int = 1000, warmup: int = 50
) -> dict:
    resultado: dict = {}
    for backend, arquivo in ARQUIVOS.items():
        inicio = time.perf_counter()
        clf = carregar_classificador(Path(diretorio_modelo), backend)
        carga = time.perf_counter() - inicio
        resultado[backend] = {
            "latencia_ms": medir(clf.prever, textos, n, warmup),
            "carga_s": carga,
            "tamanho_bytes": (Path(diretorio_modelo) / arquivo).stat().st_size,
        }
    return resultado | _speedups(resultado)


def benchmark_http(
    urls: dict[str, str], textos: Sequence[str], n: int = 1000, warmup: int = 50
) -> dict:
    import httpx

    resultado: dict = {}
    for backend, url in urls.items():
        with httpx.Client(base_url=url, timeout=10) as cliente:

            def chamar(texto: str, cliente: httpx.Client = cliente) -> None:
                cliente.post("/predict", json={"texto": texto}).raise_for_status()

            resultado[backend] = {"latencia_ms": medir(chamar, textos, n, warmup)}
    return resultado | _speedups(resultado)


def _linha(backend: str, dados: dict) -> str:
    lat = dados["latencia_ms"]
    extras = ""
    if "carga_s" in dados:
        extras = f" {dados['carga_s'] * 1000:.1f} | {dados['tamanho_bytes'] / 1024:.0f} |"
    return (
        f"| {backend} | {lat['p50']:.3f} | {lat['p95']:.3f} | {lat['p99']:.3f} | "
        f"{lat['media']:.3f} |{extras}"
    )


def gerar_relatorio_md(resultado: dict) -> str:
    amb = resultado["ambiente"]
    linhas = [
        "# Benchmark de latência — scikit-learn vs. ONNX Runtime",
        "",
        f"- Python {amb['python']} em {amb['plataforma']} ({amb['cpus']} CPUs)",
        f"- Bibliotecas: {json.dumps(amb['bibliotecas'], ensure_ascii=False)}",
        f"- {resultado['config']['n']} chamadas com batch 1, após "
        f"{resultado['config']['warmup']} de warmup; 1 thread por backend",
        "",
    ]
    if "inprocess" in resultado:
        r = resultado["inprocess"]
        linhas += [
            "## Em processo (só a predição, incluindo o pré-processamento)",
            "",
            "| Backend | P50 (ms) | P95 (ms) | P99 (ms) | Média (ms) | Carga (ms) "
            "| Artefato (KiB) |",
            "|---|---|---|---|---|---|---|",
            _linha("sklearn", r["sklearn"]),
            _linha("onnx", r["onnx"]),
            "",
            f"**Speedup ONNX:** {r['speedup_p50']:.1f}x no P50 e {r['speedup_p95']:.1f}x no P95.",
            "",
        ]
    if "http" in resultado:
        r = resultado["http"]
        linhas += [
            "## Ponta a ponta via HTTP (API em Docker)",
            "",
            "| Backend | P50 (ms) | P95 (ms) | P99 (ms) | Média (ms) |",
            "|---|---|---|---|---|",
            _linha("sklearn", r["sklearn"]),
            _linha("onnx", r["onnx"]),
            "",
            f"**Speedup ONNX:** {r['speedup_p50']:.1f}x no P50 e {r['speedup_p95']:.1f}x no P95.",
            "",
        ]
    return "\n".join(linhas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark sklearn vs. ONNX Runtime.")
    parser.add_argument("--modo", choices=["inprocess", "http", "ambos"], default="inprocess")
    parser.add_argument("--model-dir", type=Path, default=Path("models/producao"))
    parser.add_argument("--dados", type=Path, default=Path("data/raw/laudos_sinteticos.csv"))
    parser.add_argument("--url-sklearn", default="http://localhost:8001")
    parser.add_argument("--url-onnx", default="http://localhost:8002")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--saida", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    from triagem.dados import carregar_dataset

    textos = list(carregar_dataset(args.dados)["texto"].sample(n=500, random_state=42))
    resultado: dict = {
        "ambiente": {
            "python": platform.python_version(),
            "plataforma": platform.platform(),
            "cpus": os.cpu_count(),
            "bibliotecas": versoes_bibliotecas(),
        },
        "config": {"n": args.n, "warmup": args.warmup},
    }
    if args.modo in ("inprocess", "ambos"):
        resultado["inprocess"] = benchmark_inprocess(args.model_dir, textos, args.n, args.warmup)
    if args.modo in ("http", "ambos"):
        urls = {"sklearn": args.url_sklearn, "onnx": args.url_onnx}
        resultado["http"] = benchmark_http(urls, textos, args.n, args.warmup)
    args.saida.mkdir(parents=True, exist_ok=True)
    (args.saida / "latencia.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    relatorio = gerar_relatorio_md(resultado)
    (args.saida / "latencia.md").write_text(relatorio, encoding="utf-8")
    print(relatorio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
