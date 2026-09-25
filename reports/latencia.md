# Benchmark de latência — scikit-learn vs. ONNX Runtime

- Python 3.12.14 em Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.41 (8 CPUs)
- Bibliotecas: {"scikit-learn": "1.9.1", "skl2onnx": null, "onnxruntime": "1.30.0", "numpy": "2.5.3"}
- 1000 chamadas com batch 1, após 50 de warmup; 1 thread por backend

## Em processo (só a predição, incluindo o pré-processamento)

| Backend | P50 (ms) | P95 (ms) | P99 (ms) | Média (ms) | Carga (ms) | Artefato (KiB) |
|---|---|---|---|---|---|---|
| sklearn | 13.062 | 18.218 | 25.163 | 13.869 | 1020.6 | 7868 |
| onnx | 0.112 | 0.183 | 0.258 | 0.122 | 107.5 | 4536 |

**Speedup ONNX:** 117.0x no P50 e 99.4x no P95.

## Ponta a ponta via HTTP (API em Docker)

| Backend | P50 (ms) | P95 (ms) | P99 (ms) | Média (ms) |
|---|---|---|---|---|
| sklearn | 15.231 | 20.492 | 28.013 | 16.076 |
| onnx | 2.156 | 2.961 | 3.887 | 2.249 |

**Speedup ONNX:** 7.1x no P50 e 6.9x no P95.
