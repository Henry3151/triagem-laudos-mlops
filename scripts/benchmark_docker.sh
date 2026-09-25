#!/usr/bin/env bash
# Benchmark sklearn vs. ONNX (em processo e via HTTP) inteiramente dentro de uma rede Docker.
# O cliente roda em container para evitar o atraso do port-forward do Docker Desktop.
# Uso: bash scripts/benchmark_docker.sh [imagem] [n]
set -euo pipefail

IMAGEM="${1:-triagem-api:local}"
N="${2:-1000}"
REDE="triagem-bench"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"

limpar() {
  docker rm -f triagem-sk triagem-ox > /dev/null 2>&1 || true
  docker network rm "$REDE" > /dev/null 2>&1 || true
}
trap limpar EXIT
limpar

docker network create "$REDE" > /dev/null
docker run -d --name triagem-sk --network "$REDE" -e MODEL_BACKEND=sklearn "$IMAGEM" > /dev/null
docker run -d --name triagem-ox --network "$REDE" -e MODEL_BACKEND=onnx "$IMAGEM" > /dev/null

for nome in triagem-sk triagem-ox; do
  for _ in $(seq 1 30); do
    if [ "$(docker inspect -f '{{.State.Health.Status}}' "$nome")" = "healthy" ]; then break; fi
    sleep 1
  done
done

mkdir -p "$RAIZ/reports"
MSYS_NO_PATHCONV=1 docker run --rm --network "$REDE" --user "$(id -u):$(id -g)" \
  -v "$RAIZ/data/raw:/dados:ro" \
  -v "$RAIZ/reports:/reports" \
  "$IMAGEM" python -m triagem.benchmark --modo ambos --n "$N" --warmup 50 \
  --dados /dados/laudos_sinteticos.csv \
  --url-sklearn http://triagem-sk:8000 --url-onnx http://triagem-ox:8000 \
  --saida /reports
