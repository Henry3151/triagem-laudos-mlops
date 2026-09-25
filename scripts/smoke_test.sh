#!/usr/bin/env bash
# Smoke test da imagem da API: health, predição válida, payload inválido e /metrics.
set -euo pipefail

IMAGEM="${1:-triagem-api:local}"
PORTA="${2:-8000}"
URL="http://localhost:${PORTA}"
RESPOSTA="$(mktemp)"

CID=$(docker run -d -p "${PORTA}:8000" "$IMAGEM")
trap 'docker rm -f "$CID" > /dev/null; rm -f "$RESPOSTA"' EXIT

for _ in $(seq 1 30); do
  if curl -fsS "$URL/health" > /dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS "$URL/health"
echo

# Payload só com ASCII: no Git Bash (Windows) argumentos não-ASCII não chegam em UTF-8.
LAUDO='{"texto": "Tomografia de cranio. Hematoma subdural agudo com efeito de massa."}'
status=$(curl -s -o "$RESPOSTA" -w '%{http_code}' -X POST "$URL/predict" \
  -H 'Content-Type: application/json' -d "$LAUDO")
cat "$RESPOSTA"
echo
[ "$status" = "200" ] || { echo "ERRO: /predict retornou $status"; docker logs "$CID"; exit 1; }
grep -q '"classe":"urgente"' "$RESPOSTA" || { echo "ERRO: classe inesperada"; exit 1; }

status=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/predict" \
  -H 'Content-Type: application/json' -d '{"texto": ""}')
[ "$status" = "422" ] || { echo "ERRO: payload inválido retornou $status"; exit 1; }

curl -fsS "$URL/metrics" | grep -q 'triagem_http_requests_total' \
  || { echo "ERRO: métricas ausentes"; exit 1; }

echo "Smoke test OK"
