"""Gera tráfego contra a API para alimentar o dashboard (inclui ~5% de payloads inválidos).

Usa só a stdlib para rodar também dentro da rede do Compose (serviço `carga`), onde não há o
atraso do port-forward do Docker Desktop no Windows.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import random
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


def carregar_textos(caminho: Path) -> list[str]:
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        return [linha["texto"] for linha in csv.DictReader(arquivo)]


def enviar(conexao: http.client.HTTPConnection, payload: dict) -> int:
    corpo = json.dumps(payload).encode("utf-8")
    conexao.request("POST", "/predict", body=corpo, headers={"Content-Type": "application/json"})
    resposta = conexao.getresponse()
    resposta.read()
    return resposta.status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rps", type=float, default=20.0)
    parser.add_argument("--duracao", type=float, default=120.0, help="segundos")
    parser.add_argument("--taxa-invalidos", type=float, default=0.05)
    parser.add_argument("--dados", type=Path, default=Path("data/raw/laudos_sinteticos.csv"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    textos = carregar_textos(args.dados)
    status: Counter[int] = Counter()
    intervalo = 1.0 / args.rps
    fim = time.monotonic() + args.duracao
    destino = urlsplit(args.url).netloc
    conexao = http.client.HTTPConnection(destino, timeout=5)
    while time.monotonic() < fim:
        inicio = time.monotonic()
        if rng.random() < args.taxa_invalidos:
            payload: dict = rng.choice([{"texto": ""}, {"texto": 42}, {}])
        else:
            payload = {"texto": rng.choice(textos)}
        try:
            status[enviar(conexao, payload)] += 1
        except (OSError, http.client.HTTPException):
            status[0] += 1
            conexao.close()
            conexao = http.client.HTTPConnection(destino, timeout=5)
        time.sleep(max(0.0, intervalo - (time.monotonic() - inicio)))
    conexao.close()
    print("Requisições por status:", dict(sorted(status.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
