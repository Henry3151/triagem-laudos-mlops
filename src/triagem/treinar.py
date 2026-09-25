"""CLI de treino: python -m triagem.treinar (usado pelo build Docker e para uso local)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from triagem.pipeline import executar_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treina, avalia, exporta e promove o modelo.")
    parser.add_argument("--dados", type=Path, default=Path("data/raw/laudos_sinteticos.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--trabalho", type=Path, default=Path("data/processed/cli"))
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    resultado = executar_pipeline(args.dados, args.models_dir, args.trabalho, args.n_estimators)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
