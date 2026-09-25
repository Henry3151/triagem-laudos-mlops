"""Normalização de texto compartilhada entre treino e serving (evita training-serving skew)."""

import re
import unicodedata

_ESPACOS = re.compile(r"\s+")


def normalizar_texto(texto: str) -> str:
    """Minúsculas, sem acentos e com espaços colapsados.

    Remover acentos antes da vetorização também deixa a tokenização idêntica entre o
    scikit-learn e o ONNX Runtime.
    """
    sem_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return _ESPACOS.sub(" ", sem_acentos.lower()).strip()
