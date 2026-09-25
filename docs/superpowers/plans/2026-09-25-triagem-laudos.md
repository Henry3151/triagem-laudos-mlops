# Triagem de Laudos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar o Tech Challenge completo: classificador de urgência de laudos (TF-IDF + Random Forest) servido por FastAPI em Docker, otimizado com ONNX Runtime, retreinado por DAG Airflow, monitorado por Prometheus + Grafana e validado por GitHub Actions.

**Architecture:** Um pacote Python `triagem` (em `src/`) concentra toda a lógica: dados, modelo, exportação ONNX, artefatos, pipeline e API. CLI, DAG e imagem Docker são cascas finas que chamam as mesmas funções. O modelo é treinado dentro do build Docker (multi-stage) e pela DAG, sempre pelo mesmo `triagem.pipeline`.

**Tech Stack:** Python 3.12, uv 0.11, scikit-learn 1.9.1, skl2onnx 1.20.0, onnxruntime 1.30.0, pandas 3.0.6, FastAPI 0.141.1, prometheus-client 0.26.0, Airflow 3.3.2, Prometheus v3.5.0, Grafana 12.1.0, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-25-triagem-laudos-design.md`

## Global Constraints

- Python **3.12** em todo lugar: `.python-version`, `requires-python = ">=3.12,<3.13"`, imagens `python:3.12-slim` e `apache/airflow:3.3.2-python3.12`.
- Versões fixadas: scikit-learn 1.9.1, skl2onnx 1.20.0, onnx 1.23.0, onnxruntime 1.30.0, pandas 3.0.6, numpy 2.5.3, fastapi 0.141.1, uvicorn 0.54.0, prometheus-client 0.26.0, joblib 1.6.0, pydantic 2.13.5; dev: pytest 9.1.1, pytest-cov 7.1.0, httpx 0.28.1, ruff 0.16.9.
- Classes: exatamente `normal`, `atencao`, `urgente` (sem acento nos rótulos).
- Identificadores, mensagens e docs em português; código segue o `ruff` (line-length 100).
- O código de serving (`preprocessamento`, `onnx_runtime`, `classificadores`, `artefatos`, `api`) **não pode importar** pandas nem skl2onnx (a imagem runtime não os tem).
- Quality gate de promoção: F1-macro ≥ 0,80; recall de urgente ≥ 0,85; F1-macro ≥ F1 do campeão − 0,01.
- Paridade ONNX: concordância ≥ 0,99 e diferença máxima de probabilidade ≤ 1e-2.
- API: `texto` com 1 a 5.000 caracteres após o strip; 422 para inválido; 500 genérico para falha de inferência.
- Commits semânticos, com o trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Arquivos sempre com terminação LF (`.gitattributes`).

## Review Focus

- Texto com acentos, caixa mista ou espaços extras deve ter a **mesma** classificação que a versão normalizada: teste na Task 6 (`test_texto_com_acentos_equivale_a_normalizado`).
- JSON com `texto` numérico, sem o campo, ou só com espaços deve dar 422, nunca 500: testes na Task 8.
- Rota inexistente (404) não pode criar label de alta cardinalidade: `rota="desconhecida"`, testado na Task 8.
- `MODEL_DIR` sem modelo ou `MODEL_BACKEND` inválido devem falhar no startup com mensagem clara (caminho ou backend no texto): testes na Task 6.
- Promoção sem campeão (primeiro treino) deve promover se passar nos limiares, e reexecutar a DAG com o mesmo modelo não deve rebaixar a produção: testes na Task 5.

---

### Task 1: Scaffold do projeto e pré-processamento

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitattributes`, `README.md` (stub), `src/triagem/__init__.py`, `src/triagem/preprocessamento.py`
- Test: `tests/test_preprocessamento.py`

**Interfaces:**
- Produces: `triagem.__version__: str`; `triagem.preprocessamento.normalizar_texto(texto: str) -> str`

- [ ] **Step 1: Criar a configuração do projeto**

`pyproject.toml`:

```toml
[project]
name = "triagem"
version = "0.1.0"
description = "Triagem automática de laudos médicos por urgência (Tech Challenge MLOps)"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    "fastapi==0.141.1",
    "uvicorn==0.54.0",
    "prometheus-client==0.26.0",
    "pydantic==2.13.5",
    "scikit-learn==1.9.1",
    "joblib==1.6.0",
    "numpy==2.5.3",
    "onnxruntime==1.30.0",
]

[project.optional-dependencies]
treino = ["pandas==3.0.6", "skl2onnx==1.20.0", "onnx==1.23.0"]

[dependency-groups]
dev = ["pytest==9.1.1", "pytest-cov==7.1.0", "httpx==0.28.1", "ruff==0.16.9"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/triagem"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

`.python-version`: `3.12`

`.gitattributes`:

```
* text=auto eol=lf
*.png binary
```

`README.md` (stub; a Task 13 reescreve): `# Triagem de Laudos` e uma linha de descrição.

`src/triagem/__init__.py`:

```python
"""Triagem automática de laudos médicos por urgência."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Instalar o ambiente**

Run: `uv sync --all-extras`
Expected: cria `.venv` e `uv.lock` sem erros.

- [ ] **Step 3: Escrever o teste que falha**

`tests/test_preprocessamento.py`:

```python
from triagem.preprocessamento import normalizar_texto


def test_remove_acentos_e_caixa():
    assert normalizar_texto("Hemorragia INTRACRANIANA à Direita") == (
        "hemorragia intracraniana a direita"
    )


def test_colapsa_espacos_e_quebras_de_linha():
    assert normalizar_texto("  derrame\n\tpleural   leve  ") == "derrame pleural leve"


def test_cedilha_e_til():
    assert normalizar_texto("Coração e pulmões") == "coracao e pulmoes"


def test_string_vazia():
    assert normalizar_texto("") == ""
```

- [ ] **Step 4: Rodar e ver falhar**

Run: `uv run pytest tests/test_preprocessamento.py`
Expected: FAIL com `ModuleNotFoundError: No module named 'triagem.preprocessamento'`

- [ ] **Step 5: Implementar**

`src/triagem/preprocessamento.py`:

```python
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
```

- [ ] **Step 6: Rodar e ver passar; lint**

Run: `uv run pytest tests/test_preprocessamento.py && uv run ruff check . && uv run ruff format --check .`
Expected: 4 passed; ruff sem erros.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitattributes README.md src tests
git commit -m "build: scaffold do projeto com uv e normalização de texto"
```

### Task 2: Dataset sintético e contrato de dados

**Files:**
- Create: `src/triagem/dados.py`, `data/raw/laudos_sinteticos.csv` (gerado)
- Test: `tests/test_dados.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `CLASSES: tuple[str, ...] = ("normal", "atencao", "urgente")`
  - `class DatasetInvalidoError(ValueError)` com o atributo `violacoes: list[str]`
  - `gerar_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame` (colunas `id`, `texto`, `classe`)
  - `salvar_dataset(df: pd.DataFrame, caminho: Path) -> Path`
  - `carregar_dataset(caminho: Path) -> pd.DataFrame`
  - `validar_dataset(df: pd.DataFrame, minimo_linhas: int = 2000) -> None`
  - `dividir(df: pd.DataFrame, proporcao_teste: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]`
  - CLI: `python -m triagem.dados --saida data/raw/laudos_sinteticos.csv`

- [ ] **Step 1: Escrever os testes que falham**

`tests/test_dados.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from triagem.dados import (
    CLASSES,
    DatasetInvalidoError,
    carregar_dataset,
    dividir,
    gerar_dataset,
    salvar_dataset,
    validar_dataset,
)

RAIZ = Path(__file__).resolve().parents[1]
CSV_VERSIONADO = RAIZ / "data" / "raw" / "laudos_sinteticos.csv"


def test_gerador_deterministico():
    pd.testing.assert_frame_equal(gerar_dataset(n=300, seed=1), gerar_dataset(n=300, seed=1))


def test_seeds_diferentes_geram_textos_diferentes():
    assert not gerar_dataset(n=300, seed=1)["texto"].equals(gerar_dataset(n=300, seed=2)["texto"])


def test_esquema_tamanho_e_proporcoes():
    df = gerar_dataset(n=3000, seed=42)
    assert list(df.columns) == ["id", "texto", "classe"]
    assert len(df) == 3000
    assert df["id"].is_unique
    proporcoes = df["classe"].value_counts(normalize=True)
    for classe, esperado in {"normal": 0.45, "atencao": 0.35, "urgente": 0.20}.items():
        assert abs(proporcoes[classe] - esperado) <= 0.03


def test_dataset_gerado_e_valido():
    validar_dataset(gerar_dataset(n=3000, seed=42))


def test_csv_versionado_reproduz_o_gerador():
    df = carregar_dataset(CSV_VERSIONADO)
    pd.testing.assert_frame_equal(df, gerar_dataset(n=3000, seed=42))


def test_salvar_e_carregar_ida_e_volta(tmp_path):
    df = gerar_dataset(n=200, seed=3)
    caminho = salvar_dataset(df, tmp_path / "sub" / "x.csv")
    pd.testing.assert_frame_equal(carregar_dataset(caminho), df)


@pytest.mark.parametrize(
    ("mutacao", "trecho"),
    [
        (lambda d: d.drop(columns=["classe"]), "colunas ausentes"),
        (lambda d: d.assign(texto=d["texto"].where(d.index != 0, None)), "nulos"),
        (lambda d: d.assign(texto=d["texto"].where(d.index != 0, "   ")), "texto vazio"),
        (lambda d: d.assign(classe=d["classe"].where(d.index != 0, "grave")), "classes inválidas"),
        (lambda d: d.head(100), "mínimo"),
        (lambda d: d[d["classe"] != "urgente"], "classes ausentes"),
        (lambda d: d.assign(id=1), "ids duplicados"),
    ],
)
def test_validar_rejeita_violacoes(mutacao, trecho):
    df = mutacao(gerar_dataset(n=3000, seed=42))
    with pytest.raises(DatasetInvalidoError) as erro:
        validar_dataset(df)
    assert any(trecho in v for v in erro.value.violacoes)


def test_dividir_estratificado():
    df = gerar_dataset(n=1000, seed=5)
    treino, teste = dividir(df, proporcao_teste=0.2, seed=42)
    assert len(treino) == 800 and len(teste) == 200
    assert set(treino["id"]).isdisjoint(teste["id"])
    prop_total = df["classe"].value_counts(normalize=True)
    prop_teste = teste["classe"].value_counts(normalize=True)
    for classe in CLASSES:
        assert abs(prop_total[classe] - prop_teste[classe]) < 0.02
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_dados.py`
Expected: FAIL com `ModuleNotFoundError: No module named 'triagem.dados'`

- [ ] **Step 3: Implementar `src/triagem/dados.py`**

```python
"""Geração, carga, validação e divisão do dataset sintético de laudos."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES: tuple[str, ...] = ("normal", "atencao", "urgente")
PROPORCOES: dict[str, float] = {"normal": 0.45, "atencao": 0.35, "urgente": 0.20}
COLUNAS: tuple[str, ...] = ("id", "texto", "classe")
TAXA_RUIDO = 0.03

EXAMES = (
    "Radiografia de tórax em PA e perfil",
    "Tomografia computadorizada de crânio sem contraste",
    "Tomografia computadorizada de tórax",
    "Ultrassonografia de abdome total",
    "Eletrocardiograma de repouso",
    "Ressonância magnética de coluna lombar",
    "Angiotomografia de artérias pulmonares",
    "Ultrassonografia obstétrica",
)
INDICACOES = (
    "dor torácica",
    "cefaleia intensa",
    "controle de rotina",
    "trauma após queda",
    "dispneia progressiva",
    "dor abdominal",
    "febre persistente",
    "avaliação pré-operatória",
    "lombalgia crônica",
    "síncope",
)
ACHADOS: dict[str, tuple[str, ...]] = {
    "normal": (
        "Estruturas avaliadas com aspecto habitual.",
        "Parênquima pulmonar com transparência preservada.",
        "Área cardíaca dentro dos limites da normalidade.",
        "Ritmo sinusal com frequência cardíaca normal.",
        "Fígado de dimensões e ecotextura normais.",
        "Sistema ventricular de dimensões preservadas.",
        "Seios costofrênicos livres.",
        "Corpos vertebrais com altura e alinhamento preservados.",
        "Vesícula biliar de paredes finas, sem cálculos.",
        "Rins de forma, tamanho e contornos normais.",
        "Traçado sem alterações da repolarização ventricular.",
        "Aorta de calibre normal.",
    ),
    "atencao": (
        "Nódulo pulmonar sólido de 8 mm no lobo superior direito.",
        "Pequeno derrame pleural à esquerda.",
        "Esteatose hepática moderada.",
        "Discreta cardiomegalia.",
        "Cálculo renal não obstrutivo de 6 mm à direita.",
        "Protrusão discal L4-L5 com compressão radicular leve.",
        "Bloqueio de ramo direito incompleto.",
        "Espessamento parietal da vesícula biliar sem sinais de complicação.",
        "Opacidade em vidro fosco periférica de pequena extensão.",
        "Cisto ovariano complexo de 4 cm.",
        "Linfonodos mediastinais levemente aumentados.",
        "Extrassístoles ventriculares frequentes.",
    ),
    "urgente": (
        "Hemorragia intraparenquimatosa extensa com desvio da linha média.",
        "Pneumotórax hipertensivo à direita com desvio do mediastino.",
        "Supradesnivelamento do segmento ST em parede anterior.",
        "Falha de enchimento em artéria pulmonar principal compatível com tromboembolismo.",
        "Pneumoperitônio compatível com perfuração de víscera oca.",
        "Dissecção de aorta torácica tipo A.",
        "Apendicite aguda com sinais de perfuração e abscesso.",
        "Descolamento prematuro de placenta com sinais de sofrimento fetal.",
        "Fibrilação ventricular.",
        "Compressão medular aguda com sinais de mielopatia.",
        "Hematoma subdural agudo com efeito de massa.",
        "Aneurisma de aorta abdominal roto.",
    ),
}
# Negações compartilham vocabulário com achados urgentes: forçam o modelo a usar bigramas.
NEGACOES = (
    "Ausência de sinais de hemorragia.",
    "Sem evidência de pneumotórax.",
    "Não há sinais de perfuração.",
    "Sem desvio da linha média.",
    "Sem sinais de tromboembolismo.",
    "Não há supradesnivelamento do segmento ST.",
)
CONCLUSOES: dict[str, tuple[str, ...]] = {
    "normal": (
        "Exame dentro dos limites da normalidade.",
        "Sem alterações significativas.",
        "Estudo sem achados relevantes.",
    ),
    "atencao": (
        "Recomenda-se acompanhamento ambulatorial.",
        "Sugere-se correlação clínica e controle evolutivo.",
        "Achado que requer seguimento em consulta.",
    ),
    "urgente": (
        "Achado crítico: comunicado à equipe assistente.",
        "Necessita avaliação médica imediata.",
        "Resultado crítico, acionar protocolo de emergência.",
    ),
}


class DatasetInvalidoError(ValueError):
    """Dataset fora do contrato de dados (fail-fast antes do treino)."""

    def __init__(self, violacoes: list[str]):
        self.violacoes = violacoes
        super().__init__("Dataset inválido: " + "; ".join(violacoes))


def _achados(classe: str, rng: random.Random) -> list[str]:
    if classe == "normal":
        achados = rng.sample(ACHADOS["normal"], 2)
        if rng.random() < 0.5:
            achados.append(rng.choice(NEGACOES))
    elif classe == "atencao":
        achados = [rng.choice(ACHADOS["atencao"]), rng.choice(ACHADOS["normal"])]
        if rng.random() < 0.3:
            achados.append(rng.choice(NEGACOES))
    else:
        complemento = rng.choice(ACHADOS[rng.choice(("normal", "atencao"))])
        achados = [rng.choice(ACHADOS["urgente"]), complemento]
    rng.shuffle(achados)
    return achados


def _laudo(classe: str, rng: random.Random) -> str:
    partes = [
        f"{rng.choice(EXAMES)}.",
        f"Indicação: {rng.choice(INDICACOES)}.",
        *_achados(classe, rng),
    ]
    if rng.random() < 0.7:
        partes.append(f"Conclusão: {rng.choice(CONCLUSOES[classe])}")
    return " ".join(partes)


def gerar_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Gera `n` laudos sintéticos com ~3% de ruído de rótulo, de forma determinística."""
    rng = random.Random(seed)
    contagens = {c: round(n * p) for c, p in PROPORCOES.items()}
    contagens["normal"] += n - sum(contagens.values())
    classes = [c for c, k in contagens.items() for _ in range(k)]
    rng.shuffle(classes)
    textos, rotulos = [], []
    for classe in classes:
        textos.append(_laudo(classe, rng))
        if rng.random() < TAXA_RUIDO:
            classe = rng.choice([c for c in CLASSES if c != classe])
        rotulos.append(classe)
    ids = pd.Series(range(1, n + 1), dtype="int64")
    return pd.DataFrame({"id": ids, "texto": textos, "classe": rotulos})


def salvar_dataset(df: pd.DataFrame, caminho: Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8", lineterminator="\n")
    return caminho


def carregar_dataset(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho, encoding="utf-8")


def validar_dataset(df: pd.DataFrame, minimo_linhas: int = 2000) -> None:
    """Contrato de dados: lança DatasetInvalidoError com todas as violações encontradas."""
    ausentes = [c for c in COLUNAS if c not in df.columns]
    if ausentes:
        raise DatasetInvalidoError([f"colunas ausentes: {ausentes}"])
    violacoes: list[str] = []
    nulos = [c for c in COLUNAS if df[c].isna().any()]
    if nulos:
        violacoes.append(f"valores nulos em: {nulos}")
    vazios = int((df["texto"].fillna("x").astype(str).str.strip() == "").sum())
    if vazios:
        violacoes.append(f"texto vazio em {vazios} linha(s)")
    invalidas = sorted(set(df["classe"].dropna()) - set(CLASSES))
    if invalidas:
        violacoes.append(f"classes inválidas: {invalidas}")
    if len(df) < minimo_linhas:
        violacoes.append(f"{len(df)} linhas, abaixo do mínimo de {minimo_linhas}")
    faltando = sorted(set(CLASSES) - set(df["classe"].dropna()))
    if faltando:
        violacoes.append(f"classes ausentes: {faltando}")
    if not df["id"].is_unique:
        violacoes.append("ids duplicados")
    if violacoes:
        raise DatasetInvalidoError(violacoes)


def dividir(
    df: pd.DataFrame, proporcao_teste: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    treino, teste = train_test_split(
        df, test_size=proporcao_teste, stratify=df["classe"], random_state=seed
    )
    return treino.reset_index(drop=True), teste.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera o dataset sintético de laudos.")
    parser.add_argument("--saida", type=Path, default=Path("data/raw/laudos_sinteticos.csv"))
    parser.add_argument("-n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    caminho = salvar_dataset(gerar_dataset(args.n, args.seed), args.saida)
    print(f"Dataset salvo em {caminho}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Gerar o CSV versionado**

Run: `uv run python -m triagem.dados --saida data/raw/laudos_sinteticos.csv`
Expected: `Dataset salvo em data\raw\laudos_sinteticos.csv` (3.001 linhas incluindo o cabeçalho).

- [ ] **Step 5: Rodar e ver passar; lint**

Run: `uv run pytest tests/test_dados.py && uv run ruff check . && uv run ruff format --check .`
Expected: todos passam.

- [ ] **Step 6: Commit**

```bash
git add src/triagem/dados.py tests/test_dados.py data/raw/laudos_sinteticos.csv
git commit -m "feat: dataset sintético de laudos com contrato de dados"
```

### Task 3: Modelo TF-IDF + Random Forest

**Files:**
- Create: `src/triagem/modelo.py`, `tests/conftest.py`
- Test: `tests/test_modelo.py`

**Interfaces:**
- Consumes: `normalizar_texto`, `CLASSES`, `gerar_dataset`, `dividir`.
- Produces:
  - `TOKEN_PATTERN: str = r"\b\w\w+\b"`
  - `construir_pipeline(n_estimators: int = 200, seed: int = 42) -> sklearn.pipeline.Pipeline` (passos `tfidf` e `rf`)
  - `treinar(textos: Iterable[str], rotulos: Iterable[str], n_estimators: int = 200, seed: int = 42) -> Pipeline` (normaliza os textos antes do fit)
  - `avaliar(pipeline: Pipeline, textos: Iterable[str], rotulos: Iterable[str]) -> dict` com as chaves `acuracia`, `f1_macro`, `recall_urgente` (float), `matriz_confusao` (`{"rotulos": list[str], "valores": list[list[int]]}`) e `n_amostras` (int)
  - fixtures de sessão em `tests/conftest.py`: `split_pequeno -> tuple[DataFrame, DataFrame]` e `pipeline_treinado -> Pipeline`

- [ ] **Step 1: Criar as fixtures compartilhadas**

`tests/conftest.py`:

```python
import pytest

from triagem.dados import dividir, gerar_dataset
from triagem.modelo import treinar


@pytest.fixture(scope="session")
def split_pequeno():
    return dividir(gerar_dataset(n=800, seed=7), proporcao_teste=0.25, seed=42)


@pytest.fixture(scope="session")
def pipeline_treinado(split_pequeno):
    treino, _ = split_pequeno
    return treinar(treino["texto"], treino["classe"], n_estimators=40)
```

- [ ] **Step 2: Escrever os testes que falham**

`tests/test_modelo.py`:

```python
from triagem.modelo import avaliar, construir_pipeline


def test_pipeline_tem_tfidf_e_random_forest():
    pipe = construir_pipeline()
    assert list(pipe.named_steps) == ["tfidf", "rf"]
    assert pipe.named_steps["tfidf"].ngram_range == (1, 2)
    assert pipe.named_steps["rf"].n_jobs == 1


def test_modelo_treinado_atinge_limiar(pipeline_treinado, split_pequeno):
    _, teste = split_pequeno
    metricas = avaliar(pipeline_treinado, teste["texto"], teste["classe"])
    assert metricas["f1_macro"] >= 0.80
    assert metricas["recall_urgente"] >= 0.85


def test_metricas_tem_formato_esperado(pipeline_treinado, split_pequeno):
    _, teste = split_pequeno
    metricas = avaliar(pipeline_treinado, teste["texto"], teste["classe"])
    assert set(metricas) == {"acuracia", "f1_macro", "recall_urgente", "matriz_confusao", "n_amostras"}
    assert metricas["n_amostras"] == len(teste)
    assert metricas["matriz_confusao"]["rotulos"] == ["normal", "atencao", "urgente"]
    assert sum(map(sum, metricas["matriz_confusao"]["valores"])) == len(teste)


def test_classes_do_modelo(pipeline_treinado):
    assert sorted(pipeline_treinado.classes_) == ["atencao", "normal", "urgente"]
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_modelo.py`
Expected: FAIL com `ModuleNotFoundError: No module named 'triagem.modelo'`

- [ ] **Step 4: Implementar `src/triagem/modelo.py`**

```python
"""Construção, treino e avaliação do classificador TF-IDF + Random Forest."""

from __future__ import annotations

from collections.abc import Iterable

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline

from triagem.dados import CLASSES
from triagem.preprocessamento import normalizar_texto

# Mesmo padrão usado como `tokenexp` na conversão ONNX (paridade de tokenização).
TOKEN_PATTERN = r"\b\w\w+\b"


def construir_pipeline(n_estimators: int = 200, seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                    sublinear_tf=True,
                    token_pattern=TOKEN_PATTERN,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def treinar(
    textos: Iterable[str], rotulos: Iterable[str], n_estimators: int = 200, seed: int = 42
) -> Pipeline:
    pipeline = construir_pipeline(n_estimators=n_estimators, seed=seed)
    pipeline.fit([normalizar_texto(t) for t in textos], list(rotulos))
    return pipeline


def avaliar(pipeline: Pipeline, textos: Iterable[str], rotulos: Iterable[str]) -> dict:
    verdadeiros = list(rotulos)
    previstos = pipeline.predict([normalizar_texto(t) for t in textos])
    return {
        "acuracia": float(accuracy_score(verdadeiros, previstos)),
        "f1_macro": float(f1_score(verdadeiros, previstos, average="macro")),
        "recall_urgente": float(
            recall_score(verdadeiros, previstos, labels=["urgente"], average="macro")
        ),
        "matriz_confusao": {
            "rotulos": list(CLASSES),
            "valores": confusion_matrix(verdadeiros, previstos, labels=list(CLASSES)).tolist(),
        },
        "n_amostras": len(verdadeiros),
    }
```

- [ ] **Step 5: Rodar e ver passar; lint**

Run: `uv run pytest tests/test_modelo.py && uv run ruff check . && uv run ruff format --check .`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/triagem/modelo.py tests/conftest.py tests/test_modelo.py
git commit -m "feat: classificador TF-IDF + Random Forest com métricas de avaliação"
```

### Task 4: Exportação ONNX com verificação de paridade

**Files:**
- Create: `src/triagem/onnx_runtime.py` (só runtime: onnxruntime + numpy), `src/triagem/onnx_export.py` (treino: skl2onnx)
- Modify: `tests/conftest.py` (fixture `modelo_onnx`)
- Test: `tests/test_onnx_export.py`

**Interfaces:**
- Consumes: `normalizar_texto`, `TOKEN_PATTERN`, o `Pipeline` da Task 3.
- Produces:
  - `onnx_runtime.criar_sessao_onnx(modelo: bytes | str | Path, threads: int = 1) -> onnxruntime.InferenceSession`
  - `onnx_runtime.prever_probabilidades_onnx(sessao, textos_normalizados: Sequence[str]) -> np.ndarray` (shape `(n, n_classes)`, float64, na ordem de `pipeline.classes_`)
  - `onnx_export.ParidadeOnnxError(RuntimeError)`
  - `onnx_export.converter_para_onnx(pipeline: Pipeline) -> bytes`
  - `onnx_export.verificar_paridade(pipeline, modelo_onnx: bytes, textos: Iterable[str], min_concordancia: float = 0.99, max_diff_prob: float = 1e-2) -> dict` com `concordancia` e `max_diff_prob` (float); lança `ParidadeOnnxError` se violar
  - `onnx_export.exportar_onnx(pipeline, destino: Path, textos_validacao: Iterable[str]) -> dict` (grava o arquivo **só** se a paridade passar; devolve o dict de paridade)

- [ ] **Step 1: Adicionar a fixture ao `tests/conftest.py`** (no final do arquivo)

```python
@pytest.fixture(scope="session")
def modelo_onnx(pipeline_treinado) -> bytes:
    from triagem.onnx_export import converter_para_onnx

    return converter_para_onnx(pipeline_treinado)
```

- [ ] **Step 2: Escrever os testes que falham**

`tests/test_onnx_export.py`:

```python
import numpy as np
import pytest

from triagem.onnx_export import ParidadeOnnxError, exportar_onnx, verificar_paridade
from triagem.onnx_runtime import criar_sessao_onnx, prever_probabilidades_onnx
from triagem.preprocessamento import normalizar_texto


def test_paridade_com_sklearn(pipeline_treinado, modelo_onnx, split_pequeno):
    _, teste = split_pequeno
    paridade = verificar_paridade(pipeline_treinado, modelo_onnx, teste["texto"])
    assert paridade["concordancia"] >= 0.99
    assert paridade["max_diff_prob"] <= 1e-2


def test_probabilidades_somam_um(modelo_onnx, split_pequeno):
    _, teste = split_pequeno
    sessao = criar_sessao_onnx(modelo_onnx)
    probs = prever_probabilidades_onnx(sessao, [normalizar_texto(t) for t in teste["texto"][:20]])
    assert probs.shape == (20, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_paridade_violada_lanca_erro(pipeline_treinado, modelo_onnx, split_pequeno):
    _, teste = split_pequeno
    with pytest.raises(ParidadeOnnxError):
        verificar_paridade(
            pipeline_treinado, modelo_onnx, teste["texto"], min_concordancia=1.01
        )


def test_exportar_grava_arquivo(pipeline_treinado, split_pequeno, tmp_path):
    _, teste = split_pequeno
    destino = tmp_path / "modelo.onnx"
    paridade = exportar_onnx(pipeline_treinado, destino, teste["texto"])
    assert destino.stat().st_size > 0
    assert paridade["concordancia"] >= 0.99


def test_exportar_nao_grava_se_paridade_falha(pipeline_treinado, split_pequeno, tmp_path, monkeypatch):
    import triagem.onnx_export as modulo

    def falha(*_args, **_kwargs):
        raise ParidadeOnnxError("forçado")

    monkeypatch.setattr(modulo, "verificar_paridade", falha)
    destino = tmp_path / "modelo.onnx"
    _, teste = split_pequeno
    with pytest.raises(ParidadeOnnxError):
        exportar_onnx(pipeline_treinado, destino, teste["texto"])
    assert not destino.exists()
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_onnx_export.py`
Expected: FAIL com `ModuleNotFoundError: No module named 'triagem.onnx_export'`

- [ ] **Step 4: Implementar `src/triagem/onnx_runtime.py`**

```python
"""Inferência com ONNX Runtime (sem dependências de treino)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort

NOME_ENTRADA = "texto"


def criar_sessao_onnx(modelo: bytes | str | Path, threads: int = 1) -> ort.InferenceSession:
    opcoes = ort.SessionOptions()
    opcoes.intra_op_num_threads = threads
    opcoes.inter_op_num_threads = threads
    fonte = modelo if isinstance(modelo, bytes) else str(modelo)
    return ort.InferenceSession(fonte, opcoes, providers=["CPUExecutionProvider"])


def prever_probabilidades_onnx(
    sessao: ort.InferenceSession, textos_normalizados: Sequence[str]
) -> np.ndarray:
    entrada = np.array(list(textos_normalizados), dtype=object).reshape(-1, 1)
    _rotulos, probabilidades = sessao.run(None, {NOME_ENTRADA: entrada})
    return np.asarray(probabilidades, dtype=np.float64)
```

- [ ] **Step 5: Implementar `src/triagem/onnx_export.py`**

```python
"""Conversão do pipeline scikit-learn para ONNX, com verificação de paridade bloqueante."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

from triagem.modelo import TOKEN_PATTERN
from triagem.onnx_runtime import NOME_ENTRADA, criar_sessao_onnx, prever_probabilidades_onnx
from triagem.preprocessamento import normalizar_texto

TARGET_OPSET = 18


class ParidadeOnnxError(RuntimeError):
    """O modelo ONNX diverge do scikit-learn além da tolerância."""


def converter_para_onnx(pipeline: Pipeline) -> bytes:
    opcoes = {
        id(pipeline.named_steps["tfidf"]): {"tokenexp": TOKEN_PATTERN},
        id(pipeline.named_steps["rf"]): {"zipmap": False},
    }
    modelo = to_onnx(
        pipeline,
        initial_types=[(NOME_ENTRADA, StringTensorType([None, 1]))],
        options=opcoes,
        target_opset=TARGET_OPSET,
    )
    return modelo.SerializeToString()


def verificar_paridade(
    pipeline: Pipeline,
    modelo_onnx: bytes,
    textos: Iterable[str],
    min_concordancia: float = 0.99,
    max_diff_prob: float = 1e-2,
) -> dict:
    normalizados = [normalizar_texto(t) for t in textos]
    probs_sklearn = pipeline.predict_proba(normalizados)
    probs_onnx = prever_probabilidades_onnx(criar_sessao_onnx(modelo_onnx), normalizados)
    concordancia = float(np.mean(probs_sklearn.argmax(axis=1) == probs_onnx.argmax(axis=1)))
    diff = float(np.abs(probs_sklearn - probs_onnx).max())
    if concordancia < min_concordancia or diff > max_diff_prob:
        raise ParidadeOnnxError(
            f"Paridade ONNX violada: concordância {concordancia:.4f} (mín. {min_concordancia}), "
            f"diferença máxima {diff:.5f} (máx. {max_diff_prob})"
        )
    return {"concordancia": concordancia, "max_diff_prob": diff}


def exportar_onnx(pipeline: Pipeline, destino: Path, textos_validacao: Iterable[str]) -> dict:
    modelo = converter_para_onnx(pipeline)
    paridade = verificar_paridade(pipeline, modelo, textos_validacao)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(modelo)
    return paridade
```

- [ ] **Step 6: Rodar e ver passar; lint**

Run: `uv run pytest tests/test_onnx_export.py && uv run ruff check . && uv run ruff format --check .`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add src/triagem/onnx_runtime.py src/triagem/onnx_export.py tests/conftest.py tests/test_onnx_export.py
git commit -m "feat: exportação ONNX com verificação de paridade"
```

### Task 5: Artefatos versionados e quality gate de promoção

**Files:**
- Create: `src/triagem/artefatos.py`
- Test: `tests/test_artefatos.py`

**Interfaces:**
- Consumes: nada do projeto (só stdlib).
- Produces:
  - constantes `ARQUIVO_JOBLIB = "modelo.joblib"`, `ARQUIVO_ONNX = "modelo.onnx"`, `ARQUIVO_METADATA = "metadata.json"`, `ARQUIVO_METRICAS = "metricas.json"`, `DIR_PRODUCAO = "producao"`
  - `nova_versao(agora: datetime | None = None) -> str` (formato `YYYYMMDDTHHMMSSZ`, em UTC)
  - `sha256_arquivo(caminho: Path) -> str`
  - `salvar_json(caminho: Path, dados: dict) -> Path` / `ler_json(caminho: Path) -> dict`
  - `versoes_bibliotecas() -> dict[str, str | None]` (scikit-learn, skl2onnx, onnxruntime, numpy)
  - `@dataclass(frozen=True) DecisaoPromocao(promover: bool, motivo: str)`
  - `decidir_promocao(metricas: dict, metricas_campeao: dict | None, f1_minimo: float = 0.80, recall_urgente_minimo: float = 0.85, tolerancia: float = 0.01) -> DecisaoPromocao`
  - `promover(diretorio_versao: Path, models_dir: Path) -> Path` (substitui `models_dir/producao` inteiro e devolve o caminho)

- [ ] **Step 1: Escrever os testes que falham**

`tests/test_artefatos.py`:

```python
from datetime import UTC, datetime

from triagem.artefatos import (
    ARQUIVO_METADATA,
    DIR_PRODUCAO,
    decidir_promocao,
    ler_json,
    nova_versao,
    promover,
    salvar_json,
    sha256_arquivo,
    versoes_bibliotecas,
)

BOAS = {"f1_macro": 0.92, "recall_urgente": 0.95}


def test_nova_versao_formato():
    assert nova_versao(datetime(2026, 9, 25, 14, 3, 9, tzinfo=UTC)) == "20260925T140309Z"


def test_sha256_estavel(tmp_path):
    arquivo = tmp_path / "a.txt"
    arquivo.write_bytes(b"abc")
    assert sha256_arquivo(arquivo) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_json_ida_e_volta(tmp_path):
    caminho = salvar_json(tmp_path / "x" / "m.json", {"a": 1, "texto": "ação"})
    assert ler_json(caminho) == {"a": 1, "texto": "ação"}


def test_versoes_bibliotecas_tem_sklearn():
    assert versoes_bibliotecas()["scikit-learn"] == "1.9.1"


def test_primeiro_modelo_promove_se_passa_limiares():
    decisao = decidir_promocao(BOAS, None)
    assert decisao.promover
    assert "sem campeão" in decisao.motivo


def test_rejeita_f1_abaixo_do_minimo():
    decisao = decidir_promocao({"f1_macro": 0.70, "recall_urgente": 0.95}, None)
    assert not decisao.promover
    assert "F1-macro" in decisao.motivo


def test_rejeita_recall_urgente_baixo():
    decisao = decidir_promocao({"f1_macro": 0.90, "recall_urgente": 0.80}, None)
    assert not decisao.promover
    assert "recall" in decisao.motivo


def test_rejeita_quando_campeao_e_melhor():
    decisao = decidir_promocao(BOAS, {"f1_macro": 0.95, "recall_urgente": 0.97})
    assert not decisao.promover
    assert "campeão" in decisao.motivo


def test_empate_com_campeao_promove_dentro_da_tolerancia():
    assert decidir_promocao(BOAS, dict(BOAS)).promover


def test_promover_substitui_producao(tmp_path):
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / ARQUIVO_METADATA).write_text('{"versao": "v1"}')
    (v1 / "sobra.txt").write_text("x")
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / ARQUIVO_METADATA).write_text('{"versao": "v2"}')

    promover(v1, tmp_path)
    destino = promover(v2, tmp_path)

    assert destino == tmp_path / DIR_PRODUCAO
    assert ler_json(destino / ARQUIVO_METADATA) == {"versao": "v2"}
    assert not (destino / "sobra.txt").exists()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_artefatos.py`
Expected: FAIL com `ModuleNotFoundError: No module named 'triagem.artefatos'`

- [ ] **Step 3: Implementar `src/triagem/artefatos.py`**

```python
"""Versionamento de artefatos de modelo e quality gate de promoção (champion vs. challenger)."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

ARQUIVO_JOBLIB = "modelo.joblib"
ARQUIVO_ONNX = "modelo.onnx"
ARQUIVO_METADATA = "metadata.json"
ARQUIVO_METRICAS = "metricas.json"
DIR_PRODUCAO = "producao"
BIBLIOTECAS = ("scikit-learn", "skl2onnx", "onnxruntime", "numpy")


def nova_versao(agora: datetime | None = None) -> str:
    return (agora or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def sha256_arquivo(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def salvar_json(caminho: Path, dados: dict) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")
    return caminho


def ler_json(caminho: Path) -> dict:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def versoes_bibliotecas() -> dict[str, str | None]:
    versoes: dict[str, str | None] = {}
    for nome in BIBLIOTECAS:
        try:
            versoes[nome] = metadata.version(nome)
        except metadata.PackageNotFoundError:
            versoes[nome] = None
    return versoes


@dataclass(frozen=True)
class DecisaoPromocao:
    promover: bool
    motivo: str


def decidir_promocao(
    metricas: dict,
    metricas_campeao: dict | None,
    f1_minimo: float = 0.80,
    recall_urgente_minimo: float = 0.85,
    tolerancia: float = 0.01,
) -> DecisaoPromocao:
    f1 = metricas["f1_macro"]
    recall = metricas["recall_urgente"]
    if f1 < f1_minimo:
        return DecisaoPromocao(False, f"F1-macro {f1:.4f} abaixo do mínimo {f1_minimo}")
    if recall < recall_urgente_minimo:
        return DecisaoPromocao(
            False, f"recall de urgente {recall:.4f} abaixo do mínimo {recall_urgente_minimo}"
        )
    if metricas_campeao is None:
        return DecisaoPromocao(True, "sem campeão em produção; limiares atendidos")
    f1_campeao = metricas_campeao["f1_macro"]
    if f1 < f1_campeao - tolerancia:
        return DecisaoPromocao(
            False, f"F1-macro {f1:.4f} pior que o campeão ({f1_campeao:.4f}) além da tolerância"
        )
    return DecisaoPromocao(True, f"F1-macro {f1:.4f} ≥ campeão {f1_campeao:.4f} − {tolerancia}")


def promover(diretorio_versao: Path, models_dir: Path) -> Path:
    """Copia a versão para `models_dir/producao`, substituindo o conteúdo anterior."""
    models_dir = Path(models_dir)
    destino = models_dir / DIR_PRODUCAO
    temporario = models_dir / f".{DIR_PRODUCAO}_tmp"
    if temporario.exists():
        shutil.rmtree(temporario)
    shutil.copytree(diretorio_versao, temporario)
    if destino.exists():
        shutil.rmtree(destino)
    temporario.rename(destino)
    return destino
```

- [ ] **Step 4: Rodar e ver passar; lint**

Run: `uv run pytest tests/test_artefatos.py && uv run ruff check . && uv run ruff format --check .`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/triagem/artefatos.py tests/test_artefatos.py
git commit -m "feat: versionamento de artefatos e quality gate de promoção"
```

### Task 6: Classificadores de serving (sklearn e ONNX)

**Files:**
- Create: `src/triagem/classificadores.py`
- Modify: `tests/conftest.py` (fixture `diretorio_modelo`)
- Test: `tests/test_classificadores.py`

**Interfaces:**
- Consumes: `normalizar_texto`, `criar_sessao_onnx`, `prever_probabilidades_onnx`, as constantes de `artefatos` e `ler_json`.
- Produces:
  - `@dataclass(frozen=True) Predicao(classe: str, probabilidades: dict[str, float])`
  - `class Classificador(Protocol)` com os atributos `backend: str`, `versao: str` e o método `prever(texto: str) -> Predicao`
  - `SklearnClassificador(pipeline, versao: str)` com `backend = "sklearn"`
  - `OnnxClassificador(sessao, classes: Sequence[str], versao: str)` com `backend = "onnx"`
  - `BACKENDS = ("onnx", "sklearn")`
  - `carregar_classificador(diretorio: Path, backend: str = "onnx") -> Classificador`: `ValueError` para backend inválido; `FileNotFoundError` citando o caminho se faltar artefato. Lê `versao` e `classes` do `metadata.json`.
  - fixture `diretorio_modelo -> Path`: um diretório com `modelo.joblib`, `modelo.onnx` e `metadata.json`

- [ ] **Step 1: Adicionar a fixture ao `tests/conftest.py`** (no final do arquivo)

```python
@pytest.fixture(scope="session")
def diretorio_modelo(tmp_path_factory, pipeline_treinado, modelo_onnx):
    import joblib

    from triagem.artefatos import ARQUIVO_JOBLIB, ARQUIVO_METADATA, ARQUIVO_ONNX, salvar_json

    diretorio = tmp_path_factory.mktemp("modelo")
    joblib.dump(pipeline_treinado, diretorio / ARQUIVO_JOBLIB)
    (diretorio / ARQUIVO_ONNX).write_bytes(modelo_onnx)
    salvar_json(
        diretorio / ARQUIVO_METADATA,
        {"versao": "teste-v1", "classes": [str(c) for c in pipeline_treinado.classes_]},
    )
    return diretorio
```

- [ ] **Step 2: Escrever os testes que falham**

`tests/test_classificadores.py`:

```python
import math

import pytest

from triagem.classificadores import carregar_classificador

URGENTE = "Tomografia de crânio. Hemorragia intraparenquimatosa extensa com desvio da linha média."
NORMAL = "Radiografia de tórax. Parênquima pulmonar com transparência preservada. Seios costofrênicos livres."


@pytest.mark.parametrize("backend", ["sklearn", "onnx"])
def test_carrega_e_prediz(diretorio_modelo, backend):
    clf = carregar_classificador(diretorio_modelo, backend)
    assert clf.backend == backend
    assert clf.versao == "teste-v1"
    predicao = clf.prever(URGENTE)
    assert predicao.classe == "urgente"
    assert set(predicao.probabilidades) == {"atencao", "normal", "urgente"}
    assert math.isclose(sum(predicao.probabilidades.values()), 1.0, abs_tol=1e-5)


def test_backends_concordam(diretorio_modelo, split_pequeno):
    _, teste = split_pequeno
    sk = carregar_classificador(diretorio_modelo, "sklearn")
    ox = carregar_classificador(diretorio_modelo, "onnx")
    textos = list(teste["texto"][:100])
    iguais = sum(sk.prever(t).classe == ox.prever(t).classe for t in textos)
    assert iguais / len(textos) >= 0.99


@pytest.mark.parametrize("backend", ["sklearn", "onnx"])
def test_texto_com_acentos_equivale_a_normalizado(diretorio_modelo, backend):
    clf = carregar_classificador(diretorio_modelo, backend)
    original = clf.prever("  RADIOGRAFIA de Tórax.   Parênquima pulmonar com transparência PRESERVADA. ")
    normalizado = clf.prever("radiografia de torax. parenquima pulmonar com transparencia preservada.")
    assert original == normalizado


def test_backend_invalido(diretorio_modelo):
    with pytest.raises(ValueError, match="tensorflow"):
        carregar_classificador(diretorio_modelo, "tensorflow")


def test_diretorio_sem_modelo(tmp_path):
    with pytest.raises(FileNotFoundError, match=str(tmp_path.name)):
        carregar_classificador(tmp_path, "onnx")


def test_normal_classificado_como_normal(diretorio_modelo):
    assert carregar_classificador(diretorio_modelo, "onnx").prever(NORMAL).classe == "normal"
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_classificadores.py`
Expected: FAIL com `ModuleNotFoundError: No module named 'triagem.classificadores'`

- [ ] **Step 4: Implementar `src/triagem/classificadores.py`**

```python
"""Backends de inferência usados pela API: scikit-learn (original) e ONNX Runtime (otimizado)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

from triagem.artefatos import ARQUIVO_JOBLIB, ARQUIVO_METADATA, ARQUIVO_ONNX, ler_json
from triagem.onnx_runtime import criar_sessao_onnx, prever_probabilidades_onnx
from triagem.preprocessamento import normalizar_texto

BACKENDS = ("onnx", "sklearn")


@dataclass(frozen=True)
class Predicao:
    classe: str
    probabilidades: dict[str, float]


class Classificador(Protocol):
    backend: str
    versao: str

    def prever(self, texto: str) -> Predicao: ...


def _montar_predicao(classes: Sequence[str], probabilidades: np.ndarray) -> Predicao:
    mapa = {c: float(p) for c, p in zip(classes, probabilidades, strict=True)}
    return Predicao(classe=max(mapa, key=mapa.__getitem__), probabilidades=mapa)


class SklearnClassificador:
    backend = "sklearn"

    def __init__(self, pipeline, versao: str):
        self._pipeline = pipeline
        self._classes = [str(c) for c in pipeline.classes_]
        self.versao = versao

    def prever(self, texto: str) -> Predicao:
        probs = self._pipeline.predict_proba([normalizar_texto(texto)])[0]
        return _montar_predicao(self._classes, probs)


class OnnxClassificador:
    backend = "onnx"

    def __init__(self, sessao, classes: Sequence[str], versao: str):
        self._sessao = sessao
        self._classes = list(classes)
        self.versao = versao

    def prever(self, texto: str) -> Predicao:
        probs = prever_probabilidades_onnx(self._sessao, [normalizar_texto(texto)])[0]
        return _montar_predicao(self._classes, probs)


def _exigir(caminho: Path) -> Path:
    if not caminho.is_file():
        raise FileNotFoundError(f"Artefato de modelo não encontrado: {caminho}")
    return caminho


def carregar_classificador(diretorio: Path, backend: str = "onnx") -> Classificador:
    if backend not in BACKENDS:
        raise ValueError(f"Backend '{backend}' inválido; use um de {BACKENDS}")
    diretorio = Path(diretorio)
    metadata = ler_json(_exigir(diretorio / ARQUIVO_METADATA))
    if backend == "sklearn":
        pipeline = joblib.load(_exigir(diretorio / ARQUIVO_JOBLIB))
        return SklearnClassificador(pipeline, metadata["versao"])
    sessao = criar_sessao_onnx(_exigir(diretorio / ARQUIVO_ONNX))
    return OnnxClassificador(sessao, metadata["classes"], metadata["versao"])
```

- [ ] **Step 5: Rodar e ver passar; lint**

Run: `uv run pytest tests/test_classificadores.py && uv run ruff check . && uv run ruff format --check .`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/triagem/classificadores.py tests/conftest.py tests/test_classificadores.py
git commit -m "feat: backends de inferência sklearn e ONNX Runtime"
```

### Task 7: Etapas do pipeline de treino e CLI

**Files:**
- Create: `src/triagem/pipeline.py`, `src/triagem/treinar.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `carregar_dataset`, `validar_dataset`, `dividir`, `salvar_dataset`, `treinar`, `avaliar`, `exportar_onnx` e tudo de `artefatos`.
- Produces (as funções devolvem só tipos JSON-serializáveis, porque viram XCom na DAG):
  - `ingerir(caminho_csv: Path, destino: Path, seed: int = 42) -> dict[str, str]` com as chaves `treino`, `teste` (caminhos) e `sha256_dados`
  - `treinar_versao(caminho_treino: Path, models_dir: Path, versao: str | None = None, n_estimators: int = 200) -> str` (diretório da versão)
  - `avaliar_versao(diretorio_versao: Path, caminho_teste: Path) -> dict` (métricas; grava `metricas.json`)
  - `exportar_versao(diretorio_versao: Path, caminho_teste: Path, metricas: dict, sha256_dados: str) -> dict` (metadata; grava `modelo.onnx` e `metadata.json`)
  - `promover_se_aprovado(diretorio_versao: Path, models_dir: Path) -> dict` com as chaves `promovido: bool`, `motivo: str` e `versao: str`
  - `executar_pipeline(caminho_csv: Path, models_dir: Path, trabalho_dir: Path, n_estimators: int = 200) -> dict` com as chaves `versao`, `metricas`, `paridade_onnx` e `promocao`
  - CLI: `python -m triagem.treinar [--dados CSV] [--models-dir DIR] [--trabalho DIR] [--n-estimators N]`, que imprime o JSON do resultado e retorna 0

- [ ] **Step 1: Escrever os testes que falham**

`tests/test_pipeline.py`:

```python
import json

import pytest

from triagem.artefatos import (
    ARQUIVO_JOBLIB,
    ARQUIVO_METADATA,
    ARQUIVO_METRICAS,
    ARQUIVO_ONNX,
    DIR_PRODUCAO,
    ler_json,
)
from triagem.dados import DatasetInvalidoError, gerar_dataset, salvar_dataset
from triagem.pipeline import (
    avaliar_versao,
    executar_pipeline,
    exportar_versao,
    ingerir,
    promover_se_aprovado,
    treinar_versao,
)
from triagem.treinar import main


@pytest.fixture(scope="module")
def csv_dados(tmp_path_factory):
    return salvar_dataset(gerar_dataset(n=2000, seed=11), tmp_path_factory.mktemp("d") / "d.csv")


def test_etapas_encadeadas(csv_dados, tmp_path):
    split = ingerir(csv_dados, tmp_path / "proc")
    assert set(split) == {"treino", "teste", "sha256_dados"}
    diretorio = treinar_versao(split["treino"], tmp_path / "models", versao="v1", n_estimators=30)
    assert (tmp_path / "models" / "v1" / ARQUIVO_JOBLIB).is_file()
    metricas = avaliar_versao(diretorio, split["teste"])
    assert ler_json(tmp_path / "models" / "v1" / ARQUIVO_METRICAS) == metricas
    metadata = exportar_versao(diretorio, split["teste"], metricas, split["sha256_dados"])
    assert metadata["versao"] == "v1"
    assert metadata["classes"] == ["atencao", "normal", "urgente"]
    assert metadata["paridade_onnx"]["concordancia"] >= 0.99
    assert (tmp_path / "models" / "v1" / ARQUIVO_ONNX).is_file()
    decisao = promover_se_aprovado(diretorio, tmp_path / "models")
    assert decisao == {"promovido": True, "motivo": decisao["motivo"], "versao": "v1"}
    assert (tmp_path / "models" / DIR_PRODUCAO / ARQUIVO_METADATA).is_file()
    json.dumps([split, metricas, metadata, decisao])  # tudo serializável para XCom


def test_reexecucao_com_mesmo_modelo_mantem_producao(csv_dados, tmp_path):
    models = tmp_path / "models"
    primeiro = executar_pipeline(csv_dados, models, tmp_path / "t1", n_estimators=30)
    segundo = executar_pipeline(csv_dados, models, tmp_path / "t2", n_estimators=30)
    assert primeiro["promocao"]["promovido"]
    assert segundo["promocao"]["promovido"]  # empate com o campeão fica dentro da tolerância
    assert ler_json(models / DIR_PRODUCAO / ARQUIVO_METADATA)["versao"] == segundo["versao"]


def test_rejeita_challenger_pior(csv_dados, tmp_path):
    models = tmp_path / "models"
    executar_pipeline(csv_dados, models, tmp_path / "t1", n_estimators=30)
    campeao = ler_json(models / DIR_PRODUCAO / ARQUIVO_METADATA)
    campeao["metricas"]["f1_macro"] = 0.9999
    (models / DIR_PRODUCAO / ARQUIVO_METADATA).write_text(json.dumps(campeao), encoding="utf-8")
    resultado = executar_pipeline(csv_dados, models, tmp_path / "t2", n_estimators=30)
    assert not resultado["promocao"]["promovido"]
    assert ler_json(models / DIR_PRODUCAO / ARQUIVO_METADATA)["versao"] == campeao["versao"]


def test_ingerir_falha_com_dataset_invalido(tmp_path):
    csv = salvar_dataset(gerar_dataset(n=500, seed=1), tmp_path / "pequeno.csv")
    with pytest.raises(DatasetInvalidoError):
        ingerir(csv, tmp_path / "proc")


def test_cli(csv_dados, tmp_path, capsys):
    codigo = main(
        [
            "--dados", str(csv_dados),
            "--models-dir", str(tmp_path / "models"),
            "--trabalho", str(tmp_path / "trab"),
            "--n-estimators", "20",
        ]
    )
    assert codigo == 0
    saida = json.loads(capsys.readouterr().out)
    assert saida["promocao"]["promovido"] is True
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_pipeline.py`
Expected: FAIL com `ModuleNotFoundError: No module named 'triagem.pipeline'`

- [ ] **Step 3: Implementar `src/triagem/pipeline.py`**

```python
"""Etapas do pipeline de treino, reutilizadas pela DAG do Airflow e pelo CLI.

Cada etapa recebe e devolve apenas caminhos e dicionários JSON-serializáveis (XCom leve).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib

from triagem.artefatos import (
    ARQUIVO_JOBLIB,
    ARQUIVO_METADATA,
    ARQUIVO_METRICAS,
    ARQUIVO_ONNX,
    DIR_PRODUCAO,
    decidir_promocao,
    ler_json,
    nova_versao,
    promover,
    salvar_json,
    sha256_arquivo,
    versoes_bibliotecas,
)
from triagem.dados import carregar_dataset, dividir, salvar_dataset, validar_dataset
from triagem.modelo import avaliar, treinar
from triagem.onnx_export import exportar_onnx

logger = logging.getLogger(__name__)


def ingerir(caminho_csv: Path, destino: Path, seed: int = 42) -> dict[str, str]:
    df = carregar_dataset(caminho_csv)
    validar_dataset(df)
    treino, teste = dividir(df, seed=seed)
    destino = Path(destino)
    caminho_treino = salvar_dataset(treino, destino / "treino.csv")
    caminho_teste = salvar_dataset(teste, destino / "teste.csv")
    logger.info("Dados ingeridos: %d treino, %d teste", len(treino), len(teste))
    return {
        "treino": str(caminho_treino),
        "teste": str(caminho_teste),
        "sha256_dados": sha256_arquivo(caminho_csv),
    }


def treinar_versao(
    caminho_treino: Path,
    models_dir: Path,
    versao: str | None = None,
    n_estimators: int = 200,
) -> str:
    df = carregar_dataset(caminho_treino)
    pipeline = treinar(df["texto"], df["classe"], n_estimators=n_estimators)
    # nova_versao tem resolução de segundo: sufixo evita colisão em execuções seguidas.
    base = Path(models_dir) / (versao or nova_versao())
    diretorio, sufixo = base, 2
    while diretorio.exists():
        diretorio, sufixo = base.with_name(f"{base.name}-{sufixo}"), sufixo + 1
    diretorio.mkdir(parents=True)
    joblib.dump(pipeline, diretorio / ARQUIVO_JOBLIB)
    logger.info("Modelo treinado em %s", diretorio)
    return str(diretorio)


def avaliar_versao(diretorio_versao: Path, caminho_teste: Path) -> dict:
    diretorio = Path(diretorio_versao)
    pipeline = joblib.load(diretorio / ARQUIVO_JOBLIB)
    df = carregar_dataset(caminho_teste)
    metricas = avaliar(pipeline, df["texto"], df["classe"])
    salvar_json(diretorio / ARQUIVO_METRICAS, metricas)
    logger.info(
        "Métricas: F1-macro=%.4f recall_urgente=%.4f",
        metricas["f1_macro"],
        metricas["recall_urgente"],
    )
    return metricas


def exportar_versao(
    diretorio_versao: Path, caminho_teste: Path, metricas: dict, sha256_dados: str
) -> dict:
    diretorio = Path(diretorio_versao)
    pipeline = joblib.load(diretorio / ARQUIVO_JOBLIB)
    df = carregar_dataset(caminho_teste)
    paridade = exportar_onnx(pipeline, diretorio / ARQUIVO_ONNX, df["texto"])
    metadata = {
        "versao": diretorio.name,
        "criado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "classes": [str(c) for c in pipeline.classes_],
        "sha256_dados": sha256_dados,
        "metricas": metricas,
        "paridade_onnx": paridade,
        "bibliotecas": versoes_bibliotecas(),
    }
    salvar_json(diretorio / ARQUIVO_METADATA, metadata)
    return metadata


def promover_se_aprovado(diretorio_versao: Path, models_dir: Path) -> dict:
    diretorio = Path(diretorio_versao)
    metadata = ler_json(diretorio / ARQUIVO_METADATA)
    campeao_path = Path(models_dir) / DIR_PRODUCAO / ARQUIVO_METADATA
    campeao = ler_json(campeao_path)["metricas"] if campeao_path.is_file() else None
    decisao = decidir_promocao(metadata["metricas"], campeao)
    if decisao.promover:
        promover(diretorio, models_dir)
    logger.info("Promoção de %s: %s (%s)", metadata["versao"], decisao.promover, decisao.motivo)
    return {"promovido": decisao.promover, "motivo": decisao.motivo, "versao": metadata["versao"]}


def executar_pipeline(
    caminho_csv: Path, models_dir: Path, trabalho_dir: Path, n_estimators: int = 200
) -> dict:
    split = ingerir(caminho_csv, trabalho_dir)
    diretorio = treinar_versao(split["treino"], models_dir, n_estimators=n_estimators)
    metricas = avaliar_versao(diretorio, split["teste"])
    metadata = exportar_versao(diretorio, split["teste"], metricas, split["sha256_dados"])
    promocao = promover_se_aprovado(diretorio, models_dir)
    return {
        "versao": metadata["versao"],
        "metricas": metricas,
        "paridade_onnx": metadata["paridade_onnx"],
        "promocao": promocao,
    }
```

- [ ] **Step 4: Implementar `src/triagem/treinar.py`**

```python
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
```

- [ ] **Step 5: Rodar e ver passar; rodar a suíte inteira; lint**

Run: `uv run pytest && uv run ruff check . && uv run ruff format --check .`
Expected: todos passam.

- [ ] **Step 6: Treinar o modelo real localmente**

Run: `uv run python -m triagem.treinar`
Expected: JSON com `"promovido": true`, F1-macro entre 0,85 e 0,97 e `models/producao/` preenchido. Se o F1 ficar fora da faixa, ajuste `TAXA_RUIDO` ou a probabilidade de conclusão em `dados.py`, regenere o CSV e rode os testes de novo.

- [ ] **Step 7: Commit**

```bash
git add src/triagem/pipeline.py src/triagem/treinar.py tests/test_pipeline.py
git commit -m "feat: etapas do pipeline de treino e CLI"
```

### Task 8: API FastAPI instrumentada com Prometheus

**Files:**
- Create: `src/triagem/api/__init__.py` (vazio), `src/triagem/api/schemas.py`, `src/triagem/api/metricas.py`, `src/triagem/api/app.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Classificador`, `Predicao`, `carregar_classificador`, `triagem.__version__`.
- Produces:
  - `schemas.PedidoTriagem(texto: str)`, `schemas.RespostaTriagem(classe, probabilidades, versao_modelo, backend)`, `schemas.RespostaSaude(status, versao_modelo, backend)`
  - `metricas.MetricasApi(registry: CollectorRegistry | None = None)` com os atributos `registry`, `requisicoes`, `duracao`, `inferencia`, `predicoes` e `modelo`
  - `app.criar_app(classificador: Classificador | None = None) -> FastAPI`; sem classificador injetado, carrega de `MODEL_DIR` (padrão `models/producao`) e `MODEL_BACKEND` (padrão `onnx`) no lifespan
  - `app.app`: instância do módulo, usada pelo uvicorn (`triagem.api.app:app`)
  - Nomes das métricas expostas: `triagem_http_requests_total`, `triagem_http_request_duration_seconds`, `triagem_inferencia_duration_seconds`, `triagem_predicoes_total` e `triagem_modelo_info`

- [ ] **Step 1: Escrever os testes que falham**

`tests/test_api.py`:

```python
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
    resposta = cliente.post("/predict", content=b"nao e json", headers={"Content-Type": "application/json"})
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_api.py`
Expected: FAIL com `ModuleNotFoundError: No module named 'triagem.api'`

- [ ] **Step 3: Implementar `src/triagem/api/schemas.py`**

```python
"""Contratos de entrada e saída da API."""

from pydantic import BaseModel, ConfigDict, Field

EXEMPLO = (
    "Tomografia computadorizada de crânio sem contraste. Indicação: trauma após queda. "
    "Hematoma subdural agudo com efeito de massa."
)


class PedidoTriagem(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True, json_schema_extra={"examples": [{"texto": EXEMPLO}]}
    )

    texto: str = Field(min_length=1, max_length=5000, description="Texto livre do laudo")


class RespostaTriagem(BaseModel):
    classe: str
    probabilidades: dict[str, float]
    versao_modelo: str
    backend: str


class RespostaSaude(BaseModel):
    status: str
    versao_modelo: str
    backend: str
```

- [ ] **Step 4: Implementar `src/triagem/api/metricas.py`**

```python
"""Métricas Prometheus da API (registry próprio por instância de app)."""

from prometheus_client import CollectorRegistry, Counter, Histogram, Info, ProcessCollector

BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class MetricasApi:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        ProcessCollector(registry=self.registry)
        self.requisicoes = Counter(
            "triagem_http_requests",
            "Total de requisições HTTP",
            ["method", "rota", "status"],
            registry=self.registry,
        )
        self.duracao = Histogram(
            "triagem_http_request_duration_seconds",
            "Latência das requisições HTTP",
            ["method", "rota"],
            buckets=BUCKETS,
            registry=self.registry,
        )
        self.inferencia = Histogram(
            "triagem_inferencia_duration_seconds",
            "Tempo apenas da predição do modelo",
            ["backend"],
            buckets=BUCKETS,
            registry=self.registry,
        )
        self.predicoes = Counter(
            "triagem_predicoes",
            "Predições por classe",
            ["classe"],
            registry=self.registry,
        )
        self.modelo = Info("triagem_modelo", "Versão e backend do modelo", registry=self.registry)
```

- [ ] **Step 5: Implementar `src/triagem/api/app.py`**

```python
"""API de triagem de laudos (FastAPI)."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.routing import Match

from triagem import __version__
from triagem.api.metricas import MetricasApi
from triagem.api.schemas import PedidoTriagem, RespostaSaude, RespostaTriagem
from triagem.classificadores import Classificador, carregar_classificador

logger = logging.getLogger("triagem.api")
ROTA_DESCONHECIDA = "desconhecida"


def _rota(request: Request) -> str:
    """Template da rota (ex.: /predict) para manter baixa a cardinalidade dos labels."""
    for rota in request.app.router.routes:
        correspondencia, _ = rota.matches(request.scope)
        if correspondencia == Match.FULL:
            return rota.path
    return ROTA_DESCONHECIDA


def criar_app(classificador: Classificador | None = None) -> FastAPI:
    metricas = MetricasApi()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        clf = classificador or carregar_classificador(
            Path(os.environ.get("MODEL_DIR", "models/producao")),
            os.environ.get("MODEL_BACKEND", "onnx"),
        )
        app.state.classificador = clf
        metricas.modelo.info({"backend": clf.backend, "versao": clf.versao})
        logger.info("Modelo %s carregado (backend=%s)", clf.versao, clf.backend)
        yield

    app = FastAPI(
        title="Triagem de Laudos",
        description="Classifica laudos médicos em normal, atenção ou urgente.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def medir_requisicoes(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        inicio = time.perf_counter()
        status = 500
        try:
            resposta = await call_next(request)
            status = resposta.status_code
            return resposta
        finally:
            rota = _rota(request)
            metricas.requisicoes.labels(request.method, rota, str(status)).inc()
            metricas.duracao.labels(request.method, rota).observe(time.perf_counter() - inicio)

    @app.post("/predict", response_model=RespostaTriagem)
    def predict(pedido: PedidoTriagem, request: Request) -> RespostaTriagem:
        clf: Classificador = request.app.state.classificador
        inicio = time.perf_counter()
        try:
            predicao = clf.prever(pedido.texto)
        except Exception as erro:
            logger.exception("Falha na inferência")
            raise HTTPException(500, "Erro interno ao classificar o laudo") from erro
        metricas.inferencia.labels(clf.backend).observe(time.perf_counter() - inicio)
        metricas.predicoes.labels(predicao.classe).inc()
        return RespostaTriagem(
            classe=predicao.classe,
            probabilidades=predicao.probabilidades,
            versao_modelo=clf.versao,
            backend=clf.backend,
        )

    @app.get("/health", response_model=RespostaSaude)
    def health(request: Request) -> RespostaSaude:
        clf: Classificador = request.app.state.classificador
        return RespostaSaude(status="ok", versao_modelo=clf.versao, backend=clf.backend)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(metricas.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = criar_app()
```

- [ ] **Step 6: Rodar e ver passar; lint**

Run: `uv run pytest tests/test_api.py && uv run ruff check . && uv run ruff format --check .`
Expected: todos passam. Se o label de rota inexistente não aparecer como `desconhecida`, confira se `_rota` roda **depois** do `call_next` (ela roda no `finally`).

- [ ] **Step 7: Testar a API localmente de ponta a ponta**

Run: `uv run uvicorn triagem.api.app:app --port 8000` e, em outro terminal, `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d "{\"texto\": \"Pneumotórax hipertensivo à direita\"}"`
Expected: JSON com `"classe": "urgente"`. Depois, encerre o uvicorn.

- [ ] **Step 8: Commit**

```bash
git add src/triagem/api tests/test_api.py
git commit -m "feat: API FastAPI de triagem instrumentada com Prometheus"
```

### Task 9: Benchmark de latência (sklearn vs. ONNX)

**Files:**
- Create: `src/triagem/benchmark.py`
- Test: `tests/test_benchmark.py`
- Generate: `reports/latencia.json`, `reports/latencia.md`

**Interfaces:**
- Consumes: `carregar_classificador`, `carregar_dataset`, `ARQUIVO_JOBLIB`, `ARQUIVO_ONNX` e `versoes_bibliotecas`.
- Produces:
  - `percentis(amostras_ms: Sequence[float]) -> dict` com `p50`, `p95`, `p99`, `media` e `n`
  - `medir(funcao: Callable[[str], object], textos: Sequence[str], n: int = 1000, warmup: int = 50) -> dict` (os percentis)
  - `benchmark_inprocess(diretorio_modelo: Path, textos: Sequence[str], n: int, warmup: int) -> dict`, com uma entrada por backend contendo `latencia_ms`, `carga_s` e `tamanho_bytes`, mais `speedup_p50` e `speedup_p95`
  - `benchmark_http(urls: dict[str, str], textos, n, warmup) -> dict`: backend → `latencia_ms`, mais os speedups
  - `gerar_relatorio_md(resultado: dict) -> str`
  - CLI: `python -m triagem.benchmark --modo {inprocess,http,ambos} [--model-dir] [--dados] [--url-sklearn] [--url-onnx] [--n] [--warmup] [--saida reports]`

- [ ] **Step 1: Escrever os testes que falham**

`tests/test_benchmark.py`:

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_benchmark.py`
Expected: FAIL com `ModuleNotFoundError: No module named 'triagem.benchmark'`

- [ ] **Step 3: Implementar `src/triagem/benchmark.py`**

```python
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
            "| Backend | P50 (ms) | P95 (ms) | P99 (ms) | Média (ms) | Carga (ms) | Artefato (KiB) |",
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
```

- [ ] **Step 4: Rodar e ver passar; lint**

Run: `uv run pytest tests/test_benchmark.py && uv run ruff check . && uv run ruff format --check .`
Expected: 4 passed.

- [ ] **Step 5: Commit** (os relatórios reais são gerados e commitados na Task 10, com o modo `ambos`)

```bash
git add src/triagem/benchmark.py tests/test_benchmark.py
git commit -m "feat: benchmark de latência sklearn vs ONNX Runtime"
```

### Task 10: Imagem Docker e stack de monitoramento (API + Prometheus + Grafana)

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `docker-compose.modelo-local.yml`, `monitoring/prometheus/prometheus.yml`, `monitoring/grafana/provisioning/datasources/prometheus.yml`, `monitoring/grafana/provisioning/dashboards/dashboards.yml`, `monitoring/grafana/dashboards/triagem.json`, `scripts/gerar_carga.py`, `scripts/smoke_test.sh`
- Generate: `reports/latencia.json`, `reports/latencia.md`, `docs/img/dashboard.png`

**Interfaces:**
- Consumes: o CLI `python -m triagem.treinar`, `triagem.api.app:app`, as métricas da Task 8 e o CLI de benchmark.
- Produces: a imagem `triagem-api:local`; `scripts/smoke_test.sh <imagem> [porta]`, que retorna ≠0 em caso de falha (usado pelo CI na Task 12).

- [ ] **Step 1: Criar `.dockerignore`**

```
.git
.github
.venv
.claude
**/__pycache__
.pytest_cache
.ruff_cache
models
data/processed
airflow/logs
reports
docs
tests
```

- [ ] **Step 2: Criar o `Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1

# ---------- base com uv ----------
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# ---------- treino: gera dataset, treina, exporta ONNX e promove ----------
FROM base AS treino
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra treino --no-install-project
COPY src ./src
COPY data/raw ./data/raw
RUN uv sync --locked --no-dev --extra treino \
    && uv run --no-sync python -m triagem.treinar \
        --dados data/raw/laudos_sinteticos.csv --models-dir /build/models --trabalho /tmp/trabalho

# ---------- dependências de serving (sem pandas/skl2onnx) ----------
FROM base AS deps
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# ---------- runtime enxuto ----------
FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
COPY --from=treino /build/models/producao /app/models/producao
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models/producao \
    MODEL_BACKEND=onnx \
    PORT=8000
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}/health\", timeout=2)"]
CMD ["sh", "-c", "exec uvicorn triagem.api.app:app --host 0.0.0.0 --port ${PORT}"]
```

- [ ] **Step 3: Build e verificação manual**

Run: `docker build -t triagem-api:local .`
Expected: build conclui; o log do estágio `treino` mostra `"promovido": true`.

Run: `docker run --rm -d --name triagem-teste -p 8000:8000 triagem-api:local`, depois (após ~5 s) `curl localhost:8000/health` e `docker rm -f triagem-teste`
Expected: `{"status":"ok","versao_modelo":"<versao>","backend":"onnx"}`

Run: `docker image ls triagem-api:local --format "{{.Size}}"`
Expected: bem abaixo de 1 GB. Anote o tamanho para o README.

- [ ] **Step 4: Criar `scripts/smoke_test.sh`** (marcar como executável: `git update-index --chmod=+x scripts/smoke_test.sh` depois do `git add`)

```bash
#!/usr/bin/env bash
# Smoke test da imagem da API: health, predição válida, payload inválido e /metrics.
set -euo pipefail

IMAGEM="${1:-triagem-api:local}"
PORTA="${2:-8000}"
URL="http://localhost:${PORTA}"

CID=$(docker run -d -p "${PORTA}:8000" "$IMAGEM")
trap 'docker logs "$CID" > /dev/null 2>&1 || true; docker rm -f "$CID" > /dev/null' EXIT

for _ in $(seq 1 30); do
  if curl -fsS "$URL/health" > /dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS "$URL/health"
echo

LAUDO='{"texto": "Tomografia de crânio. Hematoma subdural agudo com efeito de massa."}'
status=$(curl -s -o /tmp/resposta.json -w '%{http_code}' -X POST "$URL/predict" \
  -H 'Content-Type: application/json' -d "$LAUDO")
cat /tmp/resposta.json
echo
[ "$status" = "200" ] || { echo "ERRO: /predict retornou $status"; exit 1; }
grep -q '"classe":"urgente"' /tmp/resposta.json || { echo "ERRO: classe inesperada"; exit 1; }

status=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/predict" \
  -H 'Content-Type: application/json' -d '{"texto": ""}')
[ "$status" = "422" ] || { echo "ERRO: payload inválido retornou $status"; exit 1; }

curl -fsS "$URL/metrics" | grep -q 'triagem_http_requests_total' \
  || { echo "ERRO: métricas ausentes"; exit 1; }

echo "Smoke test OK"
```

Run (Git Bash): `bash scripts/smoke_test.sh triagem-api:local 8000`
Expected: termina com `Smoke test OK`.

- [ ] **Step 5: Criar a configuração do Prometheus** (`monitoring/prometheus/prometheus.yml`)

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: triagem-api
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```

- [ ] **Step 6: Criar o provisionamento do Grafana**

`monitoring/grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      timeInterval: 5s
```

`monitoring/grafana/provisioning/dashboards/dashboards.yml`:

```yaml
apiVersion: 1
providers:
  - name: triagem
    folder: Triagem
    type: file
    disableDeletion: true
    options:
      path: /var/lib/grafana/dashboards
```

- [ ] **Step 7: Criar o dashboard** (`monitoring/grafana/dashboards/triagem.json`)

```json
{
  "uid": "triagem-api",
  "title": "Triagem de Laudos — API",
  "tags": ["triagem", "mlops"],
  "timezone": "browser",
  "schemaVersion": 41,
  "version": 1,
  "refresh": "5s",
  "time": {"from": "now-15m", "to": "now"},
  "panels": [
    {
      "id": 1, "type": "stat", "title": "Total de requisições (/predict)",
      "gridPos": {"x": 0, "y": 0, "w": 6, "h": 5},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [{"refId": "A", "expr": "sum(triagem_http_requests_total{rota=\"/predict\"})"}],
      "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "graphMode": "area"}
    },
    {
      "id": 2, "type": "stat", "title": "Modelo em produção",
      "gridPos": {"x": 6, "y": 0, "w": 6, "h": 5},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [{"refId": "A", "expr": "triagem_modelo_info", "legendFormat": "{{versao}} ({{backend}})", "instant": true}],
      "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "textMode": "name", "colorMode": "none"}
    },
    {
      "id": 3, "type": "timeseries", "title": "Requisições por segundo, por status",
      "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [{"refId": "A", "expr": "sum by (status) (rate(triagem_http_requests_total{rota=\"/predict\"}[1m]))", "legendFormat": "HTTP {{status}}"}],
      "fieldConfig": {"defaults": {"unit": "reqps"}, "overrides": []}
    },
    {
      "id": 4, "type": "timeseries", "title": "Latência HTTP de /predict (P50 / P95 / P99)",
      "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [
        {"refId": "A", "expr": "histogram_quantile(0.50, sum by (le) (rate(triagem_http_request_duration_seconds_bucket{rota=\"/predict\"}[1m])))", "legendFormat": "P50"},
        {"refId": "B", "expr": "histogram_quantile(0.95, sum by (le) (rate(triagem_http_request_duration_seconds_bucket{rota=\"/predict\"}[1m])))", "legendFormat": "P95"},
        {"refId": "C", "expr": "histogram_quantile(0.99, sum by (le) (rate(triagem_http_request_duration_seconds_bucket{rota=\"/predict\"}[1m])))", "legendFormat": "P99"}
      ],
      "fieldConfig": {"defaults": {"unit": "s"}, "overrides": []}
    },
    {
      "id": 5, "type": "timeseries", "title": "Taxa de erro (4xx + 5xx) em /predict",
      "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [{"refId": "A", "expr": "100 * sum(rate(triagem_http_requests_total{rota=\"/predict\",status=~\"4..|5..\"}[1m])) / clamp_min(sum(rate(triagem_http_requests_total{rota=\"/predict\"}[1m])), 1e-9)", "legendFormat": "% erros"}],
      "fieldConfig": {"defaults": {"unit": "percent", "min": 0}, "overrides": []}
    },
    {
      "id": 6, "type": "barchart", "title": "Predições por classe (últimos 5 min)",
      "gridPos": {"x": 0, "y": 16, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [{"refId": "A", "expr": "sum by (classe) (increase(triagem_predicoes_total[5m]))", "legendFormat": "{{classe}}", "instant": true, "format": "table"}],
      "transformations": [{"id": "reduce", "options": {"reducers": ["lastNotNull"]}}],
      "options": {"xField": "Field", "orientation": "horizontal"}
    },
    {
      "id": 7, "type": "timeseries", "title": "Latência de inferência P95 por backend",
      "gridPos": {"x": 12, "y": 16, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "targets": [{"refId": "A", "expr": "histogram_quantile(0.95, sum by (le, backend) (rate(triagem_inferencia_duration_seconds_bucket[1m])))", "legendFormat": "{{backend}}"}],
      "fieldConfig": {"defaults": {"unit": "s"}, "overrides": []}
    }
  ]
}
```

- [ ] **Step 8: Criar os arquivos do Compose**

`docker-compose.yml`:

```yaml
name: triagem

services:
  api:
    build: .
    image: triagem-api:local
    environment:
      MODEL_BACKEND: ${MODEL_BACKEND:-onnx}
    ports:
      - "8000:8000"

  prometheus:
    image: prom/prometheus:v3.5.0
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "9090:9090"
    depends_on:
      - api

  grafana:
    image: grafana/grafana:12.1.0
    environment:
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Viewer
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH: /var/lib/grafana/dashboards/triagem.json
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

`docker-compose.modelo-local.yml` (override opcional para servir o modelo promovido pela DAG):

```yaml
# Uso: docker compose -f docker-compose.yml -f docker-compose.modelo-local.yml up -d
services:
  api:
    volumes:
      - ./models/producao:/app/models/producao:ro
```

- [ ] **Step 9: Criar `scripts/gerar_carga.py`**

```python
"""Gera tráfego contra a API para alimentar o dashboard (inclui ~5% de payloads inválidos)."""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import Counter
from pathlib import Path

import httpx


def carregar_textos(caminho: Path) -> list[str]:
    with caminho.open(encoding="utf-8") as arquivo:
        return [linha["texto"] for linha in csv.DictReader(arquivo)]


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
    with httpx.Client(base_url=args.url, timeout=5) as cliente:
        while time.monotonic() < fim:
            inicio = time.monotonic()
            if rng.random() < args.taxa_invalidos:
                payload: dict = rng.choice([{"texto": ""}, {"texto": 42}, {}])
            else:
                payload = {"texto": rng.choice(textos)}
            try:
                status[cliente.post("/predict", json=payload).status_code] += 1
            except httpx.HTTPError:
                status[0] += 1
            time.sleep(max(0.0, intervalo - (time.monotonic() - inicio)))
    print("Requisições por status:", dict(sorted(status.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10: Subir a stack e validar o dashboard**

Run: `docker compose up -d --build` e depois `uv run python scripts/gerar_carga.py --duracao 90`
Expected: o resumo mostra majoritariamente status 200, com uns ~5% de 422.

Verificar: `curl -s "localhost:9090/api/v1/targets" | grep -o '"health":"up"'` deve imprimir `"health":"up"`. `curl -s localhost:3000/api/dashboards/uid/triagem-api` deve retornar o JSON do dashboard (acesso anônimo). Abra `http://localhost:3000` e confira que os 7 painéis mostram dados.

- [ ] **Step 11: Capturar o print do dashboard**

Com a carga ainda rodando (ou logo após), rode:
`& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless=new --disable-gpu --window-size=1600,1300 --virtual-time-budget=15000 --screenshot="$PWD\docs\img\dashboard.png" "http://localhost:3000/d/triagem-api?orgId=1&kiosk"`
Expected: `docs/img/dashboard.png` mostra os painéis com dados. Abra a imagem para conferir. Se o Edge não existir, use outro Chromium headless com as mesmas flags.

- [ ] **Step 12: Gerar os relatórios de latência (in-process e HTTP)**

Run:
```
docker compose down
docker run -d --rm --name triagem-sk -p 8001:8000 -e MODEL_BACKEND=sklearn triagem-api:local
docker run -d --rm --name triagem-ox -p 8002:8000 -e MODEL_BACKEND=onnx triagem-api:local
uv run python -m triagem.benchmark --modo ambos --n 1000 --warmup 50
docker rm -f triagem-sk triagem-ox
```
Expected: `reports/latencia.md` e `reports/latencia.json` gerados, com o speedup do ONNX positivo nos dois modos. Registre os números reais (sem arredondar a favor do ONNX).

- [ ] **Step 13: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml docker-compose.modelo-local.yml monitoring scripts reports docs/img
git update-index --chmod=+x scripts/smoke_test.sh
git commit -m "build: imagem Docker multi-stage e stack Prometheus + Grafana"
```

### Task 11: DAG de retreino no Airflow 3.3.2

**Files:**
- Create: `dags/triagem_retreino.py`, `airflow/Dockerfile`, `airflow/check_dag.py`, `docker-compose.airflow.yml`

**Interfaces:**
- Consumes: `triagem.pipeline.ingerir`, `treinar_versao`, `avaliar_versao`, `exportar_versao`, `promover_se_aprovado` (importadas **dentro** das tasks, para o parse da DAG ficar leve).
- Produces: a DAG `triagem_retreino` com as tasks `ingerir_dados`, `treinar_modelo`, `avaliar_modelo`, `exportar_onnx` e `promover_modelo`; `airflow/check_dag.py` (usado pelo CI) sai com código ≠0 se houver erro.

- [ ] **Step 1: Criar `airflow/Dockerfile`**

```dockerfile
FROM apache/airflow:3.3.2-python3.12
# Fixar apache-airflow impede o pip de atualizá-lo; as versões de ML batem com o uv.lock.
RUN pip install --no-cache-dir \
    "apache-airflow==3.3.2" \
    "scikit-learn==1.9.1" \
    "skl2onnx==1.20.0" \
    "onnx==1.23.0" \
    "onnxruntime==1.30.0"
ENV PYTHONPATH=/opt/airflow/src
```

- [ ] **Step 2: Criar `dags/triagem_retreino.py`**

```python
"""Retreino semanal do classificador de urgência de laudos.

ingerir_dados → treinar_modelo → avaliar_modelo → exportar_onnx → promover_modelo

Entre as tasks trafegam só caminhos e métricas (XCom leve); a lógica vive em `triagem.pipeline`.
A promoção aplica o quality gate champion vs. challenger (F1-macro e recall de urgente).
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import dag, get_current_context, task

BASE = Path(os.environ.get("TRIAGEM_BASE_DIR", "/opt/airflow"))
DADOS = BASE / "data" / "raw" / "laudos_sinteticos.csv"
PROCESSADOS = BASE / "data" / "processed"
MODELOS = BASE / "models"


@dag(
    dag_id="triagem_retreino",
    schedule="@weekly",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["triagem", "mlops"],
    doc_md=__doc__,
)
def triagem_retreino():
    @task
    def ingerir_dados() -> dict[str, str]:
        from triagem.pipeline import ingerir

        run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", get_current_context()["run_id"])
        return ingerir(DADOS, PROCESSADOS / run_id)

    @task
    def treinar_modelo(split: dict[str, str]) -> str:
        from triagem.pipeline import treinar_versao

        return treinar_versao(Path(split["treino"]), MODELOS)

    @task
    def avaliar_modelo(diretorio_versao: str, split: dict[str, str]) -> dict:
        from triagem.pipeline import avaliar_versao

        return avaliar_versao(Path(diretorio_versao), Path(split["teste"]))

    @task
    def exportar_onnx(diretorio_versao: str, split: dict[str, str], metricas: dict) -> dict:
        from triagem.pipeline import exportar_versao

        metadata = exportar_versao(
            Path(diretorio_versao), Path(split["teste"]), metricas, split["sha256_dados"]
        )
        return {"versao": metadata["versao"], "paridade_onnx": metadata["paridade_onnx"]}

    @task
    def promover_modelo(diretorio_versao: str, exportado: dict) -> dict:
        from triagem.pipeline import promover_se_aprovado

        return promover_se_aprovado(Path(diretorio_versao), MODELOS)

    split = ingerir_dados()
    versao = treinar_modelo(split)
    metricas = avaliar_modelo(versao, split)
    exportado = exportar_onnx(versao, split, metricas)
    promover_modelo(versao, exportado)


triagem_retreino()
```

- [ ] **Step 3: Criar `airflow/check_dag.py`**

```python
"""Valida que a DAG carrega sem erros e tem as tasks esperadas (usado no CI)."""

import sys

from airflow.dag_processing.dagbag import DagBag

ESPERADAS = {"ingerir_dados", "treinar_modelo", "avaliar_modelo", "exportar_onnx", "promover_modelo"}

bag = DagBag("/opt/airflow/dags")
if bag.import_errors:
    print("Erros de importação:", bag.import_errors)
    sys.exit(1)
dag = bag.dags.get("triagem_retreino")
if dag is None:
    print("DAG triagem_retreino não encontrada")
    sys.exit(1)
tarefas = {t.task_id for t in dag.tasks}
if tarefas != ESPERADAS:
    print("Tasks divergentes:", sorted(tarefas))
    sys.exit(1)
print("DAG OK:", sorted(tarefas))
```

- [ ] **Step 4: Criar `docker-compose.airflow.yml`**

```yaml
name: triagem-airflow

services:
  airflow:
    build:
      context: .
      dockerfile: airflow/Dockerfile
    image: triagem-airflow:local
    command: standalone
    user: "${AIRFLOW_UID:-50000}:0"
    environment:
      AIRFLOW__CORE__LOAD_EXAMPLES: "false"
      # Somente para desenvolvimento local: dispensa login na UI.
      AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS: "true"
      TRIAGEM_BASE_DIR: /opt/airflow
    volumes:
      - ./dags:/opt/airflow/dags
      - ./src:/opt/airflow/src
      - ./data:/opt/airflow/data
      - ./models:/opt/airflow/models
      - ./airflow/logs:/opt/airflow/logs
    ports:
      - "8080:8080"
```

- [ ] **Step 5: Build e validação estática da DAG**

Run: `docker compose -f docker-compose.airflow.yml build`
Run: `docker run --rm -v "${PWD}/dags:/opt/airflow/dags:ro" -v "${PWD}/airflow/check_dag.py:/opt/airflow/check_dag.py:ro" --entrypoint python triagem-airflow:local /opt/airflow/check_dag.py`
Expected: `DAG OK: [...5 tasks...]`

- [ ] **Step 6: Execução real da DAG**

Run: `docker compose -f docker-compose.airflow.yml run --rm airflow bash -c "airflow db migrate && airflow dags test triagem_retreino"`
Expected: as 5 tasks terminam em `success`; em `models/`, surge uma nova versão, e `models/producao/metadata.json` fica atualizado (ou a decisão de não promover aparece registrada no log com o motivo).

Depois, suba a UI: `docker compose -f docker-compose.airflow.yml up -d`, abra `http://localhost:8080`, confira a DAG e o grafo e depois rode `docker compose -f docker-compose.airflow.yml down`.

- [ ] **Step 7: Commit**

```bash
git add dags airflow/Dockerfile airflow/check_dag.py docker-compose.airflow.yml
git commit -m "feat: DAG Airflow de retreino com quality gate de promoção"
```

### Task 12: Pipeline de CI no GitHub Actions

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `scripts/smoke_test.sh`, `airflow/check_dag.py`, `airflow/Dockerfile`, `Dockerfile`, a configuração de pytest e ruff.

- [ ] **Step 1: Conferir as versões atuais das actions**

Run: `git ls-remote --tags https://github.com/actions/checkout "v*"`, e o mesmo para `astral-sh/setup-uv`, `docker/setup-buildx-action` e `docker/build-push-action`.
Expected: use a **maior tag major** existente de cada uma (ex.: `v5`). Os valores abaixo assumem `checkout@v5`, `setup-uv@v7`, `setup-buildx-action@v3` e `build-push-action@v6`; ajuste se o `ls-remote` mostrar outra major mais recente.

- [ ] **Step 2: Criar `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    name: Lint (ruff)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
      - run: uv sync --locked --all-extras
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  test:
    name: Testes (pytest)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
      - run: uv sync --locked --all-extras
      - run: uv run pytest --cov=triagem --cov-report=term-missing --cov-fail-under=80

  build:
    name: Build da imagem + smoke test
    needs: [lint, test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v6
        with:
          context: .
          load: true
          tags: triagem-api:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - run: bash scripts/smoke_test.sh triagem-api:ci 8000

  dag-check:
    name: Validação da DAG (Airflow)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v6
        with:
          context: .
          file: airflow/Dockerfile
          load: true
          tags: triagem-airflow:ci
          cache-from: type=gha,scope=airflow
          cache-to: type=gha,mode=max,scope=airflow
      - run: >
          docker run --rm
          -v "$PWD/dags:/opt/airflow/dags:ro"
          -v "$PWD/airflow/check_dag.py:/opt/airflow/check_dag.py:ro"
          --entrypoint python triagem-airflow:ci /opt/airflow/check_dag.py
```

- [ ] **Step 3: Validar localmente o que der**

Run: `uv run pytest --cov=triagem --cov-report=term-missing --cov-fail-under=80`
Expected: passa, com cobertura ≥ 80%. Se ficar abaixo, adicione testes para as linhas descobertas em vez de baixar o limite.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: workflow com lint, testes, build + smoke test e validação da DAG"
```

### Task 13: Documentação (README, arquitetura e roteiro do vídeo)

**Files:**
- Modify: `README.md` (reescrita completa)
- Create: `docs/arquitetura.md`, `docs/roteiro_video_star.md`

**Interfaces:**
- Consumes: os números reais de `reports/latencia.md`, de `models/producao/metadata.json` (métricas) e do tamanho da imagem (Task 10, Step 3). **Nenhum número inventado:** copie dos arquivos gerados.

- [ ] **Step 1: Escrever `README.md`** com estas seções, nesta ordem e todas preenchidas:
  1. **Título e resumo:** o problema clínico em 3 frases e o badge do CI (`![CI](https://github.com/<usuario>/triagem-laudos-mlops/actions/workflows/ci.yml/badge.svg)`, com o `<usuario>` real obtido na Task 14 via `gh api user -q .login`; até lá, a linha fica fora do README e entra na Task 14).
  2. **Arquitetura:** diagrama Mermaid (`flowchart LR`) com Dados CSV → DAG Airflow (5 tasks) → `models/producao` → Imagem Docker (FastAPI + ONNX Runtime) → `/metrics` → Prometheus → Grafana; e GitHub Actions (lint, test, build + smoke, dag-check) apontando para a imagem.
  3. **Decisão de deploy em nuvem:** o conteúdo da seção 3 da spec (real-time vs. batch vs. serverless, justificativa clínica, tabela GCP e equivalentes AWS/Azure, cold start e `min-instances=1`, segurança/LGPD, FinOps, fluxo de CD futuro com OIDC → Artifact Registry → Cloud Run).
  4. **Dataset:** por que é sintético, esquema, proporções, ruído e a limitação declarada.
  5. **Modelo e resultados:** o pipeline TF-IDF + RF, as métricas reais do `metadata.json` (acurácia, F1-macro, recall de urgente) e a matriz de confusão em tabela.
  6. **Otimização de latência:** o método e a tabela copiada de `reports/latencia.md` (in-process e HTTP), com uma interpretação honesta: o ganho de inferência pura é grande, mas no HTTP o overhead de rede e serialização domina.
  7. **Como executar:** os pré-requisitos (Docker, uv) e os comandos testados, cada um com a saída esperada: `uv sync --all-extras`; `uv run pytest`; `uv run python -m triagem.treinar`; `docker compose up -d --build` com as URLs (API docs em `:8000/docs`, Prometheus em `:9090`, Grafana em `:3000`); `uv run python scripts/gerar_carga.py`; o Airflow com `docker compose -f docker-compose.airflow.yml up -d --build` e a UI em `:8080` (aviso de que o login desativado é só para dev); `airflow dags test` via `run --rm`; o override `docker-compose.modelo-local.yml`; e o benchmark nos dois modos.
  8. **Monitoramento:** a lista dos 7 painéis com as queries PromQL principais, e `![Dashboard](docs/img/dashboard.png)`.
  9. **CI/CD:** os 4 jobs e o que cada um garante.
  10. **Estrutura do repositório:** a árvore da seção 11 da spec.
  11. **Limitações e próximos passos:** o que ficou fora de escopo (seção 15 da spec), com referência às aulas.
  12. **Vídeo STAR:** o link para `docs/roteiro_video_star.md` e a linha `Link do vídeo: (a ser adicionado após a gravação)`.

- [ ] **Step 2: Escrever `docs/arquitetura.md`:** o diagrama Mermaid detalhado (componentes e portas), o fluxo de uma requisição (validação → normalização → ONNX → métricas), o fluxo de retreino (5 tasks, XComs e gate) e o mapeamento de cada aula para um componente (uma tabela com os 6 temas do resumo).

- [ ] **Step 3: Escrever `docs/roteiro_video_star.md`:** roteiro de 5 minutos com tempos (S 0:00–0:40, T 0:40–1:20, A 1:20–3:30, R 3:30–5:00), falas sugeridas e a lista exata de telas e comandos a mostrar (CI verde no GitHub, `docker compose up`, `gerar_carga`, dashboard, UI do Airflow com a DAG executada, tabela de latência). Os números vêm dos relatórios reais.

- [ ] **Step 4: Verificar os comandos do README** executando cada um que ainda não tenha sido executado nas Tasks anteriores; corrija o README se algum divergir.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/arquitetura.md docs/roteiro_video_star.md
git commit -m "docs: README com arquitetura em nuvem, execução, resultados e roteiro do vídeo"
```

### Task 14: Repositório no GitHub, push e CI verde

**Files:**
- Modify: `README.md` (badge do CI)

- [ ] **Step 1: Garantir autenticação:** `gh auth status`. Se não houver login, **pare e peça ao usuário** para rodar `! gh auth login` (é um bloqueio real: exige a credencial dele).

- [ ] **Step 2: Adicionar o badge do CI ao README** com o login real (`gh api user -q .login`) e fazer o commit:

```bash
git add README.md
git commit -m "docs: badge do CI no README"
```

- [ ] **Step 3: Criar o repositório e dar push**

Run: `gh repo create triagem-laudos-mlops --public --source . --remote origin --push --description "Tech Challenge MLOps: triagem de laudos com FastAPI, ONNX, Airflow, Prometheus e Grafana"`
Expected: a URL do repositório impressa e a branch `main` enviada.

- [ ] **Step 4: Acompanhar o CI**

Run: `gh run watch --exit-status $(gh run list --workflow ci.yml --limit 1 --json databaseId -q ".[0].databaseId")`
Expected: os 4 jobs em verde. Se algum falhar, leia `gh run view --log-failed`, corrija com um commit `fix:`/`ci:`, faça o push e repita até ficar verde.
