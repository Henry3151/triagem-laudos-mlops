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
