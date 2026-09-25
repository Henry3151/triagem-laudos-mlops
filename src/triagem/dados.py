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
