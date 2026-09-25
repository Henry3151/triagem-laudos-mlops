"""Gera os gráficos do README a partir dos relatórios versionados em reports/.

Entradas: reports/latencia.json e reports/candidatos.json (nada é digitado à mão).
Saídas:   docs/img/latencia.png, docs/img/matriz_confusao.png e docs/img/candidatos.png.

Uso: uv run python scripts/gerar_graficos.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "docs" / "img"

# Paleta de referência (skill dataviz): superfície clara, tintas e slots categóricos.
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUTED = "#898781"
GRADE = "#e1e0d9"
BASE = "#c3c2b7"
AZUL = "#2a78d6"  # slot 1 — ONNX / modelo em produção
LARANJA = "#eb6834"  # slot 2 — scikit-learn original
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "axes.edgecolor": BASE,
        "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": TINTA_2,
        "ytick.labelcolor": TINTA_2,
        "grid.color": GRADE,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": SUPERFICIE,
        "savefig.dpi": 160,
    }
)


def _num(valor: float, casas: int = 2) -> str:
    """Número no padrão pt-BR (vírgula decimal), como no README."""
    return f"{valor:.{casas}f}".replace(".", ",")


def _eixo_ptbr(eixo, qual: str, casas: int) -> None:
    formatador = FuncFormatter(lambda v, _pos: _num(v, casas))
    (eixo.xaxis if qual == "x" else eixo.yaxis).set_major_formatter(formatador)


def _ler(nome: str) -> dict:
    return json.loads((RAIZ / "reports" / nome).read_text(encoding="utf-8"))


def _salvar(fig, nome: str) -> Path:
    caminho = SAIDA / nome
    fig.savefig(caminho, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return caminho


def grafico_latencia() -> Path:
    dados = _ler("latencia.json")
    percentis = ["p50", "p95", "p99"]
    modos = [("inprocess", "Em processo (só a predição)"), ("http", "Ponta a ponta via HTTP")]
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.6))
    largura = 0.36
    for eixo, (modo, titulo) in zip(eixos, modos, strict=True):
        r = dados[modo]
        xs = range(len(percentis))
        for deslocamento, backend, cor, rotulo in (
            (-largura / 2, "sklearn", LARANJA, "scikit-learn (original)"),
            (largura / 2, "onnx", AZUL, "ONNX Runtime (otimizado)"),
        ):
            valores = [r[backend]["latencia_ms"][p] for p in percentis]
            barras = eixo.bar(
                [x + deslocamento for x in xs],
                valores,
                width=largura - 0.04,
                color=cor,
                label=rotulo,
                zorder=3,
            )
            eixo.bar_label(
                barras, labels=[_num(v) for v in valores], padding=3, color=TINTA, fontsize=10
            )
        eixo.set_xticks(list(xs), [p.upper() for p in percentis])
        eixo.set_ylabel("latência (ms)")
        eixo.grid(axis="y", zorder=0)
        eixo.set_ylim(0, max(r["sklearn"]["latencia_ms"]["p99"] * 1.18, 1))
        casas = 0 if r["speedup_p50"] >= 20 else 1
        eixo.set_title(
            f"{titulo}\nONNX {_num(r['speedup_p50'], casas)}x mais rápido no P50", loc="left"
        )
    eixos[0].legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2)
    fig.suptitle(
        "Latência por requisição (batch 1, 1.000 chamadas) — scikit-learn vs. ONNX Runtime",
        x=0.01,
        ha="left",
        color=TINTA,
        fontsize=13.5,
        fontweight="bold",
    )
    fig.tight_layout()
    return _salvar(fig, "latencia.png")


def grafico_matriz_confusao() -> Path:
    producao = next(
        c for c in _ler("candidatos.json")["candidatos"] if "produção" in c["candidato"]
    )
    matriz = producao["matriz_confusao"]
    rotulos = matriz["rotulos"]
    valores = matriz["valores"]
    fig, eixo = plt.subplots(figsize=(6.4, 5.4))
    mapa = LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)
    imagem = eixo.imshow(valores, cmap=mapa, vmin=0)
    maximo = max(max(linha) for linha in valores)
    for i, linha in enumerate(valores):
        total = sum(linha)
        for j, v in enumerate(linha):
            escura = v > maximo * 0.45
            eixo.text(
                j,
                i,
                f"{v}\n{_num(100 * v / total, 1)}%",
                ha="center",
                va="center",
                fontsize=11,
                color="#ffffff" if escura else TINTA,
                fontweight="bold" if i == j else "normal",
            )
    eixo.set_xticks(range(len(rotulos)), rotulos)
    eixo.set_yticks(range(len(rotulos)), rotulos)
    eixo.set_xlabel("classe prevista")
    eixo.set_ylabel("classe real")
    eixo.spines[:].set_visible(False)
    eixo.tick_params(length=0)
    barra = fig.colorbar(imagem, ax=eixo, fraction=0.046, pad=0.04)
    barra.outline.set_visible(False)
    barra.ax.tick_params(color=MUTED, labelcolor=TINTA_2)
    eixo.set_title(
        f"Matriz de confusão — teste ({producao['n_amostras']} laudos)\n"
        f"F1-macro {_num(producao['f1_macro'], 3)} · "
        f"recall de urgente {_num(producao['recall_urgente'], 3)}",
        loc="left",
    )
    fig.tight_layout()
    return _salvar(fig, "matriz_confusao.png")


def grafico_candidatos() -> Path:
    candidatos = sorted(_ler("candidatos.json")["candidatos"], key=lambda c: c["f1_macro"])
    nomes = [c["candidato"] for c in candidatos]
    paineis = [
        ("f1_macro", "F1-macro", 0.80, "gate ≥ 0,80"),
        ("recall_urgente", "Recall de urgente", 0.85, "gate ≥ 0,85"),
    ]
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for eixo, (chave, titulo, limiar, rotulo_limiar) in zip(eixos, paineis, strict=True):
        for y, c in enumerate(candidatos):
            producao = "produção" in c["candidato"]
            cor = AZUL if producao else MUTED
            eixo.plot([limiar, c[chave]], [y, y], color=GRADE, linewidth=2, zorder=1)
            eixo.scatter(c[chave], y, s=90, color=cor, zorder=3, edgecolors=SUPERFICIE)
            eixo.annotate(
                _num(c[chave], 3),
                (c[chave], y),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                color=TINTA,
                fontsize=10,
                fontweight="bold" if producao else "normal",
            )
        eixo.axvline(limiar, color=BASE, linestyle="--", linewidth=1.2, zorder=0)
        eixo.text(limiar, -0.75, f" {rotulo_limiar}", color=MUTED, fontsize=9, va="center")
        eixo.set_xlim(limiar - 0.01, 1.0)
        eixo.set_ylim(-1.0, len(candidatos) - 0.5)
        _eixo_ptbr(eixo, "x", 2 if chave == "recall_urgente" else 3)
        eixo.set_title(titulo, loc="left")
        eixo.grid(axis="x", zorder=0)
        eixo.spines["left"].set_visible(False)
        eixo.tick_params(axis="y", length=0)
    eixos[0].set_yticks(range(len(nomes)), nomes)
    for rotulo in eixos[0].get_yticklabels():
        if "produção" in rotulo.get_text():
            rotulo.set_color(AZUL)
            rotulo.set_fontweight("bold")
    fig.suptitle(
        "Candidatos no mesmo split de teste (600 laudos, seed 42) — comparação a posteriori",
        x=0.01,
        ha="left",
        color=TINTA,
        fontsize=13.5,
        fontweight="bold",
    )
    fig.tight_layout()
    return _salvar(fig, "candidatos.png")


def main() -> int:
    SAIDA.mkdir(parents=True, exist_ok=True)
    for gerar in (grafico_latencia, grafico_matriz_confusao, grafico_candidatos):
        print(f"gerado: {gerar().relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
