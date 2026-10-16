"""Gera as figuras usadas no README a partir dos artefatos de models/."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from itbi import clean, features, train
from itbi.features import CAT_FEATURES, NUM_FEATURES, VENAL_FEATURE

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

NOMES = {
    "mediana_cep": "Mediana do CEP × área",
    "linear": "Regressão linear",
    "lgbm": "LightGBM",
    "lgbm_venal": "LightGBM + valor venal",
    "log_area": "Área construída",
    "idade": "Idade do prédio",
    "padrao": "Padrão construtivo (IPTU)",
    "log_area_terreno": "Área do terreno do condomínio",
    "fracao_ideal": "Fração ideal",
    "testada": "Testada",
    "meses": "Data da venda",
    "cep5": "CEP (5 dígitos)",
    "cep3": "CEP (3 dígitos)",
    VENAL_FEATURE: "Valor venal / m²",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10.5,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    }
)


def _brl(v: float) -> str:
    if v >= 1e6:
        return f"R$ {v / 1e6:g} mi".replace(".", ",")
    return f"R$ {v / 1e3:.0f} mil"


def fig_modelos(metrics: dict, out: Path) -> None:
    m = pd.DataFrame(metrics["modelos"]).T.sort_values("mdape", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(m))
    ax.barh(y, m["mdape"] * 100, color=BLUE, height=0.55)
    for i, v in enumerate(m["mdape"] * 100):
        ax.text(v + 0.3, i, f"{v:.1f}%".replace(".", ","), va="center", color=INK)
    ax.set_yticks(y, [NOMES[k] for k in m.index])
    ax.set_xlabel("Erro percentual mediano no teste (menor é melhor)")
    ax.set_title("Erro de cada modelo em vendas do 2º semestre de 2025")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out / "modelos.png", dpi=150)
    plt.close(fig)


def fig_real_vs_previsto(pred: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    hb = ax.hexbin(
        pred["real"],
        pred["lgbm_venal"],
        gridsize=70,
        xscale="log",
        yscale="log",
        bins="log",
        cmap="Blues",
        mincnt=1,
        linewidths=0,
    )
    lo, hi = 8e4, 1e7
    ax.plot([lo, hi], [lo, hi], color=INK_2, lw=1, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [1e5, 3e5, 1e6, 3e6, 1e7]
    ax.set_xticks(ticks, [_brl(t) for t in ticks])
    ax.set_yticks(ticks, [_brl(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Valor real da venda")
    ax.set_ylabel("Valor previsto")
    ax.set_title("Real × previsto (LightGBM + valor venal)")
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("vendas por célula", color=INK_2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "real_vs_previsto.png", dpi=150)
    plt.close(fig)


def fig_erro_faixa(pred: pd.DataFrame, out: Path) -> None:
    faixas = [0, 300e3, 500e3, 800e3, 1.2e6, 2e6, np.inf]
    rot = [
        "até\n300 mil",
        "300 a\n500 mil",
        "500 a\n800 mil",
        "800 mil\na 1,2 mi",
        "1,2 a\n2 mi",
        "acima\nde 2 mi",
    ]
    p = pred.assign(faixa=pd.cut(pred["real"], faixas, labels=rot))

    def mdape(col):
        return p.groupby("faixa", observed=True).apply(
            lambda g: np.median(np.abs(g[col] - g["real"]) / g["real"]) * 100
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rot))
    for col, cor in [("linear", ORANGE), ("lgbm_venal", BLUE)]:
        v = mdape(col)
        ax.plot(x, v.values, color=cor, lw=2, marker="o", ms=7, mec=SURFACE, mew=2)
        ax.annotate(
            NOMES[col], (x[-1], v.values[-1]), xytext=(8, 0),
            textcoords="offset points", va="center", color=INK,
        )
    ax.set_xticks(x, rot)
    ax.set_xlim(-0.3, len(rot) + 1.2)
    ax.set_ylim(0, None)
    ax.set_ylabel("Erro mediano (%)")
    ax.set_title("Onde o modelo erra mais: erro por faixa de preço")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "erro_por_faixa.png", dpi=150)
    plt.close(fig)


def fig_shap(model_dir: Path, data: Path, out: Path) -> None:
    d = clean.apartments(pd.read_parquet(data))
    _, test = train.split(d)
    cols = NUM_FEATURES + CAT_FEATURES + [VENAL_FEATURE]
    X = features.make(test)[cols].sample(5000, random_state=train.SEED)

    pipe = joblib.load(model_dir / "lgbm_venal.joblib")
    Xt = pd.DataFrame(pipe["pre"].transform(X), columns=pipe["pre"].get_feature_names_out())
    sv = shap.TreeExplainer(pipe["reg"]).shap_values(Xt)

    imp = pd.Series(np.abs(sv).mean(axis=0), index=Xt.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([NOMES[c] for c in imp.index], imp.values, color=BLUE, height=0.6)
    ax.set_xlabel("Impacto médio no log do preço (|SHAP|)")
    ax.set_title("O que mais pesa no preço")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out / "shap.png", dpi=150)
    plt.close(fig)


def main(model_dir: Path, data: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    pred = pd.read_parquet(model_dir / "predicoes_teste.parquet")
    fig_modelos(metrics, out)
    fig_real_vs_previsto(pred, out)
    fig_erro_faixa(pred, out)
    fig_shap(model_dir, data, out)


if __name__ == "__main__":
    main(Path("models"), Path("data/interim/itbi.parquet"), Path("docs/img"))
