"""Treina e compara os modelos de preço com validação temporal.

Modelos:
  mediana_cep   preço/m² mediano do CEP (5 dígitos) x área. É o que um
                corretor faria de cabeça e serve de piso.
  linear        regressão linear em log(preço)
  lgbm          LightGBM com as mesmas features
  lgbm_venal    LightGBM + valor venal de referência da prefeitura
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from itbi import clean, features
from itbi.features import CAT_FEATURES, NUM_FEATURES, VENAL_FEATURE

CUTOFF = pd.Timestamp("2025-07-01")
SEED = 42


def split(d: pd.DataFrame):
    train = d[d["data_transacao"] < CUTOFF]
    test = d[d["data_transacao"] >= CUTOFF]
    return train, test


def metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    y, p = np.exp(y_true_log), np.exp(y_pred_log)
    ape = np.abs(p - y) / y
    ss_res = np.sum((y_true_log - y_pred_log) ** 2)
    ss_tot = np.sum((y_true_log - y_true_log.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(p - y))),
        "mdape": float(np.median(ape)),
        "dentro_10pct": float(np.mean(ape <= 0.10)),
        "dentro_20pct": float(np.mean(ape <= 0.20)),
        "r2_log": float(1 - ss_res / ss_tot),
    }


def mediana_cep(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    m2_cep = train.groupby(train["cep"].str[:5])["preco_m2"].median()
    m2_cep3 = train.groupby(train["cep"].str[:3])["preco_m2"].median()
    m2 = test["cep"].str[:5].map(m2_cep)
    m2 = m2.fillna(test["cep"].str[:3].map(m2_cep3)).fillna(train["preco_m2"].median())
    return np.log(m2.to_numpy() * test["area_construida"].to_numpy())


def linear() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", TargetEncoder(cv=KFold(5, shuffle=True, random_state=SEED)), CAT_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("reg", LinearRegression())])


def lgbm(cols: list[str]) -> Pipeline:
    num = [c for c in cols if c not in CAT_FEATURES]
    pre = ColumnTransformer(
        [
            ("num", "passthrough", num),
            ("cat", TargetEncoder(cv=KFold(5, shuffle=True, random_state=SEED)), CAT_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    reg = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1,
    )
    return Pipeline([("pre", pre), ("reg", reg)])


def run(data: Path, out: Path) -> dict:
    raw = pd.read_parquet(data)
    d = clean.apartments(raw)
    train, test = split(d)
    X_tr, X_te = features.make(train), features.make(test)
    y_tr, y_te = features.target(train), features.target(test)

    results = {"n_treino": len(train), "n_teste": len(test), "modelos": {}}
    preds = {"mediana_cep": mediana_cep(train, test)}

    base = NUM_FEATURES + CAT_FEATURES
    models = {
        "linear": (linear(), base),
        "lgbm": (lgbm(base), base),
        "lgbm_venal": (lgbm(base + [VENAL_FEATURE]), base + [VENAL_FEATURE]),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, (model, cols) in models.items():
        model.fit(X_tr[cols], y_tr)
        preds[name] = model.predict(X_te[cols])
        joblib.dump(model, out / f"{name}.joblib")
        print(f"{name:12s} ok")

    for name, p in preds.items():
        results["modelos"][name] = metrics(y_te.to_numpy(), p)

    pd.DataFrame(
        {"cep": test["cep"], "area": test["area_construida"], "real": np.exp(y_te)}
        | {k: np.exp(v) for k, v in preds.items()}
    ).to_parquet(out / "predicoes_teste.parquet", index=False)

    (out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/interim/itbi.parquet")
    p.add_argument("--out", default="models")
    a = p.parse_args()

    r = run(Path(a.data), Path(a.out))
    print(f"\ntreino={r['n_treino']:,}  teste={r['n_teste']:,}")
    print(pd.DataFrame(r["modelos"]).T.round(3).to_string())
