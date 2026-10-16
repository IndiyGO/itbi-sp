from __future__ import annotations

import numpy as np
import pandas as pd

NUM_FEATURES = [
    "log_area",
    "idade",
    "padrao",
    "log_area_terreno",
    "fracao_ideal",
    "testada",
    "meses",
]
CAT_FEATURES = ["cep5", "cep3"]
VENAL_FEATURE = "log_venal_m2"

BASE_DATE = pd.Timestamp("2024-01-01")


def make(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["log_area"] = np.log(d["area_construida"])
    X["idade"] = d["data_transacao"].dt.year - d["ano_construcao"]
    # padrão IPTU: dezena = tipo, unidade = classe construtiva (A=1 ... F=6)
    X["padrao"] = (d["padrao_cod"] % 10).astype(float)
    X["log_area_terreno"] = np.log1p(d["area_terreno"])
    X["fracao_ideal"] = d["fracao_ideal"]
    X["testada"] = d["testada"]
    X["meses"] = (d["data_transacao"] - BASE_DATE).dt.days / 30.44
    X["cep5"] = d["cep"].str[:5]
    X["cep3"] = d["cep"].str[:3]
    X[VENAL_FEATURE] = np.log(d["valor_venal_ref"] / d["area_construida"])
    return X


def target(d: pd.DataFrame) -> pd.Series:
    return np.log(d["valor_transacao"])
