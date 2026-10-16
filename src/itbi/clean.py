"""Recorte usado na modelagem: apartamentos vendidos inteiros por compra e venda."""

from __future__ import annotations

import pandas as pd

# limites de sanidade; fora disso quase sempre é erro de digitação
# ou negócio que não reflete preço de mercado (venda entre parentes etc.)
AREA_MIN, AREA_MAX = 20, 600
PRECO_M2_MIN, PRECO_M2_MAX = 1_500, 60_000
RAZAO_VENAL_MIN, RAZAO_VENAL_MAX = 0.3, 5.0


def apartments(df: pd.DataFrame) -> pd.DataFrame:
    d = df[
        df["natureza"].str.startswith("1.", na=False)
        & df["uso_desc"].str.startswith("APARTAMENTO", na=False)
        & (df["proporcao_transmitida"] == 100)
        & df["cep"].notna()
    ].copy()

    d = d[d["area_construida"].between(AREA_MIN, AREA_MAX)]
    d["preco_m2"] = d["valor_transacao"] / d["area_construida"]
    d = d[d["preco_m2"].between(PRECO_M2_MIN, PRECO_M2_MAX)]

    razao = d["valor_transacao"] / d["valor_venal_ref"]
    d = d[razao.between(RAZAO_VENAL_MIN, RAZAO_VENAL_MAX)]

    ano = d["data_transacao"].dt.year
    # guias pagas com anos de atraso carregam preço de outra época
    ano_guia = d["mes_ref"].str[-4:].astype(int)
    d = d[d["ano_construcao"].between(1900, ano + 1) & (ano_guia - ano).between(0, 1)]

    # a mesma guia às vezes aparece em dois meses
    d = d.drop_duplicates(subset=["sql", "data_transacao", "valor_transacao"])
    return d.reset_index(drop=True)
