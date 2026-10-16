"""Leitura das planilhas de guias de ITBI pagas e conversão para parquet.

Cada arquivo anual tem uma aba por mês (JAN-2024, FEV-2024, ...) mais abas
de legenda. As colunas são sempre as mesmas 28, mas o cabeçalho não é
confiável: em alguns meses ele simplesmente não existe e em outros a coluna
de descrição do padrão vem com o nome trocado. Por isso a leitura é feita por
posição e os nomes são definidos aqui.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

COLUMNS = [
    "sql",
    "logradouro",
    "numero",
    "complemento",
    "bairro",
    "referencia",
    "cep",
    "natureza",
    "valor_transacao",
    "data_transacao",
    "valor_venal_ref",
    "proporcao_transmitida",
    "valor_venal_ref_prop",
    "base_calculo",
    "tipo_financiamento",
    "valor_financiado",
    "cartorio",
    "matricula",
    "situacao_sql",
    "area_terreno",
    "testada",
    "fracao_ideal",
    "area_construida",
    "uso_cod",
    "uso_desc",
    "padrao_cod",
    "padrao_desc",
    "ano_construcao",
]

NUMERIC = [
    "numero",
    "valor_transacao",
    "valor_venal_ref",
    "proporcao_transmitida",
    "valor_venal_ref_prop",
    "base_calculo",
    "valor_financiado",
    "area_terreno",
    "testada",
    "fracao_ideal",
    "area_construida",
    "uso_cod",
    "padrao_cod",
    "ano_construcao",
]

MONTH_SHEET = re.compile(r"^[A-Z]{3}-\d{4}$")


def _read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, header=None, engine="calamine")
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS[: df.shape[1]]

    # descarta linha de cabeçalho quando ela existe
    first = df.iloc[0, 0] if len(df) else None
    if isinstance(first, str) and not first.strip().isdigit():
        df = df.iloc[1:]

    df = df.dropna(subset=["sql", "valor_transacao"], how="any")
    df["mes_ref"] = sheet
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # algumas abas repetem o cabeçalho no meio dos dados
    sql = pd.to_numeric(df["sql"], errors="coerce")
    df = df[sql.notna()].copy()
    df["sql"] = sql[sql.notna()].astype("int64").astype(str).str.zfill(11)
    df["cep"] = (
        pd.to_numeric(df["cep"], errors="coerce").astype("Int64").astype(str).str.zfill(8)
    )
    df.loc[df["cep"].str.contains("<NA>"), "cep"] = pd.NA
    df["data_transacao"] = pd.to_datetime(df["data_transacao"], errors="coerce")

    for col in ["natureza", "uso_desc", "padrao_desc", "tipo_financiamento", "bairro"]:
        df[col] = df[col].astype("string").str.strip()

    text = [c for c in COLUMNS if c not in NUMERIC and c != "data_transacao"]
    df[text] = df[text].astype("string")
    return df.reset_index(drop=True)


def read_year(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sheets = pd.ExcelFile(path, engine="calamine").sheet_names
    months = [s for s in sheets if MONTH_SHEET.match(s.strip())]
    if not months:
        raise ValueError(f"nenhuma aba mensal encontrada em {path.name}")
    frames = [_read_sheet(path, s) for s in months]
    return _normalize(pd.concat(frames, ignore_index=True))


def build(raw_dir: str | Path, out: str | Path) -> pd.DataFrame:
    raw_dir, out = Path(raw_dir), Path(out)
    files = sorted(raw_dir.glob("itbi_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"nenhum itbi_*.xlsx em {raw_dir}")

    df = pd.concat([read_year(f) for f in files], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Converte as planilhas de ITBI para parquet")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/interim/itbi.parquet")
    a = p.parse_args()

    df = build(a.raw, a.out)
    print(f"{len(df):,} linhas -> {a.out}")
