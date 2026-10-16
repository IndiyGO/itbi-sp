from datetime import datetime

import pandas as pd

from itbi import ingest

LINHA = [
    100800297, "R S LAZARO", 295, None, None, None, 1103020, "1.Compra e venda",
    450000, datetime(2024, 1, 16), 190531, 100, 190531, 450000, None, 0,
    "5º Cartório de Registro de Imóvel", 42915, "Ativo Predial", 72, 5.9, 1, 120,
    10, "RESIDÊNCIA", 12, "RESIDENCIAL HORIZONTAL", 1939,
]


def _planilha(tmp_path, abas):
    path = tmp_path / "itbi_2024.xlsx"
    with pd.ExcelWriter(path) as w:
        for nome, linhas in abas.items():
            pd.DataFrame(linhas).to_excel(w, sheet_name=nome, header=False, index=False)
    return path


def test_le_abas_com_e_sem_cabecalho(tmp_path):
    cabecalho = ["Nº do Cadastro (SQL)"] + [f"col{i}" for i in range(27)]
    path = _planilha(
        tmp_path,
        {
            "JAN-2024": [LINHA, LINHA],
            "FEV-2024": [cabecalho, LINHA, cabecalho, LINHA],
            "LEGENDA": [["qualquer coisa"]],
        },
    )
    df = ingest.read_year(path)

    assert len(df) == 4
    assert set(df["mes_ref"]) == {"JAN-2024", "FEV-2024"}
    assert (df["sql"] == "00100800297").all()
    assert (df["cep"] == "01103020").all()
    assert df["valor_transacao"].dtype == float
