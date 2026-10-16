import numpy as np
import pandas as pd

from itbi import clean, features


def _row(**kw):
    base = {
        "sql": "00100800297",
        "cep": "01103020",
        "natureza": "1.Compra e venda",
        "uso_desc": "APARTAMENTO EM CONDOMÍNIO (EXIGE FRAÇÃO IDEAL)",
        "proporcao_transmitida": 100.0,
        "area_construida": 70.0,
        "valor_transacao": 500_000.0,
        "valor_venal_ref": 400_000.0,
        "data_transacao": pd.Timestamp("2024-05-10"),
        "ano_construcao": 2001.0,
        "mes_ref": "MAI-2024",
        "padrao_cod": 23.0,
        "area_terreno": 1200.0,
        "fracao_ideal": 0.01,
        "testada": 20.0,
    }
    base.update(kw)
    return base


def test_mantem_apartamento_valido():
    d = clean.apartments(pd.DataFrame([_row()]))
    assert len(d) == 1
    assert d.loc[0, "preco_m2"] == 500_000 / 70


def test_descarta_fora_do_recorte():
    rows = [
        _row(natureza="15.Permuta por escritura pública"),
        _row(uso_desc="RESIDÊNCIA"),
        _row(proporcao_transmitida=50.0),
        _row(area_construida=10.0),
        _row(valor_transacao=50_000.0),  # R$ 714/m²
        _row(valor_venal_ref=2_000_000.0),  # declarado muito abaixo do venal
        _row(ano_construcao=2030.0),
        _row(mes_ref="MAI-2026"),  # guia paga dois anos depois
    ]
    assert clean.apartments(pd.DataFrame(rows)).empty


def test_remove_guia_duplicada():
    d = clean.apartments(pd.DataFrame([_row(), _row(mes_ref="JUN-2024")]))
    assert len(d) == 1


def test_features():
    d = clean.apartments(pd.DataFrame([_row()]))
    X = features.make(d)
    assert X.loc[0, "padrao"] == 3
    assert X.loc[0, "idade"] == 23
    assert X.loc[0, "cep5"] == "01103"
    assert np.isclose(features.target(d)[0], np.log(500_000))
