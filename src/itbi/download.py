"""Baixa as planilhas anuais de guias de ITBI pagas da Prefeitura de São Paulo.

Fonte: https://prefeitura.sp.gov.br/web/fazenda/w/acesso_a_informacao/31501

Os nomes dos arquivos mudam a cada atualização, então os links ficam fixos
aqui e precisam ser revistos quando a prefeitura publicar uma versão nova.
"""

from __future__ import annotations

from pathlib import Path

import requests

BASE = "https://prefeitura.sp.gov.br/cidade/secretarias/upload/fazenda/arquivos/itbi"

URLS = {
    2024: f"{BASE}/GUIAS-DE-ITBI-PAGAS-2024.xlsx",
    2025: f"{BASE}/GUIAS%20DE%20ITBI%20PAGAS%20%2828012026%29%20XLS.xlsx",
}

# o servidor recusa requisições sem user-agent de navegador
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch(year: int, dest: Path, force: bool = False) -> Path:
    out = dest / f"itbi_{year}.xlsx"
    if out.exists() and not force:
        print(f"{out} já existe")
        return out

    dest.mkdir(parents=True, exist_ok=True)
    with requests.get(URLS[year], headers=HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = out.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(out)
    print(f"{out} ({out.stat().st_size / 1e6:.0f} MB)")
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("anos", nargs="*", type=int, default=sorted(URLS))
    p.add_argument("--dest", default="data/raw")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    for ano in a.anos:
        fetch(ano, Path(a.dest), a.force)
