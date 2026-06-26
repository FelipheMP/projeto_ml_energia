"""
Baixa o dataset 'Individual Household Electric Power Consumption' (UCI, id=235)
e o salva em data/raw/household_power_consumption.txt no formato esperado pelo
pipeline (separador ';', '?' para valores ausentes, colunas Date/Time).

Uso:
    python download_data.py
"""

import os

import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_DIR = "data/raw"
RAW_PATH = os.path.join(RAW_DIR, "household_power_consumption.txt")

# Ordem original das colunas no arquivo bruto da UCI
COLUNAS_ORDENADAS = [
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def _normalizar_datas(df: pd.DataFrame) -> pd.DataFrame:
    """Garante Date como dd/mm/yyyy e Time como HH:MM:SS (formato do cleaner)."""
    if pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")
    else:
        df["Date"] = df["Date"].astype(str)

    if pd.api.types.is_datetime64_any_dtype(df["Time"]):
        df["Time"] = df["Time"].dt.strftime("%H:%M:%S")
    else:
        df["Time"] = df["Time"].astype(str)

    return df


def main():
    print("Baixando o dataset da UCI (id=235)... pode levar alguns minutos.")
    ds = fetch_ucirepo(id=235)

    # Junta features + target num unico DataFrame
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    print(f"Colunas recebidas: {list(df.columns)}")

    faltando = [c for c in COLUNAS_ORDENADAS if c not in df.columns]
    if faltando:
        print(f"Aviso: colunas esperadas ausentes {faltando}. Salvando na ordem recebida.")
    else:
        df = df[COLUNAS_ORDENADAS]

    df = _normalizar_datas(df)

    os.makedirs(RAW_DIR, exist_ok=True)
    # na_rep='?' reescreve os nulos como '?', exatamente o que o loader espera
    df.to_csv(RAW_PATH, sep=";", index=False, na_rep="?")
    print(f"Arquivo salvo em {RAW_PATH} ({len(df):,} linhas).")


if __name__ == "__main__":
    main()
