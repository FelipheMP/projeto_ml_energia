import pandas as pd

from src.preprocessing.data_cleaner import clean_data


def _df_bruto_exemplo():
    """Cria um DataFrame bruto pequeno no mesmo formato da base original."""
    return pd.DataFrame(
        {
            "Date": ["16/12/2006", "16/12/2006", "16/12/2006"],
            "Time": ["17:24:00", "17:25:00", "17:26:00"],
            "Global_active_power": ["4.216", "5.360", None],  # último vira NaN
        }
    )


def test_clean_data_cria_indice_datetime():
    """O DataFrame limpo deve ter um índice do tipo DatetimeIndex."""
    df_limpo = clean_data(_df_bruto_exemplo())
    assert isinstance(df_limpo.index, pd.DatetimeIndex)


def test_clean_data_remove_colunas_de_texto():
    """As colunas 'Date' e 'Time' devem ser removidas após a limpeza."""
    df_limpo = clean_data(_df_bruto_exemplo())
    assert "Date" not in df_limpo.columns
    assert "Time" not in df_limpo.columns


def test_clean_data_remove_nulos_e_converte_numerico():
    """Linhas com valores nulos devem ser removidas e as colunas viram numéricas."""
    df_limpo = clean_data(_df_bruto_exemplo())
    assert len(df_limpo) == 2  # a linha com None foi descartada
    assert pd.api.types.is_numeric_dtype(df_limpo["Global_active_power"])
