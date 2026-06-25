import pandas as pd
import os

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Carrega os dados brutos de consumo de energia elétrica residencial.

    :param filepath: Caminho para o arquivo de dados brutos (.txt ou .csv).
    :return: DataFrame do pandas contendo os dados brutos.
    :raises FileNotFoundError: Se o arquivo não for encontrado no caminho especificado.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}. Verifique se o dataset está na pasta data/raw/")

    try:
        # A base da UCI usa ';' como separador e '?' para dados ausentes
        df = pd.read_csv(
            filepath,
            sep=';',
            na_values=['?'],
            low_memory=False
        )
        return df
    except Exception as e:
        raise Exception(f"Erro inesperado ao carregar os dados: {e}")
