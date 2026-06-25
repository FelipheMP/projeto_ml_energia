import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
  """
  Limpa e formata os dados brutos de consumo de energia.
  
  :param df: DataFrame contendo os dados brutos.
  :return: DataFrame limpo, com índice temporal e colunas numéricas.
  """
  df_clean = df.copy()
  
  # Combina as colunas 'Date' e 'Time' em um único objeto datetime
  # O formato original da base é dd/mm/yyyy HH:MM:SS
  df_clean['Datetime'] = pd.to_datetime(df_clean['Date'] + ' ' + df_clean['Time'], format='%d/%m/%Y %H:%M:%S')
  
  # Define o Datetime como índice (essencial para séries temporais)
  df_clean = df_clean.set_index('Datetime')
  
  # Remove as colunas antigas de texto
  df_clean = df_clean.drop(columns=['Date', 'Time'])
  
  # Força a conversão de todas as colunas restantes para numérico (float)
  for col in df_clean.columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

  # Remove as linhas que possuem valores nulos (NaN)
  df_clean = df_clean.dropna()
  
  return df_clean
