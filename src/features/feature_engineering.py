import pandas as pd

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas variáveis (features) baseadas no índice temporal para ajudar o modelo.
    
    :param df: DataFrame limpo e com índice Datetime.
    :return: DataFrame com novas colunas temporais (hour, day_of_week, month).
    """
    df_feat = df.copy()
    
    # Cria novas colunas baseadas no índice de tempo
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek # 0=Segunda, 6=Domingo
    df_feat['month'] = df_feat.index.month
    
    return df_feat
