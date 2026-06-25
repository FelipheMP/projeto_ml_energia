import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def prepare_and_split_data(
    df: pd.DataFrame, 
    target_col: str = 'Global_active_power', 
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Separa os dados em treino e teste e aplica padronização (StandardScaler).
    Justificativa: Como é uma série temporal, shuffle=False é obrigatório para manter 
    a ordem cronológica e evitar vazamento de dados do futuro para o passado (data leakage).
    
    :param df: DataFrame processado com as features.
    :param target_col: Nome da coluna alvo (target).
    :param test_size: Proporção de dados para teste (ex: 0.2 para 20%).
    :return: X_train, X_test, y_train, y_test e o objeto scaler ajustado.
    """
    # Separa as features (X) do alvo (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Separação hold-out cronológica (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Padronização (Standardization) baseada apenas no conjunto de treino
    scaler = StandardScaler()
    
    # O fit_transform aprende os padrões do treino e já aplica a transformação
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    
    # O transform apenas aplica a transformação no teste (evita data leakage)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
