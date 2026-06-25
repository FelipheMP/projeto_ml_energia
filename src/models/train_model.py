import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
  """
  Treina modelo Random Forest para predição de consumo.
  
  :param X_train: features de treino
  :param y_train: labels de treino
  :return: modelo treinado
  """
  # Instanciando o modelo (usando n_estimators=100 e random_state para reprodutibilidade)
  # n_jobs=-1 permite que o modelo use todos os núcleos do processador para treinar mais rápido
  model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
  
  # Treinando o modelo
  model.fit(X_train, y_train)
  
  return model
