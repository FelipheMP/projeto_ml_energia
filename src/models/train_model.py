import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def build_model() -> RandomForestRegressor:
  """
  Cria uma instância (não treinada) do modelo Random Forest Regressor.

  Centraliza os hiperparâmetros em um único lugar para que o treino e a validação
  cruzada usem exatamente a mesma configuração (evita código duplicado).

  :return: instância de RandomForestRegressor configurada.
  """
  # n_estimators=100: número de árvores. random_state=42: reprodutibilidade.
  # n_jobs=-1: usa todos os núcleos do processador para treinar mais rápido.
  # min_samples_leaf=1 e max_features=1.0: explicitados para deixar a configuração
  # reprodutível e clara (são os padrões do regressor).
  return RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    max_features=1.0,
  )


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
  """
  Treina modelo Random Forest para predição de consumo.

  :param X_train: features de treino
  :param y_train: labels de treino
  :return: modelo treinado
  """
  model = build_model()
  model.fit(X_train, y_train)
  return model
