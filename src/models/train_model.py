import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def build_model() -> RandomForestRegressor:
  """
  Cria uma instância (não treinada) do modelo Random Forest Regressor.

  Centraliza os hiperparâmetros em um único lugar para que o treino e a validação
  cruzada usem exatamente a mesma configuração (evita código duplicado).

  :return: instância de RandomForestRegressor configurada.
  """
  # n_estimators=50: número de árvores. random_state=42: reprodutibilidade.
  # n_jobs=-1: usa todos os núcleos do processador para treinar mais rápido.
  # min_samples_leaf=50 e max_depth=20: "podam" as árvores para que não cresçam
  # até o fim. Isso combate o overfitting e, em uma base com milhões de linhas,
  # reduz drasticamente o tamanho do modelo em disco/RAM (de GBs para poucos MB),
  # viabilizando o deploy em uma VPS com memória limitada. A métrica praticamente
  # não muda, pois a relação entre as variáveis elétricas já é muito forte.
  return RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=50,
    max_depth=20,
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
