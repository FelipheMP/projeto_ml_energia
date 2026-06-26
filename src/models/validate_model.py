import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from typing import Dict


def cross_validate_timeseries(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Valida o modelo usando validacao cruzada para series temporais (TimeSeriesSplit).

    Justificativa do metodo: em series temporais nao podemos usar o KFold tradicional,
    pois ele embaralha os dados e permitiria que o modelo "visse o futuro" durante o
    treino (data leakage). O TimeSeriesSplit cria dobras crescentes no tempo, sempre
    treinando no passado e validando no futuro, o que reflete o uso real do modelo.
    Esta validacao complementa o hold-out cronologico do conjunto de teste final.

    :param model: estimador do scikit-learn (ex: RandomForestRegressor) ainda nao treinado.
    :param X_train: features do conjunto de treino.
    :param y_train: alvo do conjunto de treino.
    :param n_splits: numero de dobras (folds) da validacao cruzada temporal.
    :return: dicionario com a media e o desvio padrao do RMSE e do R2 nas dobras.
    """
    if n_splits < 2:
        raise ValueError("n_splits deve ser >= 2 para realizar a validacao cruzada.")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # neg_root_mean_squared_error retorna valores negativos por convencao do sklearn
    neg_rmse_scores = cross_val_score(
        model, X_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error"
    )
    r2_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="r2")

    rmse_scores = -neg_rmse_scores

    return {
        "rmse_medio": float(np.mean(rmse_scores)),
        "rmse_desvio": float(np.std(rmse_scores)),
        "r2_medio": float(np.mean(r2_scores)),
        "r2_desvio": float(np.std(r2_scores)),
        "n_splits": n_splits,
    }
