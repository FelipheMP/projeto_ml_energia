import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem interface grafica (permite salvar sem abrir janelas)
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict


def evaluate_regression(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Avalia um modelo de regressao no conjunto de teste calculando as metricas padrao.

    Metricas retornadas:
        - MAE  (Mean Absolute Error): erro medio absoluto, na mesma unidade do alvo.
        - MSE  (Mean Squared Error): erro quadratico medio, penaliza erros grandes.
        - RMSE (Root Mean Squared Error): raiz do MSE, na mesma unidade do alvo.
        - R2   (Coeficiente de determinacao): fracao da variancia explicada (1.0 = perfeito).

    :param model: modelo ja treinado.
    :param X_test: features do conjunto de teste.
    :param y_test: valores reais do alvo no conjunto de teste.
    :return: dicionario com as metricas MAE, MSE, RMSE e R2.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def plot_feature_importance(
    model: BaseEstimator,
    feature_names: list,
    output_path: str,
) -> None:
    """
    Gera e salva um grafico de barras com a importancia das features do modelo.

    :param model: modelo treinado que possua o atributo feature_importances_
                  (ex: RandomForestRegressor).
    :param feature_names: lista com os nomes das features, na ordem das colunas de treino.
    :param output_path: caminho do arquivo de imagem a ser salvo (ex: metrics/importancia.png).
    :raises AttributeError: se o modelo nao expor feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("O modelo nao possui o atributo 'feature_importances_'.")

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="#2c7fb8")
    plt.title("Importancia das Features")
    plt.xlabel("Importancia relativa")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_residuals(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_path: str,
) -> None:
    """
    Gera e salva um grafico de dispersao dos residuos (erro = real - previsto).

    Interpretacao: residuos distribuidos aleatoriamente em torno de zero indicam um
    bom ajuste. Padroes (curvas, funis) sugerem que o modelo nao capturou alguma
    estrutura dos dados.

    :param y_test: valores reais do alvo.
    :param y_pred: valores previstos pelo modelo.
    :param output_path: caminho do arquivo de imagem a ser salvo.
    """
    residuos = y_test.values - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuos, s=6, alpha=0.3, color="#2c7fb8")
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1)
    plt.title("Analise de Residuos")
    plt.xlabel("Valor previsto")
    plt.ylabel("Residuo (real - previsto)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def build_metrics_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Converte o dicionario de metricas em uma tabela (DataFrame) com duas colunas.

    :param metrics: dicionario com nomes de metricas e seus valores.
    :return: DataFrame com as colunas 'Metrica' e 'Valor'.
    """
    return pd.DataFrame(
        {"Metrica": list(metrics.keys()), "Valor": list(metrics.values())}
    )


def save_metrics_table(metrics: Dict[str, float], output_path: str) -> pd.DataFrame:
    """
    Salva a tabela de metricas em CSV e tambem imprime uma versao em Markdown.

    :param metrics: dicionario com as metricas de avaliacao.
    :param output_path: caminho do arquivo CSV de saida (ex: metrics/tabela_metricas.csv).
    :return: o DataFrame da tabela gerada.
    """
    tabela = build_metrics_table(metrics)
    tabela.to_csv(output_path, index=False, encoding="utf-8")
    return tabela


def plot_predictions_vs_actual(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_path: str,
    n_points: int = 500,
) -> None:
    """
    Gera e salva um grafico comparando os valores reais e previstos ao longo do tempo.

    Como o conjunto de teste pode ser muito grande, apenas os primeiros n_points sao
    plotados para manter o grafico legivel.

    :param y_test: valores reais do alvo (com indice temporal).
    :param y_pred: valores previstos pelo modelo.
    :param output_path: caminho do arquivo de imagem a ser salvo.
    :param n_points: quantidade de pontos a exibir no grafico.
    """
    recorte = min(n_points, len(y_test))

    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values[:recorte], label="Real", color="#1f77b4", linewidth=1)
    plt.plot(y_pred[:recorte], label="Previsto", color="#ff7f0e", linewidth=1, alpha=0.8)
    plt.title(f"Real vs. Previsto (primeiros {recorte} pontos do teste)")
    plt.xlabel("Amostra")
    plt.ylabel("Global_active_power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
