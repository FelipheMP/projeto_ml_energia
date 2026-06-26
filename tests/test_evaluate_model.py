import numpy as np
import pandas as pd
import pytest

from src.models.train_model import build_model, train_random_forest
from src.models.validate_model import cross_validate_timeseries
from src.evaluation.evaluate_model import (
    evaluate_regression,
    plot_feature_importance,
    plot_predictions_vs_actual,
    plot_residuals,
    build_metrics_table,
    save_metrics_table,
)


def _dados_treino_teste(n=120):
    """Gera dados sintéticos com relação simples para treinar/avaliar um modelo."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )
    y = pd.Series(2 * X["f1"] - X["f2"] + rng.normal(scale=0.1, size=n), name="alvo")
    corte = int(n * 0.8)
    return X.iloc[:corte], X.iloc[corte:], y.iloc[:corte], y.iloc[corte:]


def test_evaluate_regression_retorna_todas_metricas():
    """O dicionário de avaliação deve conter MAE, MSE, RMSE e R2."""
    X_train, X_test, y_train, y_test = _dados_treino_teste()
    modelo = train_random_forest(X_train, y_train)
    metricas = evaluate_regression(modelo, X_test, y_test)
    assert set(metricas.keys()) == {"MAE", "MSE", "RMSE", "R2"}


def test_evaluate_regression_rmse_consistente_com_mse():
    """RMSE deve ser a raiz quadrada do MSE."""
    X_train, X_test, y_train, y_test = _dados_treino_teste()
    modelo = train_random_forest(X_train, y_train)
    metricas = evaluate_regression(modelo, X_test, y_test)
    assert metricas["RMSE"] == pytest.approx(np.sqrt(metricas["MSE"]))


def test_cross_validate_timeseries_retorna_resumo():
    """A validação cruzada temporal deve retornar média e desvio de RMSE e R2."""
    X_train, _, y_train, _ = _dados_treino_teste(150)
    resultado = cross_validate_timeseries(build_model(), X_train, y_train, n_splits=3)
    assert resultado["n_splits"] == 3
    assert "rmse_medio" in resultado
    assert "r2_medio" in resultado


def test_cross_validate_timeseries_n_splits_invalido():
    """n_splits < 2 deve lançar ValueError (verificação de entrada)."""
    X_train, _, y_train, _ = _dados_treino_teste()
    with pytest.raises(ValueError):
        cross_validate_timeseries(build_model(), X_train, y_train, n_splits=1)


def test_plot_feature_importance_gera_arquivo(tmp_path):
    """O gráfico de importância das features deve ser salvo em disco."""
    X_train, _, y_train, _ = _dados_treino_teste()
    modelo = train_random_forest(X_train, y_train)
    destino = tmp_path / "importancia.png"
    plot_feature_importance(modelo, list(X_train.columns), str(destino))
    assert destino.exists()


def test_plot_predictions_vs_actual_gera_arquivo(tmp_path):
    """O gráfico real vs. previsto deve ser salvo em disco."""
    X_train, X_test, y_train, y_test = _dados_treino_teste()
    modelo = train_random_forest(X_train, y_train)
    y_pred = modelo.predict(X_test)
    destino = tmp_path / "real_vs_previsto.png"
    plot_predictions_vs_actual(y_test, y_pred, str(destino))
    assert destino.exists()


def test_plot_residuals_gera_arquivo(tmp_path):
    """O gráfico de resíduos deve ser salvo em disco."""
    X_train, X_test, y_train, y_test = _dados_treino_teste()
    modelo = train_random_forest(X_train, y_train)
    y_pred = modelo.predict(X_test)
    destino = tmp_path / "residuos.png"
    plot_residuals(y_test, y_pred, str(destino))
    assert destino.exists()


def test_build_metrics_table_estrutura():
    """A tabela de métricas deve ter as colunas 'Metrica' e 'Valor'."""
    tabela = build_metrics_table({"MAE": 0.1, "R2": 0.95})
    assert list(tabela.columns) == ["Metrica", "Valor"]
    assert len(tabela) == 2


def test_save_metrics_table_gera_csv(tmp_path):
    """A tabela de métricas deve ser salva como CSV."""
    destino = tmp_path / "tabela.csv"
    save_metrics_table({"MAE": 0.1, "RMSE": 0.3}, str(destino))
    assert destino.exists()
