from src.ingestion.data_loader import load_raw_data
from src.preprocessing.data_cleaner import clean_data
from src.features.feature_engineering import create_temporal_features
from src.preprocessing.data_transformer import prepare_and_split_data
from src.models.train_model import build_model, train_random_forest
from src.models.validate_model import cross_validate_timeseries
from src.evaluation.evaluate_model import (
    evaluate_regression,
    plot_feature_importance,
    plot_predictions_vs_actual,
    plot_residuals,
    save_metrics_table,
)
from src.utils.persistence import save_model, save_metrics, ensure_dir


def main():
  print("Iniciando o pipeline de Inteligência Computacional...")
  filepath = "data/raw/household_power_consumption.txt"

  try:
    # 1. Ingestão
    print("1. Carregando os dados brutos (isso pode levar alguns segundos)...")
    df_raw = load_raw_data(filepath)

    # 2. Pré-processamento
    print("2. Limpando os dados e ajustando a temporalidade...")
    df_clean = clean_data(df_raw)

    # 3. Engenharia de Features
    print("3. Criando novas features temporais...")
    df_processed = create_temporal_features(df_clean)

    # 3.1 Salva o dataset processado (limpo + com features) em disco.
    # Serve de insumo para a Análise Exploratória de Dados (notebooks/) e garante
    # reprodutibilidade. Salvo ANTES da remoção das colunas triviais, pois a EDA
    # ainda analisa Global_intensity/Voltage (ex.: na matriz de correlação).
    caminho_processado = "data/processed/processed_power_consumption.csv"
    print(f"3.1. Salvando dataset processado em {caminho_processado}...")
    ensure_dir("data/processed")
    df_processed.to_csv(caminho_processado, index_label="Datetime")

    # 3.2 Remoção de variáveis "óbvias" demais.
    # A potência ativa segue, na prática, P ≈ V × I (tensão x corrente). Como a
    # tensão (Voltage) é quase constante, a corrente (Global_intensity) sozinha
    # "explica" quase toda a variável alvo, tornando o problema trivial (R² ~ 0,99)
    # e concentrando toda a importância numa única feature. Ao removê-las, forçamos
    # o modelo a aprender a partir das sub-medições e das features temporais,
    # resultando em um problema de regressão mais realista e interpretável.
    colunas_removidas = ["Global_intensity", "Voltage"]
    print(f"3.2. Removendo variaveis triviais: {colunas_removidas}")
    df_processed = df_processed.drop(columns=colunas_removidas)

    # 4. Separação Treino/Teste e Padronização
    print("4. Separando em Treino/Teste (Hold-out cronológico) e aplicando StandardScaler...")
    X_train, X_test, y_train, y_test, scaler = prepare_and_split_data(
      df=df_processed,
      target_col='Global_active_power',
      test_size=0.2
    )

    # 4.7 Validação cruzada temporal (antes do treino final)
    print("5. Validando o modelo com TimeSeriesSplit (validação cruzada temporal)...")
    cv_resultados = cross_validate_timeseries(
      model=build_model(),
      X_train=X_train,
      y_train=y_train,
      n_splits=5,
    )
    print(f"   RMSE (CV): {cv_resultados['rmse_medio']:.4f} (+/- {cv_resultados['rmse_desvio']:.4f})")
    print(f"   R2   (CV): {cv_resultados['r2_medio']:.4f} (+/- {cv_resultados['r2_desvio']:.4f})")

    # 5. Modelagem (Treinamento)
    print("6. Treinando o modelo Random Forest Regressor (isso pode levar um tempo)...")
    modelo_treinado = train_random_forest(X_train, y_train)

    # 4.8 Avaliação no conjunto de teste (hold-out)
    print("7. Avaliando o modelo no conjunto de teste...")
    metricas = evaluate_regression(modelo_treinado, X_test, y_test)
    print(f"   MAE:  {metricas['MAE']:.4f}")
    print(f"   MSE:  {metricas['MSE']:.4f}")
    print(f"   RMSE: {metricas['RMSE']:.4f}")
    print(f"   R2:   {metricas['R2']:.4f}")

    # 8. Visualização dos resultados (gráficos e tabela)
    print("8. Gerando gráficos e tabela de métricas...")
    y_pred = modelo_treinado.predict(X_test)
    plot_feature_importance(
      modelo_treinado, list(X_train.columns), "metrics/importancia_features.png"
    )
    plot_predictions_vs_actual(y_test, y_pred, "metrics/real_vs_previsto.png")
    plot_residuals(y_test, y_pred, "metrics/residuos.png")
    tabela = save_metrics_table(metricas, "metrics/tabela_metricas.csv")
    print(tabela.to_string(index=False))

    # 7. Persistência (modelo, scaler e métricas)
    print("9. Salvando modelo, scaler e métricas...")
    save_model(modelo_treinado, "models_saved/random_forest.joblib")
    save_model(scaler, "models_saved/scaler.joblib")
    save_metrics(
      {"validacao_cruzada": cv_resultados, "teste_holdout": metricas},
      "metrics/resultados.json",
    )

    print("\nPipeline concluído com sucesso!")

  except Exception as e:
    print(f"Falha na execução: {e}")


if __name__ == "__main__":
  main()
