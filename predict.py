"""
Script de inferencia: carrega o modelo e o scaler ja treinados (salvos em disco)
e gera previsoes para novos dados, sem precisar treinar tudo de novo.

Demonstra a reprodutibilidade/reexecucao exigida na secao de Persistencia:
    python predict.py
"""

import pandas as pd

from src.ingestion.data_loader import load_raw_data
from src.preprocessing.data_cleaner import clean_data
from src.features.feature_engineering import create_temporal_features
from src.utils.persistence import load_model


def gerar_previsoes(
    raw_filepath: str,
    model_path: str = "models_saved/random_forest.joblib",
    scaler_path: str = "models_saved/scaler.joblib",
    target_col: str = "Global_active_power",
    output_path: str = "metrics/previsoes.csv",
) -> pd.DataFrame:
    """
    Carrega artefatos salvos e gera previsoes para um arquivo de dados brutos.

    O fluxo reaproveita exatamente os mesmos passos do treino (limpeza, features e
    padronizacao) para garantir consistencia entre treino e inferencia.

    :param raw_filepath: caminho do arquivo de dados brutos a prever.
    :param model_path: caminho do modelo salvo (.joblib).
    :param scaler_path: caminho do scaler salvo (.joblib).
    :param target_col: nome da coluna alvo (removida das features, se presente).
    :param output_path: caminho do CSV onde as previsoes serao salvas.
    :return: DataFrame com a coluna 'previsao' (e o valor real, se existir).
    """
    # 1. Recarrega os artefatos persistidos
    modelo = load_model(model_path)
    scaler = load_model(scaler_path)

    # 2. Aplica o mesmo pre-processamento usado no treino
    df_raw = load_raw_data(raw_filepath)
    df_clean = clean_data(df_raw)
    df_feat = create_temporal_features(df_clean)

    # 3. Separa as features (preservando o alvo real para comparacao, se houver)
    y_real = df_feat[target_col] if target_col in df_feat.columns else None
    X = df_feat.drop(columns=[target_col]) if y_real is not None else df_feat

    # 3.1 Seleciona apenas as colunas que o modelo realmente usa (na mesma ordem
    # do treino). Garante consistencia mesmo que o arquivo bruto traga colunas
    # extras (ex.: Global_intensity/Voltage, removidas do conjunto de features).
    X = X[list(scaler.feature_names_in_)]

    # 4. Padroniza com o scaler ja ajustado e preve
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    previsoes = modelo.predict(X_scaled)

    resultado = pd.DataFrame({"previsao": previsoes}, index=X.index)
    if y_real is not None:
        resultado["real"] = y_real

    resultado.to_csv(output_path, encoding="utf-8")
    return resultado


def main():
    filepath = "data/raw/household_power_consumption.txt"
    try:
        print("Carregando modelo salvo e gerando previsoes...")
        resultado = gerar_previsoes(filepath)
        print(f"Previsoes geradas para {len(resultado)} registros.")
        print("Primeiras linhas:")
        print(resultado.head())
        print("\nArquivo salvo em metrics/previsoes.csv")
    except Exception as e:
        print(f"Falha na inferencia: {e}")


if __name__ == "__main__":
    main()
