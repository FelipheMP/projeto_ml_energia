from src.ingestion.data_loader import load_raw_data
from src.preprocessing.data_cleaner import clean_data
from src.features.feature_engineering import create_temporal_features
from src.preprocessing.data_transformer import prepare_and_split_data
from src.models.train_model import train_random_forest

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

    # 4. Separação Treino/Teste e Padronização
    print("4. Separando em Treino/Teste (Hold-out cronológico) e aplicando StandardScaler...")
    X_train, X_test, y_train, y_test, scaler = prepare_and_split_data(
      df=df_processed, 
      target_col='Global_active_power', 
      test_size=0.2
    )

    # 5. Modelagem (Treinamento)
    print("5. Treinando o modelo Random Forest Regressor (isso pode levar um tempo)...")
    modelo_treinado = train_random_forest(X_train, y_train)

    print("\nModelo treinado com sucesso!")

  except Exception as e:
    print(f"Falha na execução: {e}")

if __name__ == "__main__":
  main()
