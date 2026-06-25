from src.ingestion.data_loader import load_raw_data
from src.preprocessing.data_cleaner import clean_data
from src.features.feature_engineering import create_temporal_features

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
    
    print("Pré-processamento concluído com sucesso!\n")
    print(f"Novo formato (linhas, colunas): {df_processed.shape}")
    print("\nPrimeiras 5 linhas do dataset processado:")
    print(df_processed.head())
    
    # Salva uma amostra do dado processado para uso nos notebooks (Análise Exploratória)
    df_processed.to_csv('data/processed/processed_power_consumption.csv')
    print("\nDataset processado salvo em 'data/processed/processed_power_consumption.csv'")

  except Exception as e:
    print(f"Falha na execução: {e}")

if __name__ == "__main__":
  main()
