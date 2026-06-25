from src.ingestion.data_loader import load_raw_data

def main():
    print("Iniciando o pipeline de Inteligência Computacional...")
    filepath = "data/raw/household_power_consumption.txt"

    try:
        print("Carregando os dados brutos...")
        df = load_raw_data(filepath)
        print("Dados carregados com sucesso!\n")
        
        print(f"Total de registros e variáveis (linhas, colunas): {df.shape}")
        print("\nPrimeiras 5 linhas do dataset:")
        print(df.head())
        
    except Exception as e:
        print(f"Falha na execução: {e}")

if __name__ == "__main__":
    main()
