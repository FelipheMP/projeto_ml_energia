"""
Arquivo de configuracao do pytest.

A  presenca deste arquivo na raiz do projeto faz o pytest inserir este
diretorio no sys.path durante a coleta dos testes. Assim, os imports absolutos
do tipo `from src.ingestion.data_loader import load_raw_data` funcionam tanto
localmente quanto no ambiente de CI (GitHub Actions), sem depender de PYTHONPATH.
"""
