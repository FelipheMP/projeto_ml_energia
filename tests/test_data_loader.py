import pandas as pd
import pytest

from src.ingestion.data_loader import load_raw_data


def test_load_raw_data_arquivo_inexistente():
    """Deve lançar FileNotFoundError quando o arquivo não existe (verificação de entrada)."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("data/raw/arquivo_que_nao_existe.txt")


def test_load_raw_data_le_csv(tmp_path):
    """Deve carregar corretamente um arquivo no formato da base (separador ';' e '?' como NaN)."""
    conteudo = (
        "Date;Time;Global_active_power\n"
        "16/12/2006;17:24:00;4.216\n"
        "16/12/2006;17:25:00;?\n"
    )
    arquivo = tmp_path / "amostra.txt"
    arquivo.write_text(conteudo, encoding="utf-8")

    df = load_raw_data(str(arquivo))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["Date", "Time", "Global_active_power"]
    # O '?' deve ter sido convertido em NaN
    assert df["Global_active_power"].isna().sum() == 1
