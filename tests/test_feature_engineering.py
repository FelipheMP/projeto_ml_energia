import pandas as pd

from src.features.feature_engineering import create_temporal_features


def _df_temporal_exemplo():
    """DataFrame com índice temporal para testar a criação de features."""
    indice = pd.to_datetime(
        ["2006-12-16 17:24:00", "2006-12-17 09:00:00", "2007-01-01 00:00:00"]
    )
    return pd.DataFrame({"Global_active_power": [4.2, 3.1, 2.0]}, index=indice)


def test_create_temporal_features_adiciona_colunas():
    """Devem ser criadas as colunas hour, day_of_week e month."""
    df_feat = create_temporal_features(_df_temporal_exemplo())
    for coluna in ["hour", "day_of_week", "month"]:
        assert coluna in df_feat.columns


def test_create_temporal_features_valores_corretos():
    """Os valores extraídos do índice devem corresponder à data/hora."""
    df_feat = create_temporal_features(_df_temporal_exemplo())
    assert df_feat["hour"].iloc[0] == 17
    assert df_feat["day_of_week"].iloc[0] == 5  # 16/12/2006 é um sábado
    assert df_feat["month"].iloc[2] == 1  # janeiro


def test_create_temporal_features_nao_altera_original():
    """A função não deve modificar o DataFrame original (usa cópia)."""
    df_original = _df_temporal_exemplo()
    create_temporal_features(df_original)
    assert "hour" not in df_original.columns
