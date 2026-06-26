import numpy as np
import pandas as pd

from src.preprocessing.data_transformer import prepare_and_split_data


def _df_processado_exemplo(n=100):
    """DataFrame numérico com alvo e features, simulando dados já processados."""
    indice = pd.date_range("2007-01-01", periods=n, freq="min")
    return pd.DataFrame(
        {
            "Global_active_power": np.linspace(1.0, 5.0, n),
            "hour": np.arange(n) % 24,
            "month": (np.arange(n) % 12) + 1,
        },
        index=indice,
    )


def test_split_respeita_test_size():
    """A proporção treino/teste deve corresponder ao test_size informado."""
    df = _df_processado_exemplo(100)
    X_train, X_test, y_train, y_test, _ = prepare_and_split_data(df, test_size=0.2)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_split_cronologico_sem_embaralhar():
    """O split deve ser cronológico: todo o treino vem antes do teste no tempo."""
    df = _df_processado_exemplo(100)
    X_train, X_test, _, _, _ = prepare_and_split_data(df, test_size=0.2)
    assert X_train.index.max() < X_test.index.min()


def test_alvo_nao_aparece_nas_features():
    """A coluna alvo não deve estar presente em X (evita vazamento trivial)."""
    df = _df_processado_exemplo(50)
    X_train, X_test, _, _, _ = prepare_and_split_data(df, test_size=0.2)
    assert "Global_active_power" not in X_train.columns
    assert "Global_active_power" not in X_test.columns


def test_padronizacao_media_zero_no_treino():
    """Após o StandardScaler, as features de treino devem ter média ~0."""
    df = _df_processado_exemplo(100)
    X_train, _, _, _, _ = prepare_and_split_data(df, test_size=0.2)
    assert np.allclose(X_train.mean().values, 0, atol=1e-9)
