import pytest

from src.utils.persistence import save_model, load_model, save_metrics


def test_save_e_load_model(tmp_path):
    """Um objeto salvo deve ser recuperado igual ao original."""
    objeto = {"a": 1, "b": [1, 2, 3]}
    destino = tmp_path / "subpasta" / "obj.joblib"
    save_model(objeto, str(destino))
    assert destino.exists()
    assert load_model(str(destino)) == objeto


def test_load_model_arquivo_inexistente():
    """Carregar um arquivo inexistente deve lançar FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model("models_saved/nao_existe.joblib")


def test_save_metrics_gera_json(tmp_path):
    """As métricas devem ser salvas como JSON válido e legível."""
    import json

    metricas = {"MAE": 0.1, "R2": 0.95}
    destino = tmp_path / "metrics" / "resultados.json"
    save_metrics(metricas, str(destino))
    assert destino.exists()
    with open(destino, encoding="utf-8") as f:
        assert json.load(f) == metricas
