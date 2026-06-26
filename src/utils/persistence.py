import json
import os
import joblib
from typing import Any, Dict


def ensure_dir(directory: str) -> None:
    """
    Garante que um diretorio exista, criando-o se necessario.

    :param directory: caminho do diretorio.
    """
    os.makedirs(directory, exist_ok=True)


def save_model(model: Any, output_path: str) -> None:
    """
    Persiste um objeto (modelo, scaler, etc.) em disco usando joblib.

    :param model: objeto a ser salvo (ex: modelo treinado ou scaler).
    :param output_path: caminho do arquivo de saida (ex: models_saved/modelo.joblib).
    """
    ensure_dir(os.path.dirname(output_path) or ".")
    joblib.dump(model, output_path)


def load_model(input_path: str) -> Any:
    """
    Carrega um objeto previamente salvo com joblib.

    :param input_path: caminho do arquivo salvo.
    :return: objeto desserializado.
    :raises FileNotFoundError: se o arquivo nao existir.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo de modelo nao encontrado: {input_path}")
    return joblib.load(input_path)


def save_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Salva um dicionario de metricas em formato JSON (legivel e reprodutivel).

    :param metrics: dicionario com nomes de metricas e seus valores.
    :param output_path: caminho do arquivo JSON de saida (ex: metrics/resultados.json).
    """
    ensure_dir(os.path.dirname(output_path) or ".")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
