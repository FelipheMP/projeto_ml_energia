"""
API REST (FastAPI) para servir o modelo de previsao de consumo de energia.

Sobe um endpoint que recebe as variaveis de uma medicao e retorna a previsao
do Global_active_power, reutilizando o modelo e o scaler ja treinados.

Para executar localmente:
    uvicorn api:app --reload
Documentacao interativa em: http://127.0.0.1:8000/docs
"""

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.persistence import load_model

MODEL_PATH = "models_saved/random_forest.joblib"
SCALER_PATH = "models_saved/scaler.joblib"

app = FastAPI(
    title="API de Previsao de Consumo de Energia",
    description="Preve o Global_active_power a partir das medicoes eletricas.",
    version="1.0.0",
)

# Carregados sob demanda para nao quebrar a inicializacao caso ainda nao exista modelo
_modelo = None
_scaler = None


def _carregar_artefatos():
    """Carrega (uma unica vez) o modelo e o scaler salvos em disco."""
    global _modelo, _scaler
    if _modelo is None or _scaler is None:
        _modelo = load_model(MODEL_PATH)
        _scaler = load_model(SCALER_PATH)
    return _modelo, _scaler


class MedicaoEntrada(BaseModel):
    """Variaveis de entrada para uma previsao (uma medicao eletrica)."""

    Global_reactive_power: float = Field(..., example=0.418)
    Sub_metering_1: float = Field(..., example=0.0)
    Sub_metering_2: float = Field(..., example=1.0)
    Sub_metering_3: float = Field(..., example=17.0)
    hour: int = Field(..., ge=0, le=23, example=17)
    day_of_week: int = Field(..., ge=0, le=6, example=5)
    month: int = Field(..., ge=1, le=12, example=12)


class PrevisaoSaida(BaseModel):
    """Resposta da previsao."""

    global_active_power_previsto: float


@app.get("/")
def raiz():
    """Endpoint de checagem rapida (health check)."""
    return {"status": "ok", "mensagem": "API de previsao de consumo de energia no ar."}


@app.post("/prever", response_model=PrevisaoSaida)
def prever(medicao: MedicaoEntrada):
    """
    Preve o consumo (Global_active_power) para uma medicao.

    :param medicao: variaveis eletricas e temporais da medicao.
    :return: valor previsto de Global_active_power.
    """
    try:
        modelo, scaler = _carregar_artefatos()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Modelo nao encontrado. Treine o modelo antes (python main.py).",
        )

    # Monta o DataFrame e garante a MESMA ordem de colunas usada no treino
    entrada = pd.DataFrame([medicao.model_dump()])
    entrada = entrada[list(scaler.feature_names_in_)]

    entrada_padronizada = scaler.transform(entrada)
    previsao = float(modelo.predict(entrada_padronizada)[0])

    return PrevisaoSaida(global_active_power_previsto=previsao)
