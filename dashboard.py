"""
Dashboard interativo (Streamlit) para o projeto de previsao de consumo de energia.

Mostra as metricas e os graficos gerados pelo pipeline e permite fazer previsoes
manuais carregando o modelo treinado.

Para executar:
    streamlit run dashboard.py
"""

import json
import os

import pandas as pd
import streamlit as st

from src.utils.persistence import load_model

MODEL_PATH = "models_saved/random_forest.joblib"
SCALER_PATH = "models_saved/scaler.joblib"
METRICS_JSON = "metrics/resultados.json"

st.set_page_config(page_title="Previsao de Consumo de Energia", layout="wide")
st.title("Previsao de Consumo de Energia Eletrica")
st.caption("Random Forest Regressor - Household Power Consumption (UCI)")


def _secao_metricas():
    """Exibe a tabela de metricas salva pelo pipeline."""
    st.header("Metricas do modelo")
    if not os.path.exists(METRICS_JSON):
        st.warning("Métricas ainda não geradas. Rode `python main.py` primeiro.")
        return
    with open(METRICS_JSON, encoding="utf-8") as f:
        resultados = json.load(f)

    teste = resultados.get("teste_holdout", {})
    colunas = st.columns(len(teste) or 1)
    for coluna, (nome, valor) in zip(colunas, teste.items()):
        coluna.metric(nome, f"{valor:.4f}")


def _secao_graficos():
    """Exibe os graficos PNG gerados na avaliacao."""
    st.header("Visualizacoes")
    graficos = {
        "Importancia das features": "metrics/importancia_features.png",
        "Real vs. Previsto": "metrics/real_vs_previsto.png",
        "Residuos": "metrics/residuos.png",
    }
    for titulo, caminho in graficos.items():
        if os.path.exists(caminho):
            st.subheader(titulo)
            st.image(caminho, use_container_width=True)


def _secao_previsao():
    """Formulario para previsao manual usando o modelo salvo."""
    st.header("Fazer uma previsao")
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        st.warning("Modelo não encontrado. Rode `python main.py` para treinar e salvar.")
        return

    modelo = load_model(MODEL_PATH)
    scaler = load_model(SCALER_PATH)

    col1, col2, col3 = st.columns(3)
    entrada = {
        "Global_reactive_power": col1.number_input("Global_reactive_power", value=0.418),
        "Sub_metering_1": col1.number_input("Sub_metering_1", value=0.0),
        "Sub_metering_2": col2.number_input("Sub_metering_2", value=1.0),
        "Sub_metering_3": col2.number_input("Sub_metering_3", value=17.0),
        "hour": col3.slider("Hora", 0, 23, 17),
        "day_of_week": col3.slider("Dia da semana (0=Seg)", 0, 6, 5),
        "month": col3.slider("Mes", 1, 12, 12),
    }

    if st.button("Prever consumo"):
        df = pd.DataFrame([entrada])[list(scaler.feature_names_in_)]
        previsao = float(modelo.predict(scaler.transform(df))[0])
        st.success(f"Global_active_power previsto: **{previsao:.4f} kW**")


_secao_metricas()
_secao_graficos()
_secao_previsao()
