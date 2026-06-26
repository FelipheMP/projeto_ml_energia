# Projeto de Inteligência Computacional: Predição de Consumo de Energia Residencial

---

## 1. Objetivo e Descrição do Problema

O objetivo central deste projeto é estimar o consumo energético residencial com base em medições elétricas históricas, configurando um problema clássico de **Regressão**. O modelo visa compreender os padrões de consumo e prever a demanda.

A base de dados utilizada é o *Individual Household Electric Power Consumption Dataset*, obtida a partir do repositório público da UCI (University of California, Irvine). A variável alvo contínua a ser predita é a `Global_active_power` (Potência Ativa Global, medida em quilowatts).

Como os dados possuem uma forte dependência temporal (série temporal), o pipeline de pré-processamento foi projetado para aplicar uma divisão *hold-out* cronológica. Isso assegura que o modelo seja treinado com dados do passado e testado com dados do futuro, evitando o vazamento de informações (*data leakage*).

---

## 2. Estrutura do Repositório

Seguindo as boas práticas de Engenharia de Software e modularização, o projeto está organizado da seguinte forma:

```text
projeto_ml/
├── data/                  # Armazenamento de dados (ignorado no versionamento)
│   ├── raw/               # Dados brutos originais extraídos da UCI
│   └── processed/         # Dados limpos, normalizados e com engenharia de features
├── notebooks/             # Análise Exploratória de Dados (EDA) em Jupyter Notebook
├── src/                   # Código-fonte modularizado da aplicação
│   ├── ingestion/         # Módulo de carga e leitura de dados
│   ├── preprocessing/     # Módulos de limpeza, divisão temporal e padronização
│   ├── features/          # Criação de atributos baseados no tempo (hora, dia, mês)
│   ├── models/            # Treinamento (train_model) e validação temporal (validate_model)
│   ├── evaluation/        # Métricas de regressão e geração de gráficos/tabelas
│   └── utils/             # Persistência de modelos e métricas (joblib/JSON)
├── tests/                 # Testes unitários (pytest)
├── metrics/               # Resultados exportados (gráficos, tabela de métricas, JSON)
├── models_saved/          # Modelos e scaler persistidos (.joblib)
├── .github/workflows/     # Pipeline de CI/CD (GitHub Actions)
├── Dockerfile             # Containerização do projeto
├── docker-compose.yml     # Orquestração de API + dashboard + treino (deploy em VPS)
├── requirements.txt       # Dependências do projeto
├── download_data.py       # Baixa o dataset da UCI (via ucimlrepo) para data/raw/
├── main.py                # Orquestrador do pipeline (treino + avaliação + persistência)
├── predict.py             # Inferência reutilizando o modelo salvo (sem re-treinar)
├── api.py                 # API REST (FastAPI) para servir o modelo
├── dashboard.py           # Dashboard interativo (Streamlit)
└── README.md              # Documentação oficial do projeto
```

---

## 3. Tecnologias e Bibliotecas

O pipeline foi desenvolvido em Python (versão 3.10+). As principais bibliotecas:

- **Manipulação e Análise de Dados:** `pandas`, `numpy`
- **Visualização Gráfica:** `matplotlib`, `seaborn`
- **Machine Learning e Pré-processamento:** `scikit-learn` (Random Forest Regressor, StandardScaler, TimeSeriesSplit)
- **Persistência:** `joblib`
- **Testes Automatizados:** `pytest`
- **Download do dataset:** `ucimlrepo`
- **Diferenciais:** `fastapi` + `uvicorn` (API REST) e `streamlit` (dashboard)

> Todas as bibliotecas de núcleo (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `pytest`, `joblib`) seguem as recomendações do documento de requisitos da disciplina.

---

## 4. Análise Exploratória de Dados (EDA)

A EDA está no notebook [`notebooks/01_analise_exploratoria.ipynb`](notebooks/01_analise_exploratoria.ipynb) (já versionado **com os gráficos e tabelas renderizados**). Ela cobre:

- **Estatísticas descritivas** de todas as variáveis;
- **Distribuição** do consumo (histograma);
- **Análise de outliers** (boxplot + método IQR);
- **Correlação** entre atributos (heatmap de Pearson);
- **Visualização temporal** do consumo ao longo de uma semana.

> O notebook lê o arquivo `data/processed/processed_power_consumption.csv`, gerado automaticamente ao executar `python main.py`.

A EDA embasa as decisões do pipeline — em especial a **remoção das variáveis `Global_intensity` e `Voltage`** (ver Seção 6, motivada pela alta correlação observada) e a **criação das features temporais**.

---

## 5. Etapas do Pipeline

| Etapa | Módulo | Descrição |
|---|---|---|
| Ingestão | `src/ingestion/data_loader.py` | Lê o arquivo bruto da UCI (separador `;`, `?` como nulo). |
| Limpeza | `src/preprocessing/data_cleaner.py` | Cria índice temporal, converte para numérico e remove nulos. |
| Features | `src/features/feature_engineering.py` | Gera `hour`, `day_of_week` e `month` a partir do tempo. |
| Seleção de features | `main.py` | Remove `Global_intensity` e `Voltage` (ver Seção 6). |
| Split + Padronização | `src/preprocessing/data_transformer.py` | Hold-out cronológico (`shuffle=False`) + `StandardScaler`. |
| Validação | `src/models/validate_model.py` | Validação cruzada temporal com `TimeSeriesSplit`. |
| Treinamento | `src/models/train_model.py` | Treina o `RandomForestRegressor`. |
| Avaliação | `src/evaluation/evaluate_model.py` | Calcula MAE/MSE/RMSE/R² e gera gráficos e tabela. |
| Persistência | `src/utils/persistence.py` | Salva modelo, scaler e métricas em disco. |

### Treinamento e Validação

Por se tratar de **série temporal**, a validação usa `TimeSeriesSplit` (validação cruzada que sempre treina no passado e valida no futuro), complementando o hold-out cronológico do conjunto de teste final. Ambos evitam *data leakage*, que ocorreria com um `KFold` embaralhado.

Os hiperparâmetros do `RandomForestRegressor` foram **podados** propositalmente (`n_estimators=50`, `min_samples_leaf=50`, `max_depth=20`). Além de combater o *overfitting*, isso reduz drasticamente o tamanho do modelo em disco (de ~12 GB para ~98 MB numa base de milhões de linhas), viabilizando o **deploy em uma VPS com memória limitada** sem perda relevante de métrica.

---

## 6. Decisão de Engenharia de Features: remoção de variáveis triviais

A potência ativa segue, na prática, a relação física **P ≈ V × I** (potência = tensão × corrente). Na EDA, a `Global_intensity` (corrente) apresentou correlação de **~0,999** com a variável alvo, e a `Voltage` é praticamente constante.

Manter essas variáveis tornaria o problema **trivial** (R² ~ 0,99), com o modelo apenas "copiando" a corrente e concentrando toda a importância numa única feature. Por isso, ambas foram **removidas** do conjunto de features, forçando o modelo a aprender a partir das **sub-medições** e das **variáveis temporais**. O resultado é um problema de regressão mais realista e interpretável (R² ~ 0,86).

---

## 7. Instruções de Execução

### Passo 1: Clonar o repositório

```bash
git clone <URL_DO_SEU_REPOSITORIO_GITHUB>
cd projeto_ml
```

### Passo 2: Configurar o Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
# Windows: venv\Scripts\activate
```

### Passo 3: Instalar as Dependências

```bash
pip install -r requirements.txt
```

### Passo 4: Download dos Dados

Opção automática (recomendada), que baixa o dataset da UCI e o salva em `data/raw/`:

```bash
python download_data.py
```

> Alternativa manual: baixe o conjunto *Individual Household Electric Power Consumption* (UCI), extraia o `household_power_consumption.txt` e coloque-o em `data/raw/`.

### Passo 5: Executar o Pipeline Principal

```bash
python main.py
```

Ao final, o modelo (`models_saved/random_forest.joblib`), o scaler, as métricas (`metrics/resultados.json`, `metrics/tabela_metricas.csv`) e os gráficos serão gerados.

### Passo 6 (opcional): Gerar previsões com o modelo salvo

```bash
python predict.py     # carrega o modelo salvo e prevê sem re-treinar
```

---

## 8. Testes

Os testes unitários cobrem ingestão, limpeza, engenharia de features, split/padronização, validação, avaliação e persistência (incluindo verificações de entrada e tratamento de erros):

```bash
pytest -q
```

---

## 9. Resultados e Métricas

O modelo é avaliado pelas métricas de regressão **MAE**, **MSE**, **RMSE** e **R²**, salvas em `metrics/`. As visualizações geradas automaticamente são:

- `metrics/importancia_features.png` — importância das features
- `metrics/real_vs_previsto.png` — valores reais vs. previstos ao longo do tempo
- `metrics/residuos.png` — análise de resíduos (erro = real − previsto)
- `metrics/tabela_metricas.csv` — tabela consolidada das métricas

Resultados obtidos (modelo podado, **sem** `Global_intensity`/`Voltage`):

| Métrica | Validação cruzada (TimeSeriesSplit) | Teste (hold-out) |
|---|---|---|
| R² | ~0,77 | ~0,86 |
| RMSE | ~0,51 | ~0,34 |
| MAE | — | ~0,21 |

A análise de importância mostra que o consumo é puxado principalmente pelas **sub-medições** (eletrodomésticos), com influência relevante da **hora do dia** e do **mês** — um comportamento coerente e interpretável.

> Os valores numéricos são preenchidos automaticamente em `metrics/resultados.json` após executar `python main.py`.

---

## 10. Diferenciais

### API REST (FastAPI)

Serve o modelo treinado via HTTP:

```bash
uvicorn api:app --reload
```

- Documentação interativa: <http://127.0.0.1:8000/docs>
- Endpoint `POST /prever` recebe as variáveis da medição (`Global_reactive_power`, `Sub_metering_1/2/3`, `hour`, `day_of_week`, `month`) e retorna a previsão de `Global_active_power`.

### Dashboard (Streamlit)

Interface gráfica com métricas, gráficos e previsão manual:

```bash
streamlit run dashboard.py
```

### Docker

```bash
docker build -t projeto-ml-energia .
docker run --rm -p 8000:8000 projeto-ml-energia        # sobe a API
docker run --rm projeto-ml-energia python main.py      # roda o pipeline
```

### CI/CD (GitHub Actions)

O workflow em `.github/workflows/ci.yml` executa `pytest` na nuvem a cada *push* e *pull request* na branch `main`.

---

## 11. Deploy em VPS (Docker Compose)

O `docker-compose.yml` orquestra três serviços a partir de uma única imagem: **API** (porta 8000), **dashboard** (porta 8501) e um serviço **train** sob demanda. A API e o dashboard compartilham os volumes `models_saved/` e `metrics/`, portanto basta treinar uma vez.

> **Importante:** a imagem **não** contém o modelo treinado (ele é excluído via `.dockerignore` para manter a imagem leve). Por isso, antes de subir a API/dashboard é preciso garantir que exista um modelo em `models_saved/` — seja treinando na VPS, seja copiando o `.joblib` já treinado.

```bash
# 1. Coloque o dataset em data/raw/ (ou rode: docker compose run --rm train python download_data.py)

# 2. Treine o modelo uma única vez (gera models_saved/ e metrics/)
docker compose --profile train up train

# 3. Suba a API e o dashboard em background
docker compose up -d --build api dashboard

# Acompanhar logs / parar
docker compose logs -f
docker compose down
```

Após subir, a API fica em `http://<IP_DA_VPS>:8000/docs` e o dashboard em `http://<IP_DA_VPS>:8501`.

> **Alternativa rápida:** como o modelo podado tem apenas ~98 MB, é possível treinar localmente e copiar a pasta `models_saved/` para a VPS (`scp`), pulando o treino remoto. Depois disso, `docker compose up -d api dashboard` já sobe tudo funcionando.

> **Recomendado em produção:** colocar um proxy reverso (Nginx, Caddy ou Traefik) na frente para servir via domínio com HTTPS, em vez de expor as portas 8000/8501 diretamente.

---

## 12. Referências

HÉBRAIL, Georges; BÉRARD, Alice. **Individual Household Electric Power Consumption Dataset**. UCI Machine Learning Repository, 2012. Disponível em: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption. Acesso em: 20 de junho de 2026.

PEDREGOSA, Fabian et al. Scikit-learn: Machine Learning in Python. **Journal of Machine Learning Research**, v. 12, n. 85, p. 2825-2830, 2011.

PYTHON SOFTWARE FOUNDATION. **Python Language Reference, version 3.10**. Disponível em: https://www.python.org. Acesso em: 20 de junho de 2026.
