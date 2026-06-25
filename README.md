# Projeto de Inteligência Computacional: Predição de Consumo de Energia Residencial

---

## 1. Objetivo e Descrição do Problema

O objetivo central deste projeto é estimar o consumo energético residencial com base em medições elétricas históricas, configurando um problema clássico de **Regressão**. O modelo visa compreender os padrões de consumo e prever a demanda futura.

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
├── notebooks/             # Jupyter Notebooks para Análise Exploratória de Dados (EDA)
├── src/                   # Código-fonte modularizado da aplicação
│   ├── ingestion/         # Módulo de carga e leitura de dados
│   ├── preprocessing/     # Módulos de limpeza, divisão temporal e padronização
│   ├── features/          # Criação de atributos baseados no tempo (hora, dia, mês)
│   ├── models/            # Implementação e treinamento dos algoritmos (Random Forest)
│   ├── evaluation/        # Rotinas para cálculo de métricas de regressão
│   └── utils/             # Funções utilitárias e de suporte
├── metrics/               # Diretório para exportação de resultados (gráficos e tabelas)
├── models_saved/          # Modelos binários persistidos (ex: arquivos .pkl ou .joblib)
├── tests/                 # Scripts de testes unitários para as funções principais
├── requirements.txt       # Lista de dependências e bibliotecas do projeto
├── main.py                # Orquestrador principal do pipeline de dados e treinamento
└── README.md              # Documentação oficial do projeto
```

---

## 3. Tecnologias e Bibliotecas

O pipeline foi desenvolvido utilizando a linguagem de programação Python (versão 3.10+) . As seguintes bibliotecas compõem a base do projeto:

- Manipulação e Análise de Dados: `pandas`, `numpy`
- Visualização Gráfica: `matplotlib`, `seaborn`
- Machine Learning e Pré-processamento: `scikit-learn` (implementação do Random Forest Regressor e StandardScaler)
- Testes Automatizados: `pytest`

---

## 4. Instruções de Execução

Para garantir a reprodutibilidade integral do projeto, siga os passos abaixo no terminal do seu sistema operacional.

### Passo 1: Clonar o repositório

```bash
git clone <URL_DO_SEU_REPOSITORIO_GITHUB>
cd projeto_ml
```

### Passo 2: Configurar o Ambiente Virtual

É altamente recomendado isolar as dependências do projeto utilizando o `venv`:

```bash
python3 -m venv venv
source venv/bin/activate  # No Linux/macOS
# ou no Windows: venv\Scripts\activate
```

### Passo 3: Instalar as Dependências

```bash
pip install -r requirements.txt
```

### Passo 4: Download dos Dados

Baixe o conjunto de dados *Individual Household Electric Power Consumption*

Extraia o arquivo `household_power_consumption.txt` e posicione-o dentro do diretório `data/raw/` do projeto.

### Passo 5: Executar o Pipeline Principal

Com o ambiente configurado e os dados posicionados, inicie a ingestão, o pré-processamento e o treinamento executando:

```bash
python main.py
```

## 5. Resultados e Métricas

(Seção a ser atualizada após a conclusão da etapa de testes e validação)

O modelo está sendo avaliado pelas seguintes métricas de regressão:
- **MAE** (Erro Absoluto Médio)
- **MSE** (Erro Quadrático Médio)
- **RMSE** (Raiz do Erro Quadrático Médio)
- **R²** (Coeficiente de Determinação)
 
Os gráficos de distribuição de resíduos, predição versus valor real e a matriz de importância das features (Feature Importance) serão armazenados no diretório `metrics/` .

## 6. Referências

HÉBRAIL, Georges; BÉRARD, Alice. **Individual Household Electric Power Consumption Dataset**. UCI Machine Learning Repository, 2012. Disponível em: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption. Acesso em: 20 de junho de 2026.

PEDREGOSA, Fabian et al. Scikit-learn: Machine Learning in Python. **Journal of Machine Learning Research**, v. 12, n. 85, p. 2825-2830, 2011.

PYTHON SOFTWARE FOUNDATION. **Python Language Reference, version 3.10**. Disponível em: https://www.python.org. Acesso em: 20 de junho de 2026.
