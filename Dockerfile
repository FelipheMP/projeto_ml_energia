# Imagem base enxuta com Python 3.12
FROM python:3.12-slim

# Evita arquivos .pyc e garante logs sem buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala as dependencias primeiro (aproveita o cache de camadas do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do projeto
COPY . .

# Expoe a porta da API REST (FastAPI)
EXPOSE 8000

# Por padrao sobe a API. Para rodar o pipeline use:
#   docker run --rm projeto-ml-energia python main.py
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
