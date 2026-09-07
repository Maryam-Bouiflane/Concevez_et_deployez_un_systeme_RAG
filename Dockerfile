# Image Python
FROM python:3.12-slim

# Configuration Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Répertoire de travail
WORKDIR /app

# Installation de uv
RUN pip install --no-cache-dir uv

# Copie des fichiers de dépendances
COPY pyproject.toml uv.lock ./

# Installation des dépendances
RUN uv sync --frozen --no-dev

# Copie du code de l'application
COPY app ./app

# Copie de l'index FAISS et des métadonnées
COPY data ./data

# Port de l'API
EXPOSE 8000

# Démarrage de FastAPI
CMD ["uv", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

