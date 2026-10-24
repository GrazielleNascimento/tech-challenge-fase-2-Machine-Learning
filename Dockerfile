# Imagem unica do projeto: executa o pipeline DVC, o servidor MLflow
# e o serving do modelo campeao (ver docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.5.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    # O .git fica fora da imagem; evita avisos do MLflow ao buscar o SHA.
    GIT_PYTHON_REFRESH=quiet

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Dependencias primeiro: a camada fica em cache enquanto o lock nao mudar.
# Instala producao (main) + DVC (pipeline), sem o grupo dev.
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main,pipeline --no-root

COPY . .
RUN poetry install --only main,pipeline \
    # O .git nao entra na imagem (.dockerignore); o DVC roda sem SCM.
    && dvc config --local core.no_scm true

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /mlflow \
    && chown -R app:app /app /mlflow
USER app

EXPOSE 5000 8000

CMD ["dvc", "repro"]
