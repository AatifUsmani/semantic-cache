FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 app \
    && mkdir -p "$HF_HOME" \
    && chown -R app:app /app /home/app

ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}

USER app
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY --chown=app:app app.py demo.py load_test.py test_cache.py ./
COPY --chown=app:app semantic_cache ./semantic_cache

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]