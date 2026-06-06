# RAG Pipeline by Viprasol Tech Private Limited (https://viprasol.com)
FROM python:3.12-slim

LABEL org.opencontainers.image.title="rag-pipeline" \
      org.opencontainers.image.description="Production-style RAG pipeline in Python — by Viprasol Tech." \
      org.opencontainers.image.source="https://github.com/Viprasol-Tech/rag-pipeline" \
      org.opencontainers.image.url="https://viprasol.com" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Run the fully-offline demo by default.
ENTRYPOINT ["rag-pipeline"]
CMD ["demo"]
