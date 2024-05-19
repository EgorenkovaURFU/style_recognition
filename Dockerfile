FROM python:3.9-slim AS builder

WORKDIR /app

COPY requirements.txt .

# Установка зависимостей
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && pip install --upgrade pip \
    && pip install -r --no-cache-dir requirements.txt

FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /app/ .

# Задайте точку входа
CMD ["streamlit", "run", "src/main.py"] 
