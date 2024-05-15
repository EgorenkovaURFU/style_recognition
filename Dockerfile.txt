FROM python:3.9-slim
USER root

# Установка зависимостей
RUN apt-get update 
RUN apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
