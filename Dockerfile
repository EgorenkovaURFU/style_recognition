FROM python:3.9-slim

WORKDIR /app

# Обновите  источники  пакетов
RUN apt-get update

# Установите  необходимые  пакеты
RUN apt-get install -y --no-install-recommends build-essential

# Очистите  кэш  `apt`
RUN rm -rf /var/lib/apt/lists/* && apt-get clean

# Обновите  `pip`
RUN pip install --upgrade pip

# Установите  зависимости  из  `requirements.txt`
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "src/main.py"] 
