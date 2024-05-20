FROM python:3.9-slim
USER root

# Установка зависимостей
RUN apt-get update 
RUN apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /app

#COPY requirements.txt .
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# COPY . .
# RUN pytest /test

CMD ["pytest", "test"]

# CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# FROM python:3.9-slim as compiler
# ENV PYTHONUNBUFFERED 1

# WORKDIR /app/

# RUN python -m venv /opt/venv
# # Enable venv
# ENV PATH="/opt/venv/bin:$PATH"

# COPY ./requirements.txt /app/requirements.txt
# RUN pip install -Ur requirements.txt

# FROM python:3.9-slim as runner
# WORKDIR /app/
# COPY --from=compiler /opt/venv /opt/venv

# # Enable venv
# ENV PATH="/opt/venv/bin:$PATH"
# COPY . /app/
# CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
