FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        python3-dev \
        musl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY . /app/

EXPOSE 8080
EXPOSE 8501

CMD ["python", "main.py"]
