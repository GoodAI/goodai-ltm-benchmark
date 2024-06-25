# Dockerfile

FROM python:3.12

WORKDIR /app

COPY . /app

RUN mkdir -p /app/logs

RUN apt-get update && apt-get install -y dos2unix

RUN dos2unix /app/.env

RUN pip install --no-cache-dir -r docker_requirements.txt
RUN pip install fastapi uvicorn

EXPOSE 8000

COPY .env /app/.env

CMD ["sh", "-c", "printenv && uvicorn src.api:app --host 0.0.0.0 --port 8000"]
