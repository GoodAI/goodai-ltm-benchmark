FROM python:3.12

WORKDIR /app

COPY . /app

RUN mkdir -p /app/logs

RUN pip install --no-cache-dir -r docker_requirements.txt
RUN pip install fastapi uvicorn

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]