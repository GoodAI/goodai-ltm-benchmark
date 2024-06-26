FROM python:3.12

# Create a non-root user
RUN useradd -m appuser

WORKDIR /app

COPY . /app

RUN mkdir -p /app/logs /app/data
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create and activate virtual environment
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install dependencies in virtual environment
RUN pip install --no-cache-dir -r docker_requirements.txt
RUN pip install fastapi uvicorn

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]