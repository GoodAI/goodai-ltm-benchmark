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

# Upgrade pip and install dependencies with increased timeout and using a mirror
RUN pip install --upgrade pip

# Install numpy and scipy separately with increased timeout
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.14.0 --timeout 600

# Install scikit-learn separately with an even longer timeout
RUN pip install --no-cache-dir scikit-learn==1.4.0 --timeout 900

# Install other dependencies
RUN pip install --no-cache-dir -r docker_requirements.txt --timeout 600 --index-url https://pypi.org/simple --trusted-host pypi.org

RUN pip install fastapi uvicorn

RUN pip install networkx matplotlib

RUN pip install structlog==24.1.0

RUN pip install structlog aiologger

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-u", "src/utils/log_sync.py", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]