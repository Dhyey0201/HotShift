FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --no-cache-dir ".[service]"

ENV HOTSHIFT_BACKEND=cpu \
    HOTSHIFT_CONFIG=configs/demo.yaml \
    HOTSHIFT_HOST=0.0.0.0 \
    HOTSHIFT_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"
CMD ["python", "-m", "hotshift.cli", "serve", "--config", "configs/demo.yaml", "--backend", "cpu", "--host", "0.0.0.0", "--port", "8000"]
