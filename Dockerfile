# CPU image for the API. Base pinned by digest after local validation (python:3.11-slim).
FROM python@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377

WORKDIR /app
COPY requirements/cpu-py311-linux.lock requirements/cpu-py311-linux.lock
RUN python -m pip install --no-cache-dir -r requirements/cpu-py311-linux.lock
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY csrc ./csrc
COPY configs ./configs
RUN python -m pip install --no-cache-dir --no-deps -e .

ENV HOTSHIFT_BACKEND=cpu \
    HOTSHIFT_CONFIG=configs/demo.yaml \
    HOTSHIFT_ARTIFACT_DIR=/app/artifacts \
    HOTSHIFT_HOST=0.0.0.0 \
    HOTSHIFT_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"
# Generate the configured fixture on first start (skipped when a valid one is mounted), then
# serve with exactly one worker so there is one cache and one engine owner.
CMD ["sh", "-c", "python -m hotshift.cli data generate --config \"$HOTSHIFT_CONFIG\" --if-missing && python -m hotshift.cli data catalog --config \"$HOTSHIFT_CONFIG\" && exec python -m hotshift.cli serve --config \"$HOTSHIFT_CONFIG\" --backend \"$HOTSHIFT_BACKEND\" --host 0.0.0.0 --port 8000"]
