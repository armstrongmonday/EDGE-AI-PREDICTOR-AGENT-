FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .
ENV EDGEAGENT_DB=/data/edgeagent.duckdb
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "edgeagent.api.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
