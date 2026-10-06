# ==============================================================================
# Containerized Runtime for Process Hypothesis Agent (Cloud Run / Vertex AI)
# Follows Google Cloud `agent-starter-pack` production container layout
# (`Prototype to Production.pdf`, pp. 34-38)
# ==============================================================================
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

COPY pyproject.toml AGENTS.md README.md ./
COPY skills/ ./skills/
COPY app/ ./app/
COPY evals/ ./evals/
COPY tests/ ./tests/

# Pre-generate obfuscated 100x100 dataset at build time and verify test suite
RUN python3 -m unittest discover -s tests -p "test_*.py" -v

EXPOSE 8080

CMD ["python3", "app/server.py"]
