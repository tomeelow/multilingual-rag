# Frontend image — Streamlit only; deliberately not built from pyproject so it
# stays free of torch and the rest of the retrieval stack.
# Build context is the repo root: docker build -f docker/frontend/Dockerfile .
FROM python:3.12-slim

RUN pip install --no-cache-dir streamlit==1.58.0 httpx==0.28.1

WORKDIR /app
COPY frontend ./frontend

EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address", "0.0.0.0", \
     "--server.port", "8501", "--server.headless", "true"]
