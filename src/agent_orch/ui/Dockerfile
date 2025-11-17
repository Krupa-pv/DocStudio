# ui/Dockerfile  (runs streamlit against src/doc_studio/ui/app.py)
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

EXPOSE 8501
ENV STREAMLIT_API_URL=http://api:8000
CMD ["streamlit","run","src/doc_studio/ui/app.py","--server.port=8501","--server.address=0.0.0.0"]
