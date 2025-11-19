# api image (pyproject + src layout)

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install your package from pyproject + src
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# runtime working dir (code is importable as installed package)
WORKDIR /app

EXPOSE 8000
CMD ["uvicorn","doc_studio.api.app:app","--host","0.0.0.0","--port","8000"]
