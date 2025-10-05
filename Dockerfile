# Lightweight container for MLB-Network-Analysis
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (add any needed build tools for pandas / pyarrow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install fastapi uvicorn

COPY . .

# Expose API port
EXPOSE 8000

# Default command starts FastAPI (override with docker run ... for pipeline commands)
CMD ["uvicorn","api:app","--host","0.0.0.0","--port","8000"]
