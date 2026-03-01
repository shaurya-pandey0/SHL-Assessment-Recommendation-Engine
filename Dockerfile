FROM python:3.10-slim

WORKDIR /app

# Install system deps for llama-cpp-python (C++ compiler needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build embeddings at build time if model is present (faster cold starts)
# If model not present, embeddings will be built on first request
RUN python -c "from pathlib import Path; \
    models = list(Path('.').glob('nomic-embed-text-v1.5*.gguf')); \
    print(f'Found {len(models)} model(s): {models}'); \
    exec('from engine.embeddings import build_and_save; build_and_save()') if models else print('No model found, skipping pre-build')"

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
