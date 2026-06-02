# Hugging Face Spaces — BDI Hackathon CDSS
# Port 7860 is the default for HF Spaces

FROM python:3.11-slim

# Install Node.js for frontend build
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build frontend
COPY frontend/package*.json frontend/
RUN cd frontend && npm install

COPY frontend/ frontend/
RUN cd frontend && npm run build

# Copy rest of app
COPY app/ app/
COPY model/ model/
COPY data/ data/
COPY run.py .

# HF Spaces runs as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "run.py"]
