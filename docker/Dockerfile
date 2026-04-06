FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose MLflow UI port (optional)
EXPOSE 5000

# Default command (run pipeline)
CMD ["python", "-m", "src.pipelines.training_pipeline"]