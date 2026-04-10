FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies (Adding g++ for ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip to resolve NumPy 1.26.4 correctly
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# 4. Copy the project
COPY . .

# Match your original port
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
