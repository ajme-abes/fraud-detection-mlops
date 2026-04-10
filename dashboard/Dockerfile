FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies (build-essential is better for compiling)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. UPGRADE PIP FIRST (Crucial to find the correct NumPy wheels)
RUN pip install --no-cache-dir --upgrade pip

# 3. Install requirements with a longer timeout
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# 4. Copy the rest of the project
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
