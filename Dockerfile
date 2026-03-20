FROM python:3.10

WORKDIR /app

# 🔥 Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Run server (IMPORTANT: use PORT from Render)
CMD ["sh", "-c", "uvicorn yolo_api:app --host 0.0.0.0 --port ${PORT:-8000}"]