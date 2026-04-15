FROM python:3.11-slim

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV APP_DATA_DIR=/app/data

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application code needed at runtime
COPY main.py .
COPY services ./services

# Create the shared runtime data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "main.py"]
