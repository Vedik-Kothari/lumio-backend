FROM python:3.11-slim

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV APP_DATA_DIR=/app/data

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend application code
COPY . .

# Create the shared runtime data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "main.py"]
