FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed data/models

# Default: run the full demo pipeline
CMD ["python", "run_demo.py"]
