FROM python:3.10-slim

# Prevent Python from writing pyc files to disc & buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts, env variables, schemas, and source code
COPY models/ ./models/
COPY api/ ./api/
COPY .env .env

# Expose port
EXPOSE 8000

# Set Python path to ensure local modules can be loaded properly
ENV PYTHONPATH=/app/api

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
