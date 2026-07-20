FROM python:3.10-slim

# Prevent Python from writing pyc files to disc & buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend source files and environmental config
COPY frontend/ ./frontend/
COPY .env .env

# Expose port
EXPOSE 5000

# Set Python path to ensure local modules can be loaded properly
ENV PYTHONPATH=/app/frontend

CMD ["python", "frontend/app.py"]
