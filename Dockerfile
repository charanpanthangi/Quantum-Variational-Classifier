# Lightweight container for the quantum variational classifier demo
# Uses Python 3.11 slim as the base image.
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout for clarity.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install basic build tools for Python dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set a working directory inside the container.
WORKDIR /app

# Copy dependency list first to leverage Docker layer caching.
COPY requirements.txt ./

# Install Python packages including PennyLane and scikit-learn.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository into the image.
COPY . .

# Default command: run the demo with reasonable defaults.
CMD ["python", "app/main.py", "--n-samples", "150", "--epochs", "80"]
