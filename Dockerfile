# Multi-stage build for efficient training environment
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.0

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-root --only main

# Production stage
FROM base as production

# Copy source code
COPY src/ ./src/
COPY .env.example ./.env

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create directories
RUN mkdir -p /app/data /app/checkpoints /app/logs

# Default command
CMD ["python", "-m", "src.training.train"]

# Development stage with additional tools
FROM base as development

# Install development dependencies
RUN poetry install --no-root

# Copy source code and tests
COPY src/ ./src/
COPY tests/ ./tests/
COPY pytest.ini ./
COPY .env.example ./.env

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create directories
RUN mkdir -p /app/data /app/checkpoints /app/logs

# Default command for development
CMD ["bash"]
