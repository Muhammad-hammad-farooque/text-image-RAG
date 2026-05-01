FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Install dependencies (production only)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# Copy source
COPY . .

# Create logs directory
RUN mkdir -p logs

EXPOSE 8000 8001
