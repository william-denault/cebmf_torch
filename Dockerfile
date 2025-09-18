# Use minimal CUDA runtime image
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

# Install minimal system dependencies and let uv handle Python
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies first (uv will install Python 3.12 automatically)
RUN uv sync --frozen

# Copy source code after dependencies
COPY src/ src/
COPY tests/ tests/
COPY README.md ./

# Default command
CMD ["uv", "run", "python", "-c", "import cebmf_torch; print('cebmf_torch loaded successfully')"]
