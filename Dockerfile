# Use minimal CUDA runtime image
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.12 /usr/bin/python

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY tests/ tests/
COPY README.md ./

# Install dependencies and the package
RUN uv sync --frozen

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Default command
CMD ["uv", "run", "python", "-c", "import cebmf_torch; print('cebmf_torch loaded successfully')"]