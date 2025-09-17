# Use uv's Python 3.12 Debian image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

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