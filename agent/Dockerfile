FROM python:3.10


WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install --prefer-binary --default-timeout=200 -r requirements.txt


# Copy agent code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose API port
EXPOSE 8000

# Run the API
CMD ["python", "agent_api.py"]