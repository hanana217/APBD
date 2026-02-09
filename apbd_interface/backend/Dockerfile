FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install \
    --upgrade pip setuptools wheel && \
    pip install \
    --prefer-binary \
    --default-timeout=300 \
    --no-cache-dir \
    -r requirements.txt


# Copier le code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
