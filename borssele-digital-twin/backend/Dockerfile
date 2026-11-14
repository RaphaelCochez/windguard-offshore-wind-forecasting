FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train v2 residual LightGBM models during build if not already present
RUN if [ ! -f models/wind_forecast_v2/metrics_summary.json ]; then python train_models.py; fi

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
