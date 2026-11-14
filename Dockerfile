# ==============================================================================
# Multi-Stage Unified Dockerfile for Borssele Offshore Wind Digital Twin (v2)
# Builds the React/Vite dashboard + FastAPI/LightGBM v2 engine in one container
# ==============================================================================

# Stage 1: Build Frontend Static Assets
FROM node:18-alpine AS frontend-build
WORKDIR /frontend
COPY borssele-digital-twin/frontend/package*.json ./
RUN npm install
COPY borssele-digital-twin/frontend/ ./
RUN npm run build

# Stage 2: Python Backend + LightGBM v2 Model Training + Static UI Serving
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY borssele-digital-twin/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY borssele-digital-twin/backend/ ./
COPY --from=frontend-build /frontend/dist ./static

# Train the improved 24-horizon v2 LightGBM models if not already pre-generated
RUN if [ ! -f models/wind_forecast_v2/metrics_summary.json ]; then python train_models.py; fi

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
