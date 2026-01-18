FROM python:3.10-slim

WORKDIR /app

# 1. Install PyTorch CPU first (Critical for Free Tier limit)
# This ensures we won't accidentally pull the 5GB+ GPU version later
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Copy files
COPY pyproject.toml .
COPY src/ src/
COPY app.py .
COPY models/ models/

# 3. Install project + gradio (skipping torch re-install because it's already there)
RUN pip install --no-cache-dir ".[demo]"

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
