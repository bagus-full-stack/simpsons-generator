# Image unique pour l'API et le worker : seule la commande lancée diffère
# (voir docker-compose.yml). Basée sur une image PyTorch officielle avec CUDA
# déjà installé correctement, donc requirements.txt ne réinstalle pas torch.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /srv

# libgl1 / libglib2.0-0 : requis par opencv-python-headless (mode "pose" / Canny).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
