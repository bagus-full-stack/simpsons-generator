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

# Utilisateur non-root. Les dossiers montés en volume sont créés ici pour que les volumes
# nommés (db_data, hf_cache) héritent de son propriétaire au premier démarrage.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /srv/data /srv/generated_simpsons /srv/hf_cache \
    && chown -R app:app /srv
USER app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/srv/hf_cache
EXPOSE 8000

# Le port ne s'ouvre qu'une fois les modèles chargés : start_period large pour le 1er téléchargement.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
