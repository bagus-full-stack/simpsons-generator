import os
import io
import time
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # <--- NOUVEAU
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# ===== CONFIGURATION =====
LORA_PATH = "/tf/workspace/diffusion/simpsonsGenerator/simpsons_lora_results"
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_FOLDER = "/tf/workspace/diffusion/generated_simpsons"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ml_models = {}


def load_pipeline():
    print(f"🧠 Chargement du modèle {BASE_MODEL_ID} sur {DEVICE}...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            safety_checker=None
        ).to(DEVICE)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        lora_weights = os.path.join(LORA_PATH, "pytorch_lora_weights.safetensors")
        if os.path.exists(lora_weights):
            pipe.load_lora_weights(LORA_PATH, weight_name="pytorch_lora_weights.safetensors")
        return pipe
    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        raise e


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["pipe"] = load_pipeline()
    yield
    ml_models.clear()
    if DEVICE == "cuda": torch.cuda.empty_cache()


app = FastAPI(title="Simpson Generator API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Generation-Time"]
)

# 1. RENDRE LES IMAGES ACCESSIBLES VIA URL
app.mount("/images", StaticFiles(directory=OUTPUT_FOLDER), name="images")


class GenerationRequest(BaseModel):
    character_name: str
    steps: int = 30
    guidance_scale: float = 7.5
    save_to_disk: bool = True


@app.get("/")
def health_check():
    return {"status": "online"}


# 2. NOUVELLE ROUTE : RÉCUPÉRER L'HISTORIQUE
@app.get("/history")
def get_history():
    images = []
    # On liste les fichiers dans le dossier
    files = sorted(os.listdir(OUTPUT_FOLDER), key=lambda x: os.path.getctime(os.path.join(OUTPUT_FOLDER, x)),
                   reverse=True)

    for filename in files:
        if filename.endswith(".png"):
            # On essaie de deviner le nom du perso depuis le nom de fichier (ex: homer_2023.png)
            prompt_guess = filename.split("_")[0]

            images.append({
                "id": filename,
                "url": f"http://localhost:8000/images/{filename}",  # URL statique
                "prompt": prompt_guess,
                "timestamp": os.path.getctime(os.path.join(OUTPUT_FOLDER, filename)) * 1000,
                "duration": "N/A"  # On n'a pas stocké le temps de génération, donc N/A pour l'historique
            })
    return images


@app.post("/generate")
async def generate_image(request: GenerationRequest):
    pipe = ml_models.get("pipe")
    if not pipe: raise HTTPException(status_code=503, detail="Modèle non chargé")

    start_time = time.time()
    prompt = f"{request.character_name} as a simpson character, yellow skin, cartoon style, high quality, detailed"
    negative_prompt = "realistic, photograph, photo, 3d render, ugly, deformed, blurry, low quality, distorted, bad anatomy, extra limbs"

    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            height=512, width=512,
            cross_attention_kwargs={"scale": 1.0}
        ).images[0]

        # Sauvegarde disque
        if request.save_to_disk:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Nettoyage du nom de fichier pour éviter les erreurs
            clean_name = "".join(x for x in request.character_name if x.isalnum())
            filename = f"{clean_name}_{timestamp}.png"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            image.save(filepath)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        duration = time.time() - start_time
        return Response(content=img_byte_arr.getvalue(), media_type="image/png",
                        headers={"X-Generation-Time": f"{duration:.2f}"})

    except Exception as e:
        print(f"❌ Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)