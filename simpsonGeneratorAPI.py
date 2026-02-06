import os
import io
import time
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, LCMScheduler

# ===== CONFIGURATION =====
# /tf/workspace/diffusion

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_PATH = os.path.join(CURRENT_DIR, "simpsons_lora_results")
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_FOLDER = os.path.join(CURRENT_DIR, "generated_simpsons")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ml_models = {}


def load_pipeline():
    print(f"🧠 Chargement du modèle {BASE_MODEL_ID} sur {DEVICE}...")
    try:
        # 1. Chargement du modèle de base
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            safety_checker=None
        ).to(DEVICE)

        # 2. Chargement de TON LoRA Simpson (nommé 'simpson')
        if os.path.exists(os.path.join(LORA_PATH, "pytorch_lora_weights.safetensors")):
            print("🔌 Chargement LoRA Simpson...")
            pipe.load_lora_weights(LORA_PATH, weight_name="pytorch_lora_weights.safetensors", adapter_name="simpson")

        # 3. Chargement du LCM LoRA depuis Internet (nommé 'lcm')
        print("⚡ Chargement LCM LoRA...")
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")

        # Par défaut : On active QUE Simpson (1.0) et on éteint LCM (0.0)
        pipe.set_adapters(["simpson", "lcm"], adapter_weights=[1.0, 0.0])

        # Préparation des deux Schedulers
        pipe.scheduler_dpm = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler_lcm = LCMScheduler.from_config(pipe.scheduler.config)

        # On commence en mode Standard (DPM)
        pipe.scheduler = pipe.scheduler_dpm

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

app.mount("/images", StaticFiles(directory=OUTPUT_FOLDER), name="images")


# Ajout du paramètre turbo_mode
class GenerationRequest(BaseModel):
    character_name: str
    steps: int = 30
    guidance_scale: float = 7.5
    save_to_disk: bool = True
    turbo_mode: bool = False  # <--- NOUVEAU


@app.get("/")
def health_check():
    return {"status": "online"}


@app.get("/history")
def get_history():
    images = []
    files = sorted(os.listdir(OUTPUT_FOLDER), key=lambda x: os.path.getctime(os.path.join(OUTPUT_FOLDER, x)),
                   reverse=True)
    for filename in files:
        if filename.endswith(".png"):
            prompt_guess = filename.split("_")[0]
            images.append({
                "id": filename,
                "url": f"http://localhost:8000/images/{filename}",
                "prompt": prompt_guess,
                "timestamp": os.path.getctime(os.path.join(OUTPUT_FOLDER, filename)) * 1000,
                "duration": "N/A"
            })
    return images


@app.post("/generate")
async def generate_image(request: GenerationRequest):
    pipe = ml_models.get("pipe")
    if not pipe: raise HTTPException(status_code=503, detail="Modèle non chargé")

    start_time = time.time()
    prompt = f"{request.character_name} as a simpson character, yellow skin, cartoon style, high quality, detailed"
    negative_prompt = "realistic, photograph, photo, 3d render, ugly, deformed, blurry, low quality, distorted, bad anatomy, extra limbs"

    # === LOGIQUE DE BASCULE TURBO ===
    if request.turbo_mode:
        print(f"🚀 Mode TURBO activé pour {request.character_name}")
        # Active les deux LoRAs
        pipe.set_adapters(["simpson", "lcm"], adapter_weights=[1.0, 1.0])
        pipe.scheduler = pipe.scheduler_lcm

        # Paramètres forcés pour LCM (sinon ça brûle l'image)
        actual_steps = 6  # Très rapide (4 à 8 max)
        actual_guidance = 1.5  # Très bas (1.0 à 2.0 max)
    else:
        print(f"🎨 Mode STANDARD activé pour {request.character_name}")
        # Désactive LCM
        pipe.set_adapters(["simpson", "lcm"], adapter_weights=[1.0, 0.0])
        pipe.scheduler = pipe.scheduler_dpm

        # Paramètres de l'utilisateur
        actual_steps = request.steps
        actual_guidance = request.guidance_scale

    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=actual_steps,
            guidance_scale=actual_guidance,
            height=512, width=512,
            cross_attention_kwargs={"scale": 1.0}
        ).images[0]

        if request.save_to_disk:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_name = "".join(x for x in request.character_name if x.isalnum())
            suffix = "turbo" if request.turbo_mode else "hq"
            filename = f"{clean_name}_{suffix}_{timestamp}.png"
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
        # On remet le scheduler par défaut en cas d'erreur pour ne pas bloquer la suite
        pipe.scheduler = pipe.scheduler_dpm
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)