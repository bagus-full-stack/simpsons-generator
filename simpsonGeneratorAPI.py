import os
import io
import time
import torch
import uvicorn
from PIL import Image, ImageOps
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, LCMScheduler

# ===== CONFIGURATION =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_PATH = os.path.join(CURRENT_DIR, "simpsons_lora_results")
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_FOLDER = os.path.join(CURRENT_DIR, "generated_simpsons")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ml_models = {}


def load_pipelines():
    print(f"🧠 Chargement du modèle {BASE_MODEL_ID} sur {DEVICE}...")
    try:
        # 1. Pipeline TEXTE (Text2Img)
        text_pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            safety_checker=None
        ).to(DEVICE)

        # 2. Chargement des LoRAs
        simpson_weights = os.path.join(LORA_PATH, "pytorch_lora_weights.safetensors")
        if not os.path.exists(simpson_weights):
            raise FileNotFoundError(f"❌ IMPOSSIBLE DE TROUVER LE LORA : {simpson_weights}")

        print("🔌 Chargement des LoRAs...")
        text_pipe.load_lora_weights(LORA_PATH, weight_name="pytorch_lora_weights.safetensors", adapter_name="simpson")
        text_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")

        text_pipe.set_adapters(["simpson", "lcm"], adapter_weights=[1.0, 0.0])

        # 3. Création du Pipeline IMAGE (Img2Img)
        img_pipe = StableDiffusionImg2ImgPipeline(**text_pipe.components)

        # 4. CRÉATION ET ATTACHEMENT DES SCHEDULERS (POUR LES DEUX !)
        # On crée les objets scheduler une seule fois
        dpm_scheduler = DPMSolverMultistepScheduler.from_config(text_pipe.scheduler.config)
        lcm_scheduler = LCMScheduler.from_config(text_pipe.scheduler.config)

        # On les attache au Text Pipe
        text_pipe.scheduler_dpm = dpm_scheduler
        text_pipe.scheduler_lcm = lcm_scheduler

        # 👇 C'EST ICI QUE ÇA MANQUAIT : On les attache aussi au Image Pipe
        img_pipe.scheduler_dpm = dpm_scheduler
        img_pipe.scheduler_lcm = lcm_scheduler

        # On définit le scheduler par défaut pour les deux
        text_pipe.scheduler = text_pipe.scheduler_dpm
        img_pipe.scheduler = img_pipe.scheduler_dpm

        return text_pipe, img_pipe

    except Exception as e:
        print(f"❌ Erreur critique lors du chargement : {e}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    text_pipe, img_pipe = load_pipelines()
    ml_models["text_pipe"] = text_pipe
    ml_models["img_pipe"] = img_pipe
    yield
    ml_models.clear()
    if DEVICE == "cuda": torch.cuda.empty_cache()


app = FastAPI(title="Simpson Generator API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    expose_headers=["X-Generation-Time"]
)
app.mount("/images", StaticFiles(directory=OUTPUT_FOLDER), name="images")


# Modèle pour la requête JSON (Text2Img uniquement)
class GenerationRequest(BaseModel):
    character_name: str
    steps: int = 30
    guidance_scale: float = 7.5
    save_to_disk: bool = True
    turbo_mode: bool = False


@app.get("/")
def health_check(): return {"status": "online"}


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


# === UTILITAIRE : Préparer l'image ===
def process_uploaded_image(image_bytes):
    """Redimensionne et coupe l'image pour qu'elle soit carrée (512x512)"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = ImageOps.fit(img, (512, 512), method=Image.Resampling.LANCZOS)
    return img


# === ENDPOINT TEXT-TO-IMAGE ===
@app.post("/generate")
async def generate_text(request: GenerationRequest):
    pipe = ml_models.get("text_pipe")
    return run_generation(pipe, request.character_name, request.steps, request.guidance_scale, request.turbo_mode,
                          img_input=None)


# === NOUVEAU : ENDPOINT IMAGE-TO-IMAGE ===
@app.post("/generate-from-image")
async def generate_image(
        file: UploadFile = File(...),
        prompt: str = Form(...),
        steps: int = Form(30),
        strength: float = Form(0.75),  # Combien on change l'image (0.0 = rien, 1.0 = tout)
        turbo_mode: bool = Form(False)
):
    pipe = ml_models.get("img_pipe")

    # Lecture et préparation de l'image
    content = await file.read()
    init_image = process_uploaded_image(content)

    return run_generation(pipe, prompt, steps, 7.5, turbo_mode, img_input=init_image, strength=strength)


# === LOGIQUE COMMUNE ===
def run_generation(pipe, prompt_text, steps, guidance, turbo, img_input=None, strength=0.8):
    start_time = time.time()

    full_prompt = f"{prompt_text} as a simpson character, yellow skin, cartoon style, high quality, detailed"
    negative_prompt = "realistic, photograph, photo, 3d render, ugly, deformed, blurry, low quality, distorted, bad anatomy, extra limbs"

    # Configuration Turbo vs Standard
    if turbo:
        print(f"🚀 Mode TURBO pour '{prompt_text}'")
        pipe.set_adapters(["simpson", "lcm"], adapter_weights=[1.0, 1.0])
        pipe.scheduler = pipe.scheduler_lcm
        actual_steps = 6
        actual_guidance = 1.5
    else:
        print(f"🎨 Mode STANDARD pour '{prompt_text}'")
        pipe.set_adapters(["simpson", "lcm"], adapter_weights=[1.0, 0.0])
        pipe.scheduler = pipe.scheduler_dpm
        actual_steps = steps
        actual_guidance = guidance

    try:
        # Arguments dynamiques selon si c'est text2img ou img2img
        args = {
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": actual_steps,
            "guidance_scale": actual_guidance,
            "cross_attention_kwargs": {"scale": 1.0}
        }

        if img_input:
            # Paramètres spécifiques à Image-to-Image
            args["image"] = img_input
            args["strength"] = strength  # Important : Contrôle la créativité
        else:
            args["height"] = 512
            args["width"] = 512

        result = pipe(**args).images[0]

        # Sauvegarde
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = "".join(x for x in prompt_text if x.isalnum())
        mode_suffix = "img2img" if img_input else "txt2img"
        filename = f"{clean_name}_{mode_suffix}_{timestamp}.png"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        result.save(filepath)

        # Réponse
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        duration = time.time() - start_time

        return Response(content=img_byte_arr.getvalue(), media_type="image/png",
                        headers={"X-Generation-Time": f"{duration:.2f}"})

    except Exception as e:
        print(f"❌ Erreur : {e}")
        # Reset scheduler safety
        if hasattr(pipe, "scheduler_dpm"): pipe.scheduler = pipe.scheduler_dpm
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)