import os
import io
import time
import torch
import uvicorn
import cv2  # <--- On utilise maintenant OpenCV
import numpy as np
from PIL import Image, ImageOps
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionInpaintPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    LCMScheduler
)

# ===== CONFIGURATION =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_PATH = os.path.join(CURRENT_DIR, "simpsons_lora_results")
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_FOLDER = os.path.join(CURRENT_DIR, "generated_simpsons")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ml_models = {}


def load_pipelines():
    print(f"🧠 Chargement des modèles sur {DEVICE}...")
    try:
        # 1. Pipeline Principal (Text2Img)
        text_pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            safety_checker=None
        ).to(DEVICE)

        simpson_weights = os.path.join(LORA_PATH, "pytorch_lora_weights.safetensors")
        if os.path.exists(simpson_weights):
            text_pipe.load_lora_weights(LORA_PATH, weight_name="pytorch_lora_weights.safetensors",
                                        adapter_name="simpson")

        text_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
        text_pipe.set_adapters(["simpson", "lcm"], adapter_weights=[1.0, 0.0])

        # Schedulers
        dpm_scheduler = DPMSolverMultistepScheduler.from_config(text_pipe.scheduler.config)
        lcm_scheduler = LCMScheduler.from_config(text_pipe.scheduler.config)

        def configure_pipe(pipe):
            pipe.scheduler_dpm = dpm_scheduler
            pipe.scheduler_lcm = lcm_scheduler
            pipe.scheduler = dpm_scheduler
            return pipe

        configure_pipe(text_pipe)

        # 2. Pipeline Img2Img
        img_pipe = StableDiffusionImg2ImgPipeline(**text_pipe.components)
        configure_pipe(img_pipe)

        # 3. Pipeline Inpainting
        inpaint_pipe = StableDiffusionInpaintPipeline(**text_pipe.components)
        configure_pipe(inpaint_pipe)

        # 4. CONTROLNET (CANNY) - Changement ici
        print("📐 Chargement ControlNet Canny...")
        # On utilise le modèle Canny au lieu de OpenPose
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)

        canny_pipe = StableDiffusionControlNetPipeline(
            controlnet=controlnet, vae=text_pipe.vae, text_encoder=text_pipe.text_encoder,
            tokenizer=text_pipe.tokenizer, unet=text_pipe.unet, scheduler=text_pipe.scheduler,
            safety_checker=None, feature_extractor=text_pipe.feature_extractor
        ).to(DEVICE)
        configure_pipe(canny_pipe)

        # Plus besoin de pose_detector, on utilisera une fonction OpenCV locale

        return text_pipe, img_pipe, canny_pipe, inpaint_pipe

    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        raise e


@asynccontextmanager
async def lifespan(app: FastAPI):
    text_pipe, img_pipe, canny_pipe, inpaint_pipe = load_pipelines()
    ml_models["text_pipe"] = text_pipe
    ml_models["img_pipe"] = img_pipe
    ml_models["canny_pipe"] = canny_pipe
    ml_models["inpaint_pipe"] = inpaint_pipe
    yield
    ml_models.clear()
    if DEVICE == "cuda": torch.cuda.empty_cache()


app = FastAPI(title="Simpson Generator API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    expose_headers=["X-Generation-Time"]
)
app.mount("/images", StaticFiles(directory=OUTPUT_FOLDER), name="images")


class GenerationRequest(BaseModel):
    character_name: str;
    steps: int = 30;
    turbo_mode: bool = False


@app.get("/")
def health_check(): return {"status": "online"}


@app.get("/history")
def get_history():
    if not os.path.exists(OUTPUT_FOLDER): return []
    files = sorted(os.listdir(OUTPUT_FOLDER), key=lambda x: os.path.getctime(os.path.join(OUTPUT_FOLDER, x)),
                   reverse=True)
    return [{
        "id": f, "url": f"http://localhost:8000/images/{f}",
        "prompt": f.split("_")[0], "timestamp": os.path.getctime(os.path.join(OUTPUT_FOLDER, f)) * 1000,
        "duration": "N/A"
    } for f in files if f.endswith(".png")]


def process_uploaded_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return ImageOps.fit(img, (512, 512), method=Image.Resampling.LANCZOS)


# === FONCTION DE DÉTECTION CANNY (Contours) ===
def get_canny_image(image):
    # Convertir PIL Image en tableau Numpy
    image = np.array(image)

    # Détection des contours (Low threshold=100, High threshold=200)
    canny_image = cv2.Canny(image, 100, 200)

    # Reconvertir en format compatible ControlNet (3 channels)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)

    return Image.fromarray(canny_image)


@app.post("/generate")
async def generate_text(request: GenerationRequest):
    return run_generation(ml_models["text_pipe"], request.character_name, request.steps, request.turbo_mode,
                          mode="text")


@app.post("/generate-from-image")
async def generate_image(file: UploadFile = File(...), prompt: str = Form(...), steps: int = Form(30),
                         strength: float = Form(0.75), turbo_mode: bool = Form(False)):
    content = await file.read()
    return run_generation(ml_models["img_pipe"], prompt, steps, turbo_mode, mode="img2img",
                          img_input=process_uploaded_image(content), strength=strength)


# === ENDPOINT STRUCTURE (CANNY) ===
# On garde l'URL /generate-pose pour ne pas casser le frontend, mais la logique est Canny
@app.post("/generate-pose")
async def generate_structure(file: UploadFile = File(...), prompt: str = Form(...), steps: int = Form(30),
                             turbo_mode: bool = Form(False)):
    content = await file.read()
    init_image = process_uploaded_image(content)

    # On extrait les contours
    edge_map = get_canny_image(init_image)

    return run_generation(ml_models["canny_pipe"], prompt, steps, turbo_mode, mode="controlnet", img_input=edge_map)


@app.post("/generate-inpainting")
async def generate_inpainting(image_file: UploadFile = File(...), mask_file: UploadFile = File(...),
                              prompt: str = Form(...), steps: int = Form(30), turbo_mode: bool = Form(False)):
    img_content = await image_file.read()
    mask_content = await mask_file.read()
    source_img = process_uploaded_image(img_content)
    mask_img = Image.open(io.BytesIO(mask_content)).convert("L")
    mask_img = ImageOps.fit(mask_img, (512, 512), method=Image.Resampling.LANCZOS)
    return run_generation(ml_models["inpaint_pipe"], prompt, steps, turbo_mode, mode="inpaint", img_input=source_img,
                          mask_input=mask_img)


def run_generation(pipe, prompt_text, steps, turbo, mode="text", img_input=None, mask_input=None, strength=0.8):
    start_time = time.time()
    full_prompt = f"{prompt_text} as a simpson character, yellow skin, cartoon style, high quality"
    negative_prompt = "realistic, photo, ugly, deformed, blurry, bad anatomy"

    if turbo:
        pipe.set_adapters(["simpson", "lcm"], adapter_weights=[1.0, 1.0])
        pipe.scheduler = pipe.scheduler_lcm
        actual_steps, actual_guidance = 6, 1.5
    else:
        pipe.set_adapters(["simpson", "lcm"], adapter_weights=[1.0, 0.0])
        pipe.scheduler = pipe.scheduler_dpm
        actual_steps, actual_guidance = steps, 7.5

    try:
        args = {
            "prompt": full_prompt, "negative_prompt": negative_prompt,
            "num_inference_steps": actual_steps, "guidance_scale": actual_guidance
        }

        if mode == "img2img":
            args.update({"image": img_input, "strength": strength})
        elif mode == "controlnet":
            args.update({"image": img_input, "controlnet_conditioning_scale": 1.0})
        elif mode == "inpaint":
            args.update({"image": img_input, "mask_image": mask_input, "strength": 0.95})
        else:
            args.update({"height": 512, "width": 512})

        result = pipe(**args).images[0]

        from datetime import datetime
        filename = f"{prompt_text[:10]}_{mode}_{datetime.now().strftime('%H%M%S')}.png"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        result.save(filepath)

        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        return Response(content=img_byte_arr.getvalue(), media_type="image/png",
                        headers={"X-Generation-Time": f"{time.time() - start_time:.2f}"})

    except Exception as e:
        print(f"❌ Erreur : {e}")
        if hasattr(pipe, "scheduler_dpm"): pipe.scheduler = pipe.scheduler_dpm
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)