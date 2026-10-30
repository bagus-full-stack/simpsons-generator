"""Endpoints de génération.

Contrat : chaque endpoint POST dépose un job et répond tout de suite avec son
statut (au lieu de bloquer jusqu'à la fin de l'inférence comme l'ancienne API).
Le client va ensuite consulter GET /jobs/{id} jusqu'à ce que le statut soit
"done"/"failed"/"rejected" — voir simpson-front/app/lib/api.ts::pollJob.
"""

import io
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from PIL import Image
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..config import get_settings
from ..db import get_db
from ..jobs import create_job
from ..models_db import Job, JobStatus
from ..moderation import ModerationError, check_prompt
from ..queue_backend import get_executor
from ..schemas import JobOut, TextGenerationRequest
from ..security import limiter, require_api_key
from ..storage import get_storage

router = APIRouter(prefix="/jobs", tags=["jobs"])

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 Mo
# shortcut: ~27 s par job en mode normal, donc 8 jobs = ~4 min d'attente, sous les 5 min de patience du front.
MAX_PENDING_JOBS = 8
_rate = get_settings().rate_limit  # chaîne slowapi (ex: "20/minute"), résolue au chargement du module


def check_capacity(db: Session = Depends(get_db)) -> None:
    # Un seul GPU traite les jobs un par un : sans borne, un client rapide remplit la file pour des heures.
    pending = db.query(Job).filter(Job.status.in_((JobStatus.QUEUED.value, JobStatus.RUNNING.value))).count()
    if pending >= MAX_PENDING_JOBS:
        raise HTTPException(status_code=503, detail="Serveur occupé, réessayez dans quelques minutes.")


def _reject_blocked_prompt(prompt: str) -> None:
    # Refus tout de suite : sinon le visiteur attendrait son tour dans la file pour apprendre le rejet.
    # Le worker refait le contrôle (execute_job), au cas où un job arrive par une autre voie.
    try:
        check_prompt(prompt)
    except ModerationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _read_image(upload: UploadFile) -> bytes:
    # Lecture bornée : un gros fichier ne doit pas être chargé en entier en RAM pour être refusé.
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux (8 Mo max).")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception as exc:  # noqa: BLE001 — PIL lève des erreurs de types variés (format, taille, corruption)
        raise HTTPException(status_code=422, detail="Fichier image invalide.") from exc
    return data


def _store(storage, data: bytes) -> str:
    # Le nom envoyé par le client est ignoré : trop long, il faisait planter l'écriture (500).
    key = f"uploads/{uuid.uuid4().hex}"
    storage.save(data, key)
    return key


async def _dispatch(db: Session, job: Job) -> JobOut:
    await run_in_threadpool(get_executor().submit, job.id)
    db.refresh(job)
    return JobOut.model_validate(job)


@router.post("/text", response_model=JobOut, dependencies=[Depends(require_api_key), Depends(check_capacity)])
@limiter.limit(_rate)
async def submit_text_job(request: Request, payload: TextGenerationRequest, db: Session = Depends(get_db)) -> JobOut:
    _reject_blocked_prompt(payload.character_name)
    job = create_job(db, mode="text", prompt=payload.character_name, steps=payload.steps, turbo=payload.turbo_mode)
    return await _dispatch(db, job)


@router.post("/img2img", response_model=JobOut, dependencies=[Depends(require_api_key), Depends(check_capacity)])
@limiter.limit(_rate)
async def submit_img2img_job(
    request: Request,
    prompt: str = Form(..., min_length=1, max_length=200),
    steps: int = Form(30, ge=1, le=75),
    strength: float = Form(0.75, ge=0, le=1),
    turbo_mode: bool = Form(True),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> JobOut:
    _reject_blocked_prompt(prompt)
    storage = get_storage()
    input_key = _store(storage, await _read_image(file))
    job = create_job(
        db, mode="img2img", prompt=prompt, steps=steps, turbo=turbo_mode, strength=strength, input_image_key=input_key
    )
    return await _dispatch(db, job)


@router.post("/pose", response_model=JobOut, dependencies=[Depends(require_api_key), Depends(check_capacity)])
@limiter.limit(_rate)
async def submit_pose_job(
    request: Request,
    prompt: str = Form(..., min_length=1, max_length=200),
    steps: int = Form(30, ge=1, le=75),
    turbo_mode: bool = Form(True),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> JobOut:
    _reject_blocked_prompt(prompt)
    storage = get_storage()
    input_key = _store(storage, await _read_image(file))
    job = create_job(db, mode="pose", prompt=prompt, steps=steps, turbo=turbo_mode, input_image_key=input_key)
    return await _dispatch(db, job)


@router.post("/inpaint", response_model=JobOut, dependencies=[Depends(require_api_key), Depends(check_capacity)])
@limiter.limit(_rate)
async def submit_inpaint_job(
    request: Request,
    prompt: str = Form(..., min_length=1, max_length=200),
    steps: int = Form(30, ge=1, le=75),
    turbo_mode: bool = Form(True),
    image_file: UploadFile = File(...),
    mask_file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> JobOut:
    _reject_blocked_prompt(prompt)
    # Les deux fichiers sont validés avant d'écrire quoi que ce soit : pas d'image orpheline si le masque est refusé.
    image_data = await _read_image(image_file)
    mask_data = await _read_image(mask_file)
    storage = get_storage()
    input_key = _store(storage, image_data)
    mask_key = _store(storage, mask_data)
    job = create_job(
        db,
        mode="inpaint",
        prompt=prompt,
        steps=steps,
        turbo=turbo_mode,
        input_image_key=input_key,
        mask_image_key=mask_key,
    )
    return await _dispatch(db, job)


@router.get("/{job_id}", response_model=JobOut, dependencies=[Depends(require_api_key)])
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job introuvable.")
    return JobOut.model_validate(job)


@router.get("", response_model=list[JobOut], dependencies=[Depends(require_api_key)])
def list_jobs(limit: int = 60, offset: int = 0, db: Session = Depends(get_db)) -> list[JobOut]:
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    jobs = (
        db.query(Job)
        .filter(Job.status == "done")
        .order_by(Job.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [JobOut.model_validate(j) for j in jobs]
