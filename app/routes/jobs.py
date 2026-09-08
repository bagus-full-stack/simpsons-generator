"""Endpoints de génération.

Contrat : chaque endpoint POST dépose un job et répond tout de suite avec son
statut (au lieu de bloquer jusqu'à la fin de l'inférence comme l'ancienne API).
Le client va ensuite consulter GET /jobs/{id} jusqu'à ce que le statut soit
"done"/"failed"/"rejected" — voir simpson-front/app/lib/api.ts::pollJob.
"""

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..config import get_settings
from ..db import get_db
from ..jobs import create_job
from ..models_db import Job
from ..queue_backend import get_executor
from ..schemas import JobOut, TextGenerationRequest
from ..security import limiter, require_api_key
from ..storage import get_storage

router = APIRouter(prefix="/jobs", tags=["jobs"])

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 Mo
_rate = get_settings().rate_limit  # chaîne slowapi (ex: "20/minute"), résolue au chargement du module


async def _save_upload(storage, upload: UploadFile, prefix: str) -> str:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux (8 Mo max).")
    safe_name = (upload.filename or "upload").replace("/", "_")
    key = f"{prefix}/{uuid.uuid4().hex}_{safe_name}"
    storage.save(data, key)
    return key


async def _dispatch(db: Session, job: Job) -> JobOut:
    await run_in_threadpool(get_executor().submit, job.id)
    db.refresh(job)
    return JobOut.model_validate(job)


@router.post("/text", response_model=JobOut, dependencies=[Depends(require_api_key)])
@limiter.limit(_rate)
async def submit_text_job(request: Request, payload: TextGenerationRequest, db: Session = Depends(get_db)) -> JobOut:
    job = create_job(db, mode="text", prompt=payload.character_name, steps=payload.steps, turbo=payload.turbo_mode)
    return await _dispatch(db, job)


@router.post("/img2img", response_model=JobOut, dependencies=[Depends(require_api_key)])
@limiter.limit(_rate)
async def submit_img2img_job(
    request: Request,
    prompt: str = Form(...),
    steps: int = Form(30),
    strength: float = Form(0.75),
    turbo_mode: bool = Form(False),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> JobOut:
    storage = get_storage()
    input_key = await _save_upload(storage, file, "uploads")
    job = create_job(
        db, mode="img2img", prompt=prompt, steps=steps, turbo=turbo_mode, strength=strength, input_image_key=input_key
    )
    return await _dispatch(db, job)


@router.post("/pose", response_model=JobOut, dependencies=[Depends(require_api_key)])
@limiter.limit(_rate)
async def submit_pose_job(
    request: Request,
    prompt: str = Form(...),
    steps: int = Form(30),
    turbo_mode: bool = Form(False),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> JobOut:
    storage = get_storage()
    input_key = await _save_upload(storage, file, "uploads")
    job = create_job(db, mode="pose", prompt=prompt, steps=steps, turbo=turbo_mode, input_image_key=input_key)
    return await _dispatch(db, job)


@router.post("/inpaint", response_model=JobOut, dependencies=[Depends(require_api_key)])
@limiter.limit(_rate)
async def submit_inpaint_job(
    request: Request,
    prompt: str = Form(...),
    steps: int = Form(30),
    turbo_mode: bool = Form(False),
    image_file: UploadFile = File(...),
    mask_file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> JobOut:
    storage = get_storage()
    input_key = await _save_upload(storage, image_file, "uploads")
    mask_key = await _save_upload(storage, mask_file, "uploads")
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


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job introuvable.")
    return JobOut.model_validate(job)


@router.get("", response_model=list[JobOut])
def list_jobs(limit: int = 60, offset: int = 0, db: Session = Depends(get_db)) -> list[JobOut]:
    limit = max(1, min(limit, 100))
    jobs = (
        db.query(Job)
        .filter(Job.status == "done")
        .order_by(Job.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [JobOut.model_validate(j) for j in jobs]
