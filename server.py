import os
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File, Form
import httpx

app = FastAPI(title="Halloween Image API - Filesystem Mode")

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Directories (spaces fixed)
GARMENT_TEMPLATES_DIR = BASE_DIR / "Halloween Dress"
GARMENT_INPUT_DIR = BASE_DIR / "garment_input"

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# -----------------------------------------
# Ensure directories exist (avoids crashes)
# -----------------------------------------
GARMENT_TEMPLATES_DIR.mkdir(exist_ok=True)
GARMENT_INPUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------
# Safe mount: only mount folder if it exists
# -----------------------------------------
if GARMENT_TEMPLATES_DIR.exists():
    app.mount(
        "/garment_templates",
        StaticFiles(directory=str(GARMENT_TEMPLATES_DIR)),
        name="garment_templates"
    )

if GARMENT_INPUT_DIR.exists():
    app.mount(
        "/garment_input",
        StaticFiles(directory=str(GARMENT_INPUT_DIR)),
        name="garment_input"
    )


# Helper to list images
def list_folder_images(directory: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    if directory.exists():
        for p in sorted(directory.iterdir()):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                items.append({
                    "filename": p.name,
                    "url": f"/garment_templates/{p.name}"
                })

    return items


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "healthy",
        "source": "filesystem",
        "templates_exists": GARMENT_TEMPLATES_DIR.exists(),
        "input_exists": GARMENT_INPUT_DIR.exists(),
    }


@app.get("/garment/list")
def garment_list(limit: int = 10) -> Dict[str, List[Dict[str, str]]]:
    items: List[Dict[str, str]] = []
    items.extend(list_folder_images(GARMENT_TEMPLATES_DIR))
    items.extend(list_folder_images(GARMENT_INPUT_DIR))
    return {"garments": items[:limit]}


@app.get("/preview/garment/{filename}")
def preview_garment(filename: str):
    for directory in (GARMENT_TEMPLATES_DIR, GARMENT_INPUT_DIR):
        candidate = directory / filename
        if candidate.exists():
            return FileResponse(candidate)

    raise HTTPException(status_code=404, detail="File not found")


# -------------------------------
# NEW: POST /garment/transform (async)
# -------------------------------
HF_API_URL = "https://logicgoinfotechspaces-halloweenfaceswap.hf.space/face-swap"
HF_AUTH = "Bearer logicgo@123"

@app.post("/garment/transform")
async def garment_transform(
    sourceFile: UploadFile = File(...),
    garment_filename: str = Form(...)
):
    file_content = await sourceFile.read()

    files = {
        "source": (sourceFile.filename, file_content, sourceFile.content_type)
    }
    data = {
        "target": garment_filename
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            HF_API_URL,
            headers={"Authorization": HF_AUTH},
            files=files,
            data=data
        )

    if resp.status_code != 200:
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("Content-Type")
        )

    # ✅ HF RETURNS JSON WITH preview_url & filename
    hf_data = resp.json()
    filename = hf_data["filename"]

    # ✅ DOWNLOAD IMAGE FROM HF
    hf_image_url = (
        "https://logicgoinfotechspaces-halloweenfaceswap.hf.space"
        + hf_data["preview_url"]
    )

    async with httpx.AsyncClient() as client:
        img_resp = await client.get(hf_image_url)

    # ✅ SAVE LOCALLY SO /preview WORKS
    local_path = GARMENT_INPUT_DIR / filename
    with open(local_path, "wb") as f:
        f.write(img_resp.content)

    # ✅ RETURN SAME FORMAT YOU WANT
    return {
        "status": "success",
        "preview_url": f"/preview/garment/{filename}",
        "filename": filename
    }


# import os
# from pathlib import Path
# from typing import List, Dict

# from fastapi import FastAPI, HTTPException
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi import UploadFile, File, Form
# import requests
# from fastapi.responses import Response


# app = FastAPI(title="Halloween Image API - Filesystem Mode")

# # Base directory
# BASE_DIR = Path(__file__).resolve().parent

# # Directories (spaces fixed)
# GARMENT_TEMPLATES_DIR = BASE_DIR / "Halloween Dress"
# GARMENT_INPUT_DIR = BASE_DIR / "garment_input"

# ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# # -----------------------------------------
# # Ensure directories exist (avoids crashes)
# # -----------------------------------------
# GARMENT_TEMPLATES_DIR.mkdir(exist_ok=True)
# GARMENT_INPUT_DIR.mkdir(exist_ok=True)

# # -----------------------------------------
# # Safe mount: only mount folder if it exists
# # -----------------------------------------
# if GARMENT_TEMPLATES_DIR.exists():
#     app.mount(
#         "/garment_templates",
#         StaticFiles(directory=str(GARMENT_TEMPLATES_DIR)),
#         name="garment_templates"
#     )

# if GARMENT_INPUT_DIR.exists():
#     app.mount(
#         "/garment_input",
#         StaticFiles(directory=str(GARMENT_INPUT_DIR)),
#         name="garment_input"
#     )


# # Helper to list images
# def list_folder_images(directory: Path) -> List[Dict[str, str]]:
#     items: List[Dict[str, str]] = []

#     if directory.exists():
#         for p in sorted(directory.iterdir()):
#             if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
#                 items.append({
#                     "filename": p.name,
#                     "url": f"/garment_templates/{p.name}"
#                 })

#     return items


# @app.get("/health")
# def health() -> Dict[str, object]:
#     return {
#         "status": "healthy",
#         "source": "filesystem",
#         "templates_exists": GARMENT_TEMPLATES_DIR.exists(),
#         "input_exists": GARMENT_INPUT_DIR.exists(),
#     }


# @app.get("/garment/list")
# def garment_list(limit: int = 10) -> Dict[str, List[Dict[str, str]]]:
#     items: List[Dict[str, str]] = []
#     items.extend(list_folder_images(GARMENT_TEMPLATES_DIR))
#     items.extend(list_folder_images(GARMENT_INPUT_DIR))
#     return {"garments": items[:limit]}


# @app.get("/preview/garment/{filename}")
# def preview_garment(filename: str):
#     for directory in (GARMENT_TEMPLATES_DIR, GARMENT_INPUT_DIR):
#         candidate = directory / filename
#         if candidate.exists():
#             return FileResponse(candidate)

#     raise HTTPException(status_code=404, detail="File not found")
