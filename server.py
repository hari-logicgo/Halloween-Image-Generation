import os
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


app = FastAPI(title="Halloween Image API - Filesystem Mode")


BASE_DIR = Path(__file__).resolve().parent
GARMENT_TEMPLATES_DIR = BASE_DIR / "Halloween Dress"
GARMENT_INPUT_DIR = BASE_DIR / "garment_input"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_folder_images(directory: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if directory.exists():
        for p in sorted(directory.iterdir()):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                items.append({"filename": p.name, "url": f"/garment_templates/{p.name}"})
    return items


@app.get("/health")
def health() -> Dict[str, object]:
    return {"status": "healthy", "source": "filesystem"}


@app.get("/garment/list")
def garment_list() -> Dict[str, List[Dict[str, str]]]:
    items: List[Dict[str, str]] = []
    items.extend(list_folder_images(GARMENT_TEMPLATES_DIR))
    items.extend(list_folder_images(GARMENT_INPUT_DIR))
    return {"garments": items}


@app.get("/preview/garment/{filename}")
def preview_garment(filename: str):
    for directory in (GARMENT_TEMPLATES_DIR, GARMENT_INPUT_DIR):
        candidate = directory / filename
        if candidate.exists():
            return FileResponse(candidate)
    raise HTTPException(status_code=404, detail="File not found")


