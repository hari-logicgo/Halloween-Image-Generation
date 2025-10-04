import os
import uuid
import shutil
import requests
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from gradio_client import Client, handle_file
from PIL import Image
from contextlib import asynccontextmanager
import logging
from pymongo import MongoClient
from datetime import datetime

# ----------------- LOGGING -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- CONFIG -----------------
API_TOKEN = os.getenv("API_TOKEN", "logicgo@123")
BASE_DIR = os.path.dirname(__file__)

# Directories
HALLOWEEN_INPUT_DIR = os.path.join(BASE_DIR, "halloween_input")
HALLOWEEN_OUTPUT_DIR = os.path.join(BASE_DIR, "halloween_output")
GARMENT_INPUT_DIR = os.path.join(BASE_DIR, "garment_input")
GARMENT_OUTPUT_DIR = os.path.join(BASE_DIR, "garment_output")

for d in [HALLOWEEN_INPUT_DIR, HALLOWEEN_OUTPUT_DIR, GARMENT_INPUT_DIR, GARMENT_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# Predefined garment URLs
GARMENT_URLS = {
    "vampire_dress.jpg": "https://raw.githubusercontent.com/hari-logicgo/faceswap-api/main/vampire-dress.jpg",
    "witch_dress.webp": "https://raw.githubusercontent.com/hari-logicgo/faceswap-api/main/witch-dress.webp",
    "skull_dress.jpg": "https://raw.githubusercontent.com/hari-logicgo/faceswap-api/main/skull-dress.jpg"
}

# ----------------- MONGODB -----------------
mongo_client = MongoClient("mongodb+srv://harilogicgo_db_user:g6Zz4M2xWpr3B2VM@cluster0.bnzjt7f.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = mongo_client["image_transform_logs"]
logs_collection = db["api_logs"]

def log_to_mongo(endpoint: str, filename: str):
    try:
        log_entry = {
            "endpoint": endpoint,
            "filename": filename,
            "timestamp": datetime.utcnow()
        }
        logs_collection.insert_one(log_entry)
        logger.info(f"Logged {endpoint} call for {filename} to MongoDB.")
    except Exception as e:
        logger.error(f"Failed to log {endpoint} call to MongoDB: {e}")

# ----------------- GLOBAL CLIENTS -----------------
HALLOWEEN_CLIENT = None
GARMENT_CLIENT = None

# ----------------- LIFESPAN -----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global HALLOWEEN_CLIENT, GARMENT_CLIENT
    
    logger.info("Startup: downloading garment images if needed...")
    for filename, url in GARMENT_URLS.items():
        dest_path = os.path.join(GARMENT_INPUT_DIR, filename)
        if not os.path.exists(dest_path):
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                logger.info(f"Downloaded {filename}")
            except Exception as e:
                logger.error(f"Failed to download {filename} from {url}: {e}")
    
    # Initialize Gradio clients
    try:
        hf_token = os.getenv("HF_TOKEN")
        HALLOWEEN_CLIENT = Client("https://logicgoinfotechspaces-halloween-image.hf.space", hf_token=hf_token if hf_token else None)
        logger.info("Halloween client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Halloween client: {e}")
    
    try:
        GARMENT_CLIENT = Client("franciszzj/Leffa")
        logger.info("Garment client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Garment client: {e}")
    
    logger.info("Startup complete.")
    yield
    logger.info("Shutdown complete.")

# ----------------- FASTAPI APP -----------------
app = FastAPI(title="Combined Halloween + Virtual Try-On API", lifespan=lifespan)

# Mount static directories
app.mount("/halloween_input", StaticFiles(directory=HALLOWEEN_INPUT_DIR), name="halloween_input")
app.mount("/halloween_output", StaticFiles(directory=HALLOWEEN_OUTPUT_DIR), name="halloween_output")
app.mount("/garment_input", StaticFiles(directory=GARMENT_INPUT_DIR), name="garment_input")
app.mount("/garment_output", StaticFiles(directory=GARMENT_OUTPUT_DIR), name="garment_output")

# ----------------- HELPERS -----------------
def verify_token(auth_header: str):
    if not auth_header or auth_header != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized. Invalid or missing token.")

def process_garment_image(source_path, garment_path):
    if GARMENT_CLIENT is None:
        raise HTTPException(status_code=503, detail="Garment service unavailable")
    try:
        result = GARMENT_CLIENT.predict(
            src_image_path=handle_file(source_path),
            ref_image_path=handle_file(garment_path),
            ref_acceleration=False,
            step=30,
            scale=2.5,
            seed=42,
            vt_model_type="viton_hd",
            vt_garment_type="upper_body",
            vt_repaint=False,
            api_name="/leffa_predict_vt"
        )
        generated_image_path = result[0]
        return Image.open(generated_image_path)
    except Exception as e:
        logger.error(f"Error processing garment image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

def process_halloween_image(input_path, prompt):
    if HALLOWEEN_CLIENT is None:
        raise HTTPException(status_code=503, detail="Halloween service unavailable")
    try:
        result = HALLOWEEN_CLIENT.predict(
            input_image=handle_file(input_path),
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            guidance_scale=2.5,
            steps=28,
            api_name="/infer"
        )
        image_path = result[0]
        return Image.open(image_path)
    except Exception as e:
        logger.error(f"Error processing halloween image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ----------------- HEALTH CHECK -----------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": {
        "halloween": HALLOWEEN_CLIENT is not None,
        "garment": GARMENT_CLIENT is not None
    }}

# ----------------- HALLOWEEN ENDPOINTS -----------------
@app.post("/halloween/transform")
async def halloween_transform(
    file: UploadFile = File(...),
    prompt: str = "Make the person look like a vampire with pale skin, glowing red eyes, sharp fangs, and dark gothic makeup. Add a spooky background with bats.",
    authorization: str = Header(None)
):
    verify_token(authorization)
    try:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        upload_path = os.path.join(HALLOWEEN_INPUT_DIR, unique_filename)
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_image = process_halloween_image(upload_path, prompt)
        output_filename = f"{uuid.uuid4()}.webp"
        output_path = os.path.join(HALLOWEEN_OUTPUT_DIR, output_filename)
        output_image.save(output_path)

        # --- MongoDB Logging ---
        log_to_mongo("/halloween/transform", output_filename)

        return JSONResponse({
            "status": "success",
            "url": f"/halloween_output/{output_filename}"
        })
    except Exception as e:
        logger.error(f"Halloween transform error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- GARMENT ENDPOINTS -----------------
@app.post("/garment/transform")
async def garment_transform(
    source_file: UploadFile = File(...),
    garment_filename: str = Form(...),
    authorization: str = Header(None)
):
    verify_token(authorization)
    try:
        unique_filename = f"{uuid.uuid4()}_{source_file.filename}"
        source_path = os.path.join(GARMENT_INPUT_DIR, unique_filename)
        with open(source_path, "wb") as buffer:
            shutil.copyfileobj(source_file.file, buffer)

        garment_path = os.path.join(GARMENT_INPUT_DIR, garment_filename)
        if not os.path.exists(garment_path):
            raise HTTPException(status_code=404, detail="Garment image not found.")

        output_image = process_garment_image(source_path, garment_path)
        output_filename = f"{uuid.uuid4()}.webp"
        output_path = os.path.join(GARMENT_OUTPUT_DIR, output_filename)
        output_image.save(output_path)

        # --- MongoDB Logging ---
        log_to_mongo("/garment/transform", output_filename)

        return JSONResponse({
            "status": "success",
            "preview_url": f"/garment_output/{output_filename}",
            "filename": output_filename
        })
    except Exception as e:
        logger.error(f"Garment transform error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/garment/list")
async def list_garments():
    try:
        garments = os.listdir(GARMENT_INPUT_DIR)
        return JSONResponse({
            "garments": [{"filename": g, "url": f"/garment_input/{g}"} for g in garments]
        })
    except Exception as e:
        logger.error(f"List garments error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preview/garment/{filename}")
async def preview_garment(filename: str):
    file_path = os.path.join(GARMENT_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type="image/webp")

@app.get("/download/garment/{filename}")
async def download_garment(filename: str):
    file_path = os.path.join(GARMENT_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)

# ----------------- RUN APP -----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
