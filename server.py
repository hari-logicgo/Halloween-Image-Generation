import os
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import httpx
from PIL import Image
import io
# --- MongoDB Imports ---
# Make sure to install: pip install pymongo "bson[cpython]"
from bson.objectid import ObjectId
from pymongo import MongoClient
from datetime import datetime, timedelta
# -----------------------

# -----------------------------------------
# 1. LOGGING SETUP
# -----------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------
# 2. GLOBAL CONSTANTS AND DIRECTORY SETUP
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent
GARMENT_TEMPLATES_DIR = BASE_DIR / "Halloween Dress"
GARMENT_INPUT_DIR = BASE_DIR / "garment_input"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Hugging Face API Configuration
# HF_API_URL = "https://logicgoinfotechspaces-halloweenfaceswap.hf.space/face-swap"
# HF_AUTH = "Bearer logicgo@123"
# NEW
HF_API_URL = "https://logicgoinfotechspaces-faceswap.hf.space/face-swap"
HF_AUTH = "Bearer logicgo_secret_123"

# Ensure directories exist
GARMENT_TEMPLATES_DIR.mkdir(exist_ok=True)
GARMENT_INPUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------
# 3. MONGODB CONNECTION SETUP
# -----------------------------------------
_mongo_client = None
_subcategories_col = None # To fetch image URLs
_media_clicks_col = None  # To log user activity

try:
    # Use the URI provided by the user (or environment variable if available)
    _admin_mongo_uri = os.getenv("MONGODB_ADMIN_URI")
    _admin_mongo_db = "adminPanel"
    
    logger.info(f"Attempting connection to Admin DB: {_admin_mongo_db}...") 
    
    _mongo_client = MongoClient(_admin_mongo_uri, connect=False)
    _subcategories_col = _mongo_client[_admin_mongo_db]["subcategories"]
    _media_clicks_col = _mongo_client[_admin_mongo_db]["media_clicks"]
    logger.info("MongoDB client established for subcategories and media_clicks.")

except Exception as e:
    logger.error(f"FATAL: Admin MongoDB connection failed. Database features disabled. Error: {e}")
    _mongo_client = None
    _subcategories_col = None
    _media_clicks_col = None
# --- NEW: REMOTE DOWNLOAD HELPER (Robust I/O Error Handling) ---
async def download_remote_asset(url: str, directory: Path) -> str:
    """Downloads a file from a remote URL, saves it locally, and returns its local filename."""
    logger.info(f"Attempting download of remote asset from: {url}")
    
    try:
        # 1. Fetch the remote file content
        async with httpx.AsyncClient(timeout=30.0) as client: 
            resp = await client.get(url)
            resp.raise_for_status() 
    except httpx.RequestError as e:
        logger.error(f"Failed to download remote asset {url} (Network/Timeout): {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Failed to download asset from remote URL (Network): {url}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Remote asset returned status code {e.response.status_code} for {url}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Remote asset not found or inaccessible: {url}")
        
    # 2. Generate filename
    path_suffix = Path(url).name
    if not Path(path_suffix).suffix:
        path_suffix += ".jpg" 
    
    path_suffix = path_suffix.replace('..', '_').replace('/', '_')
    filename = f"remote_{int(time.time()*1000)}_{path_suffix}"
    local_path = directory / filename 
    
    # 3. Save the file locally
    try:
        with open(local_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Remote asset successfully saved to: {local_path.name}")
    except OSError as e:
        # This catches PermissionError, FileNotFoundError, etc.
        logger.error(f"FILE I/O ERROR: Failed to write {local_path} to disk: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Server File I/O Error: Cannot save downloaded asset to disk. Check directory permissions."
        )
    
    return filename
# -----------------------------------#

def sync_log_media_click(user_id_str: str, category_id_str: str):
    if _media_clicks_col is None:
        logger.warning("sync_log_media_click called but MongoDB is not connected.")
        return

    try:
        user_oid = ObjectId(user_id_str.strip())
        category_oid = ObjectId(category_id_str.strip())
        now = datetime.utcnow()

        # Normalize to UTC midnight
        today_date = datetime(now.year, now.month, now.day)

        logger.info(
            f"Attempting background write for User:{user_id_str}, Category:{category_id_str}"
        )

        # ------------------------------------------------
        # STEP 1: ENSURE USER DOC + DAILY FIELD
        # ------------------------------------------------
        _media_clicks_col.update_one(
            {"userId": user_oid},
            {
                "$setOnInsert": {
                    "userId": user_oid,
                    "createdAt": now,
                    "ai_edit_complete": 0,
                    "ai_edit_daily_count": [],
                },
                "$set": {
                    "updatedAt": now,
                    "ai_edit_last_date": now,
                },
            },
            upsert=True,
        )

        # ------------------------------------------------
        # STEP 2: DAILY COUNT LOGIC (CORRECT)
        # ------------------------------------------------
       # ------------------------------------------------
        # STEP 2: DAILY COUNT LOGIC (FIXED)
        # ------------------------------------------------
        doc = _media_clicks_col.find_one(
            {"userId": user_oid},
            {"ai_edit_daily_count": 1}
        )
        
        daily_entries = doc.get("ai_edit_daily_count", []) if doc else []
        daily_updates = []
        
        if not daily_entries:
            # First-ever usage → only today
            daily_updates.append({
                "date": today_date,
                "count": 1
            })
        else:
            # Build a set of existing dates to avoid duplicates
            existing_dates = {entry["date"].date() for entry in daily_entries}
            last_date = max(entry["date"] for entry in daily_entries)
        
            # Backfill all skipped days between last_date and today-1
            next_expected_date = last_date + timedelta(days=1)
            while next_expected_date.date() < today_date.date():
                if next_expected_date.date() not in existing_dates:
                    daily_updates.append({
                        "date": next_expected_date,
                        "count": 0
                    })
                next_expected_date += timedelta(days=1)
        
            # Add today if not already present
            if today_date.date() not in existing_dates:
                daily_updates.append({
                    "date": today_date,
                    "count": 1
                })
        
        # Push new entries if any
        if daily_updates:
            _media_clicks_col.update_one(
                {"userId": user_oid},
                {"$push": {"ai_edit_daily_count": {"$each": daily_updates}}}
            )
        
        # ------------------------------------------------
        # STEP 2.5: SORT OLDEST → NEWEST AND TRIM TO 32 DAYS
        # ------------------------------------------------
        doc = _media_clicks_col.find_one({"userId": user_oid}, {"ai_edit_daily_count": 1})
        daily_entries = doc.get("ai_edit_daily_count", []) if doc else []
        
        # Sort oldest first
        daily_entries.sort(key=lambda x: x["date"])
        
        # Trim to last 32 entries
        if len(daily_entries) > 32:
            daily_entries = daily_entries[-32:]
            _media_clicks_col.update_one(
                {"userId": user_oid},
                {"$set": {"ai_edit_daily_count": daily_entries}}
            )

        # ------------------------------------------------
        # STEP 3: AI EDIT GLOBAL COUNTER
        # ------------------------------------------------
        _media_clicks_col.update_one(
            {"userId": user_oid},
            {"$inc": {"ai_edit_complete": 1}}
        )

        # ------------------------------------------------
        # STEP 4: UPDATE EXISTING CATEGORY
        # ------------------------------------------------
        update_result = _media_clicks_col.update_one(
            {
                "userId": user_oid,
                "categories.categoryId": category_oid,
            },
            {
                "$set": {
                    "updatedAt": now,
                    "categories.$.lastClickedAt": now,
                },
                "$inc": {
                    "categories.$.click_count": 1,
                },
            },
        )

        # ------------------------------------------------
        # STEP 5: PUSH CATEGORY IF NOT EXISTS
        # ------------------------------------------------
        if update_result.matched_count == 0:
            _media_clicks_col.update_one(
                {"userId": user_oid},
                {
                    "$set": {"updatedAt": now},
                    "$push": {
                        "categories": {
                            "categoryId": category_oid,
                            "click_count": 1,
                            "lastClickedAt": now,
                        }
                    },
                },
            )

        logger.info(
            f"Media click logged for User {user_id_str} on Category {category_id_str}"
        )

    except Exception as media_err:
        logger.error(f"MEDIA_CLICK LOGGING WRITE ERROR: {media_err}")
        
MAX_COMPRESSED_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_DIMENSION = 1000  # 1000x1000 max
def compress_image_file(
    input_path,
    output_path
):
    """
    Compress image by:
    - limiting dimensions <= 1000x1000
    - reducing quality
    - ensuring size <= ~2MB
    """

    img = Image.open(input_path).convert("RGB")

    # Resize while keeping aspect ratio
    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    quality = 85
    buffer = io.BytesIO()

    while quality >= 45:
        buffer.seek(0)
        buffer.truncate()

        img.save(
            buffer,
            format="JPEG",  # JPEG gives best size control
            quality=quality,
            optimize=True,
            progressive=True
        )

        if buffer.tell() <= MAX_COMPRESSED_SIZE:
            break

        quality -= 5

    # Write compressed image to disk
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())
        
# -----------------------------------------
# 5. FASTAPI APP SETUP AND MOUNTING
# -----------------------------------------

app = FastAPI(title="Halloween Image API - Filesystem Mode")

# Safe mount: only mount folder if it exists
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
                # Use filename as a placeholder for target_category_id for local files
                target_id = p.stem 
                
                items.append({
                    "filename": p.name,
                    "url": f"/garment_templates/{p.name}",
                    # NOTE: This target_category_id is the local filename stem, 
                    # not the MongoDB ID, for local files.
                    "target_category_id": target_id 
                })

    return items

# -----------------------------------------
# 6. ENDPOINTS
# -----------------------------------------

@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "healthy",
        "source": "filesystem",
        "templates_exists": GARMENT_TEMPLATES_DIR.exists(),
        "input_exists": GARMENT_INPUT_DIR.exists(),
        "mongo_connected": _mongo_client is not None
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

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
# ... (Helper function download_remote_asset must be defined above this endpoint) ...
@app.post("/garment/transform")
async def garment_transform(
    sourceFile: UploadFile = File(..., description="The user's image file for face-swapping."),
    # If provided, target is a local filename (OLD LOGIC)
    garment_filename: Optional[str] = Form(None), 
    # If provided, target is a MongoDB asset ID (NEW LOGIC)
    target_category_id: Optional[str] = Form(None), 
    # Optional Subcategory ID for logging
    category_id: Optional[str] = Form(None), 
    # Optional User ID for logging
    user_id: Optional[str] = Form(None) 
):
    # 1. Mutual Exclusion and Input Validation
    is_filename_provided = garment_filename is not None and garment_filename.strip()
    is_target_id_provided = target_category_id is not None and target_category_id.strip()

    if is_filename_provided and is_target_id_provided:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Only one of 'garment_filename' (local file) or 'target_category_id' (DB asset ID) can be provided."
        )
    elif not is_filename_provided and not is_target_id_provided:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Must provide either 'garment_filename' or 'target_category_id'."
        )
    
    target_value = None

    # 2. Determine Target Garment (Local Filename or Remote URL)
    if is_filename_provided:
        # OLD LOGIC: Use local filename directly
        target_value = garment_filename
        logger.info(f"Target determined: Local filename {target_value}")

    elif is_target_id_provided:
        # NEW LOGIC: Look up URL in MongoDB, then DOWNLOAD IT.
        if _subcategories_col is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database lookup service is unavailable.")
        
        try:
            target_asset_oid = ObjectId(target_category_id.strip())
        except Exception:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid target_category_id format.")

        try:
            # Blocking MongoDB read operation must be run in a separate thread
            subcat_doc = await asyncio.to_thread(
                _subcategories_col.find_one,
                {"asset_images._id": target_asset_oid},
                {"asset_images.$": 1}
            )
        except Exception as e:
            logger.error(f"MongoDB query failed for asset ID {target_category_id}: {e}")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database query failed during asset lookup.")
        
        if subcat_doc and subcat_doc.get('asset_images') and subcat_doc['asset_images']:
            target_url = subcat_doc['asset_images'][0]['url']
            
            try:
                # --- CRITICAL NEW STEP: DOWNLOAD THE REMOTE ASSET ---
                local_filename = await download_remote_asset(target_url, GARMENT_TEMPLATES_DIR) 
                target_value = local_filename # PASS THE LOCAL FILENAME TO HF
                # ----------------------------------------------------
                logger.info(f"Target determined and downloaded locally: {target_value}")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Unexpected error during remote asset download: {e}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process remote asset.")
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Garment asset ID {target_category_id} not found in DB.")

    # 3. Prepare API Request (source image upload)
    file_content = await sourceFile.read()
    
    files = {
        "source": (sourceFile.filename, file_content, sourceFile.content_type)
    }
    # data = {
    #     # target_value is now guaranteed to be a local filename/reference
    #     "target": target_value 
    # }
    data = {
        "user_id": user_id,
        "new_category_id": target_category_id,
        "target_category_id": "" # mapping your category_id → new_category_id
    }
    
    logger.info(
    f"Calling FaceSwap API → new_category_id={target_category_id}, user_id={user_id}"
    )


    # 4. Call Hugging Face API
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                HF_API_URL,
                headers={"Authorization": HF_AUTH},
                files=files,
                data=data
            )
    except httpx.RequestError as e:
        logger.error(f"HF API network error: {e}")
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="External API request failed due to timeout or network error.")


    if resp.status_code != 200:
        logger.error(f"HF API failed with status {resp.status_code}: {resp.text}")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("Content-Type")
        )

    # 5. Process HF Response and Save Locally
    try:
        hf_data = resp.json()
        filename = hf_data["filename"]
        hf_image_url = (
            "https://logicgoinfotechspaces-halloweenfaceswap.hf.space"
            + hf_data["preview_url"]
        )
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Failed to parse or extract keys from HF JSON response: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid response format from external API.")

    # Download Image from HF
    try:
        async with httpx.AsyncClient() as client:
            img_resp = await client.get(hf_image_url)
            img_resp.raise_for_status() 
    except Exception as e:
        logger.error(f"Failed to download final image from HF: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve final processed image.")

    # Save Locally
    local_path = GARMENT_INPUT_DIR / filename
    with open(local_path, "wb") as f:
        f.write(img_resp.content)
    
    logger.info(f"Generated image saved locally: {filename}")
    # ---------------------------------------------------------
    # COMPRESS GENERATED IMAGE
    # ---------------------------------------------------------
    compressed_filename = filename.rsplit(".", 1)[0] + "_compressed.jpg"
    compressed_path = GARMENT_INPUT_DIR / compressed_filename
    
    try:
        compress_image_file(
            input_path=local_path,
            output_path=compressed_path
        )
        logger.info(f"Compressed image generated: {compressed_filename}")
    except Exception as e:
        logger.error(f"Image compression failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compress generated image."
        )

    # 6. Conditional Media Click Logging
    if user_id and category_id:
        try:
            ObjectId(user_id.strip())
            ObjectId(category_id.strip())
            
            asyncio.create_task(
                asyncio.to_thread(sync_log_media_click, user_id, category_id)
            )
        except Exception as log_err:
            logger.warning(f"Skipping media click log due to invalid ID format or internal error: {log_err}")
            pass

    # 7. Return Final Response
    return {
        "status": "success",
        "preview_url": f"/preview/garment/{filename}",
        "filename": filename,
        "Compressed_Image_URL": (
            f"https://halloween-image-generation.onrender.com"
            f"/preview/garment/{compressed_filename}"
        )
    }















# ... old code 



# import os
# from pathlib import Path
# from typing import List, Dict

# from fastapi import FastAPI, HTTPException
# from fastapi.responses import FileResponse, Response
# from fastapi.staticfiles import StaticFiles
# from fastapi import UploadFile, File, Form
# import httpx

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


# # -------------------------------
# # NEW: POST /garment/transform (async)
# # -------------------------------
# HF_API_URL = "https://logicgoinfotechspaces-halloweenfaceswap.hf.space/face-swap"
# HF_AUTH = "Bearer logicgo@123"

# @app.post("/garment/transform")
# async def garment_transform(
#     sourceFile: UploadFile = File(...),
#     garment_filename: str = Form(...)
# ):
#     file_content = await sourceFile.read()

#     files = {
#         "source": (sourceFile.filename, file_content, sourceFile.content_type)
#     }
#     data = {
#         "target": garment_filename
#     }

#     async with httpx.AsyncClient(timeout=120.0) as client:
#         resp = await client.post(
#             HF_API_URL,
#             headers={"Authorization": HF_AUTH},
#             files=files,
#             data=data
#         )

#     if resp.status_code != 200:
#         return Response(
#             content=resp.content,
#             status_code=resp.status_code,
#             media_type=resp.headers.get("Content-Type")
#         )

#     # ✅ HF RETURNS JSON WITH preview_url & filename
#     hf_data = resp.json()
#     filename = hf_data["filename"]

#     # ✅ DOWNLOAD IMAGE FROM HF
#     hf_image_url = (
#         "https://logicgoinfotechspaces-halloweenfaceswap.hf.space"
#         + hf_data["preview_url"]
#     )

#     async with httpx.AsyncClient() as client:
#         img_resp = await client.get(hf_image_url)

#     # ✅ SAVE LOCALLY SO /preview WORKS
#     local_path = GARMENT_INPUT_DIR / filename
#     with open(local_path, "wb") as f:
#         f.write(img_resp.content)

#     # ✅ RETURN SAME FORMAT YOU WANT
#     return {
#         "status": "success",
#         "preview_url": f"/preview/garment/{filename}",
#         "filename": filename
#     }
