import os
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import httpx

# --- MongoDB Imports ---
# Make sure to install: pip install pymongo "bson[cpython]"
from bson.objectid import ObjectId
from pymongo import MongoClient
from datetime import datetime
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
HF_API_URL = "https://logicgoinfotechspaces-halloweenfaceswap.hf.space/face-swap"
HF_AUTH = "Bearer logicgo@123"

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

# -----------------------------------------
# 4. SYNCHRONOUS LOGGING FUNCTION
# -----------------------------------------
def sync_log_media_click(user_id_str: str, category_id_str: str):
    """
    Synchronously logs a click event to the media_clicks collection.
    This function should be run using asyncio.to_thread().
    """
    if _media_clicks_col is None:
        logger.warning("sync_log_media_click called but MongoDB is not connected.")
        return

    try:
        user_oid = ObjectId(user_id_str.strip())
        category_oid = ObjectId(category_id_str.strip())
        now = datetime.utcnow()
        
        logger.info(f"Attempting background write for User:{user_id_str}, Category:{category_id_str}")
        
        # 1. Try updating an existing category entry
        update_result = _media_clicks_col.update_one(
            {
                "userId": user_oid,
                "categories.categoryId": category_oid
            },
            {
                "$set": {
                    "updatedAt": now,
                    "categories.$.lastClickedAt": now
                },
                "$inc": {
                    "categories.$.click_count": 1
                }
            }
        )
        
        # 2. If no category entry exists -> push a new one (or create user doc)
        if update_result.matched_count == 0:
            _media_clicks_col.update_one(
                { "userId": user_oid },
                {
                    "$setOnInsert": { "createdAt": now },
                    "$set": { "updatedAt": now },
                    "$push": {
                        "categories": {
                            "categoryId": category_oid,
                            "click_count": 1,
                            "lastClickedAt": now
                        }
                    }
                },
                upsert=True
            )
        logger.info(f"Media click logged for User {user_id_str} on Category {category_id_str}")
        
    except Exception as media_err:
        logger.error(f"MEDIA_CLICK LOGGING WRITE ERROR: {media_err}")
        pass

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
        # OLD LOGIC: Use local filename directly (e.g., BloodSchoolgirl.png)
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

        # Blocking MongoDB read operation must be run in a separate thread
        subcat_doc = await asyncio.to_thread(
            _subcategories_col.find_one,
            {"asset_images._id": target_asset_oid},
            {"asset_images.$": 1}
        )
        
        if subcat_doc and subcat_doc.get('asset_images') and subcat_doc['asset_images']:
            target_url = subcat_doc['asset_images'][0]['url']
            
            # --- CRITICAL STEP: DOWNLOAD THE REMOTE ASSET AND GET THE LOCAL FILENAME ---
            local_filename = await download_remote_asset(target_url, GARMENT_TEMPLATES_DIR) 
            target_value = local_filename # PASS THE LOCAL FILENAME TO HF
            # --------------------------------------------------------------------------
            
            logger.info(f"Target determined and downloaded locally: {target_value}")
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Garment asset ID {target_category_id} not found in DB.")

    # 3. Prepare API Request (source image upload)
    file_content = await sourceFile.read()
    
    files = {
        "source": (sourceFile.filename, file_content, sourceFile.content_type)
    }
    data = {
        # target_value is now guaranteed to be a local filename/reference
        "target": target_value 
    }
    
    logger.info(f"Calling HF API with local target reference: {target_value}")

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
    async with httpx.AsyncClient() as client:
        img_resp = await client.get(hf_image_url)
        img_resp.raise_for_status() # Raise an error for bad status codes

    # Save Locally
    local_path = GARMENT_INPUT_DIR / filename
    with open(local_path, "wb") as f:
        f.write(img_resp.content)
    
    logger.info(f"Generated image saved locally: {filename}")

    # 6. Conditional Media Click Logging
    if user_id and category_id:
        try:
            # Validate IDs before logging
            ObjectId(user_id.strip())
            ObjectId(category_id.strip())
            
            # Start the background logging task using create_task to prevent RuntimeWarning
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
        "filename": filename
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
