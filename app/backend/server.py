"""
ScanSight FastAPI ML Backend
=============================
Wraps the LiverTumorSegmenter (UNet) model behind REST endpoints.

Endpoints:
  GET  /health       → Server status + model availability
  POST /api/predict   → Upload CT scan (PNG/ZIP), run segmentation, return results

Run with:
  cd app/backend
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import io
import uuid
import shutil
import zipfile
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import firebase_admin
from firebase_admin import credentials, firestore, storage as fb_storage, auth as fb_auth

# ---------------------------------------------------------------------------
# Path setup — so we can import src.models.unet and src.inference
# ---------------------------------------------------------------------------
# server.py lives at: <project>/app/backend/server.py
# We need to add <project> to sys.path to import from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # liver_ai_project/
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")

MODEL_PATH = os.getenv("MODEL_PATH", "../../models/best_model.pth")
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CREDENTIALS", "../../firebase-service-account.json")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
PORT = int(os.getenv("PORT", "8000"))

# Resolve relative paths from this file's directory
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_ABS_PATH = (BACKEND_DIR / MODEL_PATH).resolve()
FIREBASE_ABS_PATH = (BACKEND_DIR / FIREBASE_CRED_PATH).resolve()

# ---------------------------------------------------------------------------
# Firebase Admin SDK init
# ---------------------------------------------------------------------------
firebase_app = None
firestore_db = None
storage_bucket = None

def init_firebase():
    """Initialize Firebase Admin SDK (Firestore + Storage)."""
    global firebase_app, firestore_db, storage_bucket
    
    if firebase_admin._apps:
        # Already initialized
        firebase_app = firebase_admin.get_app()
        firestore_db = firestore.client()
        storage_bucket = fb_storage.bucket()
        return True
    
    if not FIREBASE_ABS_PATH.exists():
        print(f"[WARN] Firebase credentials not found at: {FIREBASE_ABS_PATH}")
        print("   Download from: Firebase Console -> Project Settings -> Service Accounts")
        print("   Server will run WITHOUT Firebase integration (results won't persist)")
        return False
    
    try:
        cred = credentials.Certificate(str(FIREBASE_ABS_PATH))
        # Read project ID from the service account file to get the storage bucket
        import json
        with open(FIREBASE_ABS_PATH) as f:
            sa_data = json.load(f)
            project_id = sa_data.get("project_id", "")
        
        firebase_app = firebase_admin.initialize_app(cred, {
            'storageBucket': f'{project_id}.firebasestorage.app'
        })
        firestore_db = firestore.client()
        storage_bucket = fb_storage.bucket()
        print(f"[OK] Firebase Admin SDK initialized (project: {project_id})")
        return True
    except Exception as e:
        print(f"[WARN] Firebase init failed: {e}")
        return False

# ---------------------------------------------------------------------------
# ML Model init
# ---------------------------------------------------------------------------
segmenter = None

def init_model():
    """Load the UNet model if the checkpoint exists."""
    global segmenter
    
    if not MODEL_ABS_PATH.exists():
        print(f"[WARN] Model not found at: {MODEL_ABS_PATH}")
        print("   Server will start in degraded mode (no predictions)")
        print("   Place your best_model.pth at: models/best_model.pth")
        return False
    
    try:
        from src.inference import LiverTumorSegmenter
        segmenter = LiverTumorSegmenter(model_path=str(MODEL_ABS_PATH))
        print(f"[OK] UNet model loaded from: {MODEL_ABS_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        traceback.print_exc()
        return False

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  ScanSight ML Backend - Starting Up")
    print("=" * 60)
    init_firebase()
    init_model()
    print("=" * 60)
    print(f"  Model loaded:   {'Yes' if segmenter else 'No'}")
    print(f"  Firebase ready:  {'Yes' if firestore_db else 'No'}")
    print(f"  Listening on:    http://0.0.0.0:{PORT}")
    print("=" * 60)
    yield
    print("Shutting down...")

from fastapi.staticfiles import StaticFiles
import shutil

app = FastAPI(
    title="ScanSight ML API",
    description="Backend for Liver Tumor Segmentation",
    version="1.0.0",
    lifespan=lifespan
)

# Create static directory if it doesn't exist
os.makedirs("static/scans", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL, 
        "http://localhost:5173", 
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth dependency (optional — validates Firebase ID token)
# ---------------------------------------------------------------------------
async def verify_token(authorization: str = None):
    """Verify Firebase ID token from Authorization header. Returns uid or None."""
    if not authorization or not firebase_app:
        return None  # Allow unauthenticated in dev
    
    try:
        token = authorization.replace("Bearer ", "")
        decoded = fb_auth.verify_id_token(token)
        return decoded.get("uid")
    except Exception:
        return None  # Don't block in dev; enforce in production

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def extract_images_from_upload(file: UploadFile, temp_dir: str) -> list[str]:
    """
    Extract image file paths from an uploaded file.
    Handles: single PNG/JPG, or ZIP containing multiple PNGs.
    Returns: sorted list of image file paths.
    """
    file_content = file.file.read()
    file_name = file.filename.lower()
    image_paths = []
    
    if file_name.endswith('.zip'):
        # Extract ZIP and find all image files inside
        zip_path = os.path.join(temp_dir, "upload.zip")
        with open(zip_path, 'wb') as f:
            f.write(file_content)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
        
        # Find all image files (PNG, JPG, JPEG)
        for root, dirs, files in os.walk(temp_dir):
            for fname in sorted(files):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and not fname.startswith('._'):
                    image_paths.append(os.path.join(root, fname))
    
    elif file_name.endswith(('.png', '.jpg', '.jpeg')):
        # Single image file
        save_path = os.path.join(temp_dir, file.filename)
        with open(save_path, 'wb') as f:
            f.write(file_content)
        image_paths.append(save_path)
    
    elif file_name.endswith(('.nii', '.nii.gz')):
        # NIfTI files — not handled yet, return error message
        raise HTTPException(
            status_code=400, 
            detail="NIfTI (.nii) format support coming soon. Please convert to PNG slices first."
        )
    
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {file.filename}. Accepted: PNG, JPG, ZIP"
        )
    
    if len(image_paths) == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid image files found in the upload."
        )
    
    return sorted(image_paths)


def generate_overlay(image: np.ndarray, pred_liver: np.ndarray, pred_tumor: np.ndarray) -> np.ndarray:
    """
    Generate a color overlay image.
    Liver regions = Blue tint, Tumor regions = Red highlight.
    Returns: RGB numpy array (uint8)
    """
    # Create 3-channel image from grayscale
    overlay = np.stack([image, image, image], axis=-1)  # [H, W, 3]
    overlay = (overlay * 255).astype(np.uint8)
    
    # Liver = Blue tint
    liver_mask = pred_liver > 0.5
    overlay[liver_mask, 0] = np.clip(overlay[liver_mask, 0].astype(int) - 20, 0, 255).astype(np.uint8)
    overlay[liver_mask, 1] = np.clip(overlay[liver_mask, 1].astype(int) - 10, 0, 255).astype(np.uint8)
    overlay[liver_mask, 2] = np.clip(overlay[liver_mask, 2].astype(int) + 100, 0, 255).astype(np.uint8)
    
    # Tumor = Red highlight
    tumor_mask = pred_tumor > 0.5
    overlay[tumor_mask, 0] = 255
    overlay[tumor_mask, 1] = np.clip(overlay[tumor_mask, 1].astype(int) - 80, 0, 255).astype(np.uint8)
    overlay[tumor_mask, 2] = np.clip(overlay[tumor_mask, 2].astype(int) - 80, 0, 255).astype(np.uint8)
    
    return overlay


def upload_to_firebase_storage(local_path: str, storage_path: str) -> str:
    """Upload a file to Firebase Storage and return the public URL."""
    if not storage_bucket:
        return ""  # Firebase not available
    
    try:
        blob = storage_bucket.blob(storage_path)
        blob.upload_from_filename(local_path, content_type="image/png")
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"[WARN] Storage upload failed for {storage_path}: {e}")
        return ""


def update_firestore_scan(scan_id: str, user_id: str, result_data: dict):
    """Update the scan document in Firestore with full results.
    
    Stores both the complete result object AND top-level summary fields
    so dashboards can query without digging into nested data.
    """
    if not firestore_db:
        return  # Firebase not available
    
    # If result_data is the API response wrapper, unwrap it
    actual_result = result_data.get('result', result_data) if 'success' in result_data else result_data
    
    try:
        # Find the scan document by scanId field
        scans_ref = firestore_db.collection('scans')
        query = scans_ref.where('scanId', '==', scan_id).where('userId', '==', user_id).limit(1)
        docs = list(query.stream())
        
        if docs:
            has_anomaly = actual_result.get('hasAnomaly', False)
            docs[0].reference.update({
                # --- Status ---
                'status': 'complete',
                'analysisState': 'anomaly' if has_anomaly else 'clear',
                
                # --- Timestamps ---
                'completedAt': firestore.SERVER_TIMESTAMP,
                
                # --- Full result object (all slice data, URLs, etc.) ---
                'result': actual_result,
                
                # --- Top-level summary fields (for fast dashboard queries) ---
                'confidence': actual_result.get('confidence', 0),
                'totalSlices': actual_result.get('totalSlices', 1),
                'hasAnomaly': has_anomaly,
                'anomalySlices': actual_result.get('anomalySlices', 0),
                'estimatedStage': actual_result.get('estimatedStage', None),
                'liverVolumePercent': actual_result.get('liverVolumePercent', 0),
                'tumorToLiverRatio': actual_result.get('tumorToLiverRatio', 0),
                'processingTimeMs': actual_result.get('processingTimeMs', 0),
                'modelVersion': actual_result.get('modelVersion', 'U-Net v1.0'),
            })
            print(f"  [OK] Firestore updated for scan: {scan_id}")
        else:
            print(f"  [WARN] Scan document not found in Firestore: {scan_id}")
    except Exception as e:
        print(f"  [WARN] Firestore update failed: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Server health check."""
    return {
        "status": "online",
        "model_loaded": segmenter is not None,
        "firebase_connected": firestore_db is not None,
        "model_path": str(MODEL_ABS_PATH),
        "timestamp": datetime.utcnow().isoformat(),
    }

# Global dictionary to track progress of ongoing scans
scan_progress = {}

@app.get("/api/progress/{scan_id}")
async def get_progress(scan_id: str):
    """Get the current progress and logs of an ongoing scan analysis."""
    if scan_id in scan_progress:
        return scan_progress[scan_id]
    return {"progress": 0, "logs": []}


@app.get("/api/scans/{user_id}")
async def get_user_scans(user_id: str):
    """Return all scans for a given user, sorted by uploadedAt desc."""
    if not firestore_db:
        return []
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter
        scans_ref = firestore_db.collection('scans')
        query = scans_ref.where(filter=FieldFilter('userId', '==', user_id))
        docs = list(query.stream())
        
        scans = []
        for doc in docs:
            data = doc.data()
            # Normalize timestamps to ISO strings
            uploaded_at = data.get('uploadedAt')
            if hasattr(uploaded_at, 'isoformat'):
                uploaded_at = uploaded_at.isoformat()
            completed_at = data.get('completedAt')
            if hasattr(completed_at, 'isoformat'):
                completed_at = completed_at.isoformat()
            
            scans.append({
                "id": doc.id,
                **data,
                "uploadedAt": uploaded_at,
                "completedAt": completed_at,
            })
        
        # Sort by uploadedAt desc (client-side)
        scans.sort(key=lambda s: s.get('uploadedAt') or '', reverse=True)
        return scans[:50]
    except Exception as e:
        print(f"[WARN] Failed to fetch scans for user {user_id}: {e}")
        return []


@app.get("/api/scan/{scan_id}")
async def get_scan_by_id(scan_id: str):
    """Return a single scan document by its scanId field."""
    if not firestore_db:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter
        scans_ref = firestore_db.collection('scans')
        query = scans_ref.where(filter=FieldFilter('scanId', '==', scan_id)).limit(1)
        docs = list(query.stream())
        if not docs:
            raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
        data = docs[0].to_dict()
        uploaded_at = data.get('uploadedAt')
        if hasattr(uploaded_at, 'isoformat'):
            uploaded_at = uploaded_at.isoformat()
        completed_at = data.get('completedAt')
        if hasattr(completed_at, 'isoformat'):
            completed_at = completed_at.isoformat()
        
        return {
            "id": docs[0].id,
            **data,
            "uploadedAt": uploaded_at,
            "completedAt": completed_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[WARN] Failed to fetch scan {scan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/report/{scan_id}")
async def generate_report(scan_id: str):
    """Generate a downloadable PDF clinical report for the given scan."""
    from fastapi.responses import StreamingResponse
    from pdf_report import generate_medical_report
    
    if not firestore_db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter
        scans_ref = firestore_db.collection('scans')
        query = scans_ref.where(filter=FieldFilter('scanId', '==', scan_id)).limit(1)
        docs = list(query.stream())
        
        if not docs:
            raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
        scan_data = docs[0].to_dict()
        
        # Unwrap nested result if present
        result = scan_data.get('result', {})
        if isinstance(result, dict) and 'result' in result and 'success' in result:
            result = result['result']
            scan_data['result'] = result
        
        pdf_bytes = generate_medical_report(scan_data)
        
        safe_ref = scan_data.get('scanRef', scan_id).replace('#', '').replace(' ', '_')
        filename = f"ScanSight_Report_{safe_ref}.pdf"
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    scanId: str = Form(default=""),
    userId: str = Form(default=""),
    authorization: str = Form(default=""),
):
    """
    Run liver/tumor segmentation on uploaded CT scan.
    
    Accepts:
      - Single PNG/JPG image
      - ZIP file containing multiple PNG slices
    
    Returns JSON with per-slice results and aggregate metrics.
    """
    
    # --- Validate model is loaded ---
    if segmenter is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Place best_model.pth in the models/ directory and restart the server."
        )
    
    # --- Verify auth token (optional in dev) ---
    uid = None
    if authorization:
        uid = await verify_token(authorization)
    uid = uid or userId  # Fallback to form field
    
    # --- Generate a scanId if not provided ---
    if not scanId:
        scanId = f"scan-{int(datetime.utcnow().timestamp() * 1000)}"
    
    # --- Create scan doc in Firestore at the START (so frontend can find it) ---
    # This creates a proper medical record for the patient's scan history
    if firestore_db and uid:
        try:
            # Determine scan type from file extension
            fname = file.filename.lower()
            if fname.endswith('.dcm'):
                scan_type = 'DICOM'
            elif fname.endswith(('.nii', '.nii.gz')):
                scan_type = 'NIfTI'
            elif fname.endswith('.zip'):
                scan_type = 'CT Archive (ZIP)'
            else:
                scan_type = 'Image Slices'
            
            # Generate a human-readable scan reference
            import hashlib
            scan_ref = f"#SCN-{hashlib.md5(scanId.encode()).hexdigest()[:6].upper()}"
            
            firestore_db.collection('scans').add({
                # --- Identity ---
                'scanId': scanId,
                'scanRef': scan_ref,
                'userId': uid,
                
                # --- File Metadata ---
                'fileName': file.filename,
                'fileSize': file.size if hasattr(file, 'size') else 0,
                'scanType': scan_type,
                'modality': 'CT',
                
                # --- Status ---
                'status': 'processing',
                'analysisState': 'uploading',
                
                # --- Timestamps ---
                'uploadedAt': firestore.SERVER_TIMESTAMP,
                'completedAt': None,
                
                # --- Results (populated after analysis) ---
                'result': None,
                'confidence': None,
                'totalSlices': 1,
                'hasAnomaly': False,
                'estimatedStage': None,
            })
            print(f"  [OK] Firestore scan doc created: {scanId} ({scan_ref})")
        except Exception as e:
            print(f"  [WARN] Could not create scan doc: {e}")
    
    # --- Process in temp directory ---
    temp_dir = tempfile.mkdtemp(prefix="scansight_")
    results_dir = os.path.join(temp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    import time as _time
    _start_time = _time.time()
    
    try:
        # 1. Extract images from upload
        print(f"\n[INFO] Processing upload: {file.filename}")
        image_paths = extract_images_from_upload(file, temp_dir)
        total_slices = len(image_paths)
        print(f"   Found {total_slices} slice(s)")
        
        # Initialize progress tracking
        scan_progress[scanId] = { "progress": 0, "logs": [], "totalSlices": total_slices }
        scan_progress[scanId]["logs"].append({
            "t": 5, 
            "msg": f"Found {total_slices} slices in {file.filename}. Starting segmentation...", 
            "type": "info"
        })
        
        # 2. Run inference on each slice
        slice_results = []
        total_liver_px = 0
        total_tumor_px = 0
        anomaly_slices = 0
        total_possible_px = 0  # Track total pixel budget for liver volume estimation
        
        for i, img_path in enumerate(image_paths):
            slice_name = Path(img_path).stem
            print(f"   [INFO] Processing slice {i+1}/{total_slices}: {slice_name}")
            
            try:
                # Run model prediction
                image, pred_liver, pred_tumor = segmenter.predict_image(
                    img_path, apply_clahe=True, threshold=0.5
                )
                
                # Compute metrics
                liver_area = int(np.sum(pred_liver > 0.5))
                tumor_area = int(np.sum(pred_tumor > 0.5))
                total_pixels = pred_liver.shape[0] * pred_liver.shape[1]  # 256*256
                has_tumor = tumor_area > 50  # Minimum threshold (noise filter)
                
                total_liver_px += liver_area
                total_tumor_px += tumor_area
                total_possible_px += total_pixels
                if has_tumor:
                    anomaly_slices += 1
                
                # Calculate confidence (mock logic based on threshold confidence from UNet output)
                slice_confidence = float(pred_liver.max() * 100) if np.sum(pred_liver) > 0 else 99.0
                if has_tumor:
                    slice_confidence = min(slice_confidence, float(pred_tumor.max() * 100))
                    
                print(f"   [RESULT] Liver: {(liver_area/total_pixels)*100:.1f}%, Tumor: {(tumor_area/total_pixels)*100:.1f}%, Conf: {slice_confidence:.1f}%")
                
                # Update progress
                if scanId in scan_progress:
                    scan_progress[scanId]["progress"] = int(((i + 1) / total_slices) * 95)  # Cap at 95% until final processing
                    
                    log_type = "alert" if has_tumor else "normal"
                    conf_str = f"Conf: {slice_confidence:.1f}%"
                    scan_progress[scanId]["logs"].append({
                        "t": scan_progress[scanId]["progress"],
                        "msg": f"Processed slice {i+1}/{total_slices} | Liver: {(liver_area/total_pixels)*100:.1f}% | Tumor: {(tumor_area/total_pixels)*100:.1f}% | {conf_str}",
                        "type": log_type
                    })
                
                # Generate overlay image
                overlay = generate_overlay(image, pred_liver, pred_tumor)
                overlay_path = os.path.join(results_dir, f"overlay_{slice_name}.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
                # Generate mask image (liver=blue channel, tumor=red channel)
                mask_img = np.zeros((256, 256, 3), dtype=np.uint8)
                mask_img[pred_liver > 0.5, 2] = 255   # Blue channel = liver
                mask_img[pred_tumor > 0.5, 0] = 255   # Red channel = tumor
                mask_path = os.path.join(results_dir, f"mask_{slice_name}.png")
                cv2.imwrite(mask_path, mask_img)
                
                # Also save original input image to static for viewer
                original_url = ""
                
                # Save to local static folder instead of Firebase Storage
                overlay_url = ""
                mask_url = ""
                
                if uid and scanId:
                    scan_static_dir = os.path.join("static", "scans", uid, scanId)
                    os.makedirs(scan_static_dir, exist_ok=True)
                    
                    final_overlay_path = os.path.join(scan_static_dir, f"overlay_{slice_name}.png")
                    final_mask_path = os.path.join(scan_static_dir, f"mask_{slice_name}.png")
                    
                    # Copy from temp to static
                    shutil.copy2(overlay_path, final_overlay_path)
                    shutil.copy2(mask_path, final_mask_path)
                    
                    # Also copy original input image
                    final_original_path = os.path.join(scan_static_dir, f"original_{slice_name}.png")
                    shutil.copy2(img_path, final_original_path)
                    
                    overlay_url = f"http://localhost:8000/static/scans/{uid}/{scanId}/overlay_{slice_name}.png"
                    mask_url = f"http://localhost:8000/static/scans/{uid}/{scanId}/mask_{slice_name}.png"
                    original_url = f"http://localhost:8000/static/scans/{uid}/{scanId}/original_{slice_name}.png"
                
                slice_results.append({
                    "sliceIndex": i,
                    "sliceName": slice_name,
                    "liverAreaPx": liver_area,
                    "tumorAreaPx": tumor_area,
                    "liverPercent": round((liver_area / total_pixels) * 100, 2),
                    "tumorPercent": round((tumor_area / total_pixels) * 100, 2),
                    "hasTumor": has_tumor,
                    "overlayUrl": overlay_url,
                    "maskUrl": mask_url,
                    "originalUrl": original_url,
                })
                
            except Exception as slice_err:
                import traceback
                traceback.print_exc()
                print(f"   [WARN] Slice {slice_name} failed: {slice_err}")
                slice_results.append({
                    "sliceIndex": i,
                    "sliceName": slice_name,
                    "error": str(slice_err),
                })
        
        # 3. Aggregate results — enriched clinical data
        processing_time_ms = int((_time.time() - _start_time) * 1000)
        
        has_anomaly = anomaly_slices > 0
        confidence = round(
            (1.0 - (anomaly_slices / max(total_slices, 1))) * 100 
            if not has_anomaly 
            else (anomaly_slices / max(total_slices, 1)) * 100, 
            1
        )
        
        # Volumetric analysis
        liver_volume_percent = round((total_liver_px / max(total_possible_px, 1)) * 100, 2)
        tumor_to_liver_ratio = round((total_tumor_px / max(total_liver_px, 1)) * 100, 2)
        
        # Staging heuristic
        estimated_stage = None
        if has_anomaly:
            if tumor_to_liver_ratio < 2.0 and anomaly_slices < 5:
                estimated_stage = "Stage I (Early)"
            elif tumor_to_liver_ratio < 10.0:
                estimated_stage = "Stage II (Localized)"
            else:
                estimated_stage = "Stage III (Advanced)"
        
        # Find affected slice range and max tumor slice
        tumor_slice_indices = [s["sliceIndex"] for s in slice_results if s.get("hasTumor")]
        affected_slice_range = None
        if tumor_slice_indices:
            affected_slice_range = {"start": min(tumor_slice_indices), "end": max(tumor_slice_indices)}
        
        # Max tumor slice (the slice with the highest tumor pixel count)
        valid_slices = [s for s in slice_results if s.get("tumorAreaPx", 0) > 0 and not s.get("error")]
        max_tumor_slice = None
        if valid_slices:
            max_ts = max(valid_slices, key=lambda s: s["tumorAreaPx"])
            max_tumor_slice = {
                "sliceIndex": max_ts["sliceIndex"],
                "sliceName": max_ts["sliceName"],
                "tumorPercent": max_ts["tumorPercent"],
                "overlayUrl": max_ts.get("overlayUrl", ""),
            }
        
        # Use the first overlay as the primary result image
        primary_overlay = next(
            (s["overlayUrl"] for s in slice_results if s.get("overlayUrl")), ""
        )
        primary_mask = next(
            (s["maskUrl"] for s in slice_results if s.get("maskUrl")), ""
        )
        
        result_data = {
            "totalSlices": total_slices,
            "anomalySlices": anomaly_slices,
            "hasAnomaly": has_anomaly,
            "confidence": confidence,
            "totalLiverPx": total_liver_px,
            "totalTumorPx": total_tumor_px,
            "liverVolumePercent": liver_volume_percent,
            "tumorToLiverRatio": tumor_to_liver_ratio,
            "estimatedStage": estimated_stage,
            "affectedSliceRange": affected_slice_range,
            "maxTumorSlice": max_tumor_slice,
            "processingTimeMs": processing_time_ms,
            "modelVersion": "U-Net (FasNet-aligned) v1.0",
            "overlayImage": primary_overlay,
            "maskImage": primary_mask,
            "slices": slice_results,
        }
        
        # 4. Update Firestore scan document
        response = {
            "success": True,
            "scanId": scanId,
            "result": result_data,
        }
        
        # 5. Return response
        print(f"   [OK] Analysis complete: {total_slices} slices, {anomaly_slices} anomalies, {processing_time_ms}ms")
        # Update Firestore
        if uid and scanId:
            update_firestore_scan(scanId, uid, response)
            
        if scanId in scan_progress:
            scan_progress[scanId]["progress"] = 100
            scan_progress[scanId]["logs"].append({
                "t": 100,
                "msg": f"Analysis complete. Overall confidence: {confidence:.1f}%. Saving results...",
                "type": "info"
            })
            
        return response
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        if scanId in scan_progress:
            scan_progress[scanId]["logs"].append({
                "t": scan_progress[scanId]["progress"],
                "msg": f"Error during processing: {str(e)}",
                "type": "alert"
            })
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        # We don't delete from scan_progress immediately so the frontend has time to fetch the final 100% state.
        # Instead, it will just live in memory (in a production environment, use a background task to clean it up or Redis expiration).


# Duplicate /api/progress route removed — already defined above


@app.get("/api/model-info")
async def model_info():
    """Return model architecture and configuration info."""
    if segmenter is None:
        return {"loaded": False, "message": "Model not loaded"}
    
    model = segmenter.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "loaded": True,
        "architecture": "U-Net (FasNet-aligned)",
        "inputShape": "[B, 1, 256, 256]",
        "outputShape": "[B, 2, 256, 256]",
        "channels": {"liver": 0, "tumor": 1},
        "totalParameters": total_params,
        "trainableParameters": trainable_params,
        "preprocessing": "CLAHE (clipLimit=2.0, tileGrid=8x8) → Resize 256x256 → Normalize 0-1",
        "device": str(segmenter.device),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True)
