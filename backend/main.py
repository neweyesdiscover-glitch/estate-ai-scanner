# backend/main.py
# ✅ Estate AI Scanner backend
# - Serves frontend at /scanner
# - POST /analyze-image: runs vision model -> returns structured JSON
# - ✅ NEW: Saves scans to SQLite + saves uploaded images to disk
# - ✅ NEW: Server-side history endpoints to support estate company workflows

import os
import json
import base64
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# OpenAI SDK (your project already uses this)
from openai import OpenAI

# -----------------------------
# App + Config
# -----------------------------
app = FastAPI()

# If you later host frontend separately, you can tighten this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    # Don’t crash at import-time; only error when endpoint is used.
    pass

client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR = Path(__file__).resolve().parent  # backend/
FRONTEND_INDEX = BASE_DIR.parent / "frontend" / "index.html"

# Storage paths (use Render Disk for true persistence)
DATA_DIR = BASE_DIR  # keep in backend/ so it’s easy
DB_PATH = DATA_DIR / "scans.db"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Database
# -----------------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_path TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        conn.commit()

init_db()

# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clean_json_text(s: str) -> str:
    """
    Removes ```json fences if the model accidentally includes them.
    """
    if not s:
        return ""
    s = s.strip()
    s = s.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    return s

def safe_parse_json(raw: str) -> Dict[str, Any]:
    cleaned = clean_json_text(raw)
    try:
        return json.loads(cleaned)
    except Exception:
        return {
            "error": "Model did not return valid JSON",
            "raw": raw,
        }

def save_image_bytes(contents: bytes, original_filename: str) -> Dict[str, str]:
    """
    Saves uploaded image to UPLOAD_DIR. Returns saved filename + full path.
    """
    ext = ""
    if "." in (original_filename or ""):
        ext = "." + original_filename.rsplit(".", 1)[-1].lower().strip()

    # Basic extension guard
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif"]:
        # Not fatal; just store as .jpg-like name
        ext = ".img"

    saved_name = f"{uuid4().hex}{ext}"
    path = UPLOAD_DIR / saved_name
    path.write_bytes(contents)
    return {"filename": saved_name, "path": str(path)}

def insert_scan(scan_id: str, created_at: str, filename: str, image_path: str, result: Dict[str, Any]) -> None:
    with db() as conn:
        conn.execute(
            """
            INSERT INTO scans (id, created_at, filename, image_path, result_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (scan_id, created_at, filename, image_path, json.dumps(result)),
        )
        conn.commit()

def row_to_scan_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "filename": row["filename"],
        "image_url": f"/history/{row['id']}/image",
        "result": json.loads(row["result_json"]),
    }

# -----------------------------
# Routes: Frontend serving
# -----------------------------
@app.get("/", response_class=JSONResponse)
def home():
    return {"message": "Estate AI Scanner is LIVE 🚀", "scanner": "/scanner"}

@app.get("/scanner", response_class=HTMLResponse)
def scanner():
    """
    Serves the frontend UI.
    """
    if not FRONTEND_INDEX.exists():
        return JSONResponse(
            {"error": "index.html not found in frontend folder. Expected frontend/index.html"},
            status_code=500,
        )
    return HTMLResponse(FRONTEND_INDEX.read_text(encoding="utf-8"))

# -----------------------------
# Routes: Analyze
# -----------------------------
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Accepts image upload, calls OpenAI vision model, returns structured pricing JSON.
    Also saves scan to server-side history (SQLite + image file).
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # Save image to disk for history
    saved = save_image_bytes(contents, file.filename or "upload")

    # Convert to base64 for model
    base64_image = base64.b64encode(contents).decode("utf-8")

    # Prompt: estate-company friendly, consistent keys for UI
    system_msg = (
        "You are an expert estate-sale item identifier and reseller pricing assistant. "
        "Return ONLY valid JSON with these keys:\n"
        "item_name, brand_or_origin, estimated_value_range, suggested_listing_price, "
        "condition_assumptions, keywords_for_listing, pricing_sources\n"
        "No markdown. No code fences. No extra commentary."
    )

    user_msg = (
        "Identify the item in the image and estimate a resale price range suitable for an estate sale. "
        "Be conservative and realistic. Return the JSON only."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_msg},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ],
        max_tokens=300,
    )

    raw = response.choices[0].message.content or ""
    result = safe_parse_json(raw)

    # If model failed, return error payload but still keep image saved (optional choice)
    # For now: only insert into DB when result is successful (no "error" key)
    scan_id = uuid4().hex
    created_at = utc_now_iso()

    if isinstance(result, dict) and not result.get("error"):
        insert_scan(
            scan_id=scan_id,
            created_at=created_at,
            filename=saved["filename"],
            image_path=saved["path"],
            result=result,
        )

        # Include scan metadata for future UI tie-in
        result["_scan_id"] = scan_id
        result["_created_at"] = created_at
        result["_image_url"] = f"/history/{scan_id}/image"

    return JSONResponse(result)

# -----------------------------
# Routes: Server-side history
# -----------------------------
@app.get("/history")
def get_history(limit: int = Query(50, ge=1, le=200)):
    """
    Returns latest scans from SQLite.
    """
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM scans ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    scans = [row_to_scan_dict(r) for r in rows]
    return {"count": len(scans), "items": scans}

@app.get("/history/{scan_id}")
def get_scan(scan_id: str):
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM scans WHERE id = ?",
            (scan_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")

    return row_to_scan_dict(row)

@app.get("/history/{scan_id}/image")
def get_scan_image(scan_id: str):
    with db() as conn:
        row = conn.execute(
            "SELECT image_path, filename FROM scans WHERE id = ?",
            (scan_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")

    image_path = row["image_path"]
    if not image_path or not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found on server")

    # Let the browser infer; optionally set media_type if you track it
    return FileResponse(image_path, filename=row["filename"])

@app.delete("/history/{scan_id}")
def delete_scan(scan_id: str):
    """
    Optional admin/cleanup endpoint.
    Deletes DB record + image file if present.
    """
    with db() as conn:
        row = conn.execute("SELECT image_path FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Scan not found")

        image_path = row["image_path"]
        conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        conn.commit()

    try:
        if image_path and Path(image_path).exists():
            Path(image_path).unlink()
    except Exception:
        pass

    return {"deleted": True, "id": scan_id}
