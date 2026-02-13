import base64
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# OpenAI (make sure requirements include openai>=1.x)
from openai import OpenAI

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "app.db"
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

MAX_PHOTOS_PER_SCAN = 4
MAX_ITEMS_MULTI = 8

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# If you host frontend on same origin, CORS isn't needed.
# Keeping permissive CORS for safety during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL,
              analysis_mode TEXT NOT NULL DEFAULT 'single',
              result_json TEXT NOT NULL,
              image_path TEXT
            )
            """
        )

        # Add missing columns safely (in case table existed before)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(scans)").fetchall()}

        if "analysis_mode" not in cols:
            conn.execute("ALTER TABLE scans ADD COLUMN analysis_mode TEXT NOT NULL DEFAULT 'single'")
        if "result_json" not in cols:
            conn.execute("ALTER TABLE scans ADD COLUMN result_json TEXT NOT NULL DEFAULT '{}'")  # unlikely
        if "image_path" not in cols:
            conn.execute("ALTER TABLE scans ADD COLUMN image_path TEXT")

        conn.commit()


ensure_schema()


def _safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        # Try to extract JSON object/array if model wrapped it in text
        start_obj = s.find("{")
        end_obj = s.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            try:
                return json.loads(s[start_obj : end_obj + 1])
            except Exception:
                pass

        start_arr = s.find("[")
        end_arr = s.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            try:
                return json.loads(s[start_arr : end_arr + 1])
            except Exception:
                pass

        raise


def _to_data_url(upload: UploadFile) -> str:
    content = upload.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    mime = upload.content_type or "image/jpeg"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _save_first_image(upload: UploadFile) -> str:
    # Save only the first image for history thumbnail/display
    upload.file.seek(0)
    content = upload.file.read()
    if not content:
        return ""
    ext = ".jpg"
    if upload.content_type and "png" in upload.content_type:
        ext = ".png"
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}{ext}"
    path = IMAGES_DIR / filename
    path.write_bytes(content)
    return str(path)


def _build_prompt(analysis_mode: str) -> str:
    if analysis_mode == "multi":
        return f"""
You are an expert estate-sale item identifier and pricing assistant.

Task:
- The user provided 1 main photo plus optional close-up/detail photos.
- Identify MULTIPLE distinct items from the main photo (up to {MAX_ITEMS_MULTI} items).
- Output a JSON object ONLY, matching this schema exactly:

{{
  "items": [
    {{
      "name": "string",
      "description": "string (short)",
      "estimated_price": {{"low": number, "high": number}}
    }}
  ],
  "bundle_price_range": {{"min": number, "max": number}},
  "advice": [
    "string tips for what to look for (marks, stamps, series/collections, etc.)",
    "string prompt questions user can answer YES/NO if relevant"
  ]
}}

Rules:
- If the image is unclear or items are missing, still return JSON, but include advice telling them to take a brighter/closer photo and add detail photos of markings.
- Prices should be realistic estate-sale quick-sale pricing.
- If you suspect a series/collection/group affects value, mention it in advice and what to look for (e.g., set completeness, edition/volume numbers, maker’s marks).
- Always return valid JSON only.
""".strip()

    # single
    return f"""
You are an expert estate-sale item identifier and pricing assistant.

Task:
- The user provided 1 main photo plus optional close-up/detail photos.
- Identify the PRIMARY item and price it.
- Output a JSON object ONLY, matching this schema exactly:

{{
  "items": [
    {{
      "name": "string",
      "type": "string (e.g., Book, Furniture, Jewelry, Electronics, etc.)",
      "description": "string (short)",
      "estimated_price": number
    }}
  ],
  "bundle_price_range": {{"min": number, "max": number}},
  "advice": [
    "string tips for what to look for (marks, stamps, series/collections, etc.)",
    "string prompt questions user can answer YES/NO if relevant"
  ]
}}

Rules:
- Even though it's single-mode, keep items array with exactly 1 item when possible.
- If unclear, return JSON with a best guess and advice asking for a closer/brighter photo and detail photos.
- If series/collection/group could affect price (e.g., collectible set, matched pair), mention it in advice.
- Always return valid JSON only.
""".strip()


def _call_openai(images_data_urls: List[str], analysis_mode: str) -> Dict[str, Any]:
    prompt = _build_prompt(analysis_mode)

    # Vision message: include prompt + images
    content = [{"type": "text", "text": prompt}]
    for url in images_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
    )

    text = resp.choices[0].message.content or ""
    try:
        data = _safe_json_loads(text)
    except Exception:
        # If model fails JSON, return a safe error JSON envelope
        data = {
            "items": [],
            "bundle_price_range": {"min": 0, "max": 0},
            "advice": [
                "Results could not be parsed. Please take a brighter/closer photo and try again.",
                "Add close-ups of any stamps, markings, signatures, labels, or serial numbers."
            ],
            "_raw": text
        }

    # Normalize
    if not isinstance(data, dict):
        data = {"items": [], "bundle_price_range": {"min": 0, "max": 0}, "advice": [], "_raw": text}

    if "items" not in data or not isinstance(data["items"], list):
        data["items"] = []
    if "bundle_price_range" not in data or not isinstance(data["bundle_price_range"], dict):
        data["bundle_price_range"] = {"min": 0, "max": 0}
    if "advice" not in data or not isinstance(data["advice"], list):
        data["advice"] = []

    # Enforce multi max items
    if analysis_mode == "multi" and len(data["items"]) > MAX_ITEMS_MULTI:
        data["items"] = data["items"][:MAX_ITEMS_MULTI]
        data["advice"].insert(0, f"Note: Limited results to the first {MAX_ITEMS_MULTI} items.")

    return data


@app.post("/analyze-image")
async def analyze_image(
    # NEW style: images[] (main + optional detail)
    images: Optional[List[UploadFile]] = File(default=None),
    # OLD style compatibility:
    file: Optional[UploadFile] = File(default=None),
    detail_files: Optional[List[UploadFile]] = File(default=None),
    analysis_mode: str = Form(default="single"),
):
    analysis_mode = (analysis_mode or "single").strip().lower()
    if analysis_mode not in ("single", "multi"):
        analysis_mode = "single"

    # Collect uploads in the right order: main first, then details
    uploads: List[UploadFile] = []

    if images and len(images) > 0:
        uploads.extend(images)
    else:
        if file is not None:
            uploads.append(file)
        if detail_files:
            uploads.extend(detail_files)

    if not uploads:
        raise HTTPException(status_code=400, detail="No image uploaded.")

    # Enforce max photos (1 main + up to 3 details)
    uploads = uploads[:MAX_PHOTOS_PER_SCAN]

    # Convert to data URLs (for vision)
    data_urls: List[str] = []
    for u in uploads:
        # ensure pointer at start
        u.file.seek(0)
        data_urls.append(_to_data_url(u))
        u.file.seek(0)

    # Save first image (main) for history
    uploads[0].file.seek(0)
    saved_path = _save_first_image(uploads[0])
    uploads[0].file.seek(0)

    # Call AI
    result_obj = _call_openai(data_urls, analysis_mode)

    # Wrap in envelope your frontend expects
    envelope = {
        "scan_id": None,
        "created_at": utc_now_iso(),
        "analysis_mode": analysis_mode,
        "result": result_obj
    }

    # Save to DB
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO scans (created_at, analysis_mode, result_json, image_path) VALUES (?, ?, ?, ?)",
            (envelope["created_at"], analysis_mode, json.dumps(result_obj), saved_path),
        )
        scan_id = cur.lastrowid
        conn.commit()

    envelope["scan_id"] = scan_id
    return JSONResponse(envelope)


@app.get("/history")
def history(limit: int = 25):
    limit = max(1, min(int(limit), 100))
    with db() as conn:
        rows = conn.execute(
            "SELECT id, created_at, analysis_mode, result_json, image_path FROM scans ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    items = []
    for r in rows:
        result_obj = {}
        try:
            result_obj = json.loads(r["result_json"] or "{}")
        except Exception:
            result_obj = {}

        items.append(
            {
                "scan_id": r["id"],
                "created_at": r["created_at"],
                "analysis_mode": r["analysis_mode"],
                "image_url": f"/history/{r['id']}/image" if r["image_path"] else "",
                "result": result_obj,
            }
        )

    return {"items": items}


@app.get("/history/{scan_id}/image")
def history_image(scan_id: int):
    with db() as conn:
        row = conn.execute(
            "SELECT image_path FROM scans WHERE id = ?",
            (scan_id,),
        ).fetchone()

    if not row or not row["image_path"]:
        raise HTTPException(status_code=404, detail="Image not found.")

    path = Path(row["image_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image missing on server.")

    return FileResponse(path)
