# backend/main.py
# ✅ Estate AI Scanner backend (Company Login + Team History)
# - /scanner serves the UI
# - Company signup/login with session cookie
# - /analyze-image saves scans per company
# - /history returns ONLY that company’s scans
# - Persistent storage uses Render Disk if mounted at /var/data

import os
import json
import base64
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from openai import OpenAI

# -----------------------------
# App + Middleware
# -----------------------------
app = FastAPI()

# IMPORTANT: for cookies/sessions, allow_credentials must be True.
# Since your frontend is served from the same domain (/scanner), this is safe.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for early stage; later restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY:
    # You MUST set this on Render for login sessions to work reliably.
    # We'll only error when login endpoints are used.
    pass

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR = Path(__file__).resolve().parent  # backend/
FRONTEND_INDEX = BASE_DIR.parent / "frontend" / "index.html"

# -----------------------------
# Persistent Storage (Render Disk)
# -----------------------------
PERSISTENT_DIR = Path("/var/data")
DATA_DIR = PERSISTENT_DIR if PERSISTENT_DIR.exists() else BASE_DIR

DB_PATH = DATA_DIR / "scans.db"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Database helpers
# -----------------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                passcode TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id TEXT PRIMARY KEY,
                company_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_path TEXT NOT NULL,
                result_json TEXT NOT NULL,
                FOREIGN KEY(company_id) REFERENCES companies(id)
            )
        """)
        conn.commit()

init_db()

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clean_json_text(s: str) -> str:
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
        return {"error": "Model did not return valid JSON", "raw": raw}

def save_image_bytes(contents: bytes, original_filename: str) -> Dict[str, str]:
    ext = ""
    if "." in (original_filename or ""):
        ext = "." + original_filename.rsplit(".", 1)[-1].lower().strip()

    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif"]:
        ext = ".img"

    saved_name = f"{uuid4().hex}{ext}"
    path = UPLOAD_DIR / saved_name
    path.write_bytes(contents)
    return {"filename": saved_name, "path": str(path)}

def require_company_id(request) -> str:
    cid = request.session.get("company_id")
    if not cid:
        raise HTTPException(status_code=401, detail="Not logged in")
    return cid

def company_exists(company_id: str) -> bool:
    with db() as conn:
        row = conn.execute("SELECT id FROM companies WHERE id = ?", (company_id,)).fetchone()
        return row is not None

def row_to_scan_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "filename": row["filename"],
        "image_url": f"/history/{row['id']}/image",
        "result": json.loads(row["result_json"]),
    }

# -----------------------------
# Routes: Frontend
# -----------------------------
@app.get("/", response_class=JSONResponse)
def home():
    return {"message": "Estate AI Scanner is LIVE 🚀", "scanner": "/scanner"}

@app.get("/scanner", response_class=HTMLResponse)
def scanner():
    if not FRONTEND_INDEX.exists():
        return JSONResponse(
            {"error": "index.html not found. Expected frontend/index.html"},
            status_code=500,
        )
    return HTMLResponse(FRONTEND_INDEX.read_text(encoding="utf-8"))

# -----------------------------
# Routes: Auth (Company Login)
# -----------------------------
@app.get("/me")
def me(request):
    cid = request.session.get("company_id")
    if not cid:
        return {"logged_in": False}
    with db() as conn:
        row = conn.execute("SELECT id, name, created_at FROM companies WHERE id = ?", (cid,)).fetchone()
    if not row:
        request.session.clear()
        return {"logged_in": False}
    return {"logged_in": True, "company": {"id": row["id"], "name": row["name"], "created_at": row["created_at"]}}

@app.post("/signup-company")
async def signup_company(request):
    """
    Create a company account.
    Body JSON: { "name": "...", "passcode": "..." }
    """
    if not SECRET_KEY:
        raise HTTPException(status_code=500, detail="SECRET_KEY not set on server")

    data = await request.json()
    name = (data.get("name") or "").strip()
    passcode = (data.get("passcode") or "").strip()

    if len(name) < 2 or len(passcode) < 4:
        raise HTTPException(status_code=400, detail="Name must be 2+ chars and passcode 4+ chars")

    company_id = uuid4().hex
    created_at = utc_now_iso()

    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO companies (id, name, passcode, created_at) VALUES (?, ?, ?, ?)",
                (company_id, name, passcode, created_at),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Company name already exists")

    # Auto-login after signup
    request.session["company_id"] = company_id

    return {"ok": True, "company": {"id": company_id, "name": name, "created_at": created_at}}

@app.post("/login")
async def login(request):
    """
    Log in to a company.
    Body JSON: { "name": "...", "passcode": "..." }
    """
    if not SECRET_KEY:
        raise HTTPException(status_code=500, detail="SECRET_KEY not set on server")

    data = await request.json()
    name = (data.get("name") or "").strip()
    passcode = (data.get("passcode") or "").strip()

    with db() as conn:
        row = conn.execute(
            "SELECT id, name, passcode, created_at FROM companies WHERE name = ?",
            (name,),
        ).fetchone()

    if not row or row["passcode"] != passcode:
        raise HTTPException(status_code=401, detail="Invalid company name or passcode")

    request.session["company_id"] = row["id"]
    return {"ok": True, "company": {"id": row["id"], "name": row["name"], "created_at": row["created_at"]}}

@app.post("/logout")
def logout(request):
    request.session.clear()
    return {"ok": True}

# -----------------------------
# Routes: Analyze (company-scoped)
# -----------------------------
@app.post("/analyze-image")
async def analyze_image(request, file: UploadFile = File(...)):
    """
    Requires login.
    Saves scan per company (image+result).
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")

    company_id = require_company_id(request)
    if not company_exists(company_id):
        request.session.clear()
        raise HTTPException(status_code=401, detail="Not logged in")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    saved = save_image_bytes(contents, file.filename or "upload")
    base64_image = base64.b64encode(contents).decode("utf-8")

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

    # Only persist successful scans
    if isinstance(result, dict) and not result.get("error"):
        scan_id = uuid4().hex
        created_at = utc_now_iso()

        with db() as conn:
            conn.execute(
                """
                INSERT INTO scans (id, company_id, created_at, filename, image_path, result_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (scan_id, company_id, created_at, saved["filename"], saved["path"], json.dumps(result)),
            )
            conn.commit()

        # Include scan metadata for UI
        result["_scan_id"] = scan_id
        result["_created_at"] = created_at
        result["_image_url"] = f"/history/{scan_id}/image"

    return JSONResponse(result)

# -----------------------------
# Routes: History (company-scoped)
# -----------------------------
@app.get("/history")
def get_history(request, limit: int = Query(50, ge=1, le=200)):
    """
    Requires login.
    Returns this company's scan history only.
    """
    company_id = require_company_id(request)

    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM scans WHERE company_id = ? ORDER BY created_at DESC LIMIT ?",
            (company_id, limit),
        ).fetchall()

    items = [row_to_scan_dict(r) for r in rows]
    return {"count": len(items), "items": items}

@app.get("/history/{scan_id}")
def get_scan(request, scan_id: str):
    company_id = require_company_id(request)

    with db() as conn:
        row = conn.execute(
            "SELECT * FROM scans WHERE id = ? AND company_id = ?",
            (scan_id, company_id),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")

    return row_to_scan_dict(row)

@app.get("/history/{scan_id}/image")
def get_scan_image(request, scan_id: str):
    company_id = require_company_id(request)

    with db() as conn:
        row = conn.execute(
            "SELECT image_path, filename FROM scans WHERE id = ? AND company_id = ?",
            (scan_id, company_id),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")

    image_path = row["image_path"]
    if not image_path or not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found on server")

    return FileResponse(image_path, filename=row["filename"])
