"""
backend/main.py
✅ Estate AI Scanner — Render-ready FastAPI backend
Features:
- Serves frontend at /scanner
- POST /analyze-image (protected by company login)
- Company auth:
    - POST /company/create
    - POST /login
    - POST /logout
    - GET  /me
- Shared team history (per company) stored in SQLite:
    - GET  /team/history
Notes:
- Uses cookie-based sessions (SessionMiddleware) -> requires 'itsdangerous'
- Uses OPENAI_API_KEY from env vars
"""

import os
import json
import base64
import sqlite3
import hashlib
from datetime import datetime
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from pydantic import BaseModel

# If you are using OpenAI SDK:
# pip install openai
from openai import OpenAI


# =========================
# App + Session
# =========================

app = FastAPI()

# IMPORTANT: set this in Render env vars as SESSION_SECRET for production
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-session-secret-change-me")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax", https_only=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY".lower()) or os.getenv("OPENAI_API_KEY".upper()) or os.getenv("OPENAI_API_KEY")
# You showed your Render variable is OPENAI_API_KEY — this reads it.
if not OPENAI_API_KEY:
    # Don't crash Render; just raise when calling /analyze-image
    pass

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
FRONTEND_PATH = os.path.join(PROJECT_ROOT, "frontend", "index.html")
DB_PATH = os.path.join(PROJECT_ROOT, "backend", "scans.db")


# =========================
# DB helpers
# =========================

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def db_init() -> None:
    with db_connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            passcode_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            image_data_url TEXT,
            result_json TEXT NOT NULL
        )
        """)
        conn.commit()

db_init()


# =========================
# Auth + hashing
# =========================

def _hash_passcode(company: str, passcode: str) -> str:
    """
    Simple salted hash. Good enough for MVP.
    If you want stronger later: use passlib/bcrypt.
    """
    salt = os.getenv("PASSCODE_SALT", SESSION_SECRET)
    raw = f"{salt}|{company.lower().strip()}|{passcode}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def get_company_from_session(request: Request) -> Optional[str]:
    return request.session.get("company")

def require_company(request: Request) -> str:
    company = get_company_from_session(request)
    if not company:
        raise HTTPException(status_code=401, detail="Login required")
    return company


# =========================
# Models
# =========================

class CreateCompanyRequest(BaseModel):
    company: str
    passcode: str

class LoginRequest(BaseModel):
    company: str
    passcode: str

class AnalyzeSavePayload(BaseModel):
    # optional client-provided thumbnail/data-url for team history
    image_data_url: Optional[str] = None


# =========================
# Frontend serve
# =========================

@app.get("/", response_class=JSONResponse)
def root():
    return {"message": "Estate AI Scanner API is LIVE 🚀", "scanner": "/scanner"}

@app.get("/scanner", response_class=HTMLResponse)
def scanner():
    if not os.path.exists(FRONTEND_PATH):
        return HTMLResponse(
            content=json.dumps({"error": f"index.html not found in frontend folder. Expected {FRONTEND_PATH}"}),
            status_code=500
        )
    with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# =========================
# Company endpoints
# =========================

@app.get("/me")
def me(request: Request):
    """
    Frontend uses this to know if a company session exists.
    """
    company = get_company_from_session(request)
    return {"logged_in": bool(company), "company": company}

@app.post("/company/create")
def create_company(payload: CreateCompanyRequest):
    company = payload.company.strip().lower()
    passcode = payload.passcode.strip()

    if not company or not passcode:
        raise HTTPException(status_code=400, detail="Company and passcode are required")

    pass_hash = _hash_passcode(company, passcode)
    now = datetime.utcnow().isoformat()

    try:
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO companies (name, passcode_hash, created_at) VALUES (?, ?, ?)",
                (company, pass_hash, now)
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Company already exists")

    return {"success": True, "company": company}

@app.post("/login")
def login(request: Request, payload: LoginRequest):
    """
    ✅ Fixes your 422 by matching payload fields: {company, passcode}
    """
    company = payload.company.strip().lower()
    passcode = payload.passcode.strip()

    if not company or not passcode:
        raise HTTPException(status_code=400, detail="Company and passcode are required")

    with db_connect() as conn:
        row = conn.execute(
            "SELECT passcode_hash FROM companies WHERE name = ?",
            (company,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid login")

    expected = row["passcode_hash"]
    actual = _hash_passcode(company, passcode)
    if actual != expected:
        raise HTTPException(status_code=401, detail="Invalid login")

    request.session["company"] = company
    return {"success": True, "company": company}

@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"success": True}


# =========================
# Team history (shared)
# =========================

@app.get("/team/history")
def team_history(request: Request, limit: int = 25):
    company = require_company(request)
    limit = max(1, min(limit, 100))

    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT created_at, image_data_url, result_json
            FROM scans
            WHERE company_name = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (company, limit)
        ).fetchall()

    items = []
    for r in rows:
        try:
            result = json.loads(r["result_json"])
        except Exception:
            result = {"error": "Bad JSON in DB"}
        items.append({
            "created_at": r["created_at"],
            "image_data_url": r["image_data_url"],
            "result": result
        })

    return {"company": company, "items": items}


# =========================
# AI analyze helpers
# =========================

def _clean_json_text(s: str) -> str:
    """Remove ```json fences and trim."""
    if not s:
        return ""
    s = s.strip()
    s = s.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    return s

def _build_prompt() -> str:
    return (
        "You are an expert at identifying items from photos and estimating resale value.\n"
        "Return ONLY valid JSON (no markdown).\n\n"
        "JSON keys:\n"
        "- item_name (string)\n"
        "- brand_or_origin (string)\n"
        "- estimated_value_range (string, e.g. \"$20 - $50\")\n"
        "- suggested_listing_price (string, e.g. \"$35\")\n"
        "- condition_assumptions (string)\n"
        "- keywords_for_listing (string or array)\n"
        "- pricing_sources (string or array)\n\n"
        "Be concise and practical for resale/estate sale use.\n"
    )

def _save_scan(company: str, image_data_url: Optional[str], result_obj: Dict[str, Any]) -> None:
    now = datetime.utcnow().isoformat()
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO scans (company_name, created_at, image_data_url, result_json) VALUES (?, ?, ?, ?)",
            (company, now, image_data_url, json.dumps(result_obj))
        )
        conn.commit()


# =========================
# Analyze endpoint (protected)
# =========================

@app.post("/analyze-image")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    company = require_company(request)

    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set on server")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    base64_image = base64.b64encode(contents).decode("utf-8")

    # NOTE: this uses a vision-capable model. If you change model names, keep it consistent.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _build_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify this item and estimate resale value."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            },
        ],
        max_tokens=500,
    )

    raw = response.choices[0].message.content or ""
    cleaned = _clean_json_text(raw)

    try:
        data = json.loads(cleaned)
    except Exception:
        data = {"error": "Model did not return valid JSON", "raw": raw}

    # Optional: if frontend sends a thumbnail/data-url, it can be saved later.
    # We'll save scan without thumbnail by default; frontend can re-save with thumb via /team/save-thumb
    _save_scan(company=company, image_data_url=None, result_obj=data)

    return JSONResponse(content=data)
