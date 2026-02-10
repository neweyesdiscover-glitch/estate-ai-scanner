import base64
import datetime as dt
import json
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_302_FOUND, HTTP_401_UNAUTHORIZED, HTTP_404_NOT_FOUND

from openai import OpenAI

# -----------------------------
# Configuration
# -----------------------------
APP_NAME = "Estate AI Scanner"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "")

# v1 company login (single company credentials via env vars)
COMPANY_CODE = os.getenv("COMPANY_CODE", "demo")
COMPANY_PASSWORD = os.getenv("COMPANY_PASSWORD", "demo")

# Render Disk mount (persistent). Fallback locally for dev.
DATA_DIR = Path("/var/data") if Path("/var/data").exists() else Path("./local_data")
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "estate_scanner.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI client (the SDK reads OPENAI_API_KEY automatically, but we also validate)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title=APP_NAME)

if not SECRET_KEY:
    # You WANT this set in Render. This makes it obvious instead of silently insecure.
    print("WARNING: SECRET_KEY is not set. Sessions will be insecure / unstable.")

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY or "dev-insecure-secret",
    same_site="lax",
    https_only=False,  # Render terminates TLS upstream; cookies still work with lax. Set True if you enforce HTTPS-only.
)

# -----------------------------
# Helpers
# -----------------------------
def is_logged_in(request: Request) -> bool:
    return bool(request.session.get("company_code"))

def require_login_redirect(request: Request) -> Optional[RedirectResponse]:
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)
    return None

def require_login_json(request: Request) -> Optional[JSONResponse]:
    if not is_logged_in(request):
        return JSONResponse({"error": "Not authenticated"}, status_code=HTTP_401_UNAUTHORIZED)
    return None

def get_company_code(request: Request) -> str:
    return request.session.get("company_code", "")

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Better concurrency behavior for SQLite
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def db_init() -> None:
    conn = db_connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_code TEXT NOT NULL,
                created_at TEXT NOT NULL,
                image_path TEXT NOT NULL,
                result_json TEXT NOT NULL,
                result_text TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_company_time ON scans(company_code, created_at DESC)")
        conn.commit()
    finally:
        conn.close()

db_init()

def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse JSON even if the model wraps it in code fences or adds text.
    """
    if not text:
        return None

    # Remove code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()

    # Direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except Exception:
        pass

    # Try extracting first JSON object block
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None

def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# -----------------------------
# Routes: basic navigation
# -----------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/scanner", status_code=HTTP_302_FOUND)

# -----------------------------
# Login UI + handlers
# -----------------------------
@app.get("/login", include_in_schema=False)
def login_page(request: Request):
    # Simple HTML form. Keeps schema mismatch problems away.
    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <title>{APP_NAME} — Login</title>
        <style>
          body {{ font-family: Arial, sans-serif; padding: 24px; max-width: 520px; margin: 0 auto; }}
          .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; }}
          label {{ display: block; margin: 12px 0 6px; font-weight: 600; }}
          input {{ width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px; }}
          button {{ margin-top: 14px; padding: 10px 14px; border: 0; border-radius: 10px; cursor: pointer; }}
          .err {{ color: #b00020; margin-top: 10px; }}
          .sub {{ color: #555; margin-top: 6px; }}
        </style>
      </head>
      <body>
        <h1>{APP_NAME}</h1>
        <div class="sub">Company login</div>
        <div class="card">
          <form method="post" action="/login">
            <label>Company Code</label>
            <input name="company_code" autocomplete="username" required />

            <label>Password</label>
            <input name="company_password" type="password" autocomplete="current-password" required />

            <button type="submit">Sign in</button>
          </form>
          {"<div class='err'>Invalid company code or password.</div>" if request.query_params.get("error") == "1" else ""}
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)

@app.post("/login", include_in_schema=False)
def login_submit(
    request: Request,
    company_code: str = Form(...),
    company_password: str = Form(...),
):
    if company_code.strip() == COMPANY_CODE and company_password == COMPANY_PASSWORD:
        request.session["company_code"] = company_code.strip()
        return RedirectResponse(url="/scanner", status_code=HTTP_302_FOUND)
    return RedirectResponse(url="/login?error=1", status_code=HTTP_302_FOUND)

@app.get("/logout", include_in_schema=False)
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=HTTP_302_FOUND)

# -----------------------------
# Scanner page (serves frontend/index.html)
# -----------------------------
@app.get("/scanner", include_in_schema=False)
def scanner(request: Request):
    redirect = require_login_redirect(request)
    if redirect:
        return redirect

    # Serve the static frontend file
    index_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<h2>frontend/index.html not found</h2><p>Check repo structure.</p>",
            status_code=500,
        )

    return HTMLResponse(index_path.read_text(encoding="utf-8"))

# -----------------------------
# Analyze Image (protected) — returns 401 JSON if not logged in
# -----------------------------
@app.post("/analyze-image")
async def analyze_image(request: Request, image: UploadFile = File(...)):
    not_auth = require_login_json(request)
    if not_auth:
        return not_auth

    if client is None:
        return JSONResponse({"error": "OPENAI_API_KEY not configured"}, status_code=500)

    # Read image bytes
    image_bytes = await image.read()
    if not image_bytes:
        return JSONResponse({"error": "Empty image upload"}, status_code=400)

    # Persist image to disk
    ext = (Path(image.filename).suffix or ".jpg").lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        # still store it as .jpg-ish if unknown; but keep it simple
        ext = ".jpg"

    file_id = uuid.uuid4().hex
    saved_path = UPLOAD_DIR / f"{file_id}{ext}"
    saved_path.write_bytes(image_bytes)

    # Prepare base64 for OpenAI
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else ("image/png" if ext == ".png" else "image/webp")

    system_prompt = (
        "You are an expert estate-sale pricing assistant. "
        "Identify the item and produce a concise, customer-friendly estimate. "
        "Return ONLY valid JSON (no markdown, no extra text) with these keys:\n"
        "- title: short item name\n"
        "- category: broad category\n"
        "- condition_notes: brief notes\n"
        "- price_range_low: number (USD)\n"
        "- price_range_high: number (USD)\n"
        "- suggested_list_price: number (USD)\n"
        "- keywords: array of strings (10-20)\n"
        "- rationale: 2-4 bullet-like sentences explaining pricing factors\n"
        "- confidence: number 0-1\n"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this photo for estate sale pricing."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                },
            ],
            temperature=0.2,
        )
        result_text = resp.choices[0].message.content or ""
    except Exception as e:
        # Keep customer-friendly error
        return JSONResponse(
            {"error": "AI request failed. Please retry.", "detail": str(e)},
            status_code=502,
        )

    result_json = safe_json_parse(result_text) or {"raw": result_text}

    # Save scan record
    company_code = get_company_code(request)
    created_at = now_iso()

    conn = db_connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO scans (company_code, created_at, image_path, result_json, result_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                company_code,
                created_at,
                str(saved_path),
                json.dumps(result_json, ensure_ascii=False),
                result_text,
            ),
        )
        conn.commit()
        scan_id = cur.lastrowid
    finally:
        conn.close()

    return {
        "scan_id": scan_id,
        "created_at": created_at,
        "company_code": company_code,
        "result": result_json,
    }

# -----------------------------
# Team History (protected)
# -----------------------------
@app.get("/history")
def history(request: Request, limit: int = 50):
    not_auth = require_login_json(request)
    if not_auth:
        return not_auth

    limit = max(1, min(limit, 200))
    company_code = get_company_code(request)

    conn = db_connect()
    try:
        rows = conn.execute(
            """
            SELECT id, company_code, created_at, image_path, result_json
            FROM scans
            WHERE company_code = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (company_code, limit),
        ).fetchall()
    finally:
        conn.close()

    items = []
    for r in rows:
        parsed = None
        try:
            parsed = json.loads(r["result_json"])
        except Exception:
            parsed = {"raw": r["result_json"]}

        title = parsed.get("title") if isinstance(parsed, dict) else None
        items.append(
            {
                "scan_id": r["id"],
                "created_at": r["created_at"],
                "title": title or "Scan",
                "image_url": f"/history/{r['id']}/image",
                "result": parsed,
            }
        )

    return {"items": items}

@app.get("/history/{scan_id}/image", include_in_schema=False)
def history_image(request: Request, scan_id: int):
    not_auth = require_login_json(request)
    if not_auth:
        return not_auth

    company_code = get_company_code(request)

    conn = db_connect()
    try:
        row = conn.execute(
            """
            SELECT image_path
            FROM scans
            WHERE id = ? AND company_code = ?
            """,
            (scan_id, company_code),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return JSONResponse({"error": "Not found"}, status_code=HTTP_404_NOT_FOUND)

    image_path = Path(row["image_path"])
    if not image_path.exists():
        return JSONResponse({"error": "Image missing"}, status_code=HTTP_404_NOT_FOUND)

    return FileResponse(str(image_path))
