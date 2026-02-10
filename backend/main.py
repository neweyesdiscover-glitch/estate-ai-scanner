import os
import io
import json
import base64
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional, Any, Dict

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from starlette.middleware.sessions import SessionMiddleware

from openai import OpenAI


# =========================
# Config
# =========================
APP_NAME = "Estate AI Scanner"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SECRET_KEY = os.getenv("SECRET_KEY", "")
COMPANY_CODE = os.getenv("COMPANY_CODE", "demo")
COMPANY_PASSWORD = os.getenv("COMPANY_PASSWORD", "demo")

MAX_MULTI_ITEMS = int(os.getenv("MAX_MULTI_ITEMS", "8"))  # brief note: max items at once


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_data_root() -> str:
    # Render disk mount (persistent)
    if os.path.isdir("/var/data"):
        return "/var/data"
    # Local fallback
    return os.path.join(os.path.dirname(__file__), "data")


DATA_ROOT = get_data_root()
UPLOAD_DIR = os.path.join(DATA_ROOT, "uploads")
DB_PATH = os.path.join(DATA_ROOT, "app.db")
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================
# DB helpers
# =========================
def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_code TEXT NOT NULL,
            created_at TEXT NOT NULL,
            image_path TEXT NOT NULL,
            result_json TEXT NOT NULL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_scans_company_created ON scans(company_code, created_at);")
    conn.commit()
    conn.close()


def extract_title_for_history(result_obj: dict) -> str:
    # Single schema: {title, ...}
    # Multi schema: {mode:"multi", bundle:{title,...}, items:[...]}
    try:
        if isinstance(result_obj, dict) and result_obj.get("mode") == "multi":
            bundle = result_obj.get("bundle") or {}
            title = bundle.get("title") or "Multi-item lot"
            return str(title)
        title = result_obj.get("title") or "Scan"
        return str(title)
    except Exception:
        return "Scan"


# =========================
# Auth helpers
# =========================
def is_logged_in(request: Request) -> bool:
    return bool(request.session.get("company_code"))


def require_login_redirect(request: Request) -> str:
    if not is_logged_in(request):
        raise HTTPException(status_code=307, detail="Redirect", headers={"Location": "/login"})
    return request.session.get("company_code")


def require_login_json(request: Request) -> str:
    if not is_logged_in(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.session.get("company_code")


# =========================
# App
# =========================
app = FastAPI(title=APP_NAME)

# Session cookies
if not SECRET_KEY:
    # You REALLY want this set on Render.
    SECRET_KEY = "dev-only-secret-change-me"
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site="lax", https_only=False)


@app.on_event("startup")
def _startup():
    db_init()


# =========================
# Routes: Login + Scanner
# =========================
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/scanner", status_code=302)


@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
def login_page():
    # Simple form login page
    return HTMLResponse(
        f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{APP_NAME} — Login</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; max-width: 520px; margin: 0 auto; }}
    .card {{ border:1px solid #ddd; border-radius: 12px; padding: 16px; }}
    input {{ width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; margin: 6px 0 12px; }}
    button {{ width: 100%; padding: 10px; border-radius: 10px; border: 1px solid #2b7cff; background: #2b7cff; color: #fff; font-weight: 800; cursor: pointer; }}
    .small {{ color: #666; font-size: 12px; margin-top: 10px; }}
  </style>
</head>
<body>
  <h1>{APP_NAME}</h1>
  <div class="card">
    <h3>Company Login</h3>
    <form method="post" action="/login">
      <label>Company Code</label>
      <input name="company_code" autocomplete="username" required />
      <label>Password</label>
      <input name="password" type="password" autocomplete="current-password" required />
      <button type="submit">Login</button>
    </form>
    <div class="small">Tip: If you get logged out, refresh and log in again.</div>
  </div>
</body>
</html>
        """.strip()
    )


@app.post("/login", include_in_schema=False)
def login_submit(
    request: Request,
    company_code: str = Form(...),
    password: str = Form(...),
):
    if company_code.strip() == COMPANY_CODE and password == COMPANY_PASSWORD:
        request.session["company_code"] = company_code.strip()
        return RedirectResponse(url="/scanner", status_code=302)
    return HTMLResponse("<h3>Login failed</h3><p>Invalid code/password.</p><p><a href='/login'>Try again</a></p>", status_code=401)


@app.get("/logout", include_in_schema=False)
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@app.get("/scanner", response_class=HTMLResponse, include_in_schema=False)
def scanner(request: Request):
    # Browser-friendly: redirect if not logged in
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)

    try:
        with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(html)
    except Exception:
        return HTMLResponse("<h3>Missing frontend/index.html</h3>", status_code=500)


# =========================
# AI prompts
# =========================
def build_single_prompt(max_items: int) -> str:
    # Note about closer photo if unclear
    return f"""
You are an expert estate sale pricing assistant.

Return ONLY valid JSON (no markdown, no code fences).

Task:
- Identify the PRIMARY item in the photo.
- Provide an estate-sale-appropriate price range and suggested list price.
- If multiple items appear, treat them as a bundle ONLY IF the user chose a bundle/single analysis mode (default). Otherwise focus on the primary item.

Output JSON schema:
{{
  "title": string,
  "category": string,
  "condition_notes": string,
  "price_range_low": number,
  "price_range_high": number,
  "suggested_list_price": number,
  "keywords": string[],
  "rationale": string,
  "confidence": number,  // 0.0 to 1.0

  "clarity_notes": string,            // optional: what's unclear or missing
  "next_photo_tip": string            // optional: prompt user to take closer pic / new angle if needed
}}

Rules:
- Use USD for pricing.
- If the photo is unclear OR key markings/labels are missing, include "clarity_notes" and "next_photo_tip".
- Keep it concise and practical for an estate sale team.
""".strip()


def build_multi_prompt(max_items: int) -> str:
    return f"""
You are an expert estate sale pricing assistant.

Return ONLY valid JSON (no markdown, no code fences).

Task:
- Identify up to {max_items} prominent items in the photo.
- Provide BOTH:
  1) A BUNDLE/LOT recommendation (bundle price range + list price)
  2) A per-item breakdown (each item priced separately)

If you cannot confidently identify an item due to size/blur/angle, still list it but:
- lower confidence
- include clarity_notes + next_photo_tip prompting a closer photo.

Output JSON schema:
{{
  "mode": "multi",
  "bundle": {{
    "title": string,
    "price_range_low": number,
    "price_range_high": number,
    "suggested_list_price": number,
    "rationale": string,
    "confidence": number
  }},
  "items": [
    {{
      "id": number,
      "title": string,
      "category": string,
      "condition_notes": string,
      "price_range_low": number,
      "price_range_high": number,
      "suggested_list_price": number,
      "keywords": string[],
      "rationale": string,
      "confidence": number
    }}
  ],
  "recommendation": string,            // "Bundle" vs "Sell separately" guidance
  "additional_items_seen": string[],   // optional: if more than {max_items} items exist, list titles only

  "clarity_notes": string,             // optional
  "next_photo_tip": string             // optional: prompt user to take closer pic / new angle
}}

Rules:
- Max {max_items} items in "items".
- Use USD.
- Bundle pricing should usually reflect a realistic bundle discount vs selling separately.
- If the photo is unclear or items might be missed, include "clarity_notes" and "next_photo_tip".
""".strip()


def image_to_data_url_bytes(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def guess_mime(filename: str) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".png"):
        return "image/png"
    if fn.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def call_openai_vision(image_bytes: bytes, filename: str, mode: str) -> Dict[str, Any]:
    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    mime = guess_mime(filename)
    data_url = image_to_data_url_bytes(image_bytes, mime)

    if mode == "multi":
        prompt = build_multi_prompt(MAX_MULTI_ITEMS)
    else:
        prompt = build_single_prompt(MAX_MULTI_ITEMS)

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Analyze this photo. Mode={mode}. If anything is unclear, ask for a closer photo in next_photo_tip."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    # Ask for strict JSON
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={"type": "json_object"},
    )

    text = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        # Fall back: try best-effort parse
        return {
            "error": "Model returned invalid JSON",
            "raw": text,
            "confidence": 0.2,
            "clarity_notes": "AI output could not be parsed reliably.",
            "next_photo_tip": "Try taking a clearer, closer photo and try again.",
        }


# =========================
# Analyze + History
# =========================
@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    image: UploadFile = File(...),
    analysis_mode: str = Form("single"),  # "single" or "multi"
):
    company_code = require_login_json(request)

    mode = (analysis_mode or "single").strip().lower()
    if mode not in ("single", "multi"):
        mode = "single"

    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    # Save image to disk
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    fname = f"{uuid4().hex}{ext}"
    img_path = os.path.join(UPLOAD_DIR, fname)
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    # AI analysis
    result = call_openai_vision(img_bytes, image.filename or "image.jpg", mode)

    # Store scan
    created_at = utc_now_iso()
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO scans(company_code, created_at, image_path, result_json) VALUES (?, ?, ?, ?)",
        (company_code, created_at, img_path, json.dumps(result)),
    )
    scan_id = cur.lastrowid
    conn.commit()
    conn.close()

    return {
        "scan_id": scan_id,
        "created_at": created_at,
        "company_code": company_code,
        "result": result,
    }


@app.get("/history")
def history(request: Request, limit: int = 25):
    company_code = require_login_json(request)

    limit = max(1, min(int(limit or 25), 100))
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, created_at, image_path, result_json FROM scans WHERE company_code=? ORDER BY id DESC LIMIT ?",
        (company_code, limit),
    )
    rows = cur.fetchall()
    conn.close()

    items = []
    for r in rows:
        scan_id = int(r["id"])
        created_at = r["created_at"]
        try:
            result_obj = json.loads(r["result_json"])
        except Exception:
            result_obj = {"error": "Corrupt result_json", "confidence": 0.1}

        title = extract_title_for_history(result_obj)
        items.append(
            {
                "scan_id": scan_id,
                "created_at": created_at,
                "title": title,
                "image_url": f"/history/{scan_id}/image",
                "result": result_obj,
            }
        )

    return {"items": items}


@app.get("/history/{scan_id}/image", include_in_schema=False)
def history_image(request: Request, scan_id: int):
    company_code = require_login_json(request)

    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT image_path, company_code FROM scans WHERE id=?", (scan_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    if row["company_code"] != company_code:
        raise HTTPException(status_code=403, detail="Forbidden")

    path = row["image_path"]
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File missing")

    return FileResponse(path)
