import os
import json
import base64
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from openai import OpenAI

# =========================
# Config
# =========================
APP_TITLE = "Estate AI Scanner"
MODEL_VISION = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")

COMPANY_CODE = os.getenv("COMPANY_CODE", "demo")
COMPANY_PASSWORD = os.getenv("COMPANY_PASSWORD", "demo")

# Render disk support
DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
if not DATA_DIR.exists():
    DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "app.db"

MAX_PHOTOS_PER_SCAN = 4
MAX_ITEMS_MULTI = 8

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title=APP_TITLE)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site="lax", https_only=False)

# Serve frontend folder (optional)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# =========================
# DB helpers
# =========================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema() -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_code TEXT NOT NULL,
        created_at TEXT NOT NULL,
        analysis_mode TEXT NOT NULL,
        parent_scan_id INTEGER,
        result_json TEXT NOT NULL,
        thumb_path TEXT
      )
    """)
    conn.commit()

    # Lightweight migration: add missing columns if older DB exists
    cur.execute("PRAGMA table_info(scans)")
    cols = {r["name"] for r in cur.fetchall()}

    def add_col(name: str, ddl: str):
        nonlocal cols
        if name not in cols:
            cur.execute(f"ALTER TABLE scans ADD COLUMN {ddl}")
            conn.commit()
            cols.add(name)

    add_col("analysis_mode", "analysis_mode TEXT NOT NULL DEFAULT 'single'")
    add_col("parent_scan_id", "parent_scan_id INTEGER")
    add_col("thumb_path", "thumb_path TEXT")

    conn.close()


ensure_schema()


# =========================
# Auth helpers
# =========================
def is_logged_in(request: Request) -> bool:
    return bool(request.session.get("company_code"))


def require_login_page(request: Request) -> Optional[RedirectResponse]:
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)
    return None


def require_login_api(request: Request) -> Optional[JSONResponse]:
    if not is_logged_in(request):
        return JSONResponse({"error": "Not logged in"}, status_code=401)
    return None


# =========================
# HTML pages
# =========================
LOGIN_HTML = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{APP_TITLE} — Login</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; max-width: 520px; margin: 0 auto; }}
    h1 {{ margin: 0 0 10px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; }}
    label {{ display:block; margin: 10px 0 6px; font-weight: 700; }}
    input {{ width: 100%; padding: 10px; border-radius: 10px; border: 1px solid #ccc; }}
    button {{ margin-top: 14px; width:100%; padding: 12px; border-radius: 10px; border: 0; background:#2b7cff; color:#fff; font-weight: 800; }}
    .sub {{ color:#666; margin-top: 0; }}
    .err {{ color:#b00020; font-weight: 800; margin-top: 10px; }}
  </style>
</head>
<body>
  <h1>{APP_TITLE}</h1>
  <p class="sub">Company login required</p>
  <div class="card">
    <form method="post" action="/login">
      <label>Company Code</label>
      <input name="company_code" autocomplete="username" required />
      <label>Password</label>
      <input name="company_password" type="password" autocomplete="current-password" required />
      <button type="submit">Login</button>
    </form>
    {{ERROR}}
  </div>
</body>
</html>
"""


# =========================
# Routes
# =========================
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    # Always land on scanner (protected)
    redir = require_login_page(request)
    if redir:
        return redir
    return RedirectResponse(url="/scanner", status_code=302)


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    # If already logged in, go to scanner
    if is_logged_in(request):
        return RedirectResponse(url="/scanner", status_code=302)
    return HTMLResponse(LOGIN_HTML.replace("{ERROR}", ""))


@app.post("/login", response_class=HTMLResponse)
def login_post(
    request: Request,
    company_code: str = Form(...),
    company_password: str = Form(...)
):
    if company_code == COMPANY_CODE and company_password == COMPANY_PASSWORD:
        request.session["company_code"] = company_code
        return RedirectResponse(url="/scanner", status_code=302)

    html = LOGIN_HTML.replace("{ERROR}", '<div class="err">Invalid company code or password.</div>')
    return HTMLResponse(html, status_code=401)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@app.get("/scanner", response_class=HTMLResponse)
def scanner(request: Request):
    redir = require_login_page(request)
    if redir:
        return redir

    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Missing frontend/index.html</h1>", status_code=500)

    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# =========================
# AI helpers
# =========================
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_to_data_url(file_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _safe_json_loads(text: str) -> dict:
    # Remove accidental code fences
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def build_prompt(analysis_mode: str) -> str:
    # Force stable schema.
    # Note: return STRICT JSON only (no markdown).
    return f"""
You are an expert estate-sale pricing assistant.

Analyze the provided photo(s). If multiple photos are provided, treat them as the SAME scene/item(s) and combine evidence.

analysis_mode = "{analysis_mode}"
- "single" means: user wants a single item price AND an optional bundle/lot range if it makes sense.
- "multi" means: identify up to {MAX_ITEMS_MULTI} distinct items, each with its own price range, plus a bundle/lot range for the group.

IMPORTANT:
- If you cannot confidently identify distinct items, return items: [] and explain why in pricing_estimates, plus actionable suggestions.
- Include set/series/collection notes because completeness affects price.

Return STRICT JSON ONLY with exactly this shape:

{{
  "items": [
    {{
      "name": "string",
      "type": "string (e.g., Book, Laptop, Vase)",
      "description": "string (short, user-facing)",
      "estimated_price": 15 OR {{ "low": 10, "high": 20 }},
      "confidence": "high" | "medium" | "low",
      "set_collection_notes": {{
        "is_part_of_set_likely": "yes" | "no" | "unknown",
        "what_to_check": ["string", "..."],
        "price_impact": "string (1 sentence)"
      }}
    }}
  ],
  "bundle_price_range": {{ "min": 0, "max": 0 }},
  "pricing_estimates": "string (why these prices / or why unclear)",
  "suggestions": {{
    "markings_to_look_for": "string (helpful checks)",
    "photo_tip": "string (what photo would help next)"
  }},
  "advice": ["string", "..."]
}}
"""


def call_vision(images_data_urls: List[str], analysis_mode: str) -> dict:
    if not client:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    prompt = build_prompt(analysis_mode)

    content = [{"type": "text", "text": prompt}]
    for du in images_data_urls:
        content.append({"type": "image_url", "image_url": {"url": du}})

    resp = client.chat.completions.create(
        model=MODEL_VISION,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
    )

    text = resp.choices[0].message.content or ""
    return _safe_json_loads(text)


# =========================
# Core API
# =========================
@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    images: List[UploadFile] = File(...),
    analysis_mode: str = Form("single"),
):
    auth = require_login_api(request)
    if auth:
        return auth

    analysis_mode = (analysis_mode or "single").strip().lower()
    if analysis_mode not in ("single", "multi"):
        analysis_mode = "single"

    if not images or len(images) == 0:
        return JSONResponse({"error": "No images uploaded"}, status_code=400)

    if len(images) > MAX_PHOTOS_PER_SCAN:
        return JSONResponse({"error": f"Max photos per scan is {MAX_PHOTOS_PER_SCAN}"}, status_code=400)

    company_code = request.session["company_code"]

    # Convert images to data urls for OpenAI; save first image as thumbnail to disk
    data_urls = []
    thumb_path = None

    for idx, up in enumerate(images):
        file_bytes = await up.read()
        mime = up.content_type or "image/jpeg"
        data_urls.append(_file_to_data_url(file_bytes, mime))

        if idx == 0:
            # Save thumb image
            filename = f"{company_code}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}.jpg"
            thumb_path = str(UPLOADS_DIR / filename)
            with open(thumb_path, "wb") as f:
                f.write(file_bytes)

    try:
        result = call_vision(data_urls, analysis_mode)
    except Exception as e:
        return JSONResponse({"error": "Internal analysis error", "details": str(e)}, status_code=500)

    # Normalize bundle range keys (min/max) if missing
    bundle = result.get("bundle_price_range") or {}
    if isinstance(bundle, dict):
        # accept low/high variants
        if "min" not in bundle and "low" in bundle:
            bundle["min"] = bundle.get("low")
        if "max" not in bundle and "high" in bundle:
            bundle["max"] = bundle.get("high")
        # ensure exists
        result["bundle_price_range"] = {"min": bundle.get("min", 0), "max": bundle.get("max", 0)}
    else:
        result["bundle_price_range"] = {"min": 0, "max": 0}

    scan_row = {
        "company_code": company_code,
        "created_at": _now_iso(),
        "analysis_mode": analysis_mode,
        "parent_scan_id": None,
        "result_json": json.dumps(result, ensure_ascii=False),
        "thumb_path": thumb_path,
    }

    conn = db()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO scans (company_code, created_at, analysis_mode, parent_scan_id, result_json, thumb_path)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            scan_row["company_code"],
            scan_row["created_at"],
            scan_row["analysis_mode"],
            scan_row["parent_scan_id"],
            scan_row["result_json"],
            scan_row["thumb_path"],
        ),
    )
    scan_id = cur.lastrowid
    conn.commit()
    conn.close()

    return JSONResponse({
        "scan_id": scan_id,
        "created_at": scan_row["created_at"],
        "analysis_mode": analysis_mode,
        "result": result
    })


@app.post("/refine")
async def refine_scan(
    request: Request,
    scan_id: int = Form(...),
    images: List[UploadFile] = File(...),
):
    auth = require_login_api(request)
    if auth:
        return auth

    if not images or len(images) == 0:
        return JSONResponse({"error": "No images uploaded"}, status_code=400)
    if len(images) > MAX_PHOTOS_PER_SCAN:
        return JSONResponse({"error": f"Max photos per refine is {MAX_PHOTOS_PER_SCAN}"}, status_code=400)

    company_code = request.session["company_code"]

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM scans WHERE id=? AND company_code=?", (scan_id, company_code))
    row = cur.fetchone()
    if not row:
        conn.close()
        return JSONResponse({"error": "Scan not found"}, status_code=404)

    prior = json.loads(row["result_json"])
    analysis_mode = row["analysis_mode"] or "single"

    # Build data urls for refinement images
    data_urls = []
    thumb_path = None
    for idx, up in enumerate(images):
        file_bytes = await up.read()
        mime = up.content_type or "image/jpeg"
        data_urls.append(_file_to_data_url(file_bytes, mime))

        if idx == 0:
            filename = f"{company_code}_refine_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}.jpg"
            thumb_path = str(UPLOADS_DIR / filename)
            with open(thumb_path, "wb") as f:
                f.write(file_bytes)

    # We nudge the model to refine, using prior JSON as context.
    try:
        prompt = build_prompt(analysis_mode) + "\n\nPRIOR RESULT JSON:\n" + json.dumps(prior, ensure_ascii=False)
        content = [{"type": "text", "text": prompt}]
        for du in data_urls:
            content.append({"type": "image_url", "image_url": {"url": du}})

        resp = client.chat.completions.create(
            model=MODEL_VISION,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content or ""
        result = _safe_json_loads(text)
    except Exception as e:
        conn.close()
        return JSONResponse({"error": "Internal refine error", "details": str(e)}, status_code=500)

    # Normalize bundle keys
    bundle = result.get("bundle_price_range") or {}
    if isinstance(bundle, dict):
        if "min" not in bundle and "low" in bundle:
            bundle["min"] = bundle.get("low")
        if "max" not in bundle and "high" in bundle:
            bundle["max"] = bundle.get("high")
        result["bundle_price_range"] = {"min": bundle.get("min", 0), "max": bundle.get("max", 0)}
    else:
        result["bundle_price_range"] = {"min": 0, "max": 0}

    created_at = _now_iso()
    cur.execute(
        """INSERT INTO scans (company_code, created_at, analysis_mode, parent_scan_id, result_json, thumb_path)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            company_code,
            created_at,
            analysis_mode,
            scan_id,
            json.dumps(result, ensure_ascii=False),
            thumb_path,
        ),
    )
    new_id = cur.lastrowid
    conn.commit()
    conn.close()

    return JSONResponse({
        "scan_id": new_id,
        "created_at": created_at,
        "analysis_mode": analysis_mode,
        "parent_scan_id": scan_id,
        "result": result
    })


@app.get("/history")
def history(request: Request, limit: int = 25):
    auth = require_login_api(request)
    if auth:
        return auth

    limit = max(1, min(100, int(limit or 25)))
    company_code = request.session["company_code"]

    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, company_code, created_at, analysis_mode, parent_scan_id, result_json, thumb_path FROM scans "
        "WHERE company_code=? ORDER BY id DESC LIMIT ?",
        (company_code, limit),
    )
    rows = cur.fetchall()
    conn.close()

    items = []
    for r in rows:
        result = json.loads(r["result_json"])
        thumb_url = f"/history/{r['id']}/image" if r["thumb_path"] else None

        # Handy title for list
        title = "Scan"
        its = result.get("items") or []
        if isinstance(its, list) and len(its) > 0:
            title = its[0].get("name") or title

        items.append({
            "scan_id": r["id"],
            "created_at": r["created_at"],
            "analysis_mode": r["analysis_mode"],
            "parent_scan_id": r["parent_scan_id"],
            "title": title,
            "image_url": thumb_url,
            "result": result
        })

    return JSONResponse({"items": items})


@app.get("/history/{scan_id}/image")
def history_image(request: Request, scan_id: int):
    auth = require_login_api(request)
    if auth:
        return auth

    company_code = request.session["company_code"]
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT thumb_path FROM scans WHERE id=? AND company_code=?", (scan_id, company_code))
    row = cur.fetchone()
    conn.close()

    if not row or not row["thumb_path"]:
        return JSONResponse({"error": "Image not found"}, status_code=404)

    path = row["thumb_path"]
    if not os.path.exists(path):
        return JSONResponse({"error": "Image missing on disk"}, status_code=404)

    return FileResponse(path, media_type="image/jpeg")
