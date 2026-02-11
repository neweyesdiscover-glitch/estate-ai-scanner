import os
import json
import base64
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
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

# Brief note: max items at once for multi-item mode
MAX_MULTI_ITEMS = int(os.getenv("MAX_MULTI_ITEMS", "8"))
# Brief note: max photos allowed in one scan (multi-photo for one object)
MAX_PHOTOS_PER_SCAN = int(os.getenv("MAX_PHOTOS_PER_SCAN", "4"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_data_root() -> str:
    # Render disk mount (persistent)
    if os.path.isdir("/var/data"):
        return "/var/data"
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

    # scans table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_code TEXT NOT NULL,
            created_at TEXT NOT NULL,
            analysis_mode TEXT NOT NULL,
            result_json TEXT NOT NULL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_scans_company_created ON scans(company_code, created_at);")

    # scan_images table (supports multiple photos per scan)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER NOT NULL,
            idx INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            FOREIGN KEY(scan_id) REFERENCES scans(id)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_scan_images_scan ON scan_images(scan_id, idx);")

    conn.commit()
    conn.close()


def extract_title_for_history(result_obj: dict) -> str:
    try:
        if isinstance(result_obj, dict) and result_obj.get("mode") == "multi":
            bundle = result_obj.get("bundle") or {}
            return str(bundle.get("title") or "Multi-item lot")
        return str(result_obj.get("title") or "Scan")
    except Exception:
        return "Scan"


# =========================
# Auth helpers
# =========================
def is_logged_in(request: Request) -> bool:
    return bool(request.session.get("company_code"))


def require_login_json(request: Request) -> str:
    if not is_logged_in(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.session.get("company_code")


# =========================
# App
# =========================
app = FastAPI(title=APP_NAME)

if not SECRET_KEY:
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
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)

    try:
        with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(html)
    except Exception:
        return HTMLResponse("<h3>Missing frontend/index.html</h3>", status_code=500)


# =========================
# OpenAI helpers
# =========================
def guess_mime(filename: str) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".png"):
        return "image/png"
    if fn.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def bytes_to_data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_single_prompt() -> str:
    return f"""
You are an expert estate sale pricing assistant.

Return ONLY valid JSON (no markdown, no code fences).

The user may provide MULTIPLE photos of the SAME item (overall + markings + close-ups).
Use ALL photos together to produce one best answer.

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

  "what_to_check": [
    {{
      "label": string,
      "why_it_matters": string,
      "how_to_check": string
    }}
  ],

  "followup_questions": [
    {{
      "id": string,
      "question": string,
      "answers": ["Yes","No","Not sure"],
      "why": string
    }}
  ],

  "clarity_notes": string,
  "next_photo_tip": string
}}

Rules:
- Use USD.
- If photo(s) are unclear or you might be missing maker marks / stamps, fill clarity_notes + next_photo_tip.
- Provide specialized “what_to_check” tips (e.g., crystal edge sharpness, china backstamps, reproductions).
- Ask 1–3 followup questions that can materially change value (marks, pattern name, manufacturer, age).
""".strip()


def build_multi_prompt() -> str:
    return f"""
You are an expert estate sale pricing assistant.

Return ONLY valid JSON (no markdown, no code fences).

Task:
- Identify up to {MAX_MULTI_ITEMS} prominent items in the photo(s).
- Provide BOTH:
  1) A BUNDLE/LOT recommendation (bundle price range + list price)
  2) A per-item breakdown (each item priced separately)

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
  "recommendation": string,
  "additional_items_seen": string[],

  "what_to_check": [
    {{
      "label": string,
      "why_it_matters": string,
      "how_to_check": string
    }}
  ],

  "followup_questions": [
    {{
      "id": string,
      "question": string,
      "answers": ["Yes","No","Not sure"],
      "why": string
    }}
  ],

  "clarity_notes": string,
  "next_photo_tip": string
}}

Rules:
- "items" must include at most {MAX_MULTI_ITEMS} entries.
- Bundle pricing should reflect a realistic bundle discount vs selling separately.
- Provide specialized “what_to_check” tips and 1–3 followup questions that could change bundle vs separate decision.
- If photo is unclear or items might be missed, fill clarity_notes + next_photo_tip.
""".strip()


def build_refine_prompt() -> str:
    return f"""
You are an expert estate sale pricing assistant.

Return ONLY valid JSON (no markdown, no code fences).

You will be given:
- The previous JSON result
- User answers to follow-up questions (Yes/No/Not sure)
- Optional additional photos (close-ups, maker marks)

Update the result accordingly.
- If the user answers confirm a higher-value marker (maker stamp, genuine material), adjust the estimate.
- If answers suggest reproduction / lower-value, adjust down.
- Keep the same schema as the previous result (single stays single schema; multi stays multi schema).
- Update confidence if appropriate.
- Update what_to_check / followup_questions as needed (but keep it short).
""".strip()


def openai_analyze_images(image_bytes_list: List[bytes], filenames: List[str], analysis_mode: str) -> Dict[str, Any]:
    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    if analysis_mode == "multi":
        system_prompt = build_multi_prompt()
    else:
        system_prompt = build_single_prompt()

    # build image parts
    parts = [{"type": "text", "text": f"Analyze these photo(s). analysis_mode={analysis_mode}. If unclear, use next_photo_tip to ask for closer photo(s)."}]
    for b, fn in zip(image_bytes_list, filenames):
        mime = guess_mime(fn)
        parts.append({"type": "image_url", "image_url": {"url": bytes_to_data_url(b, mime)}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": parts},
    ]

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={"type": "json_object"},
    )

    text = resp.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except Exception:
        return {
            "error": "Model returned invalid JSON",
            "raw": text,
            "confidence": 0.2,
            "clarity_notes": "AI output could not be parsed reliably.",
            "next_photo_tip": "Try taking a clearer, closer photo (labels/maker marks) and try again.",
            "what_to_check": [],
            "followup_questions": [],
        }


def openai_refine(previous_result: Dict[str, Any], answers: Dict[str, str], extra_images: List[bytes], extra_filenames: List[str]) -> Dict[str, Any]:
    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    system_prompt = build_refine_prompt()

    parts: List[dict] = [
        {"type": "text", "text": "Refine the previous result using the user answers and any extra photos."},
        {"type": "text", "text": f"Previous result JSON:\n{json.dumps(previous_result, ensure_ascii=False)}"},
        {"type": "text", "text": f"User answers JSON:\n{json.dumps(answers, ensure_ascii=False)}"},
    ]

    for b, fn in zip(extra_images, extra_filenames):
        mime = guess_mime(fn)
        parts.append({"type": "image_url", "image_url": {"url": bytes_to_data_url(b, mime)}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": parts},
    ]

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={"type": "json_object"},
    )

    text = resp.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except Exception:
        # If refine fails, keep previous but add an error hint
        previous_result = dict(previous_result or {})
        previous_result["clarity_notes"] = (previous_result.get("clarity_notes") or "") + " | Refinement parse failed."
        previous_result["next_photo_tip"] = "Try a clearer close-up of the mark/stamp and retry."
        return previous_result


# =========================
# Analyze + History
# =========================
async def save_images_for_scan(scan_id: int, images: List[UploadFile]) -> None:
    conn = db_connect()
    cur = conn.cursor()

    for idx, up in enumerate(images):
        b = await up.read()
        if not b:
            continue
        ext = os.path.splitext(up.filename or "")[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"
        fname = f"{uuid4().hex}{ext}"
        path = os.path.join(UPLOAD_DIR, fname)
        with open(path, "wb") as f:
            f.write(b)
        cur.execute("INSERT INTO scan_images(scan_id, idx, image_path) VALUES (?, ?, ?)", (scan_id, idx, path))

    conn.commit()
    conn.close()


def load_scan(scan_id: int) -> Optional[sqlite3.Row]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM scans WHERE id=?", (scan_id,))
    row = cur.fetchone()
    conn.close()
    return row


def load_scan_images(scan_id: int) -> List[sqlite3.Row]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT idx, image_path FROM scan_images WHERE scan_id=? ORDER BY idx ASC", (scan_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    images: List[UploadFile] = File(...),          # supports multi-photo
    analysis_mode: str = Form("single"),           # "single" or "multi"
):
    company_code = require_login_json(request)

    mode = (analysis_mode or "single").strip().lower()
    if mode not in ("single", "multi"):
        mode = "single"

    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded")

    if len(images) > MAX_PHOTOS_PER_SCAN:
        raise HTTPException(status_code=400, detail=f"Too many photos. Max {MAX_PHOTOS_PER_SCAN} per scan.")

    # Read bytes for AI (we also save after creating scan id)
    img_bytes_list: List[bytes] = []
    filenames: List[str] = []
    for up in images:
        b = await up.read()
        if not b:
            continue
        img_bytes_list.append(b)
        filenames.append(up.filename or "image.jpg")

    if not img_bytes_list:
        raise HTTPException(status_code=400, detail="Empty uploads")

    # Create scan row first
    created_at = utc_now_iso()
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO scans(company_code, created_at, analysis_mode, result_json) VALUES (?, ?, ?, ?)",
        (company_code, created_at, mode, "{}"),
    )
    scan_id = cur.lastrowid
    conn.commit()
    conn.close()

    # Save images to disk (need to re-read files; easiest is to use the bytes we already read)
    # Since UploadFile was consumed, we save using img_bytes_list + filenames
    conn = db_connect()
    cur = conn.cursor()
    for idx, (b, fn) in enumerate(zip(img_bytes_list, filenames)):
        ext = os.path.splitext(fn or "")[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"
        fname = f"{uuid4().hex}{ext}"
        path = os.path.join(UPLOAD_DIR, fname)
        with open(path, "wb") as f:
            f.write(b)
        cur.execute("INSERT INTO scan_images(scan_id, idx, image_path) VALUES (?, ?, ?)", (scan_id, idx, path))
    conn.commit()
    conn.close()

    # AI analysis
    result = openai_analyze_images(img_bytes_list, filenames, mode)

    # Update scan record
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE scans SET result_json=? WHERE id=?", (json.dumps(result), scan_id))
    conn.commit()
    conn.close()

    return {
        "scan_id": scan_id,
        "created_at": created_at,
        "company_code": company_code,
        "analysis_mode": mode,
        "result": result,
    }


@app.post("/refine")
async def refine(
    request: Request,
    scan_id: int = Form(...),
    answers_json: str = Form(...),
    extra_images: List[UploadFile] = File([]),  # optional extra close-ups
):
    company_code = require_login_json(request)

    row = load_scan(scan_id)
    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")
    if row["company_code"] != company_code:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        prev = json.loads(row["result_json"])
    except Exception:
        prev = {}

    try:
        answers = json.loads(answers_json) if answers_json else {}
        if not isinstance(answers, dict):
            answers = {}
    except Exception:
        answers = {}

    extra_bytes: List[bytes] = []
    extra_fns: List[str] = []
    if extra_images:
        for up in extra_images[:MAX_PHOTOS_PER_SCAN]:
            b = await up.read()
            if not b:
                continue
            extra_bytes.append(b)
            extra_fns.append(up.filename or "extra.jpg")

    updated = openai_refine(prev, answers, extra_bytes, extra_fns)

    # Update scan + save any new images to scan_images (append after existing)
    existing = load_scan_images(scan_id)
    start_idx = len(existing)

    if extra_bytes:
        conn = db_connect()
        cur = conn.cursor()
        for i, (b, fn) in enumerate(zip(extra_bytes, extra_fns)):
            ext = os.path.splitext(fn or "")[1].lower()
            if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                ext = ".jpg"
            fname = f"{uuid4().hex}{ext}"
            path = os.path.join(UPLOAD_DIR, fname)
            with open(path, "wb") as f:
                f.write(b)
            cur.execute("INSERT INTO scan_images(scan_id, idx, image_path) VALUES (?, ?, ?)", (scan_id, start_idx + i, path))
        conn.commit()
        conn.close()

    conn = db_connect()
    cur = conn.cursor()
    cur.execute("UPDATE scans SET result_json=? WHERE id=?", (json.dumps(updated), scan_id))
    conn.commit()
    conn.close()

    return {
        "scan_id": scan_id,
        "created_at": row["created_at"],
        "company_code": company_code,
        "analysis_mode": row["analysis_mode"],
        "result": updated,
    }


@app.get("/history")
def history(request: Request, limit: int = 25):
    company_code = require_login_json(request)

    limit = max(1, min(int(limit or 25), 100))

    conn = db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, created_at, analysis_mode, result_json FROM scans WHERE company_code=? ORDER BY id DESC LIMIT ?",
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
                "analysis_mode": r["analysis_mode"],
                "image_url": f"/history/{scan_id}/image",  # first image
                "result": result_obj,
            }
        )

    return {"items": items}


@app.get("/history/{scan_id}/image", include_in_schema=False)
def history_image(request: Request, scan_id: int, idx: int = 0):
    company_code = require_login_json(request)

    row = load_scan(scan_id)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    if row["company_code"] != company_code:
        raise HTTPException(status_code=403, detail="Forbidden")

    imgs = load_scan_images(scan_id)
    if not imgs:
        raise HTTPException(status_code=404, detail="No images")
    idx = int(idx or 0)
    if idx < 0 or idx >= len(imgs):
        idx = 0

    path = imgs[idx]["image_path"]
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File missing")

    return FileResponse(path)
