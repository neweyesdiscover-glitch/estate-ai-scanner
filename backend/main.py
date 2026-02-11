import os
import json
import base64
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from starlette.middleware.sessions import SessionMiddleware
from openai import OpenAI


# ===============================
# CONFIG
# ===============================

APP_NAME = "Estate AI Scanner"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
COMPANY_CODE = os.getenv("COMPANY_CODE", "demo")
COMPANY_PASSWORD = os.getenv("COMPANY_PASSWORD", "demo")

MAX_MULTI_ITEMS = 8
MAX_PHOTOS_PER_SCAN = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_data_root() -> str:
    # Render disk mount path
    if os.path.isdir("/var/data"):
        return "/var/data"
    # local fallback
    return os.path.join(os.path.dirname(__file__), "data")


DATA_ROOT = get_data_root()
UPLOAD_DIR = os.path.join(DATA_ROOT, "uploads")
DB_PATH = os.path.join(DATA_ROOT, "app.db")
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ===============================
# DATABASE
# ===============================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_code TEXT NOT NULL,
            created_at TEXT NOT NULL,
            analysis_mode TEXT NOT NULL,
            result_json TEXT NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS scan_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER NOT NULL,
            idx INTEGER NOT NULL,
            image_path TEXT NOT NULL
        );
    """)

    conn.commit()
    conn.close()


# ===============================
# APP
# ===============================

app = FastAPI(title=APP_NAME)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


@app.on_event("startup")
def startup():
    init_db()


# ===============================
# AUTH
# ===============================

def require_login(request: Request) -> str:
    company = request.session.get("company_code")
    if not company:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return company


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return """
    <h2>Company Login</h2>
    <form method="post" action="/login">
        Company Code:<br>
        <input name="company_code"><br><br>
        Password:<br>
        <input name="password" type="password"><br><br>
        <button type="submit">Login</button>
    </form>
    """


@app.post("/login")
def login_submit(
    request: Request,
    company_code: str = Form(...),
    password: str = Form(...)
):
    if company_code == COMPANY_CODE and password == COMPANY_PASSWORD:
        request.session["company_code"] = company_code
        return RedirectResponse("/scanner", status_code=302)
    return HTMLResponse("Login failed", status_code=401)


@app.get("/scanner", response_class=HTMLResponse)
def scanner(request: Request):
    if not request.session.get("company_code"):
        return RedirectResponse("/login", status_code=302)

    with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ===============================
# OPENAI HELPERS
# ===============================

def encode_image(image_bytes: bytes, filename: str) -> str:
    mime = "image/jpeg"
    lower = (filename or "").lower()
    if lower.endswith(".png"):
        mime = "image/png"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    # common model formatting
    if t.startswith("```"):
        t = t.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    return t


# ===============================
# ANALYZE (ALWAYS RETURNS JSON)
# ===============================

@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    images: List[UploadFile] = File(...),  # frontend sends key "images"
    analysis_mode: str = Form("single")
):
    company = require_login(request)

    try:
        if not client:
            return {"error": "OpenAI API key not configured (OPENAI_API_KEY missing)"}

        if not images or len(images) == 0:
            return {"error": "No images uploaded"}

        if len(images) > MAX_PHOTOS_PER_SCAN:
            return {"error": f"Max {MAX_PHOTOS_PER_SCAN} photos allowed per scan"}

        image_bytes = []
        filenames = []

        for img in images:
            b = await img.read()
            image_bytes.append(b)
            filenames.append(img.filename or "image.jpg")

        prompt = f"""
You are an expert estate sale pricing assistant.

You MUST return VALID JSON ONLY. No markdown.

Goal:
- Identify item(s) in the photo(s)
- Provide pricing estimates & listing guidance.

If multiple distinct items appear:
- Identify up to {MAX_MULTI_ITEMS} prominent items
- Provide per-item pricing AND a bundle price range for the group.

If you are uncertain or the photo is unclear:
- Include tips asking for a closer/brighter photo
- Suggest what markings/stamps to look for (especially for antiques/crystal/china).

Return JSON only.
"""

        parts = [{"type": "text", "text": prompt}]
        for b, f in zip(image_bytes, filenames):
            parts.append({"type": "image_url", "image_url": {"url": encode_image(b, f)}})

        # Use "messages" only; we'll parse ourselves to avoid hard failures
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": parts}
            ],
        )

        raw_content = strip_code_fences(response.choices[0].message.content)

        # Strict JSON parse; if it fails, return raw so UI can show it
        try:
            result = json.loads(raw_content)
        except Exception:
            return {
                "error": "Model did not return valid JSON",
                "raw": raw_content
            }

        # Save scan + images
        conn = db()
        cur = conn.cursor()
        created = utc_now()

        cur.execute("""
            INSERT INTO scans(company_code, created_at, analysis_mode, result_json)
            VALUES (?, ?, ?, ?)
        """, (company, created, analysis_mode, json.dumps(result)))

        scan_id = cur.lastrowid

        for i, (b, f) in enumerate(zip(image_bytes, filenames)):
            ext = ".jpg"
            if (f or "").lower().endswith(".png"):
                ext = ".png"
            fname = f"{uuid4().hex}{ext}"
            path = os.path.join(UPLOAD_DIR, fname)
            with open(path, "wb") as out:
                out.write(b)

            cur.execute("""
                INSERT INTO scan_images(scan_id, idx, image_path)
                VALUES (?, ?, ?)
            """, (scan_id, i, path))

        conn.commit()
        conn.close()

        return {
            "scan_id": scan_id,
            "created_at": created,
            "analysis_mode": analysis_mode,
            "result": result
        }

    except Exception as e:
        # Critical: never let FastAPI produce HTML 500 here — always JSON
        return {
            "error": "Internal analysis error",
            "details": str(e)
        }


# ===============================
# REFINE WITH EXTRA PHOTOS
# ===============================

@app.post("/refine")
async def refine(
    request: Request,
    scan_id: int = Form(...),
    answers_json: str = Form("{}"),
    extra_images: List[UploadFile] = File([]),
):
    company = require_login(request)

    try:
        if not client:
            return {"error": "OpenAI API key not configured (OPENAI_API_KEY missing)"}

        conn = db()
        cur = conn.cursor()

        cur.execute("SELECT * FROM scans WHERE id=?", (scan_id,))
        row = cur.fetchone()

        if not row:
            conn.close()
            return {"error": "Scan not found"}

        if row["company_code"] != company:
            conn.close()
            return {"error": "Forbidden"}

        previous = json.loads(row["result_json"])
        try:
            answers = json.loads(answers_json or "{}")
        except Exception:
            answers = {"_parse_error": "answers_json was not valid JSON"}

        prompt = f"""
You are refining a previous estate sale analysis.

Return VALID JSON ONLY. No markdown.

Previous JSON:
{json.dumps(previous)}

User Answers:
{json.dumps(answers)}

Instructions:
- Update identification/pricing if answers or new photos provide better info
- If still uncertain, add clearer tips/questions about markings and what to photograph next
"""

        parts = [{"type": "text", "text": prompt}]

        extra_bytes = []
        extra_files = []
        for img in extra_images or []:
            b = await img.read()
            extra_bytes.append(b)
            extra_files.append(img.filename or "extra.jpg")
            parts.append({"type": "image_url", "image_url": {"url": encode_image(b, img.filename or "extra.jpg")}})

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": parts}
            ],
        )

        raw_content = strip_code_fences(response.choices[0].message.content)
        try:
            updated = json.loads(raw_content)
        except Exception:
            conn.close()
            return {"error": "Model did not return valid JSON during refine", "raw": raw_content}

        # Update scan
        cur.execute("UPDATE scans SET result_json=? WHERE id=?", (json.dumps(updated), scan_id))

        # Save extra images (idx starts at 100 to avoid collisions)
        for i, (b, f) in enumerate(zip(extra_bytes, extra_files)):
            ext = ".jpg"
            if (f or "").lower().endswith(".png"):
                ext = ".png"
            fname = f"{uuid4().hex}{ext}"
            path = os.path.join(UPLOAD_DIR, fname)
            with open(path, "wb") as out:
                out.write(b)

            cur.execute("""
                INSERT INTO scan_images(scan_id, idx, image_path)
                VALUES (?, ?, ?)
            """, (scan_id, 100 + i, path))

        conn.commit()
        conn.close()

        return {
            "scan_id": scan_id,
            "analysis_mode": row["analysis_mode"],
            "result": updated
        }

    except Exception as e:
        return {"error": "Internal refine error", "details": str(e)}


# ===============================
# HISTORY
# ===============================

@app.get("/history")
def history(request: Request, limit: int = 25):
    company = require_login(request)

    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, created_at, analysis_mode, result_json
        FROM scans
        WHERE company_code=?
        ORDER BY id DESC
        LIMIT ?
    """, (company, limit))

    rows = cur.fetchall()
    conn.close()

    items = []
    for r in rows:
        items.append({
            "scan_id": r["id"],
            "created_at": r["created_at"],
            "analysis_mode": r["analysis_mode"],
            "result": json.loads(r["result_json"]),
            "image_url": f"/history/{r['id']}/image"
        })

    return {"items": items}


@app.get("/history/{scan_id}/image")
def history_image(request: Request, scan_id: int):
    company = require_login(request)

    conn = db()
    cur = conn.cursor()

    # ensure scan belongs to company
    cur.execute("SELECT company_code FROM scans WHERE id=?", (scan_id,))
    scan_row = cur.fetchone()
    if not scan_row or scan_row["company_code"] != company:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")

    # get first image
    cur.execute("""
        SELECT image_path FROM scan_images
        WHERE scan_id=?
        ORDER BY idx ASC
        LIMIT 1
    """, (scan_id,))
    img_row = cur.fetchone()
    conn.close()

    if not img_row:
        raise HTTPException(status_code=404, detail="No image")

    return FileResponse(img_row["image_path"])
