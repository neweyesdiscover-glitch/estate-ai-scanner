# backend/main.py
# ✅ Customer-ready company login (simple + reliable)
# - GET /login shows login form
# - POST /login handles form fields (avoids JSON schema mismatch / 422)
# - /scanner requires login (redirects to /login)
# - /analyze-image requires login (returns 401 JSON, not redirect)

import os
import json
import base64
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "")
COMPANY_CODE = os.getenv("COMPANY_CODE", "demo").strip()
COMPANY_PASSWORD = os.getenv("COMPANY_PASSWORD", "demo").strip()

client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR = Path(__file__).resolve().parent           # backend/
INDEX_PATH = BASE_DIR.parent / "frontend" / "index.html"

# -----------------------------
# App
# -----------------------------
app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY or "dev-only-change-me",
    same_site="lax",
    https_only=False,   # Render uses https; this can be True later
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Auth helpers
# -----------------------------
def logged_in(request: Request) -> bool:
    return bool(request.session.get("company_code"))

def require_login_redirect(request: Request):
    if not logged_in(request):
        return RedirectResponse(url="/login", status_code=302)
    return None

def require_login_api(request: Request):
    if not logged_in(request):
        raise HTTPException(status_code=401, detail="Not logged in")

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=JSONResponse)
def home():
    return {"message": "Estate AI Scanner is LIVE 🚀", "login": "/login", "scanner": "/scanner"}

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if logged_in(request):
        return RedirectResponse(url="/scanner", status_code=302)

    msg = request.query_params.get("msg", "")
    msg = (msg or "").replace("<", "&lt;").replace(">", "&gt;")

    return HTMLResponse(f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Estate AI Scanner — Login</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 520px; margin: 40px auto; padding: 0 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; }}
    h1 {{ margin: 0 0 10px; }}
    .sub {{ color:#555; margin: 0 0 12px; }}
    label {{ display:block; margin-top: 10px; font-weight: 700; }}
    input {{ width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; }}
    button {{ margin-top: 14px; padding: 10px 14px; border-radius: 8px; border: 1px solid #2b7cff;
             background:#2b7cff; color:#fff; font-weight: 800; cursor:pointer; width:100%; }}
    .msg {{ color:#b00020; font-weight: 800; margin-top: 10px; }}
    .hint {{ font-size: 12px; color:#666; margin-top: 12px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Company Login</h1>
    <p class="sub">Enter your company access code.</p>

    <form method="post" action="/login">
      <label>Company Code</label>
      <input name="company_code" placeholder="e.g., demo" required />

      <label>Passcode</label>
      <input name="password" type="password" placeholder="••••••••" required />

      <button type="submit">Log In</button>
    </form>

    {f"<div class='msg'>{msg}</div>" if msg else ""}
    <div class="hint">Tip: Ask Victoria for your company code + passcode.</div>
  </div>
</body>
</html>
""")

@app.post("/login")
def do_login(request: Request, company_code: str = Form(...), password: str = Form(...)):
    company_code = company_code.strip()
    password = password.strip()

    if company_code == COMPANY_CODE and password == COMPANY_PASSWORD:
        request.session["company_code"] = company_code
        return RedirectResponse(url="/scanner", status_code=302)

    return RedirectResponse(url="/login?msg=Invalid+company+code+or+passcode", status_code=302)

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login?msg=Logged+out", status_code=302)

@app.get("/scanner")
def scanner(request: Request):
    gate = require_login_redirect(request)
    if gate:
        return gate

    if not INDEX_PATH.exists():
        return JSONResponse({"error": "frontend/index.html not found"}, status_code=500)

    return FileResponse(str(INDEX_PATH), media_type="text/html")

@app.post("/analyze-image")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    require_login_api(request)

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    base64_image = base64.b64encode(contents).decode("utf-8")

    system_msg = (
        "You are an expert estate-sale item identifier and reseller pricing assistant. "
        "Return ONLY valid JSON with these keys: "
        "item_name, brand_or_origin, estimated_value_range, suggested_listing_price, "
        "condition_assumptions, keywords_for_listing, pricing_sources. "
        "No markdown. No code fences."
    )

    user_msg = "Identify and price this item for an estate sale. Return JSON only."

    resp = client.chat.completions.create(
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
        max_tokens=350,
    )

    raw = (resp.choices[0].message.content or "").strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    try:
        return JSONResponse(json.loads(cleaned))
    except Exception:
        return JSONResponse({"error": "Model did not return valid JSON", "raw": raw})
