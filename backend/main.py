# backend/main.py
# ✅ Estate AI Scanner (Render-ready)
# Phase 2: Company Login (cookie session) + Protect scanner routes
# - /login (GET) shows simple login form
# - /login (POST) sets session if company code/password match
# - /logout clears session
# - /scanner + /analyze-image require login
# - Image analysis calls OpenAI and returns clean JSON

import os
import json
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from openai import OpenAI


# =========================
# Config
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "")

# Company login credentials (env overrides)
COMPANY_CODE = os.getenv("COMPANY_CODE", "demo")
COMPANY_PASSWORD = os.getenv("COMPANY_PASSWORD", "letmein123")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
INDEX_PATH = FRONTEND_DIR / "index.html"

if not OPENAI_API_KEY:
    # Render will still boot, but /analyze-image will fail with a helpful error.
    print("⚠️ OPENAI_API_KEY not set")
if not SECRET_KEY:
    # If SECRET_KEY is missing, SessionMiddleware will be unsafe / broken.
    print("❌ SECRET_KEY not set (add it in Render → Environment Variables)")


# =========================
# App
# =========================
app = FastAPI()

# Cookie-based session (for login)
# NOTE: Requires SECRET_KEY set in Render env vars
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY or "dev-only-change-me")

# If you later host frontend separately, adjust CORS.
# For now, same-domain is ideal.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Auth helpers
# =========================
def is_logged_in(request: Request) -> bool:
    return bool(request.session.get("company_code"))

def require_login(request: Request) -> Optional[RedirectResponse]:
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)
    return None


# =========================
# Simple pages
# =========================
@app.get("/", response_class=JSONResponse)
def home():
    return {"message": "Estate AI Scanner is LIVE 🚀", "login": "/login", "scanner": "/scanner"}

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    # If already logged in, go straight to scanner
    if is_logged_in(request):
        return RedirectResponse(url="/scanner", status_code=302)

    msg = request.query_params.get("msg", "")
    safe_msg = (msg or "").replace("<", "&lt;").replace(">", "&gt;")

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Estate AI Scanner - Login</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 520px; margin: 40px auto; padding: 0 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; }}
    h1 {{ margin: 0 0 10px; }}
    label {{ display:block; margin-top: 10px; font-weight: 700; }}
    input {{ width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; }}
    button {{ margin-top: 14px; padding: 10px 14px; border-radius: 8px; border: 1px solid #2b7cff; background:#2b7cff; color:#fff; font-weight: 800; cursor:pointer; }}
    .sub {{ color:#555; margin: 0 0 8px; }}
    .msg {{ color:#b00020; font-weight: 800; margin-top: 10px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Company Login</h1>
    <p class="sub">Enter your company access code to use the scanner.</p>

    <form method="post" action="/login">
      <label>Company Code</label>
      <input name="company_code" placeholder="e.g., demo" required />

      <label>Password</label>
      <input name="password" type="password" placeholder="••••••••" required />

      <button type="submit">Log In</button>
    </form>

    {"<div class='msg'>" + safe_msg + "</div>" if safe_msg else ""}
  </div>
</body>
</html>
"""
    return HTMLResponse(html)

@app.post("/login")
def login(company_code: str = Form(...), password: str = Form(...), request: Request = None):
    # Basic check (we’ll replace with real user/company DB later)
    if company_code.strip() == COMPANY_CODE and password == COMPANY_PASSWORD:
        request.session["company_code"] = company_code.strip()
        return RedirectResponse(url="/scanner", status_code=302)

    return RedirectResponse(url="/login?msg=Invalid+company+code+or+password", status_code=302)

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login?msg=Logged+out", status_code=302)

@app.get("/scanner")
def scanner(request: Request):
    # Protect the scanner UI
    gate = require_login(request)
    if gate:
        return gate

    if not INDEX_PATH.exists():
        return JSONResponse(
            {"error": "index.html not found in frontend folder. Expected frontend/index.html"},
            status_code=500,
        )

    # Serve your existing frontend scanner (camera + local history)
    return FileResponse(str(INDEX_PATH), media_type="text/html")


# =========================
# OpenAI result cleanup (strip ```json fences)
# =========================
def _clean_json_text(s: str) -> str:
    """Remove ```json fences if present and trim whitespace."""
    if not s:
        return ""
    s = s.strip()
    s = s.replace("```json", "").replace("```JSON", "")
    s = s.replace("```", "").strip()
    return s


# =========================
# Analyze endpoint (protected)
# =========================
@app.post("/analyze-image")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    # Protect scanning
    gate = require_login(request)
    if gate:
        return gate

    if not OPENAI_API_KEY:
        return JSONResponse({"error": "OPENAI_API_KEY is not set on the server."}, status_code=500)

    contents = await file.read()
    if not contents:
        return JSONResponse({"error": "Empty file upload."}, status_code=400)

    base64_image = base64.b64encode(contents).decode("utf-8")

    # Prompt: force JSON output
    system = (
        "You are an expert reseller and estate-sale pricer. "
        "Return ONLY valid JSON with these exact keys:\n"
        "item_name, brand_or_origin, estimated_value_range, suggested_listing_price, "
        "condition_assumptions, keywords_for_listing, pricing_sources\n"
        "No markdown. No code fences."
    )

    user = (
        "Analyze this item photo and estimate resale value. "
        "Use common online resale context (eBay/Etsy/etc) and be concise."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                },
            ],
            max_tokens=350,
        )

        raw = response.choices[0].message.content or ""
        cleaned = _clean_json_text(raw)

        # Return parsed JSON if possible
        try:
            data = json.loads(cleaned)
            return JSONResponse(data)
        except Exception:
            # If model returns non-JSON, return a useful error payload
            return JSONResponse({"error": "Model did not return valid JSON", "raw": raw}, status_code=200)

    except Exception as e:
        return JSONResponse({"error": f"Server exception: {str(e)}"}, status_code=500)
