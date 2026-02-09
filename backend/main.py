from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import base64
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path

# ✅ Load environment variables (local dev only; Render uses Environment tab)
load_dotenv()

# ✅ OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# ✅ Allow browser → backend calls (safe for early stage apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Find repo root reliably (works on Render + locally)
# This file is: <repo>/backend/main.py
REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_INDEX = REPO_ROOT / "frontend" / "index.html"

# ✅ ROOT ROUTE (simple health check)
@app.get("/")
def home():
    return {"message": "Estate AI Scanner is LIVE 🚀"}

# ✅ SCANNER UI ROUTE (serves the HTML)
@app.get("/scanner", response_class=HTMLResponse)
def scanner():
    if not FRONTEND_INDEX.exists():
        return json.dumps(
            {
                "error": "index.html not found in frontend folder. Expected frontend/index.html",
                "looked_for": str(FRONTEND_INDEX),
            }
        )
    return FRONTEND_INDEX.read_text(encoding="utf-8")

# ✅ Helper: clean up accidental ```json fences before parsing
def _clean_json_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.replace("```json", "").replace("```JSON", "")
    s = s.replace("```", "").strip()
    return s

# ✅ IMAGE ANALYSIS ROUTE (frontend posts image here)
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are an expert estate sale appraiser.

Identify the item and return STRICT JSON:

{
 "item_name": "",
 "brand_or_origin": "",
 "estimated_value_range": "",
 "suggested_listing_price": "",
 "condition_assumptions": "",
 "keywords_for_listing": "",
 "pricing_sources": ""
}
""",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify and price this item."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        max_tokens=500,
    )

    raw = response.choices[0].message.content or ""
    cleaned = _clean_json_text(raw)

    # ✅ Return parsed JSON (so your frontend can render fields cleanly)
    try:
        return json.loads(cleaned)
    except Exception:
        # ✅ If model returns non-JSON, return debug info instead of crashing
        return {
            "error": "Model did not return valid JSON",
            "raw": raw,
        }
