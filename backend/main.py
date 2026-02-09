from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import base64
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (.env locally; Render uses its own env vars)
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS (ok for early-stage; later restrict to your domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))         # .../backend
FRONTEND_INDEX = os.path.join(BASE_DIR, "..", "frontend", "index.html")

# --- Routes ---
@app.get("/")
def home():
    return {"message": "Estate AI Scanner is LIVE 🚀"}

@app.get("/scanner", response_class=HTMLResponse)
def scanner_page():
    # Serve the frontend UI from /scanner
    if os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    return {"error": "frontend/index.html not found. Expected ../frontend/index.html"}

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

Return STRICT JSON ONLY (no markdown, no backticks), matching this schema:

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

    # Safety: strip any accidental ```json fences then parse
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        # If model returns non-JSON, return a useful error payload
        return {
            "error": "Model did not return valid JSON",
            "raw": raw,
        }
