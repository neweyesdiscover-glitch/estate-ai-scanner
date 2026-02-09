# backend/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from openai import OpenAI

import base64
import json
import os

# ✅ Load environment variables (local dev uses .env; Render uses Environment Variables)
load_dotenv()

# ✅ OpenAI client (Render: set OPENAI_API_KEY in Render dashboard)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# ✅ VERY important for browser + Render communication (CORS)
# NOTE: For production, lock this down to your domain instead of "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ROOT ROUTE (quick health check)
@app.get("/")
def home():
    return {"message": "Estate AI Scanner is LIVE 🚀"}

# ✅ SCANNER UI ROUTE (serves your frontend page at /scanner)
# Expects your file at: frontend/index.html (repo root)
@app.get("/scanner")
def scanner():
    index_path = os.path.join("frontend", "index.html")
    if not os.path.exists(index_path):
        # Helpful error if file is missing / in wrong folder
        return {
            "error": "index.html not found in frontend folder. Expected frontend/index.html"
        }
    return FileResponse(index_path)

# ✅ Helper: remove ```json fences if the model wraps the JSON in code blocks
def _clean_json_text(s: str) -> str:
    """Remove ```json fences if present and trim whitespace."""
    if not s:
        return ""
    s = s.strip()
    s = s.replace("```json", "").replace("```JSON", "")
    s = s.replace("```", "").strip()
    return s

# ✅ IMAGE ANALYSIS ROUTE (receives uploaded image and returns structured JSON)
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    # ✅ Read uploaded file and convert to base64 for OpenAI vision input
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    # ✅ Call OpenAI (vision) and ask for STRICT JSON
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are an expert estate sale appraiser.

Identify the item and return STRICT JSON (no commentary, no markdown):

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

    # ✅ Return parsed JSON if possible; otherwise return a helpful payload
    try:
        return json.loads(cleaned)
    except Exception:
        return {
            "error": "Model did not return valid JSON",
            "raw": raw,
        }
