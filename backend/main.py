import os
import json
import base64
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# ENV + CLIENT
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()


# =========================
# CORS (OK for early stage)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Serve the scanner page
# =========================
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index.html"

@app.get("/")
def scanner_home():
    # Serves backend/index.html as the homepage on Render
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return JSONResponse(
        {"error": "index.html not found in backend folder. Expected backend/index.html"},
        status_code=500,
    )


# =========================
# AI IMAGE ANALYSIS (JSON)
# =========================
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):

    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    system_prompt = """
You are an expert estate sale appraiser.

Return ONLY valid JSON matching this schema (no markdown, no extra words):

{
  "item_name": "",
  "brand_or_origin": "",
  "estimated_value_range": "",
  "suggested_listing_price": "",
  "condition_assumptions": "",
  "keywords_for_listing": "",
  "pricing_sources": ""
}

Rules:
- Output MUST be JSON only.
- Use short, practical language.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify and price this item."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{file.content_type};base64,{base64_image}"},
                    },
                ],
            },
        ],
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()

    # Try to parse JSON (best effort)
    try:
        data = json.loads(raw)
        # Add filename for convenience
        data["filename"] = file.filename
        return data
    except Exception:
        # If the model returned non-JSON, return it for debugging
        return {
            "error": "Model did not return valid JSON.",
            "raw": raw,
            "filename": file.filename,
        }
