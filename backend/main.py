from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env locally (Render uses dashboard env vars; this won't hurt)
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Browser-friendly (safe for early stage)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Estate AI Scanner is LIVE 🚀"}

def _clean_json_text(s: str) -> str:
    """Remove ```json fences if present and trim whitespace."""
    if not s:
        return ""
    s = s.strip()
    s = s.replace("```json", "```").replace("```JSON", "```")
    s = s.replace("```", "").strip()
    return s

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert estate sale appraiser.\n"
                    "Return STRICT JSON only. No markdown. No extra commentary.\n\n"
                    "{\n"
                    '  "item_name": "",\n'
                    '  "brand_or_origin": "",\n'
                    '  "estimated_value_range": "",\n'
                    '  "suggested_listing_price": "",\n'
                    '  "condition_assumptions": "",\n'
                    '  "keywords_for_listing": "",\n'
                    '  "pricing_sources": ""\n'
                    "}\n"
                ),
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
        max_tokens=600,
    )

    raw = response.choices[0].message.content or ""
    cleaned = _clean_json_text(raw)

    # Return REAL JSON
    try:
        return json.loads(cleaned)
    except Exception:
        # Helpful payload for debugging
        return {
            "error": "Model did not return valid JSON",
            "raw": raw,
        }
