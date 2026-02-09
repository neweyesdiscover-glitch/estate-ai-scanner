from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import base64
from openai import OpenAI
import os
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# ✅ Allow browser + Render communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for early stage
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

############################################################
# ROOT ROUTE
############################################################

@app.get("/")
def home():
    return {"message": "Estate AI Scanner is LIVE 🚀"}

############################################################
# SERVE FRONTEND
############################################################

@app.get("/scanner")
def scanner():
    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "frontend",
        "index.html"
    )
    return FileResponse(file_path)

############################################################
# IMAGE ANALYSIS
############################################################

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
"""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify and price this item."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                ],
            }
        ],
        max_tokens=500,
    )

    raw = response.choices[0].message.content

    return {"raw": raw}
