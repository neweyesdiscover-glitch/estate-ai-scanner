from fastapi.responses import FileResponse
@app.get("/")
def home():
    return FileResponse("index.html")

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# 🔥 THIS FIXES 90% OF NGROK PROBLEMS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # safe for now — later we lock this down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):

    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    prompt = """
You are an estate sale expert.

Return ONLY valid JSON.

{
 "item": "",
 "category": "",
 "condition": "",
 "price_range": "",
 "suggested_price": "",
 "confidence": "",
 "keywords": "",
 "pricing_sources": []
}

Use realistic resale values based on secondary markets like eBay sold listings.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
        max_output_tokens=500,
    )

    raw = response.output_text

    return {"raw": raw}
