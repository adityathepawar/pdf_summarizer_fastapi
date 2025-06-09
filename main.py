from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import PyPDF2
import requests
import os

app = FastAPI()

API_KEY = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
MODEL = "deepseek/deepseek-chat:free"

def extract_text_from_pdf(file_stream):
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        raise Exception("Failed to extract text from PDF.") from e

def call_openrouter(messages, api_key=API_KEY, model=MODEL):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages
    }
    response = requests.post(url, headers=headers, json=payload, verify=False)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")

@app.post("/summarize/")
async def summarize_pdf(file: UploadFile = File(...)):
    try:
        text = extract_text_from_pdf(file.file)
        summary_prompt = [{"role": "user", "content": f"Summarize this PDF content:\n\n{text[:8000]}"}]
        summary = call_openrouter(summary_prompt)
        return JSONResponse({"summary": summary})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/chat/")
async def chat_with_pdf(question: str = Form(...), context: str = Form(...)):
    try:
        messages = [
            {"role": "system", "content": f"Answer based on this PDF context:\n\n{context[:8000]}"},
            {"role": "user", "content": question}
        ]
        answer = call_openrouter(messages)
        return JSONResponse({"response": answer})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
