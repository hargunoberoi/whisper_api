from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Tuple
from uuid import uuid4
import requests
import os
from whisper_processor import process_audio

model_path = '/Users/h3045/Desktop/aigarage/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin'
app = FastAPI()

print(">>> FastAPI initialized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def process_audio_input(
    file: Optional[UploadFile] = None, 
    audio_url: Optional[str] = None
) -> Tuple[str, str]:
    """
    Helper function to process either file upload or URL download.
    Returns tuple of (audio_path, file_extension)
    """
    if not file and not audio_url:
        raise HTTPException(status_code=400, detail="Either 'file' or 'audio_url' must be provided.")

    if file and audio_url:
        raise HTTPException(status_code=400, detail="Provide only one input: 'file' or 'audio_url', not both.")

    # Process file upload
    if file:
        audio_path = f"./audio/{uuid4().hex}_{file.filename}"
        with open(audio_path, "wb") as f_out:
            f_out.write(await file.read())
        return audio_path, os.path.splitext(file.filename)[1] if file.filename else ""
    else:
        # Process URL download
        try:
            response = requests.get(audio_url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch audio from URL: {str(e)}")

        content_type = response.headers.get("content-type", "")
        if "audio" not in content_type:
            raise HTTPException(status_code=400, detail=f"URL does not point to a valid audio file.")

        ext = content_type.split("/")[-1]
        audio_path = f"/tmp/{uuid4().hex}.{ext}"
        with open(audio_path, "wb") as f_out:
            f_out.write(response.content)
        return audio_path, f".{ext}"



@app.post("/transcribe")
async def transcribe(
    file: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None)
):
    audio_path, _ = await process_audio_input(file, audio_url)

    # Transcribe using your model
    try:
        result = process_audio(audio_path,model_path)
        return {"text": 'haha'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")
    finally:
        pass
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
