from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Tuple
from uuid import uuid4
import requests
import os
import whisper
from utils import print_model_info
from time import time
app = FastAPI()

print(">>> FastAPI initialized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("large-v3-turbo") 
# tiny, base, small, medium, large-v1, large-v2, large-v3, large, turbo or "large-v3-turbo"

# model = whisper.load_model("large")      # Uses 80 mel bins - works with your code
# model = whisper.load_model("large-v2")   # Also uses 80 mel bins - works with your code  
# model = whisper.load_model("large-v3")   # Uses 128 mel bins - causes your error

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
        audio_path = f"/tmp/{uuid4().hex}_{file.filename}"
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

@app.post("/detectLanguage")
async def detectLanguage(
    file: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None)
):
    audio_path, _ = await process_audio_input(file, audio_url)
    
    print_model_info(model)

    try:
        # Load and preprocess audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)  # trims/pads to 30s for language detection

        # Create mel spectrogram
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        # Detect the language
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        print(f"Detected language: {detected_lang}")

        return {"language": detected_lang, "confidence": probs[detected_lang]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio for language detection: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.post("/transcribe")
async def transcribe(
    file: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None)
):
    audio_path, _ = await process_audio_input(file, audio_url)

    # Transcribe using your model
    try:
        start_time = time()
        result = model.transcribe(audio_path)
        elapsed_time = time() - start_time 
        return {"text": result["text"],'elapsed_time':elapsed_time}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
