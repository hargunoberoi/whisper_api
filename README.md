# Whisper API

FastAPI service for audio transcription using Whisper models.

## Implementations

- **main.py** - Uses [OpenAI Whisper](https://github.com/openai/whisper) (Python library)
- **cpp.py** - Uses [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (faster, C++ implementation)

## Features

- Audio transcription (WAV, MP3)
- Automatic format conversion via ffmpeg
- Language detection (main.py only)
- Optional audio speedup

## Quick Start

```bash
# Install dependencies
uv sync

# Run with OpenAI Whisper
uv run uvicorn main:app --reload

# Run with whisper.cpp (faster)
uv run uvicorn cpp:app --reload
```

## API Endpoints

### POST /transcribe
Upload audio file or provide URL for transcription.

**Parameters:**
- `file` (optional): Audio file upload
- `audio_url` (optional): URL to audio file

**Response:**
```json
{
  "text": "transcribed text",
  "elapsed_time": 1.23
}
```

### POST /detectLanguage (main.py only)
Detect language from audio.

## Requirements

- Python 3.11+
- ffmpeg (for format conversion)
- whisper.cpp binary (for cpp.py)

## Performance

whisper.cpp typically provides 2-3x faster inference compared to the Python implementation.
