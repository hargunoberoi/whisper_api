import whisper 
import tempfile
import subprocess

def convert_to_wav(input_file,speedup=None):
    """Convert any audio format to 16kHz mono WAV"""

    # Create temp file for converted audio
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()

    # Use ffmpeg to convert
    ffmpeg_cmd = [
    "ffmpeg",
    "-i", input_file,
    ]
    if speedup:  # e.g. speedup=2.0
        ffmpeg_cmd += ["-filter:a", f"atempo={speedup}"]
    ffmpeg_cmd += [
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-y",
        temp_wav.name
    ]
    # ffmpeg_cmd = [
    #     'ffmpeg',
    #     '-i', input_file,
    #     '-ar', '16000',      # 16kHz sample rate
    #     '-ac', '1',          # Mono
    #     '-c:a', 'pcm_s16le', # 16-bit PCM
    #     '-y',                # Overwrite output
    #     temp_wav.name
    # ]



    try:
        subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return temp_wav.name
    except subprocess.CalledProcessError:
        raise Exception(f"Failed to convert {input_file} to WAV. Is ffmpeg installed?")



def print_model_info(model):
    """Safely print Whisper model information"""
    try:
        # Print basic model information
        print(f"Model dimensions: {model.dims}", flush=True)
        print(f"Mel frequency bins (n_mels): {model.dims.n_mels}", flush=True)
        print(f"Vocabulary size (n_vocab): {model.dims.n_vocab}", flush=True)
        print(f"Audio context (n_audio_ctx): {model.dims.n_audio_ctx}", flush=True)
        print(f"Text context (n_text_ctx): {model.dims.n_text_ctx}", flush=True)
        print(f"Audio state dim (n_audio_state): {model.dims.n_audio_state}", flush=True)
        print(f"Text state dim (n_text_state): {model.dims.n_text_state}", flush=True)
        print(f"Audio heads (n_audio_head): {model.dims.n_audio_head}", flush=True)
        print(f"Text heads (n_text_head): {model.dims.n_text_head}", flush=True)
        # Print model device and data type
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        # Check if model is multilingual
        print(f"Is multilingual: {model.is_multilingual}")
        # Get available models for reference
        print(f"Available models: {whisper.available_models()}")
        # Count actual layers in the model
        encoder_layers = len(model.encoder.blocks) if hasattr(model, 'encoder') else 'N/A'
        decoder_layers = len(model.decoder.blocks) if hasattr(model, 'decoder') else 'N/A'

        print(f"Encoder layers: {encoder_layers}", flush=True)
        print(f"Decoder layers: {decoder_layers}", flush=True)
    except AttributeError as e:
        print(f"Error accessing model attribute: {e}", flush=True)

