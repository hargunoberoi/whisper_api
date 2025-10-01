import subprocess
import sys
import os
from pathlib import Path
import os 

def process_audio(wav_file, model_path: str):
    """
    Processes an audio file using a specified model and returns the processed string.

    :param wav_file: Path to the WAV file
    :param model_name: Name of the model to use
    :return: Processed string output from the audio processing
    :raises: Exception if an error occurs during processing
    """

    model = Path(model_path)

    # Check if the file exists
    if not os.path.exists(model):
        raise FileNotFoundError(f"Model file not found: {model}")

    if not os.path.exists(wav_file):
        raise FileNotFoundError(f"WAV file not found: {wav_file}")

    script_dir = Path(__file__).parent
    binary_dir = script_dir / "whisper-cli"
    full_command = f"{binary_dir} -m {model} -f {wav_file} -otxt"

    # Execute the command
    process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Get the output and error (if any)
    output, error = process.communicate()

    # Check return code for actual errors
    if process.returncode != 0:
        raise Exception(f"Error processing audio: {error.decode('utf-8')}")

    # Read the generated .txt file
    txt_file = f"{wav_file}.txt"
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Transcription output file not found: {txt_file}")

    with open(txt_file, 'r') as f:
        processed_str = f.read().strip()

    # Clean up the .txt file
    os.remove(txt_file)

    processed_str = processed_str.replace('[BLANK_AUDIO]', '').strip()

    return processed_str

def main():
    pass

if __name__ == "__main__":
    main()
