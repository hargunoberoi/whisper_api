#%%
import whisper_processor
#%%
# get model path from whispercpp
model_path = 'Pathto/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin'
audio_path = './audio/jfk.wav'
# try:
result = whisper_processor.process_audio(audio_path,model_path)
# except Exception as e:
#     print(f"Error: {e}")