#%%
import whisper_processor
#%%
model_path = '/Users/h3045/Desktop/aigarage/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin'
audio_path = './audio/jfk.wav'
# try:
result = whisper_processor.process_audio(audio_path,model_path)
# except Exception as e:
#     print(f"Error: {e}")