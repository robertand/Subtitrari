from faster_whisper import WhisperModel

model_path = "/media/prouser/Work/Play_ground/Subtitrari/whisper-turkish-ct2"
model = WhisperModel(model_path, device="cuda", compute_type="float16")

# Înlocuiește cu un fișier audio real pe care îl ai
audio_file = "data/audio.wav"  # sau calea către un fișier audio existent

try:
    segments, info = model.transcribe(audio_file, language="tr")
    print(f"Language: {info.language}")
    print(f"Probability: {info.language_probability}")
    print("\nSegments:")
    for segment in segments:
        print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")
except Exception as e:
    print(f"Error: {e}")