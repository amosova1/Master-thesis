import torch
from transformers import pipeline
from pydub import AudioSegment
import os

MODEL_NAME = "../whisper-small-sk"
M4A_FILE = "pokus2.m4a"
TEMP_WAV_FILE = "temp_audio_to_transcribe.wav"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Kontrola existencie lokálneho modelu a audio súboru
if not os.path.exists(MODEL_NAME):
    print(f"Error: Lokálny priečinok s modelom '{MODEL_NAME}' nebol nájdený.")
    exit()

if not os.path.exists(M4A_FILE):
    print(f"Error: Súbor '{M4A_FILE}' nebol nájdený.")
    exit()

print(f"Konvertovanie {M4A_FILE} do WAV formátu pomocou pydub...")

try:
    # Načítanie M4A súboru pomocou pydub (vyžaduje ffmpeg)
    audio = AudioSegment.from_file(M4A_FILE, format="m4a")

    # Export do dočasného WAV súboru s požadovaným vzorkovaním (16kHz)
    audio.export(
        TEMP_WAV_FILE,
        format="wav",
        parameters=["-acodec", "pcm_s16le", "-ar", "16000"]
    )
    print(f"Konverzia úspešná. Načítavam model z lokálnej cesty: {MODEL_NAME} na zariadení: {DEVICE}")

    # Načítanie a spustenie Whisper pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        device=DEVICE,
        dtype=torch.float16 if DEVICE != "cpu" and torch.cuda.is_available() else torch.float32
    )

    print(f"Prepis zvukového súboru")

    result = pipe(TEMP_WAV_FILE)

    print("-" * 30)
    print("Prepis:")
    print(result["text"])
    print("-" * 30)

except Exception as e:
    print(f"Error: {e}")

finally:
    # Odstránenie dočasného WAV súboru po dokončení
    if os.path.exists(TEMP_WAV_FILE):
        os.remove(TEMP_WAV_FILE)
        print(f"🗑️ Dočasný súbor {TEMP_WAV_FILE} bol odstránený.")