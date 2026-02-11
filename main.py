from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import pyttsx3  # Simple, works offline, no extra models needed

# -----------------------------
# Load ASR model (Whisper)
# -----------------------------
print("Loading Whisper ASR model...")
asr_model = WhisperModel("small", device="cpu", compute_type="int8")

# -----------------------------
# Load Translation model (Hindi -> English)
# -----------------------------
print("Loading Translation model...")
model_name = "Helsinki-NLP/opus-mt-hi-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
mt_model = MarianMTModel.from_pretrained(model_name)

# -----------------------------
# Initialize Text-to-Speech
# -----------------------------
print("Initializing TTS engine...")
tts_engine = pyttsx3.init()

# Optional: Configure voice properties
tts_engine.setProperty('rate', 150)    # Speed (default ~200)
tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Optional: Set voice to female/male (uncomment to use)
# voices = tts_engine.getProperty('voices')
# tts_engine.setProperty('voice', voices[1].id)  # 0=male, 1=female (typically)

# -----------------------------
# Function: Translate Hindi text to English
# -----------------------------
def translate_hi_to_en(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = mt_model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# -----------------------------
# Run ASR + Translation on an audio file
# -----------------------------
audio_path = "kolla.wav"

print("Running ASR...")
segments, info = asr_model.transcribe(audio_path, language="hi")

full_text = ""
for segment in segments:
    full_text += segment.text + " "

print("\nHindi Transcript:")
print(full_text.strip())

print("\nTranslating...")
translated_text = translate_hi_to_en(full_text)

print("\nEnglish Translation:")
print(translated_text)

# -----------------------------
# Play translated text as speech
# -----------------------------
print("\nPlaying translated audio...")
tts_engine.say(translated_text)
tts_engine.runAndWait()
print("✓ Audio playback complete!")
