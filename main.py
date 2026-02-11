from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer

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
# Function: Translate Hindi text to English
# -----------------------------
def translate_hi_to_en(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = mt_model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# -----------------------------
# Run ASR + Translation on an audio file
# -----------------------------
audio_path = "pavan.wav"  # put your wav file in same folder

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
