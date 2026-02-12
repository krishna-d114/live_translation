from faster_whisper import WhisperModel
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
)
from gtts import gTTS
import pyttsx3
import sounddevice as sd
import scipy.io.wavfile as wav
import torch
import tempfile

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------
# ASR MODELS
#   English : whisper "small"  (generic, great for English)
#   Hindi   : vasista22/whisper-hindi-small
#             (whisper-small fine-tuned on Hindi by IIT Madras)
# ---------------------------------------------------------------
print("Loading English ASR model (Whisper small)...")
from faster_whisper import WhisperModel
asr_en = WhisperModel("small", device="cpu", compute_type="int8")

print("Loading Hindi ASR model (whisper-hindi-small)...")
_device     = "cuda" if torch.cuda.is_available() else "cpu"
_dtype      = torch.float16 if torch.cuda.is_available() else torch.float32
_hi_model   = AutoModelForSpeechSeq2Seq.from_pretrained(
    "vasista22/whisper-hindi-small",
    dtype=_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
_hi_model.to(_device)
_hi_proc    = AutoProcessor.from_pretrained("vasista22/whisper-hindi-small")
asr_hi_pipe = pipeline(
    "automatic-speech-recognition",
    model=_hi_model,
    tokenizer=_hi_proc.tokenizer,
    feature_extractor=_hi_proc.feature_extractor,
    dtype=_dtype,
    device=_device,
)
print("Hindi ASR model loaded!")

# ---------------------------------------------------------------
# TRANSLATION MODEL
#   facebook/nllb-200-distilled-600M
#   Much better than Helsinki opus-mt for Hindi ↔ English.
#   Handles proper nouns, names, and natural sentences correctly.
# ---------------------------------------------------------------
print("Loading NLLB translation model (facebook/nllb-200-distilled-600M)...")
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = NllbTokenizer.from_pretrained(NLLB_MODEL)
nllb_model     = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
print("Translation model loaded!")

# NLLB language codes
NLLB_LANG = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
}

# ---------------------------------------------------------------
# English TTS  — pyttsx3 (offline, fast, English only)
# ---------------------------------------------------------------
print("Initializing English TTS...")
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)


# ---------------------------------------------------------------
# TRANSCRIBE
# ---------------------------------------------------------------
def transcribe_audio(audio_path, lang):
    """
    Transcribe audio using the right ASR model per language.
      en -> faster-whisper small (generic)
      hi -> whisper-hindi-small (fine-tuned on Hindi)
    """
    if lang == "en":
        segments, _ = asr_en.transcribe(audio_path, language="en")
        return "".join(seg.text for seg in segments).strip()
    elif lang == "hi":
        result = asr_hi_pipe(audio_path)
        return result["text"].strip()
    else:
        raise ValueError(f"Unsupported language: {lang}")


# ---------------------------------------------------------------
# TRANSLATE  (NLLB-200)
# ---------------------------------------------------------------
def translate_text(text, src_lang, tgt_lang):
    """
    Translate text using Facebook NLLB-200 distilled 600M.
    Significantly better than Helsinki opus-mt for Hindi<->English.
    """
    src_code = NLLB_LANG[src_lang]
    tgt_code = NLLB_LANG[tgt_lang]

    nllb_tokenizer.src_lang = src_code
    inputs = nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    forced_bos_token_id = nllb_tokenizer.convert_tokens_to_ids(tgt_code)
    output_tokens = nllb_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=256,
    )
    return nllb_tokenizer.decode(output_tokens[0], skip_special_tokens=True)


# ---------------------------------------------------------------
# SPEAK
#   English -> pyttsx3 (offline)
#   Hindi   -> gTTS (needs internet, actually supports Devanagari)
# ---------------------------------------------------------------
def speak_text(text, lang):
    if lang == "en":
        tts_engine.say(text)
        tts_engine.runAndWait()
    elif lang == "hi":
        try:
            tts = gTTS(text=text, lang='hi', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tmp_path = f.name
            tts.save(tmp_path)
            if os.name == "nt":
                os.system(f'start /wait wmplayer "{tmp_path}"')
            elif hasattr(os, 'uname') and os.uname().sysname == "Darwin":
                os.system(f'afplay "{tmp_path}"')
            else:
                ret = os.system(f'mpg123 -q "{tmp_path}" 2>/dev/null')
                if ret != 0:
                    os.system(f'ffplay -nodisp -autoexit "{tmp_path}" 2>/dev/null')
            os.remove(tmp_path)
        except Exception as e:
            print(f"Hindi TTS error: {e}")
            print(f"  Translation was: {text}")


# ---------------------------------------------------------------
# RECORD FROM MIC
# ---------------------------------------------------------------
def record_from_microphone(duration=5, sample_rate=16000):
    print(f"\nRecording for {duration} seconds... Speak now!")
    try:
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()
        print("Recording complete!")
        tmp_wav = tempfile.mktemp(suffix=".wav")
        wav.write(tmp_wav, sample_rate, audio_data)
        return tmp_wav
    except KeyboardInterrupt:
        print("\nRecording cancelled.")
        return None
    except Exception as e:
        print(f"Microphone error: {e}")
        return None


# ---------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------
def process_audio(audio_path, source_lang, target_lang):
    print(f"\n{'='*50}")
    print(f"Processing: {audio_path}")
    print(f"{'='*50}")

    # Step 1: Transcribe
    lang_name = "Hindi" if source_lang == "hi" else "English"
    print(f"\nRunning ASR ({lang_name})...")
    full_text = transcribe_audio(audio_path, source_lang)

    if not full_text:
        print("No speech detected. Please try again.")
        return

    print(f"\n{lang_name} Transcript:")
    print("-" * 50)
    print(full_text)

    # Step 2: Translate
    print(f"\nTranslating {source_lang.upper()} -> {target_lang.upper()}...")
    translated_text = translate_text(full_text, source_lang, target_lang)

    target_lang_name = "English" if target_lang == "en" else "Hindi"
    print(f"\n{target_lang_name} Translation:")
    print("-" * 50)
    print(translated_text)

    # Step 3: Speak
    print("\nPlaying translated audio...")
    speak_text(translated_text, target_lang)
    print("Audio playback complete!")


# ---------------------------------------------------------------
# MENU
# ---------------------------------------------------------------
def show_menu():
    print("\n" + "="*50)
    print("   LIVE TRANSLATION SYSTEM  --  GLASSES DEMO")
    print("="*50)
    print("\n-- File Mode --")
    print("1. Hindi -> English   (kolla.wav)")
    print("2. English -> Hindi   (eng_test.wav)")
    print("\n-- Live Mic Mode --")
    print("3. Record Hindi  -> Translate to English")
    print("4. Record English -> Translate to Hindi")
    print("\n5. Exit")
    print("-"*50)
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            return choice
        print("Invalid choice. Please enter 1-5.")


def main():
    file_configs = {
        '1': {'path': 'kolla.wav',    'source_lang': 'hi', 'target_lang': 'en'},
        '2': {'path': 'eng_test.wav', 'source_lang': 'en', 'target_lang': 'hi'},
    }
    mic_configs = {
        '3': {'source_lang': 'hi', 'target_lang': 'en'},
        '4': {'source_lang': 'en', 'target_lang': 'hi'},
    }

    while True:
        choice = show_menu()

        if choice == '5':
            print("\nGoodbye!")
            break

        if choice in file_configs:
            config = file_configs[choice]
            if not os.path.exists(config['path']):
                print(f"\nError: File not found: {config['path']}")
                continue
            try:
                process_audio(config['path'], config['source_lang'], config['target_lang'])
            except Exception as e:
                print(f"\nError: {e}")

        elif choice in mic_configs:
            config = mic_configs[choice]
            dur_input = input("\nRecord duration in seconds [5]: ").strip()
            duration = int(dur_input) if dur_input.isdigit() and int(dur_input) > 0 else 5
            tmp_path = record_from_microphone(duration=duration)
            if tmp_path:
                try:
                    process_audio(tmp_path, config['source_lang'], config['target_lang'])
                except Exception as e:
                    print(f"\nError: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
