````md
# Live Translation MVP (Hindi → English)

This project is a simple MVP pipeline for **Live Translation** using:

- **ASR (Automatic Speech Recognition)**: Speech → Text using Whisper
- **MT (Machine Translation)**: Hindi Text → English Text using MarianMT

Current supported flow:

🎤 Hindi Audio → 📝 Hindi Transcript → 🌍 English Translation

---

## Requirements

- Python 3.9+ (recommended)
- VS Code (recommended)
- Internet connection (first run downloads models)

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-github-repo-link>
cd live_translation
````

---

### 2. Create Virtual Environment (venv)

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
```

---

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

---

### 4. Install Dependencies

```bash
pip install torch transformers sentencepiece accelerate
pip install faster-whisper
pip install sounddevice scipy
```

> Note: The first time you run the project, model files will automatically download.

---

## Project Files

* `main.py` → main inference script
* `hindi_audio.wav` → input audio file (you must add this)

---

## How to Run

### 1. Add Input Audio File

Place your Hindi audio file in the project folder and name it:

```
hindi_audio.wav
```

---

### 2. Run the Script

```bash
python main.py
```

---

## Output Example

The script prints:

* Hindi transcript (from ASR)
* English translation (from MT)

Example format:

```
Hindi Transcript:
<recognized Hindi text>

English Translation:
<translated English text>
```

---

## Notes

* Whisper model used: `small`
* Translation model used: `Helsinki-NLP/opus-mt-hi-en`
* Works best when audio is clear and background noise is minimal.

---

## Next Steps (Planned)

* Add microphone streaming mode (real-time audio input)
* Convert pipeline into FastAPI backend
* Build a barebones phone app to connect to backend
* Add Hindi ↔ English toggle mode

---

```
```
