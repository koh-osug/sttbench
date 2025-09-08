# STT Benchmark (WAV) – Deutsch

Dieses Projekt hilft dir, **mehrere Open‑Source Speech‑to‑Text (STT)‑Engines** auf einem Ordner mit **WAV‑Dateien** laufen zu lassen und die Qualität **vergleichbar** zu machen (WER/CER).

Unterstützte Engines:
- 
- **Whisper** (openai‑whisper, lokal)
- **Faster‑Whisper** (CTranslate2)
- **Vosk** (offline)
- **Hugging Face** (z.B. wav2vec2‑de Pipeline)
- **whisper.cpp** (externes Binary; optional)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Modelle (Deutsch)

```bash
./download_all_de.sh ./models
```
## Wavs Erstellen

~~~shell
./run_records.sh
~~~

## Vergleichslauf (Deutsch)

```bash
./run_de_compare.sh ./wavs ./results_DE_compare
python3 analyze_results.py ./results_DE_compare
```

# Tools

## Vosk Permanent Ausführen

~~~shell
python3 vosk_transcribe.py --model models/vosk/vosk-model-small-de-0.15
~~~



