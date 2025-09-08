#!/usr/bin/env bash
set -euo pipefail
# download_all_de.sh
# Bequemer Downloader für deutsche STT-Modelle.

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

DEST="${1:-./models}"

echo "==> Download destination: $DEST"
mkdir -p "$DEST"

# Whisper (openai-whisper) & Faster-Whisper & HF wav2vec2-de & Vosk-de
python download_models.py \
  --dest "$DEST" \
  --whisper-models medium \
  --faster-whisper-models medium \
  --hf-models jonatasgrosman/wav2vec2-large-xlsr-53-german \
  --vosk-url "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip"

cat <<EOF

Done. Models placed under: $DEST

Included:
 - Whisper:           medium          (openai-whisper)
 - Faster-Whisper:    medium          (Systran/faster-whisper-medium)
 - Hugging Face:      wav2vec2-de     (jonatasgrosman/wav2vec2-large-xlsr-53-german)
 - Vosk:              de-small 0.15   (offline)

Next:
  bash run_de_compare.sh ./wavs ./results_DE_compare
EOF
