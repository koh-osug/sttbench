#!/usr/bin/env bash
set -euo pipefail
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"
AUDIO_DIR="${1:-./wavs}"
OUT_DIR="${2:-./results_DE_compare}"
python stt_benchmark.py \
  --audio-dir "$AUDIO_DIR" \
  --engines whisper faster_whisper hf_wav2vec2 vosk \
  --whisper.model models/whisper/medium/medium.pt \
  --vosk.model models/vosk/vosk-model-small-de-0.15 \
  --faster_whisper.model models/faster_whisper/medium \
  --faster_whisper.device auto \
  --faster_whisper.compute_type int8_float32 \
  --hf.model models/hf/jonatasgrosman__wav2vec2-large-xlsr-53-german \
  --language de \
  --out-dir "$OUT_DIR"
echo "Done -> $OUT_DIR"
