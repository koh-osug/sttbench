#!/usr/bin/env bash
set -euo pipefail
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"
AUDIO_DIR="${1:-./wavs}"
python record.py \
  --audio-dir "$AUDIO_DIR" \
