#!/usr/bin/env python3
"""
download_models.py
Download/cache models for different STT engines so your first benchmark run is fast and offline-friendly.

Examples:
  # Whisper + Faster-Whisper + HF wav2vec2 (German) + Vosk (German small)
  python download_models.py \
    --whisper-models medium \
    --faster-whisper-models medium \
    --hf-models jonatasgrosman/wav2vec2-large-xlsr-53-german \
    --vosk-url https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip \
    --dest ./models

  # Only Hugging Face model
  python download_models.py --hf-models openai/whisper-small --dest ./models

Notes:
- Whisper/openai-whisper models are pulled implicitly by whisper.load_model() into ~/.cache/whisper.
  This script also copies them into --dest for portability.
- Faster-Whisper and HF models are cached via huggingface_hub.snapshot_download().
- Vosk and whisper.cpp accept direct URLs (zip/bin). We'll download and unpack if needed.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def download_whisper(models: List[str], dest: Path):
    if not models:
        return []
    print(f"[whisper] downloading: {models}")
    try:
        import whisper  # openai-whisper
    except Exception as e:
        print("[whisper] ERROR: openai-whisper is not installed. pip install openai-whisper", file=sys.stderr)
        raise

    paths = []
    for m in models:
        try:
            whisper.load_model(m)  # uses global cache (~/.cache/whisper)
        except Exception as e:
            print(f"[whisper] WARN loading '{m}': {e}", file=sys.stderr)
            continue
        cache_dir = Path.home() / ".cache" / "whisper"
        target = ensure_dir(dest / "whisper" / m)
        if cache_dir.exists():
            for f in cache_dir.glob(f"{m}*"):
                try:
                    shutil.copy2(f, target / f.name)
                except Exception:
                    pass
        paths.append(str(target))
    return paths

def snapshot_hf(repo_id: str, dest_dir: Path) -> str:
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=None,
    )
    return local_dir

def download_faster_whisper(models: List[str], dest: Path):
    """
    faster-whisper models live under Hugging Face org 'Systran', e.g.:
      Systran/faster-whisper-medium
    """
    if not models:
        return []
    print(f"[faster-whisper] downloading: {models}")
    out = []
    for m in models:
        repo = f"Systran/faster-whisper-{m}"
        try:
            local = snapshot_hf(repo, dest / "faster_whisper" / m)
            out.append(local)
        except Exception as e:
            print(f"[faster-whisper] WARN '{repo}': {e}", file=sys.stderr)
    return out

def download_hf_models(models: List[str], dest: Path):
    if not models:
        return []
    print(f"[hf] downloading: {models}")
    out = []
    for repo in models:
        try:
            local = snapshot_hf(repo, dest / "hf" / repo.replace('/', '__'))
            out.append(local)
        except Exception as e:
            print(f"[hf] WARN '{repo}': {e}", file=sys.stderr)
    return out

def _download_file(url: str, dest_file: Path):
    import requests
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

def download_vosk(url: Optional[str], dest: Path):
    if not url:
        return None
    ensure_dir(dest / "vosk")
    print(f"[vosk] downloading: {url}")
    import zipfile, io, requests
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(dest / "vosk")
    roots = [p for p in (dest / "vosk").iterdir() if p.is_dir()]
    path = str(roots[0] if roots else dest / "vosk")
    print(f"[vosk] extracted to: {path}")
    return path

def download_whisper_cpp(url: Optional[str], dest: Path):
    if not url:
        return None
    ensure_dir(dest / "whisper_cpp")
    print(f"[whisper.cpp] downloading: {url}")
    import requests
    target = dest / "whisper_cpp" / Path(url).name
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
    print(f"[whisper.cpp] saved: {target}")
    return str(target)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=str, default="./models", help="Destination directory for downloads")
    ap.add_argument("--whisper-models", nargs="*", default=[], help="openai-whisper model names (e.g., tiny base small medium large-v3)")
    ap.add_argument("--faster-whisper-models", nargs="*", default=[], help="faster-whisper sizes (e.g., tiny base small medium large-v2)")
    ap.add_argument("--hf-models", nargs="*", default=[], help="Hugging Face repo IDs (e.g., jonatasgrosman/wav2vec2-large-xlsr-53-german)")
    ap.add_argument("--vosk-url", type=str, default=None, help="URL to a Vosk model .zip (e.g., German small)")
    ap.add_argument("--whispercpp-url", type=str, default=None, help="URL to a whisper.cpp GGML/GGUF model file")
    args = ap.parse_args()

    dest = ensure_dir(Path(args.dest))

    paths = {}
    try:
        if args.whisper_models:
            paths["whisper"] = download_whisper(args.whisper_models, dest)
    except Exception as e:
        print(f"[whisper] WARN: {e}", file=sys.stderr)

    try:
        if args.faster_whisper_models:
            paths["faster_whisper"] = download_faster_whisper(args.faster_whisper_models, dest)
    except Exception as e:
        print(f"[faster-whisper] WARN: {e}", file=sys.stderr)

    try:
        if args.hf_models:
            paths["hf"] = download_hf_models(args.hf_models, dest)
    except Exception as e:
        print(f"[hf] WARN: {e}", file=sys.stderr)

    try:
        if args.vosk_url:
            paths["vosk"] = download_vosk(args.vosk_url, dest)
    except Exception as e:
        print(f"[vosk] WARN: {e}", file=sys.stderr)

    try:
        if args.whispercpp_url:
            paths["whisper_cpp"] = download_whisper_cpp(args.whispercpp_url, dest)
    except Exception as e:
        print(f"[whisper.cpp] WARN: {e}", file=sys.stderr)

    print("\nDownloaded/Prepared paths:")
    for k, v in paths.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    main()
