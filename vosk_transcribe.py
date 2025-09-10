#!/usr/bin/env python3
# live_vosk_de.py
# Continuous German speech-to-text with Vosk using the microphone.

import argparse
import json
import queue
import sys
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel


def list_devices_and_exit():
    print(sd.query_devices())
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description="Live German STT with Vosk (microphone).")
    ap.add_argument("--model", required=True, help="Path to Vosk German model directory (unzipped).")
    ap.add_argument("--samplerate", type=int, default=16000, help="Sample rate (Hz), typically 16000 for Vosk models.")
    ap.add_argument("--device", type=str, default=None, help="Input device index/name for sounddevice. Use --list-devices to inspect.")
    ap.add_argument("--buffer", type=int, default=8000, help="Input blocksize (frames). Lower for lower latency.")
    ap.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    ap.add_argument("--no-partials", action="store_true", help="Disable printing partial results.", default=False)
    # Filtering options to reduce false single-word finals like "nun"
    ap.add_argument("--min-words", type=int, default=1, help="Minimum words required to print a final result (after filtering).")
    ap.add_argument(
        "--min-avg-conf",
        type=float,
        default=0.6,
        help="Minimum average word confidence for short (<=2 words) finals to be printed.",
    )
    ap.add_argument(
        "--ignore-singletons",
        type=str,
        default="nun,äh,hm,ah,öh,öhm,öhm,ähm,und",
        help="Comma-separated list of single words to ignore when they appear alone as a final result.",
    )
    ap.add_argument(
        "--no-filter",
        action="store_true",
        default=False,
        help="Disable final-result filtering (print everything Vosk returns).",
    )
    args = ap.parse_args()

    if args.list_devices:
        list_devices_and_exit()

    model_path = Path(args.model)
    if not model_path.exists() or not model_path.is_dir():
        print(f"Model directory not found: {model_path}", file=sys.stderr)
        sys.exit(2)

    # Reduce Vosk logging noise
    SetLogLevel(-1)

    print(f"Loading Vosk model from: {model_path} ...")
    model = Model(str(model_path))
    print("Model loaded. Listening... Press Ctrl+C to stop.\n")

    # Build ignore set once
    ignore_singletons = set(w.strip().lower() for w in args.ignore_singletons.split(",") if w.strip())

    q = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            # You could log status messages if needed
            pass
        # RawInputStream passes a CFFI buffer; convert to Python-owned bytes
        try:
            data = bytes(indata)
        except Exception:
            # Fallback for safety if backend changes (e.g., NumPy array)
            data = indata.tobytes() if hasattr(indata, "tobytes") else bytes(memoryview(indata))
        q.put(data)

    # Create recognizer
    rec = KaldiRecognizer(model, args.samplerate)
    rec.SetWords(True)

    # Helper to decide whether to print a final result
    def should_print_final(result_obj, text_str):
        if args.no_filter:
            return bool(text_str)

        text_norm = text_str.strip().lower()
        if not text_norm:
            return False

        words = text_norm.split()
        n_words = len(words)

        # Enforce minimum words if configured (>1 to be effective)
        if n_words < max(1, args.min_words):
            # Allow printing if confidence is clearly good (for true one-word commands),
            # else suppress; we check conf below.
            pass

        # Compute average confidence if available
        word_items = result_obj.get("result") or []
        if word_items:
            # Some models may provide "conf" per word; fall back to printing if missing
            confs = [wi.get("conf") for wi in word_items if isinstance(wi.get("conf"), (int, float))]
            if confs:
                avg_conf = sum(confs) / len(confs)
                # Stricter on very short outputs
                if n_words <= 2 and avg_conf < args.min_avg_conf:
                    return False

        # Final minimal length check
        if n_words < max(1, args.min_words):
            return False

        return True

    # Helper to remove ignored words from any position
    def filter_ignored_words(text: str) -> str:
        if not text:
            return text
        words = text.split()
        kept = [w for w in words if w.lower() not in ignore_singletons]
        return " ".join(kept)

    # Open microphone stream
    try:
        with sd.RawInputStream(
            samplerate=args.samplerate,
            blocksize=args.buffer,
            device=args.device,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            partial_prev = ""
            while True:
                data = q.get()
                if data is None:
                    continue
                # data is int16 bytes already (RawInputStream + dtype="int16")
                if rec.AcceptWaveform(data):
                    # Finalized segment
                    result = json.loads(rec.Result())
                    text_raw = result.get("text", "").strip()
                    text = filter_ignored_words(text_raw)
                    if text and should_print_final(result, text):
                        # Clear partial line before printing final
                        if partial_prev:
                            print("\r" + " " * len(partial_prev) + "\r", end="")
                            partial_prev = ""
                        print(text)
                else:
                    if not args.no_partials:
                        partial_obj = json.loads(rec.PartialResult())
                        partial_raw = partial_obj.get("partial", "").strip()
                        partial = filter_ignored_words(partial_raw)
                        if partial:
                            # Overwrite the same line for partials
                            line = f"[partial] {partial}"
                            # Pad/clear previous line if needed
                            pad = max(0, len(partial_prev) - len(line))
                            print("\r" + line + (" " * pad), end="")
                            partial_prev = line
                        else:
                            # Clear partial line if recognizer returns empty
                            if partial_prev:
                                print("\r" + " " * len(partial_prev) + "\r", end="")
                                partial_prev = ""
    except KeyboardInterrupt:
        # Print any remaining final result
        try:
            final_obj = json.loads(rec.FinalResult())
            final_raw = final_obj.get("text", "").strip()
            final = filter_ignored_words(final_raw)
            if final and should_print_final(final_obj, final):
                print()
                print(final)
        except Exception:
            pass
        print("\nStopped.")
    except Exception as e:
        print(f"Audio error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()