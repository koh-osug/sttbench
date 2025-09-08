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
                    text = result.get("text", "").strip()
                    if text:
                        # Clear partial line before printing final
                        if partial_prev:
                            print("\r" + " " * len(partial_prev) + "\r", end="")
                            partial_prev = ""
                        print(text)
                else:
                    if not args.no_partials:
                        partial = json.loads(rec.PartialResult()).get("partial", "").strip()
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
            final = json.loads(rec.FinalResult()).get("text", "").strip()
            if final:
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