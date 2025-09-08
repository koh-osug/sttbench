#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import datetime as dt
import os
import sys
import threading
import queue
import time
from pathlib import Path

# Lazy imports to provide a clear error if missing
try:
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
except Exception as e:
    missing = []
    try:
        import sounddevice  # noqa: F401
    except Exception:
        missing.append("sounddevice")
    try:
        import soundfile  # noqa: F401
    except Exception:
        missing.append("soundfile")
    try:
        import numpy  # noqa: F401
    except Exception:
        missing.append("numpy")
    if missing:
        sys.stderr.write(
            "Fehlende Python-Pakete: " + ", ".join(missing) + "\n"
            "Bitte stelle sicher, dass diese Pakete in deiner Umgebung installiert sind.\n"
        )
        sys.exit(2)
    else:
        # Falls ein anderer Importfehler auftritt:
        raise

# OS-abhängige Einzel-Tastendruck-Erkennung (Enter / Esc)
IS_WIN = os.name == "nt"
if IS_WIN:
    import msvcrt
else:
    import termios
    import tty
    import select

ESC_CODE = b"\x1b"  # ASCII Escape
ENTER_CODES = {b"\r", b"\n"}  # CR/LF, je nach OS/Terminal


def get_single_key_blocking() -> bytes:
    """
    Liest ein einzelnes Tastatur-Ereignis (blocking).
    Gibt ein Byte zurück (z. B. b'\\x1b' für ESC, b'\\r' oder b'\\n' für Enter).
    """
    if IS_WIN:
        ch = msvcrt.getch()
        # Bei Sondertasten liefert Windows oft ein Präfix (b'\x00' oder b'\xe0'), das nächste Byte ist die eigentliche Taste.
        if ch in (b"\x00", b"\xe0"):
            ch2 = msvcrt.getch()
            return ch + ch2
        return ch
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            # cbreak statt raw: SIGINT (Ctrl+C) bleibt aktiv
            tty.setcbreak(fd)
            # Warten, bis was lesbar ist
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                ch = os.read(fd, 1)
                # Ctrl+C (ETX) explizit als KeyboardInterrupt behandeln (Fallback)
                if ch == b"\x03":
                    raise KeyboardInterrupt
                # ESC: kurz warten, ob eine Folge-Sequenz kommt (z. B. Pfeiltaste)
                if ch == ESC_CODE:
                    # Warte minimal auf weitere Bytes der Sequenz
                    time_wait = 0.05  # 50 ms
                    r2, _, _ = select.select([sys.stdin], [], [], time_wait)
                    if r2:
                        # weitere Bytes zusammensammeln und als Sequenz zurückgeben
                        seq = [ch]
                        # alles, was derzeit verfügbar ist, lesen
                        while True:
                            r3, _, _ = select.select([sys.stdin], [], [], 0)
                            if not r3:
                                break
                            seq.append(os.read(fd, 1))
                        return b"".join(seq)
                    # keine Folgebytes: echtes ESC
                    return ch
                return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class Recorder:
    def __init__(self, samplerate: int = 16000, channels: int = 1, device: str | int | None = None):
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self._q: queue.Queue = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._running = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Status kann Über-/Unterläufe enthalten
            sys.stderr.write(f"Audio-Status: {status}\n")
        # Kopie, um nicht auf geteilten Speicher zu zeigen
        self._q.put(indata.copy())

    def start(self):
        if self._running:
            return
        self._frames.clear()
        self._q = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
            callback=self._callback,
            blocksize=0,  # niedrige Latenz
        )
        self._stream.start()
        self._running = True

    def stop(self) -> np.ndarray:
        if not self._running:
            return np.zeros((0, self.channels), dtype=np.float32)
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
            self._running = False
        # Alle restlichen Frames aus der Queue holen
        while True:
            try:
                block = self._q.get_nowait()
                self._frames.append(block)
            except queue.Empty:
                break
        if self._frames:
            audio = np.concatenate(self._frames, axis=0)
        else:
            audio = np.zeros((0, self.channels), dtype=np.float32)
        self._frames.clear()
        return audio


def write_wav(path: Path, audio: np.ndarray, samplerate: int):
    # soundfile erwartet (frames, channels) float32 ok
    sf.write(str(path), audio, samplerate, subtype="PCM_16")


def append_to_refs_csv(csv_path: Path, filename: str, transcript: str):
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["filename", "transcript"])
        writer.writerow([filename, transcript])


def make_filename(index: int) -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"rec_{ts}_{index:03d}.wav"


def print_instructions():
    print("")
    print("Anleitung:")
    print("  - Drücke Enter, um eine Aufnahme zu STARTEN.")
    print("  - Während der Aufnahme: Enter = Aufnahme STOPPEN und Transkript eingeben.")
    print("  - Drücke ESC zu jeder Zeit, um das Programm zu BEENDEN.")
    print("")


def main():
    parser = argparse.ArgumentParser(description="Mehrere WAV-Aufnahmen mit Enter/ESC-Steuerung.")
    parser.add_argument("--audio-dir", required=True, help="Zielverzeichnis für WAV-Dateien")
    parser.add_argument("--samplerate", type=int, default=16000, help="Abtastrate (Hz), Standard: 16000")
    parser.add_argument("--channels", type=int, default=1, help="Kanäle (1=mono, 2=stereo), Standard: 1")
    parser.add_argument("--device", default=None, help="Sounddevice-Name oder Index (optional)")
    parser.add_argument("--csv", default="refs_template.csv", help="CSV-Datei für Referenzen, Standard: refs_template.csv")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).expanduser().resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv).expanduser().resolve()

    rec = Recorder(samplerate=args.samplerate, channels=args.channels, device=args.device)

    print("Recorder bereit.")
    print_instructions()

    take_idx = 1
    try:
        while True:
            print("Bereit: Enter = Aufnahme starten | ESC = Beenden")
            key = get_single_key_blocking()
            if key_is_esc(key):
                print("Beendet.")
                break
            if not key_is_enter(key):
                # Ignoriere andere Tasten bis Enter/ESC
                continue

            # Aufnahme starten
            print("Aufnahme läuft... (Enter = Stopp, ESC = Abbrechen/Beenden)")
            rec.start()

            # Warten, bis Enter (Stop) oder ESC (Beenden)
            while True:
                k = get_single_key_blocking()
                if key_is_esc(k):
                    print("ESC erkannt: Aufnahme verworfen, Programm wird beendet.")
                    rec.stop()  # stream sauber schließen
                    return
                if key_is_enter(k):
                    print("Aufnahme gestoppt.")
                    break
                # sonst weiter warten

            audio = rec.stop()
            if audio.size == 0:
                print("Warnung: Leere Aufnahme, wird verworfen.")
                continue

            # Transkript abfragen
            print("Bitte Transkript für diese Aufnahme eingeben und mit Enter bestätigen:")
            # Auf Unix ist Terminal evtl. im raw-Modus – sicherstellen, dass Zeileneingabe wieder normal ist
            if not IS_WIN:
                # Keine Aktion nötig: get_single_key_blocking stellt den Modus im finally wieder her.
                pass
            transcript = input().strip()

            # Datei speichern
            filename = make_filename(take_idx)
            filepath = audio_dir / filename
            try:
                write_wav(filepath, audio, args.samplerate)
            except Exception as e:
                print(f"Fehler beim Speichern der WAV-Datei: {e}")
                continue

            # In CSV protokollieren (Pfad relativ oder nur Dateiname? Vorgabe: filename = reine WAV-Datei)
            try:
                append_to_refs_csv(csv_path, filename, transcript)
            except Exception as e:
                print(f"Fehler beim Schreiben in CSV ({csv_path}): {e}")
                # Aufnahme wurde bereits gespeichert; weiter mit nächster
            else:
                print(f"Gespeichert: {filepath.name}  |  CSV: {csv_path.name}")

            take_idx += 1

    except KeyboardInterrupt:
        print("\nAbbruch (Strg+C).")
    finally:
        try:
            rec.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()