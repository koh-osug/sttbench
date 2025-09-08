#!/usr/bin/env python3

# (Short header, full content omitted here for brevity in this cell)
# The full script was provided in prior step; for safety, include the full content again:

"""
STT Benchmark: run multiple open-source speech-to-text engines on a folder of WAV files
and compare quality with WER/CER (if references are available).
Supported engines: whisper, faster_whisper, vosk, hf_wav2vec2, whisper_cpp
"""

import argparse, csv, json, os, subprocess, sys, tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

try:
    import jiwer
except Exception:
    jiwer = None

def _ensure_wav_pcm16_mono_16k(src_path: Path) -> Path:
    import soundfile as sf, numpy as np, librosa, tempfile
    data, sr = sf.read(str(src_path), always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        import numpy as _np
        data = _np.mean(data, axis=1)
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    if data.dtype != 'int16':
        if data.dtype.kind == 'f':
            import numpy as _np
            data = (data * 32767.0).clip(-32768, 32767).astype('int16')
        else:
            data = data.astype('int16')
    import soundfile as sf2
    fd, tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    sf2.write(tmp, data, 16000, subtype='PCM_16')
    return Path(tmp)

class BaseEngine:
    name = "base"
    def __init__(self, args): self.args = args
    def transcribe(self, audio_path: Path, language: Optional[str]) -> str: raise NotImplementedError

class WhisperEngine(BaseEngine):
    name = "whisper"
    def __init__(self, args):
        import whisper
        self.model = whisper.load_model(getattr(args,"whisper_model","base"), device=None if getattr(args,"whisper_device","auto")=="auto" else getattr(args,"whisper_device","auto"))
    def transcribe(self, audio_path: Path, language: Optional[str]) -> str:
        r = self.model.transcribe(str(audio_path), language=language) if language else self.model.transcribe(str(audio_path)); return r.get("text","").strip()

class FasterWhisperEngine(BaseEngine):
    name = "faster_whisper"
    def __init__(self, args):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(getattr(args,"faster_whisper_model","base"), device=getattr(args,"faster_whisper_device","auto"), compute_type=getattr(args,"faster_whisper_compute_type","default"))
    def transcribe(self, audio_path: Path, language: Optional[str]) -> str:
        segs,_ = self.model.transcribe(str(audio_path), language=language, beam_size=5); return " ".join(s.text for s in segs).strip()

class VoskEngine(BaseEngine):
    name = "vosk"
    def __init__(self, args):
        from vosk import Model, SetLogLevel
        mp = getattr(args,"vosk_model",None)
        if not mp or not Path(mp).exists(): raise ValueError("--vosk.model must point to a Vosk model directory")
        SetLogLevel(-1); self.model = Model(mp)
    def transcribe(self, audio_path: Path, language: Optional[str]) -> str:
        import wave, json as _json
        from vosk import KaldiRecognizer
        wav = _ensure_wav_pcm16_mono_16k(audio_path)
        wf = wave.open(str(wav),"rb"); rec = KaldiRecognizer(self.model,16000); rec.SetWords(True)
        text=""
        while True:
            data = wf.readframes(4000)
            if len(data)==0: break
            if rec.AcceptWaveform(data): text += " " + _json.loads(rec.Result()).get("text","")
        text += " " + _json.loads(rec.FinalResult()).get("text","")
        return text.strip()

class HFEngine(BaseEngine):
    name = "hf_wav2vec2"
    def __init__(self, args):
        from transformers import pipeline
        m = getattr(args,"hf_model",None)
        if not m: raise ValueError("--hf.model must be provided")
        self.pipe = pipeline("automatic-speech-recognition", model=m)
    def transcribe(self, audio_path: Path, language: Optional[str]) -> str:
        out = self.pipe(str(audio_path)); return (out.get("text","") if isinstance(out,dict) else str(out)).strip()

class WhisperCppEngine(BaseEngine):
    name = "whisper_cpp"
    def __init__(self, args):
        bp = getattr(args,"whisper_cpp_bin",None); mp = getattr(args,"whisper_cpp_model",None)
        if not bp or not Path(bp).exists(): raise ValueError("--whisper_cpp.bin must exist")
        if not mp or not Path(mp).exists(): raise ValueError("--whisper_cpp.model must exist")
        self.bin, self.model = Path(bp), Path(mp)
    def transcribe(self, audio_path: Path, language: Optional[str]) -> str:
        import tempfile
        lang = language or ""
        with tempfile.TemporaryDirectory() as td:
            outp = Path(td)/"out"
            cmd=[str(self.bin),"-m",str(self.model),"-f",str(audio_path),"-otxt","-of",str(outp)]
            if lang: cmd+=["-l",lang]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            txt = Path(str(outp)+".txt")
            return txt.read_text(encoding="utf-8",errors="ignore").strip() if txt.exists() else ""

ENGINES = {"whisper":WhisperEngine,"faster_whisper":FasterWhisperEngine,"vosk":VoskEngine,"hf_wav2vec2":HFEngine,"whisper_cpp":WhisperCppEngine}

def load_references(audio_files: List[Path], refs_path: Optional[Path], refs_sep: str) -> Dict[str,str]:
    refs={}
    def norm(t:str)->str: return t.strip()
    if refs_path and refs_path.exists():
        if refs_path.suffix.lower()==".jsonl":
            for line in refs_path.read_text(encoding="utf-8").splitlines():
                try:
                    o=json.loads(line); fn=o.get("filename"); tr=o.get("transcript")
                    if fn and tr: refs[Path(fn).stem]=norm(tr)
                except: pass
        else:
            import csv
            with open(refs_path,"r",encoding="utf-8") as f:
                reader=csv.DictReader(f, delimiter=refs_sep)
                for row in reader:
                    fn=row.get("filename") or row.get("file") or row.get("path")
                    tr=row.get("transcript") or row.get("text") or row.get("gt") or row.get("reference")
                    if fn and tr: refs[Path(fn).stem]=norm(tr)
    else:
        for wav in audio_files:
            side=wav.with_suffix(".txt")
            if side.exists(): refs[wav.stem]=norm(side.read_text(encoding="utf-8",errors="ignore"))
    return refs

def compute_metrics(references: Dict[str,str], hyps: Dict[str,str]):
    if not jiwer: return None, None
    commons=[(r,hyps.get(k)) for k,r in references.items() if hyps.get(k) is not None]
    if not commons: return None, None
    refs=[r for r,_ in commons]; preds=[h for _,h in commons]
    tf=jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    wer=jiwer.wer(refs,preds, truth_transform=tf, hypothesis_transform=tf)
    cer=jiwer.cer(refs,preds)
    return float(wer), float(cer)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--engines", nargs="+", required=True, choices=list(ENGINES.keys()))
    ap.add_argument("--language", default=None)
    ap.add_argument("--refs", default=None)
    ap.add_argument("--refs-sep", default=",")
    ap.add_argument("--whisper.model", dest="whisper_model", default="base")
    ap.add_argument("--whisper.device", dest="whisper_device", default="auto")
    ap.add_argument("--faster_whisper.model", dest="faster_whisper_model", default="base")
    ap.add_argument("--faster_whisper.device", dest="faster_whisper_device", default="auto")
    ap.add_argument("--faster_whisper.compute_type", dest="faster_whisper_compute_type", default="default")
    ap.add_argument("--vosk.model", dest="vosk_model", default=None)
    ap.add_argument("--hf.model", dest="hf_model", default=None)
    ap.add_argument("--whisper_cpp.bin", dest="whisper_cpp_bin", default=None)
    ap.add_argument("--whisper_cpp.model", dest="whisper_cpp_model", default=None)
    args=ap.parse_args()

    audio_dir=Path(args.audio_dir); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    wavs=sorted([p for p in audio_dir.glob("*.wav") if p.is_file()])
    if not wavs: print(f"No WAV files found in {audio_dir}", file=sys.stderr) or sys.exit(2)

    engines=[ENGINES[e](args) for e in args.engines]
    refs=load_references(wavs, Path(args.refs) if args.refs else None, args.refs_sep)

    import pandas as pd
    rows=[]; per={e.name:{} for e in engines}
    with open(out_dir/"log.txt","w",encoding="utf-8") as log:
        log.write(f"Engines: {', '.join(e.name for e in engines)}\n")
        log.write(f"Language: {args.language}\n")
        log.write(f"Num files: {len(wavs)}\n")
        for wav in wavs:
            for e in engines:
                try: hyp=e.transcribe(wav, args.language)
                except Exception as ex: hyp=f"__ERROR__: {type(ex).__name__}: {ex}"
                rows.append({"filename":wav.name,"engine":e.name,"hypothesis":hyp,"reference":refs.get(wav.stem,"")})
                per[e.name][wav.stem]=hyp
            log.write(f"Processed: {wav.name}\n")

    pd.DataFrame(rows).to_csv(out_dir/"hypotheses.csv", index=False)
    metrics=[]
    if refs:
        for e in engines:
            wer,cer=compute_metrics(refs, per[e.name])
            metrics.append({"engine":e.name,"WER":wer if wer is not None else "","CER":cer if cer is not None else "","num_eval_utts":len([k for k in per[e.name].keys() if k in refs])})
    pd.DataFrame(metrics).to_csv(out_dir/"metrics.csv", index=False)
    print(f"Wrote: {out_dir/'hypotheses.csv'}"); print(f"Wrote: {out_dir/'metrics.csv'}"); print(f"Wrote: {out_dir/'log.txt'}")

if __name__=="__main__": main()
