#!/usr/bin/env python3
"""
analyze_results.py
Rich analysis for STT benchmark outputs.

Usage:
  python analyze_results.py [RESULTS_DIR]

What it does:
  - Loads metrics.csv and hypotheses.csv from a results directory.
  - Prints metrics sorted by WER and by CER (ascending).
  - If references are present, computes per-utterance WER/CER per engine
    and writes per_utt_metrics.csv. Also prints the "hardest" files.
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

try:
    import jiwer
except Exception:
    jiwer = None

def fmt(x):
    if x=="" or x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.3f}" if isinstance(x, float) else x

def compute_wer_cer(truths: List[str], hyps: List[str]) -> Tuple[float, float]:
    if not jiwer:
        return float("nan"), float("nan")
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    wer = jiwer.wer(truths, hyps, truth_transform=transform, hypothesis_transform=transform)
    cer = jiwer.cer(truths, hyps)
    return float(wer), float(cer)

def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "./results_DE_compare")

    m_path = out / "metrics.csv"
    h_path = out / "hypotheses.csv"

    if not h_path.exists():
        print(f"{h_path} not found. Did you run the benchmark?")
        sys.exit(0)

    # Show aggregated metrics (if present)
    if m_path.exists():
        m = pd.read_csv(m_path)
        if len(m):
            m1 = m.copy()
            # pretty print
            m1["WER"] = m1["WER"].apply(fmt)
            m1["CER"] = m1["CER"].apply(fmt)
            # Sort by WER
            print("\n=== Aggregated metrics (sorted by WER asc) ===")
            print(m.sort_values(by="WER", ascending=True, na_position="last").to_string(index=False))
            print("\n=== Aggregated metrics (sorted by CER asc) ===")
            print(m.sort_values(by="CER", ascending=True, na_position="last").to_string(index=False))
        else:
            print("metrics.csv is empty (no references?)")
    else:
        print("No metrics.csv (provide refs to compute WER/CER).")

    # Per-utterance analysis
    H = pd.read_csv(h_path)
    have_refs = H["reference"].fillna("").str.strip().str.len().gt(0).any()
    if not have_refs:
        print("\nNo references in hypotheses.csv -> skipping per-utterance analysis.")
        sys.exit(0)

    # Compute WER/CER per utterance per engine
    per_rows = []
    for (fn, eng), df in H.groupby(["filename", "engine"]):
        ref = df["reference"].iloc[0] if len(df) else ""
        hyp = df["hypothesis"].iloc[0] if len(df) else ""
        if ref and jiwer:
            wer, cer = compute_wer_cer([ref], [hyp])
        else:
            wer, cer = (float("nan"), float("nan"))
        per_rows.append({"filename": fn, "engine": eng, "WER": wer, "CER": cer, "ref_len": len(str(ref).split()), "hypothesis": hyp, "reference": ref})
    PER = pd.DataFrame(per_rows)
    per_csv = out / "per_utt_metrics.csv"
    PER.to_csv(per_csv, index=False)
    print(f"\nWrote per-utterance metrics: {per_csv}")

    # Find hardest utterances (highest average WER across engines)
    try:
        PIV = PER.pivot(index="filename", columns="engine", values="WER")
        PIV["avg_WER"] = PIV.mean(axis=1, skipna=True)
        hardest = PIV.sort_values("avg_WER", ascending=False).head(10)
        print("\n=== Hardest 10 files by avg WER (across engines) ===")
        print(hardest.to_string())
    except Exception:
        pass

    # Quick best-engine-per-utt table
    try:
        best = PER.loc[PER.groupby("filename")["WER"].idxmin()].sort_values("WER")
        best2 = best[["filename", "engine", "WER"]].copy()
        best2["WER"] = best2["WER"].apply(fmt)
        print("\n=== Best engine per utterance (by WER) ===")
        print(best2.head(20).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
