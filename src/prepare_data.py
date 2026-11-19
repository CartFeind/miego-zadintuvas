"""Žali PhysioNet failai -> viena požymių lentelė data/processed/epochs.csv.

Naudojimas:
    python src/prepare_data.py [kelias_iki_duomenų]

Numatytas kelias: data/raw (ten turi būti katalogai motion/ ir labels/).
"""
import sys
from pathlib import Path

import pandas as pd

from features import build_subject

ROOT = Path(__file__).resolve().parents[1]
MIN_VALID_SHARE = 0.5  # žmogus atmetamas, jei < 50 % epochų turi judesio duomenis


def find_raw(path):
    path = Path(path)
    if (path / "motion").exists():
        return path
    hits = list(path.rglob("motion"))
    if not hits:
        sys.exit(f"Nerastas katalogas 'motion' po {path}")
    return hits[0].parent


def main():
    raw = find_raw(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "raw")
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts, dropped = [], []
    for lab in sorted((raw / "labels").glob("*_labeled_sleep.txt")):
        sid = lab.name.split("_")[0]
        mot = raw / "motion" / f"{sid}_acceleration.txt"
        if not mot.exists():
            dropped.append((sid, "nėra judesio failo"))
            continue
        d = build_subject(mot, lab)
        share = float((d["n"] >= 100).mean())
        if share < MIN_VALID_SHARE:
            dropped.append((sid, f"tik {share:.0%} epochų su duomenimis"))
            continue
        parts.append(d)
        print(f"{sid}: {len(d)} epochų, su duomenimis {share:.0%}")
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out_dir / "epochs.csv", index=False, float_format="%.6g")
    print(f"\nIšsaugota: {len(df)} epochų iš {df['subject'].nunique()} žmonių")
    for sid, why in dropped:
        print(f"Atmesta {sid}: {why}")


if __name__ == "__main__":
    main()
