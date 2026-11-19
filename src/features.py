"""Duomenų nuskaitymas ir 30 s epochų požymių skaičiavimas.

Naudojami TIK judesio (akselerometro) duomenys, nes telefonas ant čiužinio
širdies ritmo nematuoja. Kiekvienai 30 s epochai apskaičiuojami judesio
požymiai, kuriuos vėliau naudoja modeliai.
"""
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH = 30  # sekundės

# PSG žymos: 0 budrumas, 1 N1, 2 N2, 3 N3, 4 (sena N4 -> N3), 5 REM, -1 nevertinta
STAGE_NAMES = {0: "Budrumas", 1: "Lengvas (N1/N2)", 2: "Gilus (N3)", 3: "REM"}


def map_stage(raw):
    """PSG stadija -> 4 klasės: 0 budrumas, 1 lengvas, 2 gilus, 3 REM."""
    return {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}.get(int(raw), -1)


def load_labels(path):
    df = pd.read_csv(path, sep=" ", header=None, names=["t", "stage"])
    df["t"] = df["t"].astype(float)
    return df


def load_motion(path):
    df = pd.read_csv(path, sep=" ", header=None, names=["t", "x", "y", "z"],
                     dtype=float, engine="c")
    df = df[df["t"] >= 0].sort_values("t")  # tik PSG nakties laikas
    return df


FS = 10  # Hz – visi įrašai suvienodinami iki 10 Hz (realu telefonui)
FEATURES = ["mag_std", "act", "act_max", "ang_change", "move_frac", "axis_std"]


def resample_epoch(t, x, y, z, start):
    """Vienos epochos signalą suvidurkina į 0,1 s langelius (10 Hz)."""
    b = np.floor((t - start) * FS).astype(int)
    b = np.clip(b, 0, EPOCH * FS - 1)
    cnt = np.bincount(b, minlength=EPOCH * FS)
    ok = cnt > 0
    out = []
    for v in (x, y, z):
        s = np.bincount(b, weights=v, minlength=EPOCH * FS)
        out.append(s[ok] / cnt[ok])
    return out


def epoch_features(motion, epoch_starts):
    """Grąžina požymių lentelę: viena eilutė = viena 30 s epocha."""
    t = motion["t"].to_numpy()
    x, y, z = (motion[c].to_numpy() for c in "xyz")
    idx = np.searchsorted(t, epoch_starts)
    idx_end = np.searchsorted(t, epoch_starts + EPOCH)

    rows = []
    for start, a, b in zip(epoch_starts, idx, idx_end):
        if b - a < 2:
            rows.append(dict(n=0, **{f: np.nan for f in FEATURES}))
            continue
        ex, ey, ez = resample_epoch(t[a:b], x[a:b], y[a:b], z[a:b], start)
        n = len(ex)
        if n < 100:  # < 10 s duomenų (pvz., išsikrovė baterija)
            rows.append(dict(n=n, **{f: np.nan for f in FEATURES}))
            continue
        m = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
        ang = np.degrees(np.arctan2(ez, np.sqrt(ex ** 2 + ey ** 2)))
        dm = np.abs(np.diff(m))
        rows.append(dict(
            n=n,
            mag_std=float(np.std(m)),                        # judesio intensyvumas
            act=float(np.sum(dm) * 300 / n),                 # "aktyvumo skaičius"
            act_max=float(np.max(dm)),                       # staigiausias judesys
            ang_change=float(np.sum(np.abs(np.diff(ang))) * 300 / n),  # padėties keitimas
            move_frac=float(np.mean(dm > 0.01)),             # judesio dalis epochoje
            axis_std=float(np.std(ex) + np.std(ey) + np.std(ez)),
        ))
    return pd.DataFrame(rows)


def build_subject(motion_path, label_path):
    """Vieno žmogaus nakties epochų lentelė su požymiais ir žyma."""
    sid = Path(label_path).name.split("_")[0]
    lab = load_labels(label_path)
    mot = load_motion(motion_path)
    feats = epoch_features(mot, lab["t"].to_numpy())
    out = pd.concat([lab.reset_index(drop=True), feats], axis=1)
    out.insert(0, "subject", sid)
    out["stage4"] = out["stage"].map(map_stage)
    return out


def add_context(df, cols=("act", "mag_std", "ang_change", "move_frac", "axis_std"),
                windows=(5, 11, 21), causal=False):
    """Kaimyninių epochų požymiai: miego stadija priklauso nuo to,
    kas vyko kelias minutes prieš ir po, ir nuo laiko nuo nakties pradžios.

    causal=False – centruoti langai (naudoja ir „ateities“ epochas; tinka
                   nakties ataskaitai, kuri daroma ryte).
    causal=True  – tik praeities epochos (tinka žadintuvui realiu laiku)."""
    df = df.copy()
    out = []
    for _, g in df.groupby("subject", sort=False):
        g = g.sort_values("t").copy()
        for c in cols:
            lg = np.log1p(g[c])
            g[f"{c}_log"] = lg
            for w in windows:
                r = lg.rolling(w, center=not causal, min_periods=1)
                g[f"{c}_m{w}"] = r.mean()
                g[f"{c}_s{w}"] = r.std().fillna(0)
            # tik praeitis (realaus laiko žadintuvui)
            g[f"{c}_past10"] = lg.rolling(10, min_periods=1).mean()
        # minutės nuo paskutinio didesnio judesio
        moved = (g["act"] > g["act"].quantile(0.75)).to_numpy()
        since, cnt = [], 0
        for mv in moved:
            cnt = 0 if mv else cnt + 1
            since.append(cnt)
        g["since_move"] = np.log1p(np.array(since) / 2.0)
        g["hours"] = g["t"] / 3600.0                       # laikas nuo nakties pradžios
        g["cos_hours"] = np.cos(2 * np.pi * g["hours"] / 24)
        out.append(g)
    return pd.concat(out, ignore_index=True)
