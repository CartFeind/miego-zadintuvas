"""Treniravimo grafikas: kaip keičiasi F1 didinant medžių skaičių
(5 dalių GroupKFold pagal žmones, modelis B – gilus miegas)."""
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from train import load_dataset, context_cols, TASKS, RES, SEED
warnings.filterwarnings("ignore")

_, v = load_dataset()
ctx = context_cols(v)
fig, ax = plt.subplots(figsize=(7, 3.6))
for tkey, color in [("A_miegas_budrumas", "#d95f02"), ("B_gilus_miegas", "#1f4e79")]:
    y = TASKS[tkey]["target"](v).to_numpy()
    tr_s, te_s = [], []
    for tr, te in GroupKFold(5).split(v, y, v["subject"]):
        m = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, l2_regularization=1.0,
                                           class_weight="balanced", random_state=SEED, early_stopping=False)
        m.fit(v.iloc[tr][ctx], y[tr])
        tr_s.append([f1_score(y[tr], p, average="macro") for p in m.staged_predict(v.iloc[tr][ctx])][9::10])
        te_s.append([f1_score(y[te], p, average="macro") for p in m.staged_predict(v.iloc[te][ctx])][9::10])
    it = np.arange(10, 601, 10)
    ax.plot(it, np.mean(tr_s, 0), "--", color=color, label=f"{TASKS[tkey]['title']} – mokymo")
    ax.plot(it, np.mean(te_s, 0), "-", color=color, label=f"{TASKS[tkey]['title']} – testavimo")
    print(tkey, "test F1 at 100/300/600:", [round(np.mean(te_s, 0)[i], 3) for i in (9, 29, 59)])
ax.axvline(300, color="grey", lw=0.8, ls=":")
ax.set_xlabel("Medžių (iteracijų) skaičius")
ax.set_ylabel("F1 (macro)")
ax.set_title("Mokymosi kreivės (5-fold pagal žmones)", fontsize=10)
ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(RES / "mokymosi_kreive.png", dpi=150)
