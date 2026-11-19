"""Eksperimentai: bazinis (Baseline) vs patobulintas (Proposed) modelis.

Paleidimas (iš projekto šakninio katalogo):
    python src/prepare_data.py   # vieną kartą: žali failai -> požymiai
    python src/train.py          # eksperimentai, grafikai, modeliai

Vertinimas: Leave-One-Subject-Out (LOSO) – kiekvieną kartą modelis
mokomas su 29 žmonėmis ir tikrinamas su 30-u, kurio nematė. Taip
išvengiama „duomenų nutekėjimo“ tarp to paties žmogaus epochų.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from alarm import simulate_alarm_nights
from features import add_context

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 42

BASE_COLS = ["act_log", "mag_std_log", "ang_change_log", "move_frac_log", "axis_std_log"]


def context_cols(df):
    pref = ("act_", "mag_std_", "ang_change_", "move_frac_", "axis_std_")
    cols = [c for c in df.columns if c.startswith(pref) and c != "act_max"]
    return cols + ["since_move", "hours"]


def load_dataset():
    df = pd.read_csv(ROOT / "data" / "processed" / "epochs.csv", dtype={"subject": str})
    df = add_context(df, causal=True)  # tik praeities duomenys -> veikia realiu laiku
    valid = df[(df["n"] >= 100) & (df["stage4"] >= 0)].reset_index(drop=True)
    return df, valid


def models():
    return {
        "Baseline 1: dažniausia klasė": (DummyClassifier(strategy="most_frequent"), "base"),
        "Baseline 2: LogReg (1 epocha)": (make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced")), "base"),
        "Baseline 3: tik laikas nuo nakties pradžios": (HistGradientBoostingClassifier(
            max_iter=100, class_weight="balanced", random_state=SEED), "time"),
        "Proposed: Gradient Boosting (kontekstas)": (HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight="balanced", random_state=SEED), "ctx"),
    }


TASKS = {
    "A_miegas_budrumas": dict(
        title="A: miegas / budrumas",
        target=lambda d: (d["stage4"] > 0).astype(int),
        labels=["Budrumas", "Miegas"]),
    "B_gilus_miegas": dict(
        title="B: gilus miegas (N3) / kita",
        target=lambda d: (d["stage4"] == 2).astype(int),
        labels=["Ne gilus", "Gilus (N3)"]),
    "C_4_klases": dict(
        title="C: 4 stadijos (papildomas)",
        target=lambda d: d["stage4"].astype(int),
        labels=["Budrumas", "Lengvas", "Gilus", "REM"]),
}


def metrics_row(y, p):
    return dict(
        accuracy=accuracy_score(y, p),
        f1_macro=f1_score(y, p, average="macro"),
        precision_macro=precision_score(y, p, average="macro", zero_division=0),
        recall_macro=recall_score(y, p, average="macro"),
        kappa=cohen_kappa_score(y, p),
    )


def plot_cm(y, p, labels, title, path):
    cm = confusion_matrix(y, p, normalize="true")
    fig, ax = plt.subplots(figsize=(4.2 + 0.4 * len(labels), 3.6 + 0.3 * len(labels)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Modelio prognozė")
    ax.set_ylabel("Tikroji stadija (PSG)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_data_overview(valid):
    names = ["Budrumas", "Lengvas\n(N1/N2)", "Gilus (N3)", "REM"]
    cnt = valid["stage4"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(names, cnt.values, color=["#d95f02", "#7fb3d5", "#1f4e79", "#2ca25f"])
    for i, v in enumerate(cnt.values):
        axes[0].text(i, v, f"{v}\n({v / cnt.sum():.0%})", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Klasių pasiskirstymas (30 s epochos)")
    axes[0].set_ylabel("Epochų skaičius")
    axes[0].set_ylim(0, cnt.max() * 1.25)
    data = [valid.loc[valid["stage4"] == k, "act_log"] for k in range(4)]
    axes[1].boxplot(data, showfliers=False)
    axes[1].set_xticks(range(1, 5), names)
    axes[1].set_title("Judesio aktyvumas pagal stadiją")
    axes[1].set_ylabel("log(1 + aktyvumas)")
    fig.tight_layout()
    fig.savefig(RES / "duomenu_apzvalga.png", dpi=150)
    plt.close(fig)


def plot_hypnogram(valid, proba_deep, pred_sleep, subject, path):
    s = valid[valid["subject"] == subject]
    idx = s.index
    h = s["hours"].to_numpy()
    order = {0: 3, 3: 2, 1: 1, 2: 0}  # budrumas viršuje, gilus apačioje
    true_y = s["stage4"].map(order).to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1, 1.3]))
    axes[0].step(h, true_y, where="post", color="#1f4e79")
    axes[0].set_yticks([0, 1, 2, 3], ["Gilus", "Lengvas", "REM", "Budrumas"])
    axes[0].set_title(f"Tiriamojo {subject} naktis: tikroji hipnograma (PSG)")
    axes[1].fill_between(h, 0, 1 - pred_sleep[idx], step="post", color="#d95f02", alpha=0.7)
    axes[1].set_yticks([0, 1], ["miega", "budi"])
    axes[1].set_title("Modelio A prognozė: budrumas", fontsize=9)
    axes[2].plot(h, proba_deep[idx], color="#1f4e79")
    axes[2].fill_between(h, 0, (s["stage4"] == 2).to_numpy(), step="post",
                         color="grey", alpha=0.25, label="tikras gilus miegas")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("P(gilus)")
    axes[2].set_xlabel("Valandos nuo įrašo pradžios")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("Modelio B tikimybė: gilus miegas", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df, valid = load_dataset()
    ctx = context_cols(valid)
    groups = valid["subject"]
    cv = LeaveOneGroupOut()
    info = dict(n_subjects=int(groups.nunique()), n_epochs=int(len(valid)),
                n_features_base=len(BASE_COLS), n_features_ctx=len(ctx),
                class_counts={int(k): int(v) for k, v in valid["stage4"].value_counts().items()})
    print(info)
    plot_data_overview(valid)

    rows, oof = [], {}
    for tkey, task in TASKS.items():
        y = task["target"](valid)
        for mname, (model, cset) in models().items():
            cols = {"base": BASE_COLS, "time": ["hours"], "ctx": ctx}[cset]
            method = "predict_proba" if "Proposed" in mname else "predict"
            out = cross_val_predict(model, valid[cols], y, groups=groups, cv=cv,
                                    n_jobs=-1, method=method)
            if method == "predict_proba":
                oof[tkey] = out
                pred = out.argmax(1)
            else:
                pred = out
            r = metrics_row(y, pred)
            r.update(task=task["title"], model=mname)
            rows.append(r)
            print(f"{task['title']:32s} {mname:42s} acc={r['accuracy']:.3f} "
                  f"F1={r['f1_macro']:.3f} kappa={r['kappa']:.3f}")
            short = mname.split(":")[0].replace(" ", "")
            plot_cm(y, pred, task["labels"], f"{task['title']}\n{mname}",
                    RES / f"cm_{tkey}_{short}.png")
    res = pd.DataFrame(rows)[["task", "model", "accuracy", "f1_macro",
                              "precision_macro", "recall_macro", "kappa"]]
    res.to_csv(RES / "metrikos.csv", index=False, float_format="%.3f")

    # per-subject accuracy (proposed, A and B)
    yA = TASKS["A_miegas_budrumas"]["target"](valid)
    yB = TASKS["B_gilus_miegas"]["target"](valid)
    predA = oof["A_miegas_budrumas"].argmax(1)
    predB = oof["B_gilus_miegas"].argmax(1)
    per = valid[["subject"]].assign(okA=(predA == yA), okB=(predB == yB))
    per = per.groupby("subject").mean()
    per.to_csv(RES / "tikslumas_pagal_zmogu.csv", float_format="%.3f")

    # hipnogramos pavyzdžiai: geriausias ir blogiausias žmogus pagal B
    best, worst = per["okB"].idxmax(), per["okB"].idxmin()
    plot_hypnogram(valid, oof["B_gilus_miegas"][:, 1], predA, best, RES / "hipnograma_geriausias.png")
    plot_hypnogram(valid, oof["B_gilus_miegas"][:, 1], predA, worst, RES / "hipnograma_blogiausias.png")

    # žadintuvo simuliacija su LOSO tikimybėmis (modelis nematė to žmogaus)
    sim = simulate_alarm_nights(valid, oof["B_gilus_miegas"][:, 1], oof["A_miegas_budrumas"][:, 0])
    valid.assign(p_wake=oof["A_miegas_budrumas"][:, 0], p_deep=oof["B_gilus_miegas"][:, 1])[
        ["subject", "t", "hours", "stage4", "act_log", "p_wake", "p_deep"]].to_csv(
        RES / "loso_prognozes.csv", index=False, float_format="%.4f")
    sim["details"].to_csv(RES / "zadintuvo_simuliacija.csv", index=False, float_format="%.3f")
    print(sim["summary"])

    # paaiškinamumas (XAI): permutacinė požymių svarba
    final = {}
    for tkey in ["A_miegas_budrumas", "B_gilus_miegas"]:
        y = TASKS[tkey]["target"](valid)
        m = models()["Proposed: Gradient Boosting (kontekstas)"][0]
        m.fit(valid[ctx], y)
        final[tkey] = m
        sample = valid.sample(4000, random_state=SEED)
        pi = permutation_importance(m, sample[ctx], TASKS[tkey]["target"](sample),
                                    scoring="f1_macro", n_repeats=5, random_state=SEED, n_jobs=-1)
        imp = pd.Series(pi.importances_mean, index=ctx).sort_values(ascending=False)
        imp.to_csv(RES / f"svarba_{tkey}.csv", float_format="%.4f")
        top = imp.head(12)[::-1]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(top.index, top.values, color="#1f4e79")
        ax.set_xlabel("F1 sumažėjimas sumaišius požymį")
        ax.set_title(f"Svarbiausi požymiai – {TASKS[tkey]['title']}", fontsize=10)
        fig.tight_layout()
        fig.savefig(RES / f"svarba_{tkey}.png", dpi=150)
        plt.close(fig)

    joblib.dump(dict(model_sleep=final["A_miegas_budrumas"],
                     model_deep=final["B_gilus_miegas"], features=ctx),
                RES / "modeliai.joblib")
    with open(RES / "santrauka.json", "w", encoding="utf-8") as f:
        json.dump(dict(info=info, alarm=sim["summary"], best_subject=best,
                       worst_subject=worst), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
