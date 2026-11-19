"""Išmaniojo žadintuvo logika ir jos patikrinimas (simuliacija).

Taisyklė: vartotojas nustato žadinimo langą (pvz., 06:30–07:00).
Lango viduje žadintuvas suskamba pirmą kartą, kai
  * modelis A rodo, kad žmogus jau pabudo, ARBA
  * žmogus pajudėjo (aktyvumas didesnis nei jo paties įprastas tą naktį)
    IR modelis B rodo mažą gilaus miego tikimybę.
Jei tinkamas momentas nerandamas – skamba lango pabaigoje (kaip įprastas
žadintuvas), todėl vartotojas niekada nepramiega.
Visi sprendimai priimami tik iš praeities duomenų (realiu laiku).
AI pats nieko nevaldo be žmogaus: langą nustato vartotojas.

Pastaba: taisyklę rinkomės iš kelių variantų (žr. ataskaitos VI sk.):
„P(gilus) < riba N epochų iš eilės“ variantai davė mažesnę naudą.
"""
import numpy as np
import pandas as pd

DEEP_THR = 0.30   # P(gilus) riba
MOVE_K = 0.30     # kiek log-aktyvumas turi viršyti nakties medianą


def choose_ring_index(p_deep, awake, act_log, run_median, start_i, end_i,
                      thr=DEEP_THR, k=MOVE_K):
    """Grąžina epochos indeksą [start_i, end_i], kada skambinti."""
    for i in range(start_i, end_i + 1):
        if awake[i]:
            return i
        if act_log[i] > run_median[i] + k and p_deep[i] < thr:
            return i
    return end_i


def running_median(act_log):
    """Aktyvumo mediana nuo nakties pradžios iki dabar (be ateities)."""
    return pd.Series(act_log).expanding().median().to_numpy()


def simulate_alarm_nights(valid, p_deep, p_wake, window_min=30, step_min=15,
                          first_alarm_h=3.0):
    """Kiekvienai nakčiai „nustatome“ daug žadintuvų (kas 15 min nuo 3 val.)
    ir lyginame, kokioje TIKROJE (PSG) stadijoje žmogus pažadinamas:
    įprastas žadintuvas (lango pabaigoje), atsitiktinis momentas lange
    ir išmanusis žadintuvas."""
    rows = []
    valid = valid.reset_index(drop=True)
    for sid, s in valid.groupby("subject", sort=False):
        idx = s.index.to_numpy()
        t = s["t"].to_numpy() / 3600.0
        stage = s["stage4"].to_numpy()
        act = s["act_log"].to_numpy()
        med = running_median(act)
        pdp = p_deep[idx]
        aw = p_wake[idx] > 0.5
        T = first_alarm_h
        while T <= t[-1]:
            end_i = int(np.searchsorted(t, T, side="right") - 1)
            start_i = int(np.searchsorted(t, T - window_min / 60.0, side="left"))
            if end_i - start_i >= window_min:  # bent ~15 min duomenų lange
                ring = choose_ring_index(pdp, aw, act, med, start_i, end_i)
                win = stage[start_i:end_i + 1]
                rows.append(dict(
                    subject=sid, alarm_h=T,
                    fixed_stage=int(stage[end_i]),
                    smart_stage=int(stage[ring]),
                    minutes_early=(t[end_i] - t[ring]) * 60,
                    window_deep_share=float(np.mean(win == 2)),
                    window_light_wake_share=float(np.mean(np.isin(win, [0, 1]))),
                ))
            T += step_min / 60.0
    d = pd.DataFrame(rows)
    pct = lambda x: round(100 * float(np.mean(x)), 1)
    summary = dict(
        n_nights=int(d["subject"].nunique()),
        n_alarms=int(len(d)),
        fixed_light_or_wake_pct=pct(d["fixed_stage"].isin([0, 1])),
        random_light_or_wake_pct=pct(d["window_light_wake_share"]),
        smart_light_or_wake_pct=pct(d["smart_stage"].isin([0, 1])),
        fixed_deep_pct=pct(d["fixed_stage"] == 2),
        random_deep_pct=pct(d["window_deep_share"]),
        smart_deep_pct=pct(d["smart_stage"] == 2),
        fixed_rem_pct=pct(d["fixed_stage"] == 3),
        smart_rem_pct=pct(d["smart_stage"] == 3),
        mean_minutes_early=round(float(d["minutes_early"].mean()), 1),
    )
    return dict(details=d, summary=summary)


def night_report(stage_pred_sleep, hours):
    """Paprasta nakties ataskaita iš modelio A prognozių (1 = miega)."""
    s = np.asarray(stage_pred_sleep)
    h = np.asarray(hours)
    asleep_idx = np.flatnonzero(s == 1)
    if len(asleep_idx) == 0:
        return dict(sleep_onset_min=None, total_sleep_h=0.0, efficiency=0.0, awakenings=0)
    onset = asleep_idx[0]
    after = s[onset:]
    # pabudimas = bent 2 epochos (1 min) budrumo iš eilės po užmigimo
    wakes, run = 0, 0
    for v in after:
        run = run + 1 if v == 0 else 0
        if run == 2:
            wakes += 1
    return dict(
        sleep_onset_min=round(float(h[onset] * 60), 1),
        total_sleep_h=round(float(np.sum(s == 1) * 30 / 3600), 2),
        efficiency=round(float(np.mean(s == 1)), 3),
        awakenings=int(wakes),
    )
