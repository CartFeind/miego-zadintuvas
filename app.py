"""Demonstracija: išmanusis miego ciklų žadintuvas.

Paleidimas:  streamlit run app.py

Programa „atkuria“ vieno žmogaus naktį iš duomenų rinkinio taip, lyg
telefonas būtų ją įrašęs. Prognozės paimtos iš LOSO vertinimo, t. y.
modelis to žmogaus mokymo metu nematė.
"""
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from alarm import DEEP_THR, MOVE_K, choose_ring_index, night_report, running_median  # noqa: E402

STAGE_NAME = {0: "Budrumas", 1: "Lengvas miegas", 2: "Gilus miegas", 3: "REM miegas"}
STAGE_ORDER = ["Budrumas", "REM miegas", "Lengvas miegas", "Gilus miegas"]
STAGE_COLOR = ["#e8833a", "#2ca25f", "#7fb3d5", "#1f4e79"]
STAGE_LOW = {0: "budrumo", 1: "lengvo miego", 2: "gilaus miego", 3: "REM miego"}  # kilmininkas
GOOD_WAKE = {0, 1}  # budrumas arba lengvas miegas – geras momentas pažadinti
BASE_DAY = datetime(2026, 1, 1)
WINDOW_COLOR = "#f2c94c"
ALARM_COLOR = "#e5484d"


@st.cache_data
def load():
    df = pd.read_csv(ROOT / "results" / "loso_prognozes.csv", dtype={"subject": str})
    subjects = sorted(df["subject"].unique())
    return df, {sid: i + 1 for i, sid in enumerate(subjects)}


def hhmm(dt):
    return dt.strftime("%H:%M")


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def window_and_alarm_layers(win_start, win_end, ring_dt):
    """Geltonas žadinimo langas ir raudona žadintuvo linija – bendri visiems grafikams."""
    return [
        {"data": {"values": [{"pradzia": iso(win_start), "pabaiga": iso(win_end),
                              "pav": "Žadinimo langas"}]},
         "mark": {"type": "rect", "color": WINDOW_COLOR, "opacity": 0.3},
         "encoding": {
             "x": {"field": "pradzia", "type": "temporal"},
             "x2": {"field": "pabaiga"},
             "tooltip": [{"field": "pav", "type": "nominal", "title": " "},
                         {"field": "pradzia", "type": "temporal", "format": "%H:%M", "title": "Nuo"},
                         {"field": "pabaiga", "type": "temporal", "format": "%H:%M", "title": "Iki"}]}},
        {"data": {"values": [{"laikas": iso(ring_dt), "pav": "Žadintuvas suskamba"}]},
         "mark": {"type": "rule", "color": ALARM_COLOR, "strokeWidth": 3},
         "encoding": {
             "x": {"field": "laikas", "type": "temporal"},
             "tooltip": [{"field": "pav", "type": "nominal", "title": " "},
                         {"field": "laikas", "type": "temporal", "format": "%H:%M", "title": "Laikas"}]}},
    ]


def x_axis(show_labels=True):
    return {"field": "laikas", "type": "temporal", "title": "Laikas" if show_labels else None,
            "axis": {"format": "%H:%M", "labels": show_labels, "tickCount": 10, "grid": False}}


def chart(height, main_layer, extra_layers):
    return {"width": "container", "height": height,
            "layer": [*extra_layers[:1], main_layer, *extra_layers[1:]]}


def show(spec):
    """Grafikas per visą plotį (veikia ir naujose, ir senesnėse Streamlit versijose)."""
    try:
        st.vega_lite_chart(spec, width="stretch")
    except Exception:
        st.vega_lite_chart(spec, use_container_width=True)


# ---------------------------------------------------------------- puslapis
st.set_page_config(page_title="Miego ciklų žadintuvas", layout="wide")
df, night_no = load()

st.title("Miego ciklų žadintuvas")
st.markdown(
    "Išmanusis žadintuvas **be išmaniojo laikrodžio**. Telefonas guli ant čiužinio ir "
    "jaučia, kaip judate miegodami. Dirbtinis intelektas kas 30 sekundžių įvertina, "
    "kaip giliai miegate, ir pažadina tada, kai miegas lengviausias – todėl keliatės "
    "mažiau apsnūdę.")

st.markdown("""
<style>
/* skirtukai – dideli suapvalinti mygtukai (veikia su įvairiomis Streamlit versijomis) */
.stTabs [role="tablist"] {gap: 12px !important; border-bottom: none !important; padding: 6px 2px 10px 2px;}
.stTabs button[role="tab"], .stTabs [data-baseweb="tab"] {
    height: auto !important; min-height: 48px; padding: 12px 30px !important; margin: 0 !important;
    border-radius: 16px !important; border: 1px solid rgba(76, 120, 168, 0.35) !important;
    background: rgba(76, 120, 168, 0.12) !important;
}
.stTabs button[role="tab"] p, .stTabs [data-baseweb="tab"] p {font-size: 1.1rem; font-weight: 600; margin: 0;}
.stTabs button[role="tab"]:hover {background: rgba(76, 120, 168, 0.25) !important;}
.stTabs button[role="tab"][aria-selected="true"], .stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #4c78a8 !important; border-color: #4c78a8 !important;
}
.stTabs button[role="tab"][aria-selected="true"] p {color: #ffffff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none !important;}

/* šoninis skydelis – telpa be slinkimo */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding-top: 0.5rem !important; padding-bottom: 1rem !important;}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {gap: 1.25rem;}
section[data-testid="stSidebar"] label p {font-size: 1.05rem; font-weight: 600;}
section[data-testid="stSidebar"] h2 {font-size: 1.6rem; padding-top: 0;}
section[data-testid="stSidebar"] hr {margin: 0.25rem 0;}
section[data-testid="stSidebar"] * {scrollbar-width: none;}
section[data-testid="stSidebar"] *::-webkit-scrollbar {display: none;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Nustatymai")
    subjects = list(night_no)
    sid = st.selectbox(
        "Pasirinkite naktį", subjects, index=subjects.index("1455390"),
        format_func=lambda s: f"Naktis Nr. {night_no[s]}",
        help="Kiekviena naktis – vieno tikro žmogaus miegas, įrašytas miego laboratorijoje. "
             "Programa atkuria ją taip, lyg telefonas būtų gulėjęs ant čiužinio.")
    st.caption(f"Duomenų rinkinio tiriamojo kodas: {sid}")

    bed_t = st.time_input("Atsigulimo laikas", time(0, 0), step=timedelta(minutes=15),
                          help="Kada padėjote telefoną ir atsigulėte.")
    night = df[df["subject"] == sid].reset_index(drop=True)
    bed_dt = datetime.combine(BASE_DAY, bed_t)
    first = bed_dt + timedelta(hours=1)
    last = bed_dt + timedelta(hours=float(night["hours"].max()))
    last = last.replace(second=0, microsecond=0) - timedelta(minutes=last.minute % 15)
    default = min(bed_dt + timedelta(hours=6, minutes=30), last)
    alarm_dt = st.slider("Vėliausias kėlimosi laikas", min_value=first, max_value=last,
                         value=default, step=timedelta(minutes=15), format="HH:mm",
                         help="Žadintuvas niekada neskambės vėliau už šį laiką.")
    window = st.slider("Žadinimo lango ilgis (min.)", 10, 60, 30, 5,
                       help="Per kiek laiko iki kėlimosi žadintuvas gali jus pažadinti, "
                            "jei aptinka lengvą miegą.")
    st.caption(f"Žadintuvas ieškos tinkamo momento nuo "
               f"**{hhmm(alarm_dt - timedelta(minutes=window))}** iki **{hhmm(alarm_dt)}**.")
    st.divider()
    st.caption("Naktys paimtos iš atviro PhysioNet duomenų rinkinio (30 žmonių, miego "
               "laboratorija). Tai mokomasis projektas, ne medicininis prietaisas.")

# ---------------------------------------------------------------- skaičiavimai
h = night["hours"].to_numpy()
alarm_h = (alarm_dt - bed_dt).total_seconds() / 3600
end_i = int(np.searchsorted(h, alarm_h, side="right") - 1)
start_i = int(np.searchsorted(h, alarm_h - window / 60, side="left"))
start_i = min(start_i, end_i)
awake = night["p_wake"].to_numpy() > 0.5
ring = choose_ring_index(night["p_deep"].to_numpy(), awake, night["act_log"].to_numpy(),
                         running_median(night["act_log"].to_numpy()), start_i, end_i,
                         DEEP_THR, MOVE_K)
clock = [bed_dt + timedelta(hours=float(x)) for x in h]
ring_dt, end_dt = clock[ring], clock[end_i]
win_start = alarm_dt - timedelta(minutes=window)
smart_stage = int(night["stage4"][ring])
fixed_stage = int(night["stage4"][end_i])
early = round((end_dt - ring_dt).total_seconds() / 60)

st.caption("Pasirinkite skiltį:")
tab_alarm, tab_report, tab_how = st.tabs(["Žadintuvas", "Nakties ataskaita", "Kaip tai veikia"])

# ---------------------------------------------------------------- 1 skirtukas
with tab_alarm:
    c1, c2, c3 = st.columns(3)
    c1.metric("Išmanusis žadintuvas suskamba", hhmm(ring_dt),
              f"{early} min. anksčiau nei vėliausias laikas" if early > 0 else "vėliausiu laiku",
              delta_color="off",
              help="Pirmas momentas lange, kai AI nusprendė, kad miegas lengvas arba "
                   "žmogus jau pabudo. Neradęs tokio momento, žadintuvas skamba vėliausiu laiku.")
    c2.metric("Miego stadija tuo metu", STAGE_NAME[smart_stage],
              help="Tikroji stadija pagal miego laboratorijos tyrimą (polisomnografiją). "
                   "Telefonas jos nemato – ji rodoma tik tam, kad galėtume patikrinti AI.")
    c3.metric(f"Įprastas žadintuvas ({hhmm(end_dt)}) pažadintų iš", STAGE_NAME[fixed_stage],
              help="Kas būtų, jei žadintuvas skambėtų tiksliai nustatytu laiku.")

    if smart_stage in GOOD_WAKE and fixed_stage not in GOOD_WAKE:
        st.success(f"Išmanusis žadintuvas pažadino lengvo miego metu, o įprastas būtų "
                   f"pažadinęs iš {STAGE_LOW[fixed_stage]}.")
    elif smart_stage in GOOD_WAKE:
        st.success("Pažadinta lengvo miego metu – tinkamas momentas keltis.")
    elif ring == end_i:
        st.warning("Lange tinkamo momento nerasta, todėl žadintuvas suskambo vėliausiu laiku – "
                   "kaip įprastas žadintuvas.")
    else:
        st.warning(f"AI suklydo: pažadino iš {STAGE_LOW[smart_stage]}. "
                   "Pagal judesį ramų REM miegą sunku atskirti nuo kitų stadijų.")

    layers = window_and_alarm_layers(win_start, alarm_dt, ring_dt)

    # --- hipnograma
    st.subheader("Miego stadijos per naktį")
    st.caption("Taip miegą įvertino miego laboratorija (polisomnografija). Telefonas šių duomenų "
               "nemato – jie naudojami tik patikrinti, ar AI pažadino tinkamu metu. "
               "Užveskite pelę ant linijos, kad pamatytumėte laiką ir stadiją.")
    hyp = pd.DataFrame({"laikas": [iso(c) for c in clock],
                        "stadija": night["stage4"].map(STAGE_NAME)})
    show(chart(230, {
        "data": {"values": hyp.to_dict("records")},
        "mark": {"type": "line", "interpolate": "step-after", "color": "#4c78a8", "strokeWidth": 2},
        "encoding": {
            "x": x_axis(False),
            "y": {"field": "stadija", "type": "ordinal", "sort": STAGE_ORDER, "title": None,
                  "scale": {"domain": STAGE_ORDER}},
            "tooltip": [{"field": "laikas", "type": "temporal", "format": "%H:%M", "title": "Laikas"},
                        {"field": "stadija", "type": "nominal", "title": "Stadija"}]}},
        layers))

    # --- judesys
    st.subheader("Ką mato telefonas: judesys")
    st.caption("Kiekvieno 30 s intervalo judesio stiprumas. Smailės – apsivertimai ar pabudimai; "
               "ilgi ramūs tarpai dažniausiai reiškia gilų arba REM miegą.")
    mov = pd.DataFrame({"laikas": [iso(c) for c in clock],
                        "judesys": night["act_log"].round(3)})
    show(chart(150, {
        "data": {"values": mov.to_dict("records")},
        "mark": {"type": "area", "color": "#8a8a8a", "opacity": 0.8, "line": {"color": "#666"}},
        "encoding": {
            "x": x_axis(False),
            "y": {"field": "judesys", "type": "quantitative", "title": "Judesio stiprumas"},
            "tooltip": [{"field": "laikas", "type": "temporal", "format": "%H:%M", "title": "Laikas"},
                        {"field": "judesys", "type": "quantitative", "format": ".2f",
                         "title": "Judesio stiprumas"}]}},
        layers))

    # --- AI tikimybės
    st.subheader("Ką galvoja AI")
    st.caption("Dvi AI modelio prognozės kas 30 s: kokia tikimybė, kad žmogus šiuo metu budi "
               "(oranžinė), ir kad miega giliai (mėlyna). Žadintuvas skamba, kai žmogus budi "
               "arba pajuda ir gilaus miego tikimybė mažesnė nei 30 %.")
    prob = pd.concat([
        pd.DataFrame({"laikas": [iso(c) for c in clock], "kas": "Tikimybė, kad žmogus budi",
                      "reiksme": night["p_wake"].round(3)}),
        pd.DataFrame({"laikas": [iso(c) for c in clock], "kas": "Tikimybė, kad miegas gilus",
                      "reiksme": night["p_deep"].round(3)}),
    ])
    show(chart(210, {
        "data": {"values": prob.to_dict("records")},
        "mark": {"type": "line", "strokeWidth": 1.6},
        "encoding": {
            "x": x_axis(True),
            "y": {"field": "reiksme", "type": "quantitative", "title": "Tikimybė",
                  "axis": {"format": ".0%"}, "scale": {"domain": [0, 1]}},
            "color": {"field": "kas", "type": "nominal", "title": None,
                      "scale": {"domain": ["Tikimybė, kad žmogus budi", "Tikimybė, kad miegas gilus"],
                                "range": ["#e8833a", "#1f4e79"]},
                      "legend": {"orient": "bottom"}},
            "tooltip": [{"field": "laikas", "type": "temporal", "format": "%H:%M", "title": "Laikas"},
                        {"field": "kas", "type": "nominal", "title": "Prognozė"},
                        {"field": "reiksme", "type": "quantitative", "format": ".0%", "title": "Tikimybė"}]}},
        layers))

# ---------------------------------------------------------------- 2 skirtukas
with tab_report:
    st.subheader("Nakties ataskaita")
    st.caption("Ryte telefonas parodo nakties suvestinę. Šalia – tikros reikšmės iš miego "
               "laboratorijos, kad matytųsi, kiek AI tikslus.")
    pred = night_report((~awake).astype(int), h)
    true = night_report((night["stage4"] > 0).astype(int).to_numpy(), h)

    def onset(r):
        return "–" if r["sleep_onset_min"] is None else \
            hhmm(bed_dt + timedelta(minutes=r["sleep_onset_min"]))

    def dur(r):
        hrs = int(r["total_sleep_h"])
        return f"{hrs} val. {round((r['total_sleep_h'] - hrs) * 60)} min."

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Užmigote", onset(pred), help="Pirmasis momentas, kai AI nustatė miegą.")
    m2.metric("Miegojote", dur(pred), help="Visų miego intervalų trukmė, be pabudimų.")
    m3.metric("Miego efektyvumas", f"{pred['efficiency']:.0%}",
              help="Kiek lovoje praleisto laiko iš tikrųjų miegojote. Sveikiems "
                   "suaugusiesiems paprastai daugiau nei 85 %.")
    m4.metric("Pabudimų", pred["awakenings"],
              help="Kiek kartų pabudote bent 1 minutei po užmigimo.")

    st.table(pd.DataFrame({
        "Rodiklis": ["Užmigote", "Miego trukmė", "Miego efektyvumas", "Pabudimų skaičius"],
        "AI įvertinimas (telefonas)": [onset(pred), dur(pred), f"{pred['efficiency']:.0%}",
                                       str(pred["awakenings"])],
        "Tikra reikšmė (miego laboratorija)": [onset(true), dur(true), f"{true['efficiency']:.0%}",
                                              str(true["awakenings"])],
    }).set_index("Rodiklis"))
    st.info("Tai nėra medicininis prietaisas ir jis nenustato miego sutrikimų. "
            "Jei miegate prastai ilgą laiką, kreipkitės į gydytoją.")

# ---------------------------------------------------------------- 3 skirtukas
with tab_how:
    st.subheader("Kaip tai veikia")
    a, b, c = st.columns(3)
    a.markdown("**1. Telefonas ant čiužinio**\n\nPadėkite telefoną šalia pagalvės. Jo "
               "akselerometras (judesio jutiklis) 10 kartų per sekundę matuoja čiužinio virpesius.")
    b.markdown("**2. AI vertina miegą**\n\nKas 30 s du AI modeliai įvertina, ar budite ir ar "
               "miegate giliai. Jie remiasi tik tuo, kaip judėjote paskutines kelias minutes.")
    c.markdown("**3. Pažadina tinkamu metu**\n\nNustatytame lange žadintuvas laukia lengvo "
               "miego. Jei jo neranda, skamba vėliausiu laiku – niekada nepramiegosite.")

    st.subheader("Sąvokos")
    st.markdown("""
| Sąvoka | Ką reiškia |
|---|---|
| **Polisomnografija (PSG)** | Išsamus miego tyrimas miego laboratorijoje: matuojamos smegenų bangos, akių judesiai, raumenų įtampa, kvėpavimas. Laikomas tiksliausiu būdu nustatyti miego stadijas. Mūsų AI buvo mokomas pagal PSG rezultatus. |
| **Miego stadijos** | Per naktį miegas kelis kartus kartoja ~90 min. ciklą: lengvas miegas → gilus miegas → REM miegas. |
| **Lengvas miegas (N1, N2)** | Daugiau nei pusė nakties. Iš jo pabusti lengviausia. |
| **Gilus miegas (N3)** | Kūnas ilsisi ir atsistato. Pažadintas iš gilaus miego žmogus jaučiasi apsnūdęs (vadinamoji miego inercija). |
| **REM miegas** | Sapnų miegas. Smegenys aktyvios, bet kūno raumenys atsipalaidavę, todėl kūnas beveik nejuda. |
| **Hipnograma** | Grafikas, rodantis, kurioje miego stadijoje žmogus buvo kiekvienu nakties momentu. |
| **Žadinimo langas** | Laiko tarpas prieš vėliausią kėlimosi laiką, kuriame žadintuvas gali jus pažadinti anksčiau. |
| **Tikimybė** | Kiek AI tikras savo prognoze: 0 % – tikrai ne, 100 % – tikrai taip. |
""")

    st.subheader("Kiek tai tikslu")
    st.markdown(
        "AI patikrintas su 30 žmonių naktimis (25 350 intervalų po 30 s); kiekvienas žmogus "
        "tikrintas modeliu, kuris jo duomenų nematė.\n\n"
        "- Miegą nuo budrumo atskiria **91,5 %** atvejų.\n"
        "- Gilų miegą nuo kitų stadijų atskiria **87 %** atvejų.\n"
        "- Simuliacijoje išmanusis žadintuvas pažadino lengvo miego metu **70 %** atvejų, "
        "įprastas – **65 %**.\n\n"
        "**Ribotumai:** duomenys užrašyti ant riešo, ne čiužinio; tirti tik sveiki suaugusieji; "
        "ramų REM miegą pagal judesį sunku atskirti nuo gilaus miego.\n\n"
        "Duomenys: Walch O. (2019), *Motion and heart rate from a wrist-worn wearable and "
        "labeled sleep from polysomnography*, PhysioNet.")
