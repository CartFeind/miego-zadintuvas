# Miego ciklų žadintuvas be išmaniųjų laikrodžių

Autoriai: Jonas Olišauskas, Danielius Mažeika, Nedas Anuškevičius

Telefonas, padėtas ant čiužinio, matuoja kūno judesius akselerometru.
AI modelis kas 30 sekundžių įvertina miego būseną ir:

1. **pažadina lengvo miego metu** vartotojo nustatytame lange (pvz., 06:30–07:00);
2. **sudaro nakties ataskaitą**: užmigimo laikas, miego trukmė, efektyvumas, pabudimai.

Naudojami **tik judesio duomenys** – jokio širdies ritmo ar išmaniojo laikrodžio.

## Duomenys

PhysioNet *Motion and heart rate from a wrist-worn wearable and labeled sleep
from polysomnography* (Walch, 2019), https://doi.org/10.13026/hmhs-py35.
31 žmogus, viena naktis miego laboratorijoje, stadijos pažymėtos pagal PSG.
Licencija: Open Data Commons Attribution v1.0. Duomenys į repozitoriją neįkelti.

## Paleidimas (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

:: 1) atsisiųskite ZIP iš PhysioNet ir išskleiskite į data\raw
python src\prepare_data.py data\raw

:: 2) eksperimentai (~10 min, rezultatai -> results\)
python src\train.py

:: 3) demonstracija naršyklėje
streamlit run app.py
```

Demonstracijai (3 žingsnis) duomenų atsisiųsti nebūtina – reikalingas failas
`results/loso_prognozes.csv` jau yra repozitorijoje.

## Struktūra

| Failas | Paskirtis |
|---|---|
| `src/features.py` | 30 s epochų judesio požymiai (10 Hz), konteksto požymiai |
| `src/prepare_data.py` | žali failai → `data/processed/epochs.csv` |
| `src/train.py` | Baseline vs Proposed, LOSO vertinimas, grafikai, XAI |
| `src/alarm.py` | žadintuvo taisyklė, simuliacija, nakties ataskaita |
| `src/learning_curve.py` | mokymosi kreivės grafikas |
| `app.py` | Streamlit demonstracija |
| `results/` | metrikos, konfuzijos matricos, grafikai, modeliai |

## Pagrindiniai rezultatai (Leave-One-Subject-Out, 30 žmonių)

| Užduotis | Geriausias baseline F1 | Proposed F1 | Proposed tikslumas |
|---|---|---|---|
| A: miegas / budrumas | 0,673 | **0,761** | 91,5 % |
| B: gilus miegas / kita | 0,631 | **0,720** | 87,0 % |
| C: 4 stadijos | 0,354 | 0,535 | 56,2 % |

Žadintuvo simuliacija: lengvo miego / budrumo metu pažadina 70,2 % (įprastas 64,7 %).
Visos metrikos: `results/metrikos.csv`.

## Apribojimai

Tai mokomasis projektas, ne medicininis prietaisas. Duomenys užrašyti ant riešo,
o ne ant čiužinio; tiriamieji – sveiki suaugusieji (miego apnėja atmesta).
