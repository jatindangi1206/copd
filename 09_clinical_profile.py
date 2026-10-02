"""The rest of the datasheet: what each section holds, and the profile it gives.

    python 09_clinical_profile.py

09a: % of cells filled in each datasheet section (33 watch patients).
09b: symptoms, history and exam findings (yes/no fields), plus etiotype,
     smoking status and symptom control.
09c: lung function (spirometry, oscillometry) and the six-minute walk test.
numbers/comorbidities.csv, imaging.csv, treatment.csv: small tables for the report.

The sheet repeats column names (DUR_MNTHS_0, NOR_ABN, TRTMNT_0 ...), so columns
are read by position within the raw sheet. Values are as recorded; the only
change is the sheet's own coding (Yes=1/No=2, ND/NK/NA blank).
"""
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import BLUE, ORANGE, INK2, NUMBERS, SHEET, save
from codings import MISSING, clean

raw = pd.ExcelFile(SHEET).parse("Baseline", header=None)
section = raw.iloc[1].ffill().astype(str).str.strip().str.rstrip(":").str.strip()
names = raw.iloc[2].astype(str).str.strip()
d = raw.iloc[3:]
pid = d[names[names == "PARTCPNT_ID"].index[0]].astype(str).str.strip().str.lower()
d = d[pid.str.match(r"^c\d{3}$")]          # every patient row in the sheet
N = len(d)


def col(name, k=0):
    return d[[i for i in names.index if names[i] == name][k]]


def txt(s):
    return s.astype(str).str.strip().str.upper()


def filled(s):
    t = txt(s)
    return s.notna() & ~t.isin(MISSING) & (t != "NAN")


# ---- 09a: completeness by section
LABEL = {"DEMOGRAPHIC DETAILS": "Age and sex", "CURRENT SYMPTOMS": "Current symptoms",
         "DISEASE SPECIFIC DETAILS": "Disease history", "ETIOTYPE": "Etiotype",
         "GOLD CLASSIFICATION": "GOLD group, smoking, exposures, comorbidity",
         "PHYSICAL EXAMINATION": "Physical examination", "RESPIRATORY SYMPTOM": "Chest examination",
         "ONGOING MANAGEMENT": "Treatment", "CBC BASELINE": "Blood count", "LFT BASELINE": "Liver tests",
         "KFT BASELINE": "Kidney tests", "LIPID PROFILE BASELINE": "Lipids",
         "THYROID PROFILE BASELINE": "Thyroid", "DIABETIC PROFILE BASELINE": "Blood sugar",
         "IgE": "IgE and other tests", "IMAGING": "Imaging (X-ray, CT, ECG, echo)",
         "SIX_MINUTE_WALK_TEST": "Six-minute walk test", "SPIROMETRY_PRE": "Spirometry, before bronchodilator",
         "SPIROMETRY_POST": "Spirometry, after bronchodilator", "IMPULSE_OSCILLOMETRY": "Oscillometry",
         "DETAILS RELATED TO WEARABLES & TR EATMENT PLAN (SMART WATCH)": "Diagnosis and watch details",
         "CAAT ASSESSMENT": "CAT score", "BODE_ INDEX": "BODE score"}
rows = []
for c in d.columns:
    if names[c] in ("S_NO", "PARTCPNT_ID"):
        continue
    rows.append((LABEL.get(section[c], section[c]), filled(d[c]).mean()))
C = pd.DataFrame(rows, columns=["section", "fill"])
C = C.groupby("section", sort=False).agg(columns=("fill", "size"), pct=("fill", "mean"))
C["pct"] *= 100
C.to_csv(NUMBERS / "datasheet_sections.csv")
fig, ax = plt.subplots(figsize=(11, 8))
ax.barh(C.index[::-1], C.pct[::-1], color=BLUE, height=.7)
for i, (p, n) in enumerate(zip(C.pct[::-1], C["columns"][::-1])):
    ax.text(p + 1, i, f"{p:.0f}%  ({n} column{'s' if n > 1 else ''})", va="center", color=INK2, fontsize=9.5)
ax.set(xlim=(0, 118), xlabel=f"% of cells filled ({N} patients)")
ax.grid(axis="y", visible=False)
ax.set_title("What the clinical datasheet holds, section by section")
save(fig, "09a-datasheet-completeness")

# ---- 09b: symptoms, history, exam
yn = lambda s: clean(s).map({1: 1.0, 2: 0.0, 0: 0.0})
fam = txt(col("H/O_FAM_COPD"))
groups = {
    "Current symptoms": [("Cough", yn(col("COUGH_0"))), ("Breathlessness", yn(col("BRTHLSNS_0"))),
                         ("Sputum", yn(col("SPUTUM_0"))), ("Wheezing", yn(col("WHZNG_0"))),
                         ("Palpitation", yn(col("PALPITATION_0"))), ("Fever", yn(col("FEVER_0"))),
                         ("Pedal oedema", yn(col("PEDAL_EDEMA_0")))],
    "History": [("Smoking", yn(col("SMOKE_HX"))), ("Occupational exposure", yn(col("OCC_EXP"))),
                ("Comorbidity", yn(col("H/O_COMORB"))), ("Biomass fuel exposure", yn(col("H/O_BMFE"))),
                ("Family history of COPD", fam.map(lambda v: 0.0 if v == "2" else 1.0)),
                ("Sleep disturbance", yn(col("SLP_DIST"))), ("Past TB", yn(col("H/O_TB")))],
    "Examination": [("Reduced breath sounds", ((clean(col("BRTH_SOUNDS_RT_0")) == 2) |
                                               (clean(col("BRTH_SOUNDS_LT_0")) == 2)).astype(float)),
                    ("Clubbing", yn(col("CLUB_0"))), ("Cyanosis", yn(col("CYAN_0"))),
                    ("Oedema", yn(col("EDEMA_0")))],
}
cats = [("Etiotype", col("ETIOTYPE"), {"C": "cigarette smoking", "P": "pollution"}),
        ("Smoking now", col("CURNT_STATUS"), {"1": "continues", "2": "stopped", "0": "never smoked"}),
        ("Symptom control", col("CRNT_STAT_SYM_CNTRL"), {"1": "controlled", "2": "partly", "3": "uncontrolled"})]
fig = plt.figure(figsize=(16, 8.5))
gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], hspace=.55, wspace=.45)
labels, vals, ns, ypos, y = [], [], [], [], 0
axl = fig.add_subplot(gs[:, 0])
heads = []
for g, items in groups.items():
    heads.append((y, g))
    y += 1
    for lab, s in items:
        labels.append(lab); vals.append(int(s.sum())); ns.append(int(s.notna().sum())); ypos.append(y)
        y += 1
    y += .3
axl.barh(ypos, vals, color=BLUE, height=.7)
for yy, v, n in zip(ypos, vals, ns):
    axl.text(v + .3, yy, f"{v} of {n}", va="center", color=INK2, fontsize=9.5)
axl.set(yticks=[h for h, _ in heads] + ypos, xlim=(0, N + 5), xlabel="patients recorded as yes")
axl.set_yticklabels([g.upper() for _, g in heads] + labels)
for t in axl.get_yticklabels()[:len(heads)]:
    t.set_fontweight("bold"); t.set_fontsize(9.5)
axl.invert_yaxis()
axl.grid(axis="y", visible=False)
axl.set_title("Symptoms, history and examination at enrolment")
for k, (title, s, m) in enumerate(cats):
    a = fig.add_subplot(gs[k, 1])
    v = txt(s).map(lambda x: m.get(x, "not recorded" if x in ("NAN", "") else x)).value_counts()
    v = v.reindex([*m.values(), "not recorded"]).dropna().astype(int)
    v = v[v > 0]
    a.barh(v.index[::-1], v.values[::-1], color=ORANGE, height=.6)
    for i, x in enumerate(v.values[::-1]):
        a.text(x + .3, i, str(x), va="center", color=INK2, fontsize=9.5)
    a.set(xlim=(0, N + 3))
    a.grid(axis="y", visible=False)
    a.set_title(title)
save(fig, "09b-symptoms-and-history")

# ---- 09c: lung function and walk test
panels = [("FEV1, % of predicted (spirometry)", clean(col("FEV 1 [L]_%PRED_PRE_0")), np.arange(0, 110, 5)),
          ("FVC, % of predicted (spirometry)", clean(col("FVC [L]_%PRED_PRE_0")), np.arange(0, 140, 5)),
          ("R5, % of predicted (oscillometry)", clean(col("R5_%PRED_0")), 12),
          ("X5, % of predicted (oscillometry)", clean(col("X5_%PRED_0")), 12),
          ("Six-minute walk distance (m)", clean(col("DIS_COVRD_0")), np.arange(0, 720, 50)),
          ("Lowest SpO2 during the walk (%)", clean(col("MIN_SPO2_0")), np.arange(64, 100, 2))]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for a, (t, s, bins) in zip(axes.ravel(), panels):
    s = s.dropna()
    a.hist(s, bins=bins, color=BLUE, rwidth=.9)
    a.axvline(s.median(), color=ORANGE, lw=2)
    a.set(ylabel="patients", title=f"{t}\nn={len(s)}, median {s.median():.0f}")
fig.suptitle("Lung function and walk test at enrolment (before bronchodilator)", y=1.0)
fig.tight_layout()
save(fig, "09c-lung-and-walk")

# ---- tables
diag = txt(col("DIAGNOSIS_PROB_CAUSE")) + " " + txt(col("IF_YES_SPCFY", 0))
KEYS = [("Hypertension", r"\bS?HTN\b"), ("Diabetes", r"\b(?:T2DM|DM2?|TYPE-II DM)\b"),
        ("Sleep apnoea (OSA)", r"\bOSA\b"), ("Dyslipidaemia", r"DYSLIPIDEMIA"),
        ("Coronary artery disease", r"\bCAD\b"), ("Hypothyroidism", r"HYPOTHYROID"),
        ("Chronic kidney disease", r"\bCKD\b")]
pd.DataFrame([(k, int(diag.str.contains(p, regex=True).sum())) for k, p in KEYS],
             columns=["condition", "patients"]).to_csv(NUMBERS / "comorbidities.csv", index=False)

img = []
for k, test in enumerate(["CXR (PA view)", "CT chest", "ECG", "2D echo"]):
    avail = clean(d[[i for i in names.index if names[i] in ("CXR_PA_VIEW", "CT Chest / Thorax", "ECG", "2D ECHO")][k]])
    res = txt(col("NOR_ABN", k))
    img.append(dict(test=test, available=int((avail == 1).sum()), normal=int((res == "1").sum()),
                    abnormal=int((res == "2").sum()), coded_0=int((res == "0").sum()),
                    blank=int(res.isin(["NAN", ""]).sum())))
pd.DataFrame(img).to_csv(NUMBERS / "imaging.csv", index=False)

tr = pd.concat([txt(col("TRTMNT_0", k)) for k in range(5)] + [txt(col("TRTMNT_5"))], axis=1)
n_listed = (~tr.isin(["NAN", "", "0"])).sum(axis=1)
first = txt(col("TRTMNT_0", 0))
pd.DataFrame({"first_listed": first.value_counts().index, "patients": first.value_counts().values}) \
    .to_csv(NUMBERS / "treatment.csv", index=False)
n_listed.value_counts().sort_index().rename("patients").rename_axis("treatments_listed") \
    .to_csv(NUMBERS / "treatment_count.csv")
print(C.round(0).to_string())
