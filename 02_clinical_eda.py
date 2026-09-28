"""Baseline clinical EDA for the 33 COPD patients who also have wearable data.

    python 02_clinical_eda.py

figs/clin_cohort.png       who these patients are
figs/clin_severity.png     severity and exacerbation burden
figs/clin_corr.png         correlations among the pre-specified variables
figs/clin_vs_wearable.png  clinic vitals against the watch's own readings

VARIABLE CHOICE. n = 33. A correlation matrix over the sheet's 217 columns would
be ~23,000 comparisons on 33 subjects and would manufacture "findings" from noise,
so the set below is pre-specified on physiological grounds and kept small. No
p-values are reported: at this n they would mislead. Effect sizes and scatter
plots with n on every panel instead.

Excluded, with reasons, in numbers_clinical.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codings import load_baseline, load_exacerbations, clean, yes_no, mmrc, DECODE

from common import FIGS, WEAR as MASTER, BLUE as C1, ORANGE as C2, INK2, GREY
ROOT = Path(__file__).resolve().parent
out = []
def say(s=""):
    print(s); out.append(s)

B = load_baseline()
E = load_exacerbations()

# ---------------------------------------------------- the pre-specified frame
X = pd.DataFrame({"pid": B.pid.values})
X["age"] = clean(B["AGE"]).values
X["sex"] = clean(B["GENDER"]).map(DECODE["GENDER"]).values
X["bmi"] = clean(B["BMI_kg/M2_0"]).values
X["mmrc"] = mmrc(B["mMRC_GRD_0"]).values
X["gold"] = B["GOLD_CLSFCTN"].astype(str).str.strip().values
X["cat"] = clean(B["CAAT_TOT_SCORE_0"]).values
X["bode"] = clean(B["BODE_INDXTOT_SCORE_0"]).values
X["exac_12m"] = clean(B["NO_OF_EXACERBTNS_IN_12_MNTHS"]).values
X["smoke_index"] = clean(B["SMK_INDX"]).values
X["biomass"] = yes_no(B["H/O_BMFE"]).values
X["comorbid"] = yes_no(B["H/O_COMORB"]).values
X["sleep_dist"] = yes_no(B["SLP_DIST"]).values
X["pulse"] = clean(B["PULSE_PM_0"]).values
X["spo2"] = clean(B["SPO2_%_0"]).values
X["rr"] = clean(B["RR_PM_0"]).values
X["bp_sys"] = B.bp_systolic.values
X["walk_dist"] = clean(B["DIS_COVRD_0"]).values
X["walk_hr_avg"] = clean(B["AVG__HR_0"]).values
X["walk_spo2_min"] = clean(B["MIN_SPO2_0"]).values
X["fev1_pct"] = clean(B["FEV 1 [L]_%POSTD_POST_0"]).values
X["hb"] = clean(B["Hb_g/dl_0"]).values
X["hba1c"] = clean(B["HbA1C_%_0"]).values
cnt = E.groupby("pid").size()
X["exac_dated"] = X.pid.map(cnt).fillna(0).astype(int)
X.to_csv(Path(__file__).resolve().parent / "clinical_table.csv", index=False)

say("# COPD baseline clinical EDA")
say()
say(f"**{len(X)} patients** — every wearable patient has a datasheet row "
    f"(33/33, no id casualties). 8 further sheet patients (c034–c041) have no "
    f"wearable data and are excluded.")
say()
say("## Cohort")
say()
say(f"- Age **{X.age.min():.0f}–{X.age.max():.0f}**, median {X.age.median():.0f}")
say(f"- Sex: " + ", ".join(f"{k} {v}" for k, v in X.sex.value_counts().items()))
say(f"- BMI median {X.bmi.median():.1f} (range {X.bmi.min():.1f}–{X.bmi.max():.1f})")
say(f"- GOLD: " + ", ".join(f"{k} {v}" for k, v in X.gold.value_counts().sort_index().items()))
say(f"- mMRC median {X.mmrc.median():.1f}; CAT median {X.cat.median():.0f}; "
    f"BODE median {X.bode.median():.0f}")
say(f"- Ever-smoker {int((clean(B['SMOKE_HX'])==1).sum())}/33, "
    f"biomass exposure {int(X.biomass.sum())}/33, "
    f"comorbidity {int(X.comorbid.sum())}/33, "
    f"**sleep disturbance {int(X.sleep_dist.sum())}/33**")
say()

# ------------------------------------------------------------- fig 1: cohort
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5)); ax = ax.ravel()
ax[0].hist(X.age.dropna(), bins=np.arange(0, 90, 5), color=C1); ax[0].set(xlabel="age (years, as recorded)", ylabel="patients")
ax[0].set_title(f"Age (n={X.age.notna().sum()})", loc="left", fontweight="bold")
v = X.sex.value_counts(); ax[1].bar(v.index, v.values, color=C1, width=.5)
for i, k in enumerate(v.values): ax[1].text(i, k, str(k), ha="center", va="bottom", fontweight="bold")
ax[1].set(ylabel="patients"); ax[1].set_title("Sex", loc="left", fontweight="bold")
ax[2].hist(X.bmi.dropna(), bins=np.arange(14, 42, 2), color=C1); ax[2].set(xlabel="BMI")
ax[2].set_title("BMI", loc="left", fontweight="bold")
v = X.gold.value_counts().sort_index(); ax[3].bar(v.index, v.values, color=C2, width=.5)
for i, k in enumerate(v.values): ax[3].text(i, k, str(k), ha="center", va="bottom", fontweight="bold")
ax[3].set(ylabel="patients"); ax[3].set_title("GOLD group (A/B low-exac, E high)", loc="left", fontweight="bold")
ax[4].hist(X.mmrc.dropna(), bins=np.arange(-.25, 4, .5), color=C2); ax[4].set(xlabel="mMRC grade")
ax[4].set_title("mMRC (ranges at midpoint)", loc="left", fontweight="bold")
ax[5].hist(X.smoke_index.dropna(), bins=12, color=C2); ax[5].set(xlabel="smoking index")
ax[5].set_title(f"Smoking index ({int((X.smoke_index==0).sum())} never-smokers)", loc="left", fontweight="bold")
fig.suptitle(f"COPD cohort with wearable data (n={len(X)})", fontweight="bold", y=1.0)
fig.tight_layout(); fig.savefig(FIGS/"clin_cohort.png", dpi=145); plt.close(fig)

# ---------------------------------------------------------- fig 2: severity
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
ax[0].scatter(X.cat, X.bode, s=55, color=C1, alpha=.8)
ax[0].set(xlabel="CAT score (0-40)", ylabel="BODE index (0-10)")
ax[0].set_title(f"CAT vs BODE   r={X.cat.corr(X.bode):+.2f}  n={int((X.cat.notna()&X.bode.notna()).sum())}",
                loc="left", fontweight="bold")
ax[1].scatter(X.mmrc, X.walk_dist, s=55, color=C2, alpha=.8)
ax[1].set(xlabel="mMRC grade", ylabel="6MWT distance (m)")
ax[1].set_title(f"Breathlessness vs walk distance   r={X.mmrc.corr(X.walk_dist):+.2f}  "
                f"n={int((X.mmrc.notna()&X.walk_dist.notna()).sum())}", loc="left", fontweight="bold")
b = X.groupby("exac_dated").size()
ax[2].bar(b.index, b.values, color=C2, width=.6); ax[2].set_xticks(b.index)
for i, k in b.items(): ax[2].text(i, k, str(k), ha="center", va="bottom", fontweight="bold")
ax[2].set(xlabel="dated exacerbation episodes during monitoring", ylabel="patients")
ax[2].set_title(f"{int(X.exac_dated.sum())} episodes in {int((X.exac_dated>0).sum())} patients",
                loc="left", fontweight="bold")
fig.suptitle("Severity and exacerbation burden", fontweight="bold", y=1.0)
fig.tight_layout(); fig.savefig(FIGS/"clin_severity.png", dpi=145); plt.close(fig)

# ------------------------------------------------------------- fig 3: corr
NUM = ["age","bmi","mmrc","cat","bode","exac_12m","exac_dated","smoke_index",
       "pulse","spo2","rr","bp_sys","walk_dist","walk_hr_avg","walk_spo2_min",
       "fev1_pct","hb","hba1c"]
M = X[NUM]
corr = M.corr(method="spearman")
cnts = (~M.isna()).astype(int).T.dot((~M.isna()).astype(int))
fig, ax = plt.subplots(figsize=(11.5, 9.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set(xticks=range(len(NUM)), yticks=range(len(NUM)))
ax.set_xticklabels(NUM, rotation=90); ax.set_yticklabels(NUM)
for i in range(len(NUM)):
    for j in range(len(NUM)):
        if i != j and cnts.iloc[i, j] >= 10:
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}".replace("0.", "."), ha="center",
                    va="center", fontsize=7.5,
                    color="white" if abs(corr.iloc[i,j]) > .55 else "0.2")
fig.colorbar(im, ax=ax, shrink=.8, label="Spearman r")
ax.set_title(f"Spearman correlations, {len(X)} patients (each cell uses those with both values)\n"
             "blank cells have fewer than 10 complete pairs; no p-values by design",
             loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(FIGS/"clin_corr.png", dpi=145); plt.close(fig)

# -------------------------------------------------- fig 4: clinic vs wearable
wear = []
for pid in X.pid:
    f = MASTER / f"{pid}.csv.gz"
    if not f.exists(): continue
    m = pd.read_csv(f, usecols=["hr","spo2","hrv"])
    wear.append(dict(pid=pid, w_hr=m.hr.median(), w_spo2=m.spo2.median(),
                     w_hrv=m.hrv.median(), n_hrv=int(m.hrv.notna().sum())))
W = X.merge(pd.DataFrame(wear), on="pid", how="left")
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
for k, (a, b_, la, lb) in enumerate((
        ("pulse","w_hr","clinic: pulse (per min)","watch: median heart rate (per min)"),
        ("spo2","w_spo2","clinic: SpO2 (%)","watch: median SpO2 (%)"),
        ("walk_hr_avg","w_hr","clinic: average heart rate in walk test","watch: median heart rate (per min)"))):
    d = W.dropna(subset=[a, b_])
    ax[k].scatter(d[a], d[b_], s=55, color=GREY, alpha=.85)
    lim = [min(d[a].min(), d[b_].min())-4, max(d[a].max(), d[b_].max())+4]
    ax[k].plot(lim, lim, ls="--", color=INK2, lw=1)
    ax[k].set_xlabel(la, color=C2, fontweight="bold"); ax[k].set_ylabel(lb, color=C1, fontweight="bold")
    ax[k].tick_params(axis="x", colors=C2); ax[k].tick_params(axis="y", colors=C1)
    ax[k].spines["bottom"].set_color(C2); ax[k].spines["left"].set_color(C1)
    ax[k].set_title(f"r={d[a].corr(d[b_]):+.2f}   n={len(d)}", loc="left", fontweight="bold")
fig.suptitle("Clinic measurement against the watch's own readings", fontweight="bold", y=1.0)
fig.tight_layout(); fig.savefig(FIGS/"clin_vs_wearable.png", dpi=145); plt.close(fig)

say("## Correlations worth reporting (Spearman, n in brackets)")
say()
say("| pair | r | n |")
say("|---|---|---|")
for a, b_ in [("cat","bode"),("mmrc","walk_dist"),("bode","walk_dist"),("fev1_pct","bode"),
              ("age","walk_dist"),("spo2","walk_spo2_min"),("smoke_index","fev1_pct"),
              ("bmi","bode"),("exac_12m","cat")]:
    d = X.dropna(subset=[a, b_])
    if len(d) >= 10:
        say(f"| {a} vs {b_} | {d[a].corr(d[b_], method='spearman'):+.2f} | {len(d)} |")
say()
say("## Clinic vs wearable")
say()
for a, b_, lab in (("pulse","w_hr","clinic pulse vs wearable median HR"),
                   ("spo2","w_spo2","clinic SpO2 vs wearable median SpO2")):
    d = W.dropna(subset=[a, b_])
    say(f"- {lab}: r = **{d[a].corr(d[b_]):+.2f}** (n={len(d)}), "
        f"median difference {(d[b_]-d[a]).median():+.1f}")
say()
(ROOT/"numbers"/"clinical.md").write_text("\n".join(out)+"\n")
print(f"\n-> 4 figures in figs/, numbers in numbers/clinical.md")
