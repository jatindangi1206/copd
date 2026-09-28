"""Sleep stages (deep / light / almost awake) and HRV.

    python 08_sleep_stages.py

The watch does not label minutes with a stage. Each sleep block comes with how
many of its minutes were deep, light and almost awake, so everything here is
per block: a block is a run of sleep_active == 1 in the master, its stage
minutes are summed, and its HRV is the mean of the readings inside it. Same
method as copd_eda/19_sleep_stages.py.

08a: each patient's split of scored sleep into the three stages.
08b: per patient, median block HRV in deep-dominated vs light-dominated blocks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import BLUE, ORANGE, INK2, LINE, NUMBERS, STAGE_COLORS, master, patients, save

DOMINANT = 70      # % of a block's scored minutes one stage needs to "dominate" it
MIN_HRV = 3        # HRV readings a block needs to get a mean
STAGES = [("deep", "deep"), ("light", "light"), ("awake", "almost awake")]

rows = []
for pid in patients():
    m = master(pid, ["time", "hrv", "sleep_active", "deep_sleep", "light_sleep", "almost_awake"])
    st = m[["deep_sleep", "light_sleep", "almost_awake"]].fillna(0)
    if st.to_numpy().sum() == 0:
        continue
    sa = m.sleep_active.fillna(0).astype(int).to_numpy()
    m = m.assign(run=(np.diff(np.r_[0, sa]) == 1).cumsum() * sa,
                 deep=st.deep_sleep, light=st.light_sleep, awake=st.almost_awake)
    b = m[m.run > 0].groupby("run").agg(deep=("deep", "sum"), light=("light", "sum"),
                                         awake=("awake", "sum"), n_hrv=("hrv", "count"),
                                         hrv=("hrv", "mean"), minutes=("time", "size"))
    b = b[b[["deep", "light", "awake"]].sum(axis=1) > 0].assign(patient=pid)
    rows.append(b.reset_index(drop=True))
B = pd.concat(rows, ignore_index=True)
B["scored"] = B[["deep", "light", "awake"]].sum(axis=1)
for s, _ in STAGES:
    B[f"{s}_pct"] = 100 * B[s] / B.scored
B.to_csv(NUMBERS / "sleep_blocks.csv", index=False)

# ---- 08a: stage split per patient
pp = B.groupby("patient")[["deep", "light", "awake"]].sum()
pp = 100 * pp.div(pp.sum(axis=1), axis=0).sort_values("deep")
tot = 100 * B[["deep", "light", "awake"]].sum() / B.scored.sum()
pp.loc["all patients"] = tot
fig, ax = plt.subplots(figsize=(11, 9))
left = np.zeros(len(pp))
for s, lab in STAGES:
    ax.barh(pp.index, pp[s], left=left, color=STAGE_COLORS[s], label=lab, height=.75)
    left += pp[s].values
ax.axhline(len(pp) - 1.5, color=INK2, lw=.8)
ax.set(xlim=(0, 100), xlabel="% of scored sleep minutes")
ax.grid(axis="y", visible=False)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.0))
ax.set_title(f"Sleep stages per patient ({B.patient.nunique()} patients, {len(B):,} sleep blocks)",
             pad=28)
ax.tick_params(axis="y", labelsize=8.5)
save(fig, "08a-sleep-stage-split")

# ---- 08b: deep vs light blocks, within patient
H = B[B.n_hrv >= MIN_HRV].copy()
H["dominant"] = np.select([H.deep_pct >= DOMINANT, H.light_pct >= DOMINANT], ["deep", "light"], "mixed")
D = H[H.dominant != "mixed"].groupby(["patient", "dominant"]).hrv.median().unstack().dropna()
D["diff"] = D.deep - D.light
D = D.sort_values("diff").reset_index()
D.to_csv(NUMBERS / "sleep_stage_hrv.csv", index=False)
fig, ax = plt.subplots(figsize=(13, 5))
ax.vlines(D.index, D.deep, D.light, color=LINE, lw=2.5)
ax.scatter(D.index, D.light, color=STAGE_COLORS["light"], s=45, zorder=3, label="light-sleep blocks",
           edgecolor=INK2, lw=.5)
ax.scatter(D.index, D.deep, color=BLUE, s=45, zorder=3, label="deep-sleep blocks")
ax.set(xticks=D.index, ylabel="median HRV of the blocks")
ax.set_xticklabels(D.patient, rotation=90, fontsize=8)
ax.legend(loc="upper left")
ax.set_title(f"HRV in deep vs light sleep: lower in deep for {(D['diff'] < 0).sum()} of {len(D)} patients")
save(fig, "08b-hrv-deep-vs-light")
print(f"{B.patient.nunique()} patients, {len(B):,} blocks; stage share "
      + ", ".join(f"{k} {v:.1f}%" for k, v in tot.items())
      + f"; deep lower in {(D['diff'] < 0).sum()} of {len(D)}")
