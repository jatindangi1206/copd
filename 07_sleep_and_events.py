"""HRV during recorded sleep, and HRV around each dated exacerbation.

    python 07_sleep_and_events.py

07a: per patient, median HRV inside vs outside the watch's sleep blocks.
07b: every HRV reading within WINDOW days of each dated exacerbation, with the
     daily median. Descriptive only - no test.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import BLUE, ORANGE, GREY, INK2, LINE, NUMBERS, master, patients, save
from codings import load_exacerbations

WINDOW = 14        # days either side of an exacerbation date
MIN_EACH = 10      # readings a patient needs in each state for 07a

H = pd.concat([master(p, ["time", "hrv", "sleep_active"]).assign(pid=p) for p in patients()])
H = H[H.hrv.notna()]

# ---- 07a: sleep vs outside sleep, within patient
rows = []
for pid, d in H.groupby("pid"):
    a, b = d[d.sleep_active == 1].hrv, d[d.sleep_active != 1].hrv
    if len(a) >= MIN_EACH and len(b) >= MIN_EACH:
        rows.append(dict(patient=pid, sleep=a.median(), outside=b.median(),
                         n_sleep=len(a), n_outside=len(b)))
S = pd.DataFrame(rows)
S["diff"] = S.sleep - S.outside
S = S.sort_values("diff").reset_index(drop=True)
S.to_csv(NUMBERS / "hrv_sleep.csv", index=False)

fig, ax = plt.subplots(figsize=(13, 5))
ax.vlines(S.index, S.sleep, S.outside, color=LINE, lw=2.5)
ax.scatter(S.index, S.outside, color=BLUE, s=45, zorder=3, label="outside recorded sleep")
ax.scatter(S.index, S.sleep, color=ORANGE, s=45, zorder=3, label="inside recorded sleep")
ax.set(xticks=S.index, ylabel="median HRV")
ax.set_xticklabels(S.patient, rotation=90, fontsize=8)
ax.legend(frameon=False, loc="upper left")
ax.set_title(f"Median HRV inside vs outside sleep: lower in sleep for "
             f"{(S['diff'] < 0).sum()} of {len(S)} patients")
save(fig, "07a-hrv-sleep")

# ---- 07b: HRV around each dated exacerbation
E = load_exacerbations()
ncol = 4
nrow = int(np.ceil(len(E) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.8 * nrow), sharey=True)
axes = axes.ravel()
for ax, (_, e) in zip(axes, E.iterrows()):
    h = H[H.pid == e.pid]
    rel = (h.time - e.date).dt.total_seconds() / 86400
    w = rel[(rel >= -WINDOW) & (rel < WINDOW + 1)]
    ax.axvspan(0, 1, color=ORANGE, alpha=.2, lw=0)
    if len(w):
        v = h.hrv[w.index]
        ax.scatter(w, v, s=2, color=GREY, alpha=.35, rasterized=True)
        dm = v.groupby(np.floor(w)).median()
        ax.plot(dm.index + .5, dm.values, "o-", color=BLUE, ms=3, lw=1.4)
        ax.axhline(h.hrv.median(), color=INK2, ls="--", lw=1)
    else:
        ax.text(.5, .5, "no HRV readings\nin these 4 weeks", transform=ax.transAxes,
                ha="center", va="center", color=INK2)
    ax.set(xlim=(-WINDOW, WINDOW + 1), ylim=(15, 135))
    ax.set_title(f"{e.pid}   {e.date:%d %b %Y}", fontsize=10)
    ax.tick_params(labelsize=8)
for ax in axes[len(E):]:
    ax.axis("off")
fig.supxlabel("days from the recorded exacerbation date (shaded)")
fig.supylabel("HRV")
fig.suptitle("HRV two weeks either side of each exacerbation   "
             "(grey: readings, blue: daily median, dashed: patient's usual median)", y=1.0)
fig.tight_layout()
save(fig, "07b-hrv-around-exacerbations")
print(S.to_string(index=False))
