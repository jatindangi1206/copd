"""How HRV is distributed, per patient and overall.

    python 06_hrv_distributions.py
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from common import BLUE, ORANGE, master, patients, save

MIN_READINGS = 200
GRID = np.linspace(20, 130, 300)

curves, vals = {}, []
for pid in patients():
    v = master(pid, ["hrv"]).hrv.dropna()
    if len(v):
        vals.append(pd.DataFrame({"patient": pid, "hrv": v.values}))
    if len(v) >= MIN_READINGS and v.std() > 0:
        curves[pid] = gaussian_kde(v)(GRID)
R = pd.concat(vals, ignore_index=True)

fig, ax = plt.subplots(figsize=(10, 6))
for c in curves.values():
    ax.plot(GRID, c, color=BLUE, lw=1.2, alpha=.55)
ax.set(xlabel="HRV", ylabel="density", xlim=(20, 130))
ax.set_title(f"One curve per patient ({len(curves)} patients)")
save(fig, "06a-hrv-curves-per-patient")

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.hist(R.hrv, bins=np.arange(20, 132, 1), color=BLUE)
ax.set(xlabel="HRV", ylabel="readings")
ax.set_title("All readings")
save(fig, "06b-hrv-histogram")

pids = list(curves)
ncol, nrow = 4, int(np.ceil(len(pids) / 4))
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.4 * nrow), sharex=True)
axes = np.atleast_1d(axes).ravel()
for ax, pid in zip(axes, pids):
    ax.fill_between(GRID, curves[pid], color=BLUE, alpha=.5)
    ax.set(title=pid, xlim=(20, 130), yticks=[])
    ax.tick_params(labelsize=8)
for ax in axes[len(pids):]:
    ax.axis("off")
fig.supxlabel("HRV")
fig.tight_layout()
save(fig, "06c-hrv-curve-each-patient")

fig, ax = plt.subplots(figsize=(11, 5.5))
med = R[R.patient.isin(curves)].groupby("patient").hrv.median().sort_values()
ax.bar(med.index, med.values, color=BLUE)
ax.set(ylabel="median HRV", xlabel="")
ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.set_title(f"Median HRV per patient ({len(med)} patients with at least {MIN_READINGS} readings)")
save(fig, "06d-median-hrv-per-patient")
print(f"{len(R):,} readings, {R.patient.nunique()} patients, "
      f"{len(curves)} with at least {MIN_READINGS}")
