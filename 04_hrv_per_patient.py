"""Raw HRV over time, one panel per patient.

    python 04_hrv_per_patient.py

Every recorded HRV reading as a dot, on that patient's own timeline. Gaps on
screen are gaps in the data - nothing is filled.
"""
import numpy as np
import matplotlib.pyplot as plt

from common import BLUE, INK2, master, patients, save
from codings import load_baseline

pids = sorted(load_baseline().pid)          # every patient in the sheet
have = set(patients())
ncol, nrow = 4, int(np.ceil(len(pids) / 4))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.5 * nrow))
axes = np.atleast_1d(axes).ravel()
for ax, pid in zip(axes, pids):
    d = master(pid, ["time", "hrv"]).dropna(subset=["hrv"]) if pid in have else None
    if d is None or d.empty:
        ax.text(.5, .5, "no watch data" if d is None else "no HRV readings", transform=ax.transAxes,
                ha="center", va="center", color=INK2)
        ax.set(title=pid, xticks=[], yticks=[])
        continue
    ax.scatter(d.time, d.hrv, s=1.2, color=BLUE, alpha=.5, rasterized=True)
    ax.set(ylim=(0, 140), title=pid)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
for ax in axes[len(pids):]:
    ax.axis("off")
fig.supylabel("HRV (as reported by the watch)")
fig.tight_layout()
save(fig, "04-hrv-over-time-per-patient")
