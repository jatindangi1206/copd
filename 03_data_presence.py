"""How much data each vital has, per patient.

    python 03_data_presence.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import VITALS, BLUE, FIGS, NUMBERS, master, patients, save

rows = []
for pid in patients():
    m = master(pid, ["time"] + VITALS + ["sleep_active", "steps_active"])
    r = {"patient": pid, "minutes": len(m),
         "days": (m.time.max() - m.time.min()).days + 1}
    for v in VITALS:
        r[v] = int(m[v].notna().sum())
    r["sleep"] = int(m.sleep_active.sum())
    rows.append(r)
P = pd.DataFrame(rows)
P.to_csv(NUMBERS / "data_presence.csv", index=False)

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(P))
w = 0.16
for i, v in enumerate(VITALS):
    ax.bar(x + (i - 2) * w, P[v], w, label=v)
ax.set_yscale("log")
ax.set(xticks=x, ylabel="readings", xlabel="")
ax.set_xticklabels(P.patient, rotation=90, fontsize=8)
ax.legend(frameon=False, ncol=5, fontsize=10)
ax.set_title("Readings per patient")
save(fig, "03a-readings-per-patient")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
tot = [P[v].sum() for v in VITALS]
ax[0].bar(VITALS, tot, color=BLUE, width=.6)
for i, v in enumerate(tot):
    ax[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
ax[0].set(ylabel="readings", ylim=(0, max(tot) * 1.18))
ax[0].set_title("Total readings")

ax[1].bar(P.patient, P.days, color=BLUE)
ax[1].set(ylabel="days", xlabel="")
ax[1].tick_params(axis="x", rotation=90, labelsize=8)
ax[1].set_title("Days from first to last reading (any signal)")
save(fig, "03b-totals-and-days")
print(P.to_string(index=False))
