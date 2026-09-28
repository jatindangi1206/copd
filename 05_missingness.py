"""HRV missingness: how often a reading arrives, and where the gaps are.

    python 05_missingness.py

HRV is reported every 10 minutes when the watch is worn. So the gap between one
reading and the next is 10 minutes when nothing is missing, and a multiple of 10
when readings are dropped. These figures show that gap distribution, which is
what a coverage threshold has to be chosen against.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import BLUE, CMAP, ORANGE, NUMBERS, master, patients, save

CADENCE = 10          # minutes between HRV readings when the watch is worn

gaps, rows, daily = [], [], []
for pid in patients():
    m = master(pid, ["time", "hrv"])
    d = m[m.hrv.notna()]
    if len(d) < 2:
        rows.append(dict(patient=pid, readings=len(d), days=0, coverage=np.nan,
                         median_gap=np.nan, p90_gap=np.nan, longest_gap_h=np.nan))
        continue
    g = d.time.diff().dt.total_seconds().div(60).dropna()
    gaps.append(pd.DataFrame({"patient": pid, "gap": g.values}))
    span_min = (d.time.max() - d.time.min()).total_seconds() / 60
    dd = d.set_index("time").hrv.resample("D").count()
    daily.append(pd.DataFrame({"patient": pid, "date": dd.index, "n": dd.values}))
    rows.append(dict(patient=pid, readings=len(d),
                     days=int(d.time.dt.date.nunique()),
                     coverage=100 * len(d) / (span_min / CADENCE),
                     median_gap=g.median(), p90_gap=g.quantile(.9),
                     longest_gap_h=g.max() / 60))
S = pd.DataFrame(rows)
G = pd.concat(gaps, ignore_index=True)
D = pd.concat(daily, ignore_index=True)
S.to_csv(NUMBERS / "hrv_missingness.csv", index=False)

# ---- 05a: the gap distribution
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].hist(G.gap[G.gap <= 120], bins=np.arange(0, 121, 2), color=BLUE)
ax[0].set_yscale("log")
ax[0].set(xlabel="minutes since the previous reading", ylabel="readings",
          xticks=np.arange(0, 121, 10))
ax[0].set_title("Gap between readings, up to 2 hours")

ax[1].hist(np.log10(G.gap[G.gap > 0]), bins=80, color=BLUE)
ax[1].set_yscale("log")
t = [1, 10, 60, 60 * 6, 60 * 24, 60 * 24 * 7]
ax[1].set(xlabel="minutes since the previous reading", ylabel="readings",
          xticks=np.log10(t))
ax[1].set_xticklabels(["1 min", "10 min", "1 h", "6 h", "1 day", "1 week"])
ax[1].set_title("Gap between readings, all gaps")
save(fig, "05a-gap-between-readings")

# ---- 05b: what a coverage cut keeps
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
s = S.dropna(subset=["coverage"]).sort_values("coverage", ascending=False)
ax[0].bar(s.patient, s.coverage, color=BLUE)
ax[0].set(ylabel="% of 10-minute slots with a reading", xlabel="")
ax[0].tick_params(axis="x", rotation=90, labelsize=8)
ax[0].set_title("Coverage per patient")

cuts = np.arange(0, 101, 5)
ax[1].plot(cuts, [(s.coverage >= c).sum() for c in cuts], "o-", color=ORANGE, lw=2)
ax[1].set(xlabel="coverage threshold (%)", ylabel="patients kept",
          xticks=np.arange(0, 101, 10))
ax[1].set_title("Patients kept at each threshold")
save(fig, "05b-coverage-per-patient")

# ---- 05c: readings per day, per patient
fig, ax = plt.subplots(figsize=(13, 7))
pids = list(S.dropna(subset=["coverage"]).patient)
pos = {p: i for i, p in enumerate(pids)}
d = D[D.patient.isin(pids)].copy()
sc = ax.scatter(d.date, d.patient.map(pos), c=d.n, cmap=CMAP, s=12,
                vmin=0, vmax=144)
ax.set(yticks=range(len(pids)), xlabel="", ylabel="")
ax.set_yticklabels(pids, fontsize=8)
fig.colorbar(sc, ax=ax, label="readings that day")
ax.set_title("Readings per day")
save(fig, "05c-readings-per-day")

# ---- 05d: how much of each day is covered
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(100 * D.n / (24 * 60 / CADENCE), bins=np.arange(0, 102, 2), color=BLUE)
ax.set(xlabel="% of the day with readings", ylabel="patient-days")
ax.set_title("Daily coverage")
save(fig, "05d-daily-coverage")

# ---- 05e: minute difference at three resolutions, no cut-off drawn
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for ax, (hours, step) in zip(axes, ((3, 5), (6, 10), (12, 10))):
    edges = np.arange(0, hours * 60 + step, step)
    ax.hist(G.gap[G.gap <= hours * 60], bins=edges, color=BLUE, rwidth=.85)
    ax.set_yscale("log")
    ax.set(xlim=(0, hours * 60), xticks=np.arange(0, hours * 60 + 1, 30 if hours <= 6 else 60),
           ylabel="number of gaps (log scale)",
           title=f"Up to {hours} hours ({step}-minute bins)")
axes[-1].set_xlabel("minute difference (time of a reading minus time of the previous reading)")
fig.tight_layout()
save(fig, "05e-minute-difference")

print(S.to_string(index=False))
print(f"\ngaps: median {G.gap.median():.0f} min   "
      f"{100*(G.gap <= CADENCE).mean():.1f}% are {CADENCE} min or less")
for t in (10, 20, 30, 60, 180, 1440):
    print(f"  gaps over {t:>5} min: {100*(G.gap > t).mean():>5.1f}%")
