"""HRV during recorded sleep, and HRV around each dated exacerbation.

    python 07_sleep_and_events.py

07a: per patient, median HRV inside vs outside the watch's sleep blocks.
07b: every HRV reading within WINDOW days of each dated exacerbation, with the
     daily median, the day's median step count and hours of recorded sleep, each
     against the patient's usual median. Descriptive only - no test.
     numbers/exacerbation_day0.csv: the recorded date itself vs the usual median.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import BLUE, ORANGE, GREY, INK2, LINE, NUMBERS, master, patients, save
from codings import load_exacerbations

warnings.filterwarnings("ignore", "Mean of empty slice")   # median of a day with no readings: blank

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

# ---- 07b: HRV around each dated exacerbation, with that day's step count and sleep
E = load_exacerbations()
W = {p: master(p, ["time", "hrv", "steps", "sleep_active"]) for p in E.pid.unique()}


def blank(ax, what):
    ax.text(.5, .5, f"no {what} in these 4 weeks", transform=ax.transAxes,
            ha="center", va="center", color=INK2, fontsize=8)


ncol = 4
nrow = int(np.ceil(len(E) / ncol))
fig = plt.figure(figsize=(16, 4.4 * nrow), layout="constrained")
day0 = []                                  # the recorded date against the patient's usual median
for sf, (_, e) in zip(fig.subfigures(nrow, ncol).ravel(), E.iterrows()):
    m = W[e.pid]
    rel = (m.time - e.date).dt.total_seconds() / 86400
    m = m.assign(day=np.floor(rel))[(rel >= -WINDOW) & (rel < WINDOW + 1)]
    a_hrv, a_st, a_sl = sf.subplots(3, 1, sharex=True, gridspec_kw=dict(height_ratios=[3, 1, 1]))
    for ax in (a_hrv, a_st, a_sl):
        ax.axvspan(0, 1, color=ORANGE, alpha=.2, lw=0)
        ax.tick_params(labelsize=8)
    # HRV: readings, daily median, the patient's usual median
    h = m[m.hrv.notna()]
    if len(h):
        a_hrv.scatter(h.time.sub(e.date).dt.total_seconds() / 86400, h.hrv, s=2, color=GREY,
                      alpha=.35, rasterized=True)
        dm = h.groupby("day").hrv.median()
        a_hrv.plot(dm.index + .5, dm.values, "o-", color=BLUE, ms=3, lw=1.4)
        a_hrv.axhline(W[e.pid].hrv.median(), color=INK2, ls="--", lw=1)
    else:
        blank(a_hrv, "HRV readings")
    # steps: median of the day's step readings (each a 10-minute count)
    s = m[m.steps.notna()]
    if len(s):
        dm = s.groupby("day").steps.median()
        a_st.plot(dm.index + .5, dm.values, "o-", color=ORANGE, ms=2.5, lw=1.2)
        a_st.axhline(W[e.pid].steps.median(), color=INK2, ls="--", lw=1)
    else:
        blank(a_st, "step readings")
    # sleep: hours of recorded sleep per day
    all_sleep = W[e.pid].groupby(W[e.pid].time.dt.normalize()).sleep_active.sum().div(60)
    sl = m.groupby("day").sleep_active.sum().div(60)
    sl = sl[sl > 0]
    if len(sl):
        a_sl.bar(sl.index + .5, sl.values, .7, color=BLUE, alpha=.6)
        a_sl.axhline(all_sleep[all_sleep > 0].median(), color=INK2, ls="--", lw=1)
    else:
        blank(a_sl, "recorded sleep")
    at0 = m[m.day == 0]
    day0.append(dict(patient=e.pid, date=e.date.date(),
                     hrv_day0=at0.hrv.median(), hrv_usual=W[e.pid].hrv.median(),
                     steps_day0=at0.steps.median(), steps_usual=W[e.pid].steps.median(),
                     sleep_day0=sl.get(0.0), sleep_usual=all_sleep[all_sleep > 0].median()))
    a_hrv.set(xlim=(-WINDOW, WINDOW + 1), ylim=(15, 135))
    a_sl.set_xticks([-13.5, -6.5, .5, 7.5, 14.5],     # centre of each day; shared by all three strips
                    ["14 days\nbefore", "7 days\nbefore", "exacerbation\nday", "7 days\nafter", "14 days\nafter"],
                    fontsize=7)
    a_hrv.set_ylabel("HRV", fontsize=8)
    a_st.set_ylim(bottom=0)
    a_st.set_ylabel("steps", fontsize=8)
    a_sl.set_ylim(bottom=0)
    a_sl.set_ylabel("sleep h", fontsize=8)
    a_hrv.set_title(f"{e.pid}   {e.date:%d %b %Y}", fontsize=10)
fig.suptitle("HRV, steps and sleep two weeks either side of each exacerbation\n"
             "grey: HRV readings · blue line: daily median HRV · orange: daily median step count "
             "(10-minute counts) · bars: hours of recorded sleep · dashed: the patient's usual median")
save(fig, "07b-hrv-around-exacerbations")
D0 = pd.DataFrame(day0)
D0.to_csv(NUMBERS / "exacerbation_day0.csv", index=False)
print(S.to_string(index=False))
print(D0.round(1).to_string(index=False))
