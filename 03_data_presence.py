"""How much data each vital has, per patient, and since enrolment.

    python 03_data_presence.py

Every patient in the datasheet is included; one without a watch file shows as zero
readings, labelled "no watch data". Enrolment = the date the smart watch was provided
(codings.load_enrolment). 03c-03e count only readings on or after it, up to the last
reading in the export.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import VITALS, BLUE, GREY, INK2, FIGS, NUMBERS, master, patients, save
from codings import load_enrolment

EN = load_enrolment().set_index("pid")
COLS = ["time"] + VITALS + ["sleep_active", "steps_active"]
NONE = pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "time" else float) for c in COLS})
have = set(patients())
rows, daily = [], {}
for pid in EN.index:                          # every patient in the sheet
    m = master(pid, COLS) if pid in have else NONE
    r = {"patient": pid, "minutes": len(m),
         "days": (m.time.max() - m.time.min()).days + 1 if len(m) else 0}
    for v in VITALS:
        r[v] = int(m[v].notna().sum())
    r["sleep"] = int(m.sleep_active.sum())
    # since enrolment
    e = EN.enrolment[pid]
    has = m[VITALS].notna().any(axis=1)
    s = m[(m.time >= e) & has]
    r.update(enrolment=e.date(), enrolment_source=EN.source[pid],
             minutes_before_enrolment=int((has & (m.time < e)).sum()),
             last_reading=m.time[has].max(),
             days_with_data=s.time.dt.normalize().nunique())
    for v in VITALS:
        r[f"{v}_since"] = int(s[v].notna().sum())
    daily[pid] = s[VITALS].notna().groupby((s.time - e).dt.days).sum()
    rows.append(r)
P = pd.DataFrame(rows)
END = P.last_reading.max()
P["days_since_enrolment"] = [(END.normalize() - pd.Timestamp(e)).days + 1 for e in P.enrolment]  # both ends counted
P.to_csv(NUMBERS / "data_presence.csv", index=False)
assert P.enrolment.notna().all(), "every patient needs an enrolment date"
assert (P.days_with_data <= P.days_since_enrolment).all()
no_watch = P.index[P.minutes == 0]


def mark_no_watch(ax, y):
    for i in no_watch:
        ax.text(i, y, "no watch data", rotation=90, ha="center", va="bottom", fontsize=7, color=INK2)


fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(P))
w = 0.16
for i, v in enumerate(VITALS):
    ax.bar(x + (i - 2) * w, P[v], w, label=v)
ax.set_yscale("log")
ax.set(xticks=x, ylabel="readings", xlabel="")
ax.set_xticklabels(P.patient, rotation=90, fontsize=8)
ax.legend(frameon=False, ncol=5, fontsize=10)
mark_no_watch(ax, 1.5)
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
mark_no_watch(ax[1], 3)
ax[1].set(ylabel="days", xlabel="")
ax[1].tick_params(axis="x", rotation=90, labelsize=8)
ax[1].set_title("Days from first to last reading (any signal)")
save(fig, "03b-totals-and-days")

# ---- 03c: time since enrolment
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x, P.days_since_enrolment, .7, color=GREY, alpha=.45, label="days since enrolment")
ax.bar(x, P.days_with_data, .38, color=BLUE, label="days with any watch reading")
ax.set(xticks=x, ylabel="days", xlabel="")
ax.set_xticklabels(P.patient, rotation=90, fontsize=8)
ax.legend(frameon=False, ncol=2, loc="upper right")
ax.set_title(f"Time since enrolment, up to {END:%d %b %Y} (last reading in the export)")
save(fig, "03c-time-since-enrolment")

# ---- 03d: 03a, counting only readings since enrolment
fig, ax = plt.subplots(figsize=(13, 5))
for i, v in enumerate(VITALS):
    ax.bar(x + (i - 2) * w, P[f"{v}_since"], w, label=v)
ax.set_yscale("log")
ax.set(xticks=x, ylabel="readings", xlabel="")
ax.set_xticklabels(P.patient, rotation=90, fontsize=8)
ax.legend(frameon=False, ncol=5, fontsize=10)
mark_no_watch(ax, 1.5)
ax.set_title("Readings per patient since enrolment")
save(fig, "03d-readings-since-enrolment")

# ---- 03e: readings per day since enrolment, one panel per patient
ncol = 6
nrow = int(np.ceil(len(P) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(17, 2.3 * nrow), sharex=True, sharey=True)
axes = axes.ravel()
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for ax, r in zip(axes, P.itertuples()):
    d = daily[r.patient].reindex(range(r.days_since_enrolment), fill_value=0)
    for i, v in enumerate(VITALS):
        ax.plot(d.index, d[v].where(d[v] > 0), color=colors[i], lw=.9, label=v)
    ax.set_yscale("log")
    ax.set_title(f"{r.patient}   enrolled {pd.Timestamp(r.enrolment):%d %b}", fontsize=9)
    ax.tick_params(labelsize=8)
    if not r.days_with_data:
        ax.text(.5, .5, "no watch data" if not r.minutes else "no readings\nsince enrolment", transform=ax.transAxes,
                ha="center", va="center", color=INK2, fontsize=8)
for ax in axes[len(P):]:
    ax.axis("off")
for ax in axes[len(P) - ncol:len(P)]:     # lowest panel of each column shows the day axis
    ax.xaxis.set_tick_params(labelbottom=True)
axes[0].set_ylim(.8, 2500)
fig.legend(*axes[0].get_legend_handles_labels(), ncol=len(VITALS), frameon=False,
           loc="upper center", bbox_to_anchor=(.5, 1.0))
fig.supxlabel("days since enrolment")
fig.supylabel("readings that day (log scale)")
fig.suptitle("Watch readings per day since enrolment, per patient (a break in a line: no readings that day)",
             y=1.025)
fig.tight_layout()
save(fig, "03e-readings-per-day-since-enrolment")
print(P.drop(columns=[f"{v}_since" for v in VITALS]).to_string(index=False))
