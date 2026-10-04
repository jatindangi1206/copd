"""Modelling-ready data: segments, imputation masks and forecasting splits.

    python 11_model_data.py

model_data/model_10min.csv.gz   one row per patient per 10-minute slot (the HRV cadence),
                                from each patient's first to last HRV reading
model_data/model_daily.csv      one row per patient-day, for day-level forecasting
model_data/segments.csv         one row per segment (only once X is set)
model_data/patients.csv         per patient: coverage, segments
model_data/threshold_scan.csv   for each candidate X: HRV kept, share missing inside

SEGMENTS. A segment ends where HRV is missing for more than X consecutive slots.
X is set below (18 slots = 180 min for this pilot). If it is set to None, the script
writes everything that does not depend on it (the 10-minute table with
missing_run, the daily table, the scan of candidate X) and leaves segment,
split and the masks blank.

IMPUTATION. Inside segments, MASK_PCT % of observed HRV slots are held out:
mask_random picks them at random (the core test); mask_block removes whole runs
with lengths drawn from the real missing runs, a harder check closer to real gaps.
FORECASTING. Inside each segment the first TRAIN_PCT % of slots are train, the
rest test - the model always predicts the later part. model_daily.csv does the
same per patient over days.

Nothing is filled in: hrv is blank wherever the watch gave no reading.
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import BLUE, ORANGE, GREY, INK2, LINE, FIGS, HERE, save, master, patients
from codings import load_baseline, load_exacerbations, clean

SLOT = "10min"
PER_HOUR = 6
CANDIDATES = [1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 72, 144]    # missing slots in a row
X = 18                    # missing slots in a row that end a segment (180 min), set for this pilot
MIN_SEG_HOURS = 24        # a segment must cover at least one full day
MASK_PCT = 20             # % of observed HRV held out for imputation
TRAIN_PCT = 80            # chronological train share for forecasting
SEED = 0
OUT = HERE / "model_data"
OUT.mkdir(exist_ok=True)
STATIC = ["age", "sex", "bmi", "mmrc", "gold", "cat", "fev1_pre", "walk_dist", "spo2", "exac_12m"]


# ------------------------------------------------------------------ 10-min grid
def grid(pid):
    m = master(pid, ["time", "hr", "hrv", "temp", "spo2", "steps", "sleep_active", "steps_active"])
    h = m[m.hrv.notna()]
    if h.empty:
        return None
    m = m[(m.time >= h.time.min().floor(SLOT)) & (m.time <= h.time.max())].set_index("time")
    r = m.resample(SLOT)
    g = pd.DataFrame({"hrv": r.hrv.mean(), "hrv_n": r.hrv.count(), "hr": r.hr.mean(), "hr_n": r.hr.count(),
                      "temp": r.temp.mean(), "spo2": r.spo2.mean(), "steps": r.steps.sum(min_count=1),
                      "sleep_frac": r.sleep_active.mean(), "steps_active_frac": r.steps_active.mean()})
    g = g.reset_index().assign(pid=pid)
    g[["sleep_frac", "steps_active_frac"]] = g[["sleep_frac", "steps_active_frac"]].fillna(0)
    return g


def missing_runs(obs):
    """Length of the missing run each slot sits in (0 where observed)."""
    out = np.zeros(len(obs), int)
    i = 0
    while i < len(obs):
        if not obs[i]:
            j = i
            while j < len(obs) and not obs[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def segment(obs, runs, x, min_len):
    """Segment number per slot (-1 outside). Cut at missing runs longer than x."""
    seg = np.full(len(obs), -1)
    cut = runs > x
    k, i = 0, 0
    while i < len(obs):
        if cut[i]:
            i += 1
            continue
        j = i
        while j < len(obs) and not cut[j]:
            j += 1
        idx = np.flatnonzero(obs[i:j])
        if len(idx):
            a, b = i + idx[0], i + idx[-1] + 1          # trim to first/last reading
            if b - a >= min_len:
                seg[a:b] = k
                k += 1
        i = j
    return seg


def score(G, x):
    kept = slots = seen = 0
    for obs, runs in G:
        s = segment(obs, runs, x, MIN_SEG_HOURS * PER_HOUR)
        slots += (s >= 0).sum()
        kept += obs[s >= 0].sum()
        seen += obs.sum()
    return (100 * kept / seen if seen else 0, 100 * (1 - kept / slots) if slots else np.nan)


frames = [g for g in (grid(p) for p in patients()) if g is not None]
arrays = {g.pid.iat[0]: (g.hrv.notna().to_numpy(), missing_runs(g.hrv.notna().to_numpy())) for g in frames}

cohort = [(x, *score(arrays.values(), x)) for x in CANDIDATES]
per_patient = {p: [(x, *score([a], x)) for x in CANDIDATES] for p, a in arrays.items()}
pd.DataFrame(cohort, columns=["x_slots", "kept_pct", "missing_inside_pct"]).to_csv(OUT / "threshold_scan.csv", index=False)

# ------------------------------------------------------------------ build rows
rng = np.random.default_rng(SEED)
E = load_exacerbations()
exac = set(zip(E.pid, E.date.dt.normalize()))
static = pd.read_csv(HERE / "clinical_table.csv").set_index("pid")
B = load_baseline()
static["fev1_pre"] = pd.Series(clean(B["FEV 1 [L]_%PRED_PRE_0"]).values, index=B.pid.values)
static = static[STATIC]
all_runs = []
rows, seg_rows, pat_rows = [], [], []
for g in frames:
    pid = g.pid.iat[0]
    obs, runs = arrays[pid]
    s = segment(obs, runs, X, MIN_SEG_HOURS * PER_HOUR) if X else np.full(len(obs), -1)
    g["missing_run"] = runs
    g["segment"] = np.where(s >= 0, [f"{pid}_s{k:02d}" for k in s], "")
    g["split"] = ""
    g["mask_random"] = 0
    g["mask_block"] = 0
    for k in np.unique(s[s >= 0]):
        idx = np.flatnonzero(s == k)
        cut = idx[int(len(idx) * TRAIN_PCT / 100)]
        g.loc[idx, "split"] = np.where(idx < cut, "train", "test")
        seen = idx[obs[idx]]
        n_mask = int(round(len(seen) * MASK_PCT / 100))
        g.loc[rng.choice(seen, n_mask, replace=False), "mask_random"] = 1
        inside = runs[idx][(runs[idx] > 0)]
        all_runs.extend(np.unique(inside))
        seg_rows.append(dict(segment=f"{pid}_s{k:02d}", pid=pid, start=g.time.iat[idx[0]],
                             end=g.time.iat[idx[-1]], hours=len(idx) / PER_HOUR,
                             hrv_readings=int(obs[idx].sum()),
                             missing_pct=100 * (1 - obs[idx].mean()),
                             train_slots=int((idx < cut).sum()), test_slots=int((idx >= cut).sum()),
                             masked=n_mask))
    pat_rows.append(dict(pid=pid, hrv_readings=int(obs.sum()), slots=len(obs),
                         coverage_pct=100 * obs.mean(),
                         segments=len(np.unique(s[s >= 0])) if X else None,
                         kept_pct=100 * obs[s >= 0].sum() / obs.sum() if X else None))
    rows.append(g)

# block masks: whole runs, lengths drawn from real missing runs inside segments
run_pool = np.array(all_runs) if all_runs else np.array([1])
for g in rows:
    for seg_id, idx in g[g.segment != ""].groupby("segment").groups.items():
        idx = np.asarray(idx)
        target = int(round(g.loc[idx, "hrv"].notna().sum() * MASK_PCT / 100))
        hit, tries = 0, 0
        while hit < target and tries < 1000:
            L = int(rng.choice(run_pool))
            a = int(rng.integers(0, max(1, len(idx) - L)))
            span = idx[a:a + L]
            fresh = span[(g.loc[span, "hrv"].notna()) & (g.loc[span, "mask_block"] == 0)]
            g.loc[fresh, "mask_block"] = 1
            hit += len(fresh)
            tries += 1

T = pd.concat(rows, ignore_index=True)
T["hour"] = T.time.dt.hour
T["date"] = T.time.dt.normalize()
T["exac_day"] = [int((p, d) in exac) for p, d in zip(T.pid, T.date)]
T = T.join(static.add_prefix("c_"), on="pid")
cols = ["pid", "time", "hour", "hrv", "hrv_n", "hr", "hr_n", "temp", "spo2", "steps", "sleep_frac",
        "steps_active_frac", "exac_day", "missing_run", "segment", "split", "mask_random", "mask_block"] \
    + [f"c_{c}" for c in STATIC]
T[cols].to_csv(OUT / "model_10min.csv.gz", index=False, float_format="%.4g", compression={"method": "gzip", "mtime": 0})   # mtime 0: same bytes every run
SEG = pd.DataFrame(seg_rows)
if X:
    SEG.to_csv(OUT / "segments.csv", index=False)
else:
    (OUT / "segments.csv").unlink(missing_ok=True)
PAT = pd.DataFrame(pat_rows)
PAT.to_csv(OUT / "patients.csv", index=False)

# daily table: last TRAIN_PCT-complement of each patient's HRV days are test
D = T.groupby(["pid", "date"]).agg(hrv_median=("hrv", "median"), hrv_n=("hrv_n", "sum"),
                                   hr_mean=("hr", "mean"), temp_mean=("temp", "mean"),
                                   steps_total=("steps", "sum"), sleep_hours=("sleep_frac", "sum"),
                                   exac_day=("exac_day", "max")).reset_index()
D["sleep_hours"] = D.sleep_hours / PER_HOUR
D["coverage_pct"] = 100 * D.hrv_n.clip(upper=24 * PER_HOUR) / (24 * PER_HOUR)
D = D[D.hrv_n > 0].copy()
D["split"] = D.groupby("pid").date.transform(
    lambda d: np.where(np.arange(len(d)) < int(len(d) * TRAIN_PCT / 100), "train", "test"))
D.to_csv(OUT / "model_daily.csv", index=False, float_format="%.4g")

# ------------------------------------------------------------------ figures
C = pd.DataFrame(cohort, columns=["x", "kept", "miss"])
hours = C.x * 10 / 60
fig, ax = plt.subplots(1, 2, figsize=(16, 5.2))
ax[0].plot(hours, C.kept, "o-", color=BLUE, lw=2, label="% of HRV readings kept in segments")
ax[0].plot(hours, C.miss, "o-", color=ORANGE, lw=2, label="% of segment slots with no reading")
ax[0].set(xscale="log", ylim=(0, 108), xlabel="longest allowed run of missing HRV (hours, log scale)",
          ylabel="%", title=f"Cohort: segments of at least {MIN_SEG_HOURS} h")
ax[0].set_xticks([1/6, .5, 1, 2, 4, 6, 12, 24])
ax[0].set_xticklabels(["10 min", "30 min", "1 h", "2 h", "4 h", "6 h", "12 h", "24 h"])
ax[0].legend(loc="upper left")
for p, r in per_patient.items():
    k = pd.DataFrame(r, columns=["x", "kept", "miss"])
    ax[1].plot(k.x * 10 / 60, k.kept, color=GREY, lw=1, alpha=.6)
ax[1].plot(hours, C.kept, color=BLUE, lw=2.5, label="cohort")
ax[1].set(xscale="log", ylim=(0, 105), xlabel="longest allowed run of missing HRV (hours, log scale)",
          ylabel="% of that patient's HRV readings kept", title="Each patient (grey) against the cohort (blue)")
ax[1].set_xticks([1/6, .5, 1, 2, 4, 6, 12, 24])
ax[1].set_xticklabels(["10 min", "30 min", "1 h", "2 h", "4 h", "6 h", "12 h", "24 h"])
ax[1].legend(loc="lower right")
fig.tight_layout()
save(fig, "11a-segment-threshold")

if not X:
    (FIGS / "11b-segments.png").unlink(missing_ok=True)
    (OUT / "config.json").write_text(json.dumps(dict(
        x_slots=None, min_segment_hours=MIN_SEG_HOURS, mask_pct=MASK_PCT, train_pct=TRAIN_PCT, seed=SEED,
        candidates_slots=CANDIDATES), indent=1))
    print(f"X not set: wrote the 10-minute table ({len(T):,} rows), daily table and threshold scan; "
          "segments, splits and masks wait for X")
    raise SystemExit
pids = sorted(T.pid.unique())
fig, ax = plt.subplots(figsize=(14, 7.5))
for i, p in enumerate(pids):
    t = T[T.pid == p]
    ax.plot([t.time.min(), t.time.max()], [i, i], color=LINE, lw=6, solid_capstyle="butt")
    for _, s in SEG[SEG.pid == p].iterrows():
        cut = s.start + (s.end - s.start) * TRAIN_PCT / 100
        ax.plot([s.start, cut], [i, i], color=BLUE, lw=6, solid_capstyle="butt")
        ax.plot([cut, s.end], [i, i], color=ORANGE, lw=6, solid_capstyle="butt")
ax.plot([], [], color=LINE, lw=6, label="HRV span, not in a segment")
ax.plot([], [], color=BLUE, lw=6, label=f"segment: first {TRAIN_PCT}% (train)")
ax.plot([], [], color=ORANGE, lw=6, label=f"segment: last {100 - TRAIN_PCT}% (test)")
ax.set(yticks=range(len(pids)), ylim=(-1, len(pids)))
ax.set_yticklabels(pids, fontsize=8.5)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.0))
ax.set_title(f"{len(SEG)} segments in {SEG.pid.nunique()} patients (cut at more than {X * 10} min without HRV)",
             pad=30)
save(fig, "11b-segments")
print(f"X = {X} slots ({X * 10} min); {len(SEG)} segments, {SEG.pid.nunique()} patients; "
      f"{len(T):,} rows; {int(T.mask_random.sum()):,} random-masked, {int(T.mask_block.sum()):,} block-masked")

(OUT / "config.json").write_text(json.dumps(dict(
    x_slots=int(X), x_minutes=int(X) * 10, min_segment_hours=MIN_SEG_HOURS,
    mask_pct=MASK_PCT, train_pct=TRAIN_PCT, seed=SEED, candidates_slots=CANDIDATES), indent=1))

# ------------------------------------------------------------------ self-check
inseg = T.segment != ""
assert T.loc[T.mask_random == 1, "hrv"].notna().all() and T.loc[T.mask_block == 1, "hrv"].notna().all(), "mask on a blank"
assert not (T.mask_random.astype(bool) | T.mask_block.astype(bool) | (T.split != ""))[~inseg].any(), "mask/split outside a segment"
last_train = T[T.split == "train"].groupby("segment").time.max()
first_test = T[T.split == "test"].groupby("segment").time.min()
assert (last_train < first_test.reindex(last_train.index)).all(), "test must come after train"
assert (T.loc[inseg, "missing_run"] <= X).all(), "a segment holds a run longer than X"
print("ok: masks only on observed HRV inside segments; test always after train; no run > X inside")
