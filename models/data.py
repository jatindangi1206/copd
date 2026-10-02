"""Shared inputs, scoring and saving. Every model sees the same data and is
scored the same way; a model file only has to implement

    impute(series, ctx)            -> list of arrays, one per series, same length,
                                      every blank HRV filled (hidden ones included)
    forecast(history, horizons, ctx) -> list of arrays, horizons[i] values each

series / history are lists of Series. ctx is a dict: device, cores, quick, seed,
period (slots in one cycle: 144 for 10-minute data, 7 for daily), covs.

impute   : one Series per segment; hidden readings (mask_random or mask_block)
           are blanked in x.hrv and scored against their real value.
forecast : one Series per segment, train part only. x.hrv has its gaps filled by
           the chosen imputer, x.hrv_obs keeps the blanks for models that handle
           missing data themselves. Scored on the test part's real readings.
daily    : the same over days (model_daily.csv), one Series per patient.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "model_data"
RESULTS = ROOT / "results"
COVS = ["hr", "temp", "steps", "sleep_frac", "steps_active_frac"]
HRV_MAX = 129      # the watch's ceiling (fig 06); a prediction above it, or at or below 0, is impossible
DAILY_COVS = {"hr_mean": "hr", "temp_mean": "temp", "steps_total": "steps", "sleep_hours": "sleep_frac"}


@dataclass
class Series:
    id: str
    pid: str
    x: pd.DataFrame          # inputs: hrv (blank = missing or hidden), hrv_obs, covs, t
    truth: np.ndarray        # real HRV for the scored positions' reference
    score: np.ndarray        # bool: positions (in x for impute, in the horizon for forecast) to score


def _t(frame, daily):
    """Position in the cycle: 10-min slot of the day (0-143) or day of week (0-6)."""
    if daily:
        return pd.to_datetime(frame.date).dt.dayofweek.to_numpy()
    return (frame.time.dt.hour * 6 + frame.time.dt.minute // 10).to_numpy()


def hp(ctx, name, default):
    """A hyperparameter: ctx['params'][name] if tuning (or a tuned run) set it, else the default."""
    return ctx.get("params", {}).get(name, default)


def _quick(series):
    """The 6 longest series for --quick / check (the first 6 were all short, so long-segment failures never showed)."""
    return sorted(series, key=lambda s: len(s.x), reverse=True)[:6]


def load_impute(mask="random", quick=False):
    T = pd.read_csv(MD / "model_10min.csv.gz", parse_dates=["time"])
    T = T[T.segment.notna()]
    out = []
    for sid, g in T.groupby("segment", sort=False):
        g = g.reset_index(drop=True)
        truth = g.hrv.to_numpy(float)
        hidden = g[f"mask_{mask}"].to_numpy(bool)
        x = g[["time"] + COVS].copy()
        x["hrv"] = np.where(hidden, np.nan, truth)
        x["hrv_obs"] = x.hrv
        x["t"] = _t(g, False)
        out.append(Series(sid, g.pid.iat[0], x, truth, hidden))
    return _quick(out) if quick else out


def load_forecast(quick=False):
    T = pd.read_csv(MD / "model_10min.csv.gz", parse_dates=["time"])
    T = T[T.segment.notna()]
    hist, tests = [], []
    for sid, g in T.groupby("segment", sort=False):
        tr, te = g[g.split == "train"], g[g.split == "test"]
        x = tr[["time"] + COVS].reset_index(drop=True)
        x["hrv"] = x["hrv_obs"] = tr.hrv.to_numpy(float)
        x["t"] = _t(tr, False)
        truth = te.hrv.to_numpy(float)
        hist.append(Series(sid, g.pid.iat[0], x, truth, ~np.isnan(truth)))
    return _quick(hist) if quick else hist


def load_daily(quick=False):
    D = pd.read_csv(MD / "model_daily.csv", parse_dates=["date"])
    hist = []
    for pid, g in D.groupby("pid", sort=False):
        full = g.set_index("date").reindex(pd.date_range(g.date.min(), g.date.max(), freq="D"))
        full.index.name = "date"
        full = full.reset_index()
        last_train = g.date[g.split == "train"].max()
        tr, te = full[full.date <= last_train], full[full.date > last_train]
        if tr.hrv_median.notna().sum() < 3 or te.empty:
            continue
        x = tr.rename(columns=DAILY_COVS)[["date"] + list(DAILY_COVS.values())].reset_index(drop=True)
        x["hrv"] = x["hrv_obs"] = tr.hrv_median.to_numpy(float)
        x["t"] = _t(tr, True)
        truth = te.hrv_median.to_numpy(float)
        hist.append(Series(pid, pid, x, truth, ~np.isnan(truth)))
    return _quick(hist) if quick else hist


# ------------------------------------------------------------------ helpers for models
def lin_fill(y):
    """Straight-line fill between readings, flat at the ends. The imputation reference."""
    y = np.asarray(y, float)
    ok = ~np.isnan(y)
    if ok.sum() == 0:
        return np.full_like(y, np.nan)
    i = np.arange(len(y))
    return np.interp(i, i[ok], y[ok])


def zstats(series):
    """Per-patient mean and sd of log HRV, from the readings a model is allowed to see (H10)."""
    v = pd.concat([pd.DataFrame({"pid": s.pid, "y": np.log(s.x.hrv_obs)}) for s in series]).dropna()
    st = v.groupby("pid").y.agg(["mean", "std"]).fillna({"std": 1.0})
    st["std"] = st["std"].clip(lower=1e-3)
    glob = (v.y.mean(), max(v.y.std(), 1e-3))
    return {p: (r["mean"], r["std"]) for p, r in st.iterrows()}, glob


def to_z(y, pid, stats):
    m, s = stats[0].get(pid, stats[1])
    return (np.log(y) - m) / s


def from_z(z, pid, stats):
    m, s = stats[0].get(pid, stats[1])
    return np.exp(np.asarray(z) * s + m)


def phase(t, period):
    a = 2 * np.pi * np.asarray(t) / period
    return np.sin(a), np.cos(a)


def future_t(s, h, period):
    return (s.x.t.iat[-1] + np.arange(1, h + 1)) % period


def cov_matrix(series, covs):
    """Covariates standardised over all series; blanks become 0 with a 0/1 seen-flag."""
    allc = pd.concat([s.x[covs] for s in series])
    mu, sd = allc.mean(), allc.std().replace(0, 1).fillna(1)
    out = []
    for s in series:
        c = ((s.x[covs] - mu) / sd).to_numpy(float)
        out.append(np.concatenate([np.nan_to_num(c), (~np.isnan(c)).astype(float)], axis=1))
    return out


# ------------------------------------------------------------------ scoring and saving
def score(task, series, preds, horizons=None):
    rows = []
    for s, p in zip(series, preds):
        p = np.asarray(p, float)
        if task == "impute":
            idx = np.flatnonzero(s.score)
            rows.append(pd.DataFrame({"series": s.id, "pid": s.pid, "step": idx,
                                      "truth": s.truth[idx], "pred": p[idx]}))
        else:
            assert len(p) == len(s.truth), f"{s.id}: {len(p)} predictions for horizon {len(s.truth)}"
            idx = np.flatnonzero(s.score)
            rows.append(pd.DataFrame({"series": s.id, "pid": s.pid, "step": idx + 1,
                                      "truth": s.truth[idx], "pred": p[idx]}))
    P = pd.concat(rows, ignore_index=True)
    err = P.pred - P.truth
    bad = int(err.isna().sum())
    e = err.dropna()
    m = dict(n=int(len(P)), n_missing_pred=bad, mae=float(e.abs().mean()),
             rmse=float(np.sqrt((e ** 2).mean())),
             mae_below_120=float(e[P.truth[e.index] < 120].abs().mean()),
             mae_120_up=float(e[P.truth[e.index] >= 120].abs().mean()),
             mae_per_patient_median=float(e.abs().groupby(P.pid[e.index]).mean().median()),
             n_nonfinite=int(np.isinf(P.pred).sum()), n_nonpositive=int((P.pred <= 0).sum()),
             pct_above_max=float(100 * (P.pred > HRV_MAX).mean()))
    return P, m


def save(task, name, P, m, extra):
    out = RESULTS / task / name
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "predictions.csv.gz", index=False, float_format="%.4g")
    m = {**m, **extra, "finished": time.strftime("%Y-%m-%d %H:%M")}
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    row = pd.DataFrame([{"task": task, "run": name, **m}])
    summ = RESULTS / ("summary_tuned.csv" if extra.get("tuned") else "summary.csv")
    row.to_csv(summ, mode="a", header=not summ.exists(), index=False)
    return out
