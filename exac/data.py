"""Data: days, minutes and the clinical table of one set; each sample's window of W days (copied from run_exac.py).

load() fills the module state below once per run; every other module reads it as data.D, data.X, ...
"""
import warnings

import numpy as np
import pandas as pd

from exac import config as C

D = X = S = POS = DAYMAX = P = None     # days of the set, minutes.npy, clinical table, (pid, date) -> row, ...
W = M = None                            # window in days and in minutes
device = "cpu"


def load(set_name, days, dev, horizon=C.HORIZON):
    global D, X, S, POS, DAYMAX, P, W, M, device
    W, M, device = days, days * 1440, dev
    cohorts, clinical = C.SETS[set_name]
    D = pd.read_csv(C.MD / "days.csv", parse_dates=["date"])
    X = np.load(C.MD / "minutes.npy", mmap_mode="r")        # (stored days, 1440, 7), blank = NaN
    D["row"] = np.arange(len(D))
    if horizon != C.HORIZON:            # onset within fewer days: the same samples, only the label narrows (EXAC_DESIGN 9)
        lead = (pd.to_datetime(D.onset_id.str.rsplit("_", n=1).str[1], format="%Y%m%d") - D.date).dt.days
        assert 1 <= horizon < C.HORIZON and lead[D.onset_label == 1].between(1, C.HORIZON).all()
        keep = (D.onset_label == 1) & (lead <= horizon)   # onset_id is the nearest onset: no closer one is missed
        D["onset_label"], D["onset_id"] = keep.astype(int), D.onset_id.where(keep)
    S = pd.read_csv(C.MD / "static.csv").set_index("pid") if clinical else None
    D = D[D.cohort.isin(cohorts) & (D.pid.isin(S.index) if clinical else True)].reset_index(drop=True)
    POS = {(p_, d_): r for p_, d_, r in zip(D.pid, D.date, D.row)}
    with warnings.catch_warnings():                         # a day with no reading of a signal gives NaN here
        warnings.simplefilter("ignore", RuntimeWarning)
        DAYMAX = np.concatenate([np.nanmax(np.abs(X[D.row.to_numpy()[k:k + 2000]]), axis=1) for k in range(0, len(D), 2000)])
    P = D[D["sample"]].groupby("pid").agg(group=("group", "first"), pos=("label", "max"))
    for g, d in D[D["sample"]].groupby("group"):
        o = D[(D.group == g) & D.onset_valid]
        print(f"{set_name} / {g}: {d.pid.nunique()} patients | detect {len(d):,} days, {d.label.sum()} positive | "
              f"onset {len(o):,} days, {o.onset_label.sum()} positive")


def window_rows(T):
    """(n, W) rows of minutes.npy for each sample's window, oldest day first; -1 = no data that day."""
    return np.array([[POS.get((p_, d_ - pd.Timedelta(days=W - 1 - k)), -1) for k in range(W)]
                     for p_, d_ in zip(T.pid, T.date)], dtype=np.int64).reshape(len(T), W)


def window(rows):
    """(n, W) rows -> (n, M, 7) raw minutes of the window, NaN = no reading."""
    v = np.asarray(X[np.maximum(rows, 0).ravel()], dtype=np.float32).reshape(len(rows), W, 1440, -1)
    v[rows < 0] = np.nan
    return v.reshape(len(rows), M, -1)


def tree_matrix(T):
    """One row per sample: every raw value of its window (blanks stay blank), + the clinical values."""
    rows, k_s = window_rows(T), (S.shape[1] if S is not None else 0)
    out = np.empty((len(T), M * X.shape[2] + k_s), np.float32)
    for k in range(0, len(T), 256):
        out[k:k + 256, :M * X.shape[2]] = window(rows[k:k + 256]).reshape(len(rows[k:k + 256]), -1)
    if k_s:
        out[:, M * X.shape[2]:] = S.reindex(T.pid).to_numpy(np.float32)
    return out


def scales(train):
    """Each signal's (and clinical value's) largest absolute value in the training patients."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = np.nanmax(DAYMAX[D.pid.isin(train).to_numpy()], axis=0)
        st = np.nanmax(np.abs(S[S.index.isin(train)].to_numpy(float)), axis=0) if S is not None else None
    fix = lambda x: None if x is None else np.where(np.isfinite(x) & (x > 0), x, 1).astype(np.float32)
    return fix(s), fix(st)


def static(T, sscale):
    """Scaled clinical values per sample (copd_clinical), or None."""
    return np.nan_to_num(S.reindex(T.pid).to_numpy(np.float32) / sscale) if S is not None else None


def n_static():
    return S.shape[1] if S is not None else 0
