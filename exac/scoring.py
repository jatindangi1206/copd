"""Scores (copied from run_exac.py): per day, per event, 95% intervals by resampling patients, calibration,
slices by wear coverage, longest gap and days of history."""
import numpy as np
import pandas as pd

from exac import config as C
from exac import data


def avg_rank(s):
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=r) / cnt)[inv]


def auroc(y, s):
    """Chance that a random positive day scores above a random negative one (ties count half)."""
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    return float((avg_rank(np.asarray(s, float))[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def auprc(y, s):
    """Average precision as scikit-learn computes it: the sum over score thresholds of recall step x precision."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    if not y.sum():
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    last = np.r_[np.flatnonzero(np.diff(s[o])), len(s) - 1]               # last row of each distinct score
    tp = np.cumsum(y[o])[last]
    return float(np.sum(np.diff(np.r_[0, tp / y.sum()]) * tp / (last + 1)))


def core(y, s, ev):
    """The scores that get 95% intervals. ev: event code per day (-1 on negative days)."""
    f = s >= C.THRESHOLD
    tp, fp, tn, fn = (f & (y == 1)).sum(), (f & (y == 0)).sum(), (~f & (y == 0)).sum(), (~f & (y == 1)).sum()
    div = lambda u, v: u / v if v else float("nan")
    evs = np.unique(ev[y == 1])
    return dict(auroc=auroc(y, s), auprc=auprc(y, s), sensitivity=div(tp, tp + fn), specificity=div(tn, tn + fp),
                ppv=div(tp, tp + fp), brier=float(np.mean((s - y) ** 2)) if len(y) else float("nan"),
                event_sensitivity=div(len(np.unique(ev[(y == 1) & f])), len(evs)),
                false_alarms_per_30_days=div(30 * fp, fp + tn))


def evaluate(task, T, s, rng):
    lab, evc = C.EVENT[task]
    y, s = T[lab].to_numpy(int), np.asarray(s, float)
    codes = pd.factorize(T[evc].where(y == 1))[0]                          # -1 on negative days
    f = s >= C.THRESHOLD
    tp, fp, tn, fn = (int(x) for x in ((f & (y == 1)).sum(), (f & (y == 0)).sum(), (~f & (y == 0)).sum(), (~f & (y == 1)).sum()))
    div = lambda u, v: u / v if v else float("nan")
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    m = dict(n_patients=T.pid.nunique(), n_pos_patients=T.pid[y == 1].nunique(), n_days=len(y), n_pos_days=int(y.sum()),
             tp=tp, fp=fp, tn=tn, fn=fn, accuracy=div(tp + tn, len(y)), balanced_accuracy=(sens + spec) / 2,
             npv=div(tn, tn + fn), f1=div(2 * tp, 2 * tp + fp + fn), events=len(np.unique(codes[y == 1])),
             events_caught=len(np.unique(codes[(y == 1) & f])), lead_days_median=float("nan"))
    m.update(core(y, s, codes))
    if task == "onset" and m["events_caught"]:
        hit = T[(y == 1) & f]
        first = hit.groupby(evc).date.min()
        onset = pd.to_datetime(first.index.str.rsplit("_", n=1).str[1], format="%Y%m%d")
        m["lead_days_median"] = float(np.median((onset - pd.DatetimeIndex(first.to_numpy())).days))
    groups = [np.flatnonzero(T.pid.to_numpy() == p_) for p_ in T.pid.unique()]
    boot = []
    for _ in range(C.BOOT):                                                # resample patients, keep their days together
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        ev = np.concatenate([np.where(codes[groups[i]] >= 0, codes[groups[i]] + j * (codes.max() + 1), -1)
                             for j, i in enumerate(pick)])                 # a patient drawn twice: separate events
        boot.append(core(y[idx], s[idx], ev))
    B = pd.DataFrame(boot)
    for k in B:
        m[f"{k}_lo"], m[f"{k}_hi"] = (float(x) for x in np.nanpercentile(B[k], [2.5, 97.5])) if B[k].notna().any() else (float("nan"),) * 2
    return m


def extras(task, T, s):
    """Calibration table and scores by wear coverage, longest gap in the day and days of history (thirds)."""
    y, s = T[C.EVENT[task][0]].to_numpy(int), np.asarray(s, float)
    band = np.minimum((s * 10).astype(int), 9)
    cal = [dict(band=f"{k / 10:.1f}-{(k + 1) / 10:.1f}", n=int((band == k).sum()),
                predicted=float(s[band == k].mean()) if (band == k).any() else None,
                observed=float(y[band == k].mean()) if (band == k).any() else None) for k in range(10)]
    rows = data.window_rows(T)
    hr_min = data.D.set_index("row").hr_minutes
    info = {"coverage": np.array([hr_min.reindex(r[r >= 0]).sum() / data.M for r in rows]),
            "history_days": (rows >= 0).sum(1).astype(float),
            "longest_gap_minutes": np.array([np.diff(np.r_[-1, np.flatnonzero(np.isfinite(data.X[r, :, C.HR])), 1440]).max() - 1
                                             for r in rows[:, -1]], float)}
    out = {}
    for k, x in info.items():
        cut = np.digitize(x, np.unique(np.quantile(x, [1 / 3, 2 / 3])))    # thirds of the test days' own values
        out[k] = [dict(part=int(j), low=float(x[cut == j].min()), high=float(x[cut == j].max()), n_days=int((cut == j).sum()),
                       n_pos_days=int(y[cut == j].sum()), auroc=auroc(y[cut == j], s[cut == j]),
                       sensitivity=float(((s >= C.THRESHOLD) & (y == 1))[cut == j].sum() / max(y[cut == j].sum(), 1)),
                       specificity=float(((s < C.THRESHOLD) & (y == 0))[cut == j].sum() / max((y[cut == j] == 0).sum(), 1)))
                  for j in np.unique(cut)]
    return dict(calibration=cal, slices=out)
