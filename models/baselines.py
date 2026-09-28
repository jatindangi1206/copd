"""References every model has to beat.

linear         impute   : straight line between the readings either side
last_value     forecast : repeat the last real reading
patient_median forecast : the patient's median over the history given
"""
import numpy as np

from .data import lin_fill


def impute(series, ctx):
    return [lin_fill(s.x.hrv) for s in series]


def forecast_last(history, horizons, ctx):
    out = []
    for s, h in zip(history, horizons):
        v = s.x.hrv_obs.dropna()
        out.append(np.full(h, v.iat[-1] if len(v) else np.nan))
    return out


def forecast_median(history, horizons, ctx):
    med = {}
    for s in history:
        med.setdefault(s.pid, []).extend(s.x.hrv_obs.dropna())
    return [np.full(h, np.median(med[s.pid]) if med[s.pid] else np.nan) for s, h in zip(history, horizons)]
