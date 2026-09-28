"""XGBoost and CatBoost. One global model across all patients, on per-patient
standardised log HRV (H10). Both libraries take blanks as input natively (H8).

impute  : each reading is predicted from K readings before and after it (blank
          where missing), the nearest real reading on each side and how far away
          it is, the other vitals, and time of day. Trained on the visible
          readings, applied to the blanks.
forecast: direct multi-step. From an origin o, predict the value h steps ahead
          from the last K values, the recent level and spread, the vitals at o,
          h, and the time of day at o and at o+h. Only data up to o is used (H9).
"""
import numpy as np
import pandas as pd

from .data import COVS, phase, to_z, from_z, zstats

K = 6          # neighbours each side (imputation) / lags (forecast)
PAIRS = 3000   # (origin, h) training pairs per series for forecasting


def _regressor(lib, ctx):
    n = 50 if ctx["quick"] else 600
    gpu = ctx["device"] == "cuda"
    if lib == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=n, max_depth=6, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", n_jobs=ctx["cores"],
                                device="cuda" if gpu else "cpu", random_state=ctx["seed"])
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=n, depth=6, learning_rate=0.05, loss_function="RMSE",
                             thread_count=ctx["cores"], task_type="GPU" if gpu else "CPU",
                             random_seed=ctx["seed"], verbose=0, allow_writing_files=False)


def _nearest(z):
    """Value of, and distance to, the nearest reading before and after each slot."""
    i = np.arange(len(z))
    ok = ~np.isnan(z)
    prev_i = pd.Series(np.where(ok, i, np.nan)).shift(1).ffill().to_numpy()
    next_i = pd.Series(np.where(ok, i, np.nan)).shift(-1).bfill().to_numpy()
    take = lambda idx: np.where(np.isnan(idx), np.nan, z[np.nan_to_num(idx).astype(int)])
    return take(prev_i), i - prev_i, take(next_i), next_i - i


def _impute_features(s, z, period):
    f = {f"lag{k}": np.r_[np.full(k, np.nan), z[:-k]] for k in range(1, K + 1)}
    f.update({f"lead{k}": np.r_[z[k:], np.full(k, np.nan)] for k in range(1, K + 1)})
    f["prev"], f["dprev"], f["next"], f["dnext"] = _nearest(z)
    f["sin"], f["cos"] = phase(s.x.t, period)
    for c in COVS:
        if c in s.x:
            f[c] = s.x[c].to_numpy(float)
    return pd.DataFrame(f)


def make_impute(lib):
    def impute(series, ctx):
        st = zstats(series)
        zs = [to_z(s.x.hrv.to_numpy(float), s.pid, st) for s in series]
        F = [_impute_features(s, z, ctx["period"]) for s, z in zip(series, zs)]
        X = pd.concat(F, ignore_index=True)
        y = np.concatenate(zs)
        seen = ~np.isnan(y)
        m = _regressor(lib, ctx).fit(X[seen], y[seen])
        out, pred = [], m.predict(X)
        i = 0
        for s, z in zip(series, zs):
            p = pred[i:i + len(z)]
            i += len(z)
            out.append(from_z(np.where(np.isnan(z), p, z), s.pid, st))
        return out
    return impute


def _forecast_features(s, z, o, h, period, k):
    w = z[max(0, o - period + 1):o + 1]
    lags = [z[o - j] if o - j >= 0 else np.nan for j in range(k)]
    so, co = phase(s.x.t.iat[o], period)
    st_, ct_ = phase((s.x.t.iat[o] + h) % period, period)
    covs = [s.x[c].iat[o] if c in s.x else np.nan for c in COVS]
    return lags + [np.nanmean(w), np.nanstd(w), h, so, co, st_, ct_] + covs


def make_forecast(lib):
    def forecast(history, horizons, ctx):
        rng = np.random.default_rng(ctx["seed"])
        period, hmax = ctx["period"], max(horizons)
        k = min(2 * K, period)
        st = zstats(history)
        rows, ys = [], []
        for s in history:
            z = to_z(s.x.hrv.to_numpy(float), s.pid, st)          # gap-filled history
            zo = to_z(s.x.hrv_obs.to_numpy(float), s.pid, st)     # targets: real readings only
            n = len(z)
            if n < k + 2:
                continue
            for _ in range(200 if ctx["quick"] else PAIRS):
                o = int(rng.integers(k - 1, n - 1))
                h = int(rng.integers(1, min(hmax, n - 1 - o) + 1))
                if not np.isnan(zo[o + h]):
                    rows.append(_forecast_features(s, z, o, h, period, k))
                    ys.append(zo[o + h])
        m = _regressor(lib, ctx).fit(np.array(rows, float), np.array(ys))
        out = []
        for s, H in zip(history, horizons):
            z = to_z(s.x.hrv.to_numpy(float), s.pid, st)
            o = len(z) - 1
            X = np.array([_forecast_features(s, z, o, h, period, k) for h in range(1, H + 1)], float)
            out.append(from_z(m.predict(X), s.pid, st))
        return out
    return forecast
