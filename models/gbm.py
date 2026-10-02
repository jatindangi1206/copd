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

from .data import COVS, hp, phase, to_z, from_z, zstats

K = 6          # neighbours each side (imputation) / lags (forecast)
PAIRS = 3000   # (origin, h) training pairs per series for forecasting


def _regressor(lib, ctx):
    n = 50 if ctx["quick"] else hp(ctx, "n_estimators", 600)
    depth, lr = hp(ctx, "depth", 6), hp(ctx, "lr", 0.05)
    gpu = ctx["device"] == "cuda"
    if lib == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=n, max_depth=depth, learning_rate=lr,
                                subsample=hp(ctx, "subsample", 0.8), colsample_bytree=hp(ctx, "colsample", 0.8),
                                min_child_weight=hp(ctx, "min_child_weight", 1), reg_lambda=hp(ctx, "reg_lambda", 1.0),
                                tree_method="hist", n_jobs=ctx["cores"],
                                device="cuda" if gpu else "cpu", random_state=ctx["seed"])
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=n, depth=depth, learning_rate=lr, loss_function="RMSE",
                             l2_leaf_reg=hp(ctx, "l2_leaf_reg", 3.0),
                             thread_count=ctx["cores"],
                             # CPU by default: on a shared GPU CatBoost stalls when memory is short, and at
                             # this size the CPU is just as quick
                             task_type="GPU" if gpu and hp(ctx, "catboost_gpu", False) else "CPU",
                             random_seed=ctx["seed"], verbose=0, allow_writing_files=False)


def _nearest(z):
    """Value of, and distance to, the nearest reading before and after each slot."""
    i = np.arange(len(z))
    ok = ~np.isnan(z)
    prev_i = pd.Series(np.where(ok, i, np.nan)).shift(1).ffill().to_numpy()
    next_i = pd.Series(np.where(ok, i, np.nan)).shift(-1).bfill().to_numpy()
    take = lambda idx: np.where(np.isnan(idx), np.nan, z[np.nan_to_num(idx).astype(int)])
    return take(prev_i), i - prev_i, take(next_i), next_i - i


def _impute_features(s, z, period, K=K, drop=()):
    f = {f"lag{k}": np.r_[np.full(k, np.nan), z[:-k]] for k in range(1, K + 1)}
    f.update({f"lead{k}": np.r_[z[k:], np.full(k, np.nan)] for k in range(1, K + 1)})
    f["prev"], f["dprev"], f["next"], f["dnext"] = _nearest(z)
    f["sin"], f["cos"] = phase(s.x.t, period)
    for c in COVS:
        if c in s.x and c not in drop:
            f[c] = s.x[c].to_numpy(float)
    return pd.DataFrame(f)


def fit_impute(lib, series, ctx):
    """Fit on the visible readings. Returns the model, one feature table per series, the z-scored
    series and the per-patient stats (shap_analysis.py reads the model and tables)."""
    st = zstats(series)
    zs = [to_z(s.x.hrv.to_numpy(float), s.pid, st) for s in series]
    F = [_impute_features(s, z, ctx["period"], hp(ctx, "K", K), hp(ctx, "drop", ())) for s, z in zip(series, zs)]
    X = pd.concat(F, ignore_index=True)
    y = np.concatenate(zs)
    seen = ~np.isnan(y)
    m = _regressor(lib, ctx).fit(X[seen], y[seen])
    return m, F, zs, st


def make_impute(lib):
    def impute(series, ctx):
        m, F, zs, st = fit_impute(lib, series, ctx)
        out = []
        for s, z, f in zip(series, zs, F):
            out.append(from_z(np.where(np.isnan(z), m.predict(f), z), s.pid, st))
        return out
    return impute


def _forecast_features(s, z, o, h, period, k):
    w = z[max(0, o - period + 1):o + 1]
    lags = [z[o - j] if o - j >= 0 else np.nan for j in range(k)]
    so, co = phase(s.x.t.iat[o], period)
    st_, ct_ = phase((s.x.t.iat[o] + h) % period, period)
    covs = [s.x[c].iat[o] if c in s.x else np.nan for c in COVS]
    return lags + [np.nanmean(w), np.nanstd(w), h, so, co, st_, ct_] + covs


def forecast_names(k):
    return [f"lag{j}" for j in range(k)] + ["mean_day", "sd_day", "h", "sin_now", "cos_now", "sin_target", "cos_target"] + COVS


def fit_forecast(lib, history, horizons, ctx):
    """Fit, then build the feature table the test predictions use (one row per horizon step).
    Returns the model, one table per series, the per-patient stats and the feature names."""
    rng = np.random.default_rng(ctx["seed"])
    period, hmax = ctx["period"], max(horizons)
    k = min(2 * hp(ctx, "K", K), period)
    st = zstats(history)
    rows, ys = [], []
    for s in history:
        z = to_z(s.x.hrv.to_numpy(float), s.pid, st)          # gap-filled history
        zo = to_z(s.x.hrv_obs.to_numpy(float), s.pid, st)     # targets: real readings only
        n = len(z)
        if n < k + 2:
            continue
        for _ in range(200 if ctx["quick"] else hp(ctx, "pairs", PAIRS)):
            o = int(rng.integers(k - 1, n - 1))
            h = int(rng.integers(1, min(hmax, n - 1 - o) + 1))
            if not np.isnan(zo[o + h]):
                rows.append(_forecast_features(s, z, o, h, period, k))
                ys.append(zo[o + h])
    m = _regressor(lib, ctx).fit(np.array(rows, float), np.array(ys))
    Xs = []
    for s, H in zip(history, horizons):
        z = to_z(s.x.hrv.to_numpy(float), s.pid, st)
        o = len(z) - 1
        Xs.append(np.array([_forecast_features(s, z, o, h, period, k) for h in range(1, H + 1)], float))
    return m, Xs, st, forecast_names(k)


def make_forecast(lib):
    def forecast(history, horizons, ctx):
        m, Xs, st, _ = fit_forecast(lib, history, horizons, ctx)
        return [from_z(m.predict(X), s.pid, st) for s, X in zip(history, Xs)]
    return forecast
