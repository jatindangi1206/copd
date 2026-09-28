"""TimesFM 3.0 (google-research/timesfm, `pip install timesfm[torch]`), zero-shot.
Weights (google/timesfm-3.0-pytorch) download from Hugging Face on first use.
Licence: code Apache-2.0; the 3.0 WEIGHTS are non-commercial, non-production only.
TimesFM normalises its inputs itself, so HRV goes in as reported.

forecast: the gap-filled history (last CONTEXT slots) -> horizon. With
          USE_TIME_COVARIATE, time of day goes in as a past-and-future covariate.
impute  : TimesFM only forecasts, so each run of blanks is forecast forward from
          the readings before it and backward (reversed series) from the readings
          after it; the two are blended by position in the run (both sides).
"""
import numpy as np

from .data import lin_fill, phase

CHECKPOINT = "google/timesfm-3.0-pytorch"
CONTEXT, GAP_CONTEXT, BATCH = 2048, 512, 32
USE_TIME_COVARIATE = False


def _model(ctx):
    try:
        from timesfm3 import ModelConfig, TimesFM3Evaluator            # README (Aug 2026)
        return TimesFM3Evaluator(ModelConfig(checkpoint_path=CHECKPOINT, per_core_batch_size=BATCH,
                                             device=ctx["device"]))
    except ImportError:
        from timesfm3 import TimesFM3Forecaster                        # name used for the MLX mirror
        return TimesFM3Forecaster.from_pretrained(CHECKPOINT)


def _predict(m, contexts, h, covs=None):
    kw = {"past_future_covariates": covs} if covs is not None else {}
    outs = m.predict_batch([c.astype(np.float32) for c in contexts], horizon=h, return_quantiles=False,
                           make_positive=True, **kw)                      # HRV is positive (H1)
    return [np.asarray(o.forecast, float).reshape(-1)[:h] for o in outs]


def forecast(history, horizons, ctx):
    m, hmax, period = _model(ctx), max(horizons), ctx["period"]
    contexts = [s.x.hrv.to_numpy(float)[-CONTEXT:] for s in history]
    covs = None
    if USE_TIME_COVARIATE:
        covs = []
        for s, c in zip(history, contexts):
            tt = np.r_[s.x.t.to_numpy()[-len(c):], (s.x.t.iat[-1] + np.arange(1, hmax + 1)) % period]
            covs.append(np.vstack(phase(tt, period)).astype(np.float32))
        contexts = [c[None] for c in contexts]
    preds = _predict(m, contexts, hmax, covs)
    return [p[:h] for p, h in zip(preds, horizons)]


def impute(series, ctx):
    m = _model(ctx)
    jobs = []                                       # (series, start, length)
    fills = [lin_fill(s.x.hrv) for s in series]
    for k, s in enumerate(series):
        miss = np.isnan(s.x.hrv.to_numpy(float))
        edges = np.flatnonzero(np.diff(np.r_[0, miss.astype(int), 0]))
        jobs += [(k, a, b - a) for a, b in zip(edges[::2], edges[1::2])]
    fwd, bwd = {}, {}
    for L in sorted({j[2] for j in jobs}):
        group = [j for j in jobs if j[2] == L]
        left = [(j, fills[j[0]][:j[1]][-GAP_CONTEXT:]) for j in group if j[1] > 0]
        right = [(j, fills[j[0]][j[1] + L:][::-1][-GAP_CONTEXT:]) for j in group if j[1] + L < len(fills[j[0]])]
        for store, items in ((fwd, left), (bwd, right)):
            for a in range(0, len(items), 1024):
                chunk = items[a:a + 1024]
                for (j, _), p in zip(chunk, _predict(m, [c for _, c in chunk], L)):
                    store[j] = p
    out = [f.copy() for f in fills]
    for j in jobs:
        k, a, L = j
        f, b = fwd.get(j), bwd.get(j)
        b = b[::-1] if b is not None else None
        if f is None or b is None:
            out[k][a:a + L] = f if f is not None else b
        else:
            w = (np.arange(L) + 1) / (L + 1)                   # nearer the right edge -> more backward forecast
            out[k][a:a + L] = (1 - w) * f + w * b
    return out
