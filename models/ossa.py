"""OSSA: singular spectrum analysis (Golyandina & Zhigljavsky 2013), on log HRV
so reconstructions stay positive (H1). numpy only.

impute  : iterative SSA gap filling (Kondrashov & Ghil 2006): start the blanks
          from a straight-line fill, rebuild the series from its RANK leading
          SSA components, replace only the blanks, repeat until they settle.
forecast: online, adaptive window: SSA on the trailing window of the history
          (up to WINDOWS cycles, shorter when less history exists), then the
          SSA linear recurrence continues the rebuilt series forward (H9).
"""
import numpy as np

from .data import hp, lin_fill

RANK, ITERS, WINDOWS = 3, 30, 4
NU2_MAX = 0.95      # above this the forecast recurrence is not trusted


def _embed_len(n, period, frac=1.0):
    return int(max(2, min(period * frac, n // 2)))


def _reconstruct(y, L, r):
    n = len(y)
    X = np.lib.stride_tricks.sliding_window_view(y, L).T          # (L, n-L+1) Hankel
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Xr = (U[:, :r] * s[:r]) @ Vt[:r]
    out, cnt = np.zeros(n), np.zeros(n)
    for j in range(Xr.shape[1]):                                     # diagonal averaging
        out[j:j + L] += Xr[:, j]
        cnt[j:j + L] += 1
    return out / cnt, U[:, :r]


def impute(series, ctx):
    out = []
    for s in series:
        ly = np.log(s.x.hrv.to_numpy(float))
        miss = np.isnan(ly)
        z = lin_fill(ly)
        L = _embed_len(len(z), ctx["period"], hp(ctx, "embed_frac", 1.0))
        lo, hi = np.nanmin(ly), np.nanmax(ly)
        for _ in range(3 if ctx["quick"] else hp(ctx, "iters", ITERS)):
            rec, _ = _reconstruct(z, L, min(hp(ctx, "rank", RANK), L))
            z = np.where(miss, np.clip(rec, lo, hi), ly)      # H1: stay inside the observed range
        out.append(np.exp(z))
    return out


def forecast(history, horizons, ctx):
    out = []
    for s, h in zip(history, horizons):
        ly = np.log(s.x.hrv.to_numpy(float))[-hp(ctx, "windows", WINDOWS) * ctx["period"]:]
        L = _embed_len(len(ly), ctx["period"], hp(ctx, "embed_frac", 1.0))
        r = min(hp(ctx, "rank", RANK), L - 1)
        lo, hi = np.nanmin(ly), np.nanmax(ly)
        rec, U = _reconstruct(ly, L, r)
        pi = U[-1, :r]
        nu2 = float(pi @ pi)
        if nu2 >= hp(ctx, "nu2_max", NU2_MAX):                  # the recurrence divides by 1 - nu2 and explodes as nu2 -> 1
            out.append(np.full(h, np.exp(rec.mean())))     # fall back to the window's level
            continue
        R = (U[:-1, :r] @ pi) / (1 - nu2)   # linear recurrence
        z = list(rec)
        for _ in range(h):
            z.append(float(np.clip(R @ np.array(z[-(L - 1):]), lo, hi)))   # H1: stay inside the observed range
        out.append(np.exp(np.array(z[-h:])))
    return out
