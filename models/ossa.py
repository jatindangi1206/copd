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

from .data import lin_fill

RANK, ITERS, WINDOWS = 3, 30, 4


def _embed_len(n, period):
    return int(max(2, min(period, n // 2)))


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
        L = _embed_len(len(z), ctx["period"])
        for _ in range(3 if ctx["quick"] else ITERS):
            rec, _ = _reconstruct(z, L, min(RANK, L))
            z = np.where(miss, rec, ly)
        out.append(np.exp(z))
    return out


def forecast(history, horizons, ctx):
    out = []
    for s, h in zip(history, horizons):
        ly = np.log(s.x.hrv.to_numpy(float))[-WINDOWS * ctx["period"]:]
        L = _embed_len(len(ly), ctx["period"])
        r = min(RANK, L - 1)
        rec, U = _reconstruct(ly, L, r)
        pi = U[-1, :r]
        nu2 = float(pi @ pi)
        R = (U[:-1, :r] @ pi) / (1 - nu2) if nu2 < 1 else np.zeros(L - 1)   # linear recurrence
        z = list(rec)
        for _ in range(h):
            z.append(float(R @ np.array(z[-(L - 1):])))
        out.append(np.exp(np.array(z[-h:])))
    return out
