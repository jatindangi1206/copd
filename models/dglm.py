"""CD-Gamma-DGLM: Gamma dynamic generalised linear model with a log link and
discount factors (West, Harrison & Migon 1985; West & Harrison 1997, ch. 14).
PyBATS has DGLMs for Poisson/Bernoulli/Normal/Binomial but not Gamma, so the
filter is written out here (numpy + scipy).

state      theta = [level, c1, c2]; G keeps the level and rotates (c1, c2) by
           2*pi/period each slot (daily rhythm, H5)
predictor  eta = level + c1;  mean mu = exp(eta) > 0 (H1)
reading    y ~ Gamma(shape NU, mean mu)
evolution  a = G m, R = G C G' / DELTA  (discounting: old data is gradually forgotten, H4)
update     (f, q) of eta -> conjugate Gamma prior on 1/mu via trigamma matching,
           conjugate update with y, mapped back (linear Bayes). Blank: evolve only (H8).
"continuous-discrete": the data are on a 10-minute grid, so elapsed time is
counted in slots and a gap of k slots is k evolution steps without updates.

impute  : forward filter + backward smoother (both sides): E[y] = exp(f + q/2).
forecast: forward filter to the end of the history, evolve forward (H9).
NU is set per series from the spread of its log readings.
"""
import numpy as np
from scipy.special import digamma, polygamma

DELTA = 0.98


def _inv_trigamma(q):
    """alpha with trigamma(alpha) = q (Newton, a few steps)."""
    a = 0.5 + 1.0 / q
    for _ in range(8):
        a = np.maximum(a - (polygamma(1, a) - q) / polygamma(2, a), 1e-6)
    return a


def _design(period, t0, ly):
    w = 2 * np.pi / period
    G = np.array([[1, 0, 0], [0, np.cos(w), -np.sin(w)], [0, np.sin(w), np.cos(w)]])
    F = np.array([1.0, 1.0, 0.0])
    ok = ~np.isnan(ly)
    m0 = np.array([np.nanmean(ly) if ok.any() else 4.0, 0.0, 0.0])
    C0 = np.diag([1.0, 0.1, 0.1])
    dl = np.diff(ly[ok]) if ok.sum() > 2 else np.array([0.3])
    nu = float(np.clip(1.0 / max(0.5 * np.var(dl), 1e-3), 1.0, 500.0))
    return G, F, m0, C0, nu


def _filter(y, G, F, m, C, nu):
    T = len(y)
    ms, Cs, As, Rs = np.zeros((T, 3)), np.zeros((T, 3, 3)), np.zeros((T, 3)), np.zeros((T, 3, 3))
    for t in range(T):
        a, R = G @ m, G @ C @ G.T / DELTA
        As[t], Rs[t] = a, R
        if not np.isnan(y[t]):
            f, q = F @ a, max(F @ R @ F, 1e-8)
            alpha = _inv_trigamma(q)                     # prior on lam = 1/mu: Gamma(alpha, beta)
            beta = np.exp(digamma(alpha) + f)            # so that E[log mu] = f
            a2, b2 = alpha + nu, beta + nu * y[t]
            f2, q2 = np.log(b2) - digamma(a2), polygamma(1, a2)
            RF = R @ F
            m = a + RF * (f2 - f) / q
            C = R - np.outer(RF, RF) * (1 - q2 / q) / q
        else:
            m, C = a, R
        C = (C + C.T) / 2                                # keep it symmetric: rounding made it singular
        ms[t], Cs[t] = m, C
    return ms, Cs, As, Rs


def _smooth(G, ms, Cs, As, Rs):
    sm, sC = ms.copy(), Cs.copy()
    for t in range(len(ms) - 2, -1, -1):
        B = Cs[t] @ G.T @ np.linalg.pinv(Rs[t + 1])      # pinv: R can be (near) singular on long segments
        sm[t] = ms[t] + B @ (sm[t + 1] - As[t + 1])
        sC[t] = Cs[t] + B @ (sC[t + 1] - Rs[t + 1]) @ B.T
    return sm, sC


def impute(series, ctx):
    out = []
    for s in series:
        y = s.x.hrv.to_numpy(float)
        G, F, m0, C0, nu = _design(ctx["period"], s.x.t.iat[0], np.log(y))
        sm, sC = _smooth(G, *_filter(y, G, F, m0, C0, nu))
        f, q = sm @ F, np.einsum("i,tij,j->t", F, sC, F)
        out.append(np.where(np.isnan(y), np.exp(f + q / 2), y))
    return out


def forecast(history, horizons, ctx):
    out = []
    for s, h in zip(history, horizons):
        y = s.x.hrv_obs.to_numpy(float)
        G, F, m0, C0, nu = _design(ctx["period"], s.x.t.iat[0], np.log(y))
        ms, Cs, _, _ = _filter(y, G, F, m0, C0, nu)
        m, C, preds = ms[-1], Cs[-1], []
        Wv = (1 - DELTA) / DELTA * G @ C @ G.T      # evolution variance fixed at the origin (W&H 6.3):
        for _ in range(h):                           # uncertainty grows linearly, not by 1/DELTA per step
            m, C = G @ m, G @ C @ G.T + Wv
            preds.append(np.exp(F @ m + F @ C @ F / 2))
        out.append(np.array(preds))
    return out
