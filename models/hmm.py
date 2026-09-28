"""Hidden Markov model (hmmlearn GaussianHMM) on per-patient standardised log HRV.

One global HMM is fitted with hmmlearn on the unbroken runs of real readings.
hmmlearn cannot take blanks, so inference uses the fitted parameters in a short
forward-backward that simply skips the emission term where the reading is
blank - the HMM then carries its state across gaps by the transition matrix (H8).

impute  : posterior mean over states at each blank (forward-backward, both sides).
forecast: filtered state at the end of the history, pushed forward by the
          transition matrix (H9).
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from .data import from_z, to_z, zstats

STATES = 3


def _runs(z):
    ok = ~np.isnan(z)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return [z[a:b] for a, b in zip(edges[::2], edges[1::2])]


def _fit(zs, ctx):
    from hmmlearn.hmm import GaussianHMM
    runs = [r for z in zs for r in _runs(z) if len(r) >= 2]
    m = GaussianHMM(n_components=STATES, covariance_type="diag", n_iter=10 if ctx["quick"] else 200,
                    random_state=ctx["seed"])
    m.fit(np.concatenate(runs)[:, None], lengths=[len(r) for r in runs])
    return m


def _loglik(m, z):
    ll = norm.logpdf(z[:, None], m.means_[:, 0], np.sqrt(m.covars_[:, 0, 0]))
    ll[np.isnan(z)] = 0.0                     # blank: no evidence
    return ll


def _forward(m, ll):
    A = np.log(m.transmat_)
    a = np.empty_like(ll)
    a[0] = np.log(m.startprob_ + 1e-300) + ll[0]
    for t in range(1, len(ll)):
        a[t] = logsumexp(a[t - 1][:, None] + A, axis=0) + ll[t]
    return a


def impute(series, ctx):
    st = zstats(series)
    zs = [to_z(s.x.hrv.to_numpy(float), s.pid, st) for s in series]
    m = _fit(zs, ctx)
    A = np.log(m.transmat_)
    out = []
    for s, z in zip(series, zs):
        ll = _loglik(m, z)
        a = _forward(m, ll)
        b = np.zeros_like(ll)
        for t in range(len(ll) - 2, -1, -1):
            b[t] = logsumexp(A + ll[t + 1] + b[t + 1], axis=1)
        post = a + b
        post = np.exp(post - logsumexp(post, axis=1, keepdims=True))
        zhat = post @ m.means_[:, 0]
        out.append(from_z(np.where(np.isnan(z), zhat, z), s.pid, st))
    return out


def forecast(history, horizons, ctx):
    st = zstats(history)
    zs = [to_z(s.x.hrv_obs.to_numpy(float), s.pid, st) for s in history]
    m = _fit(zs, ctx)
    out = []
    for s, z, h in zip(history, zs, horizons):
        a = _forward(m, _loglik(m, z))[-1]
        p = np.exp(a - logsumexp(a))
        preds = []
        for _ in range(h):
            p = p @ m.transmat_
            preds.append(p @ m.means_[:, 0])
        out.append(from_z(np.array(preds), s.pid, st))
    return out
