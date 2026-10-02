"""RS-DPF: regime-switching differentiable particle filter (after Li et al. 2023,
"Differentiable Bootstrap Particle Filters for Regime-Switching Models",
arXiv:2302.10319). Written in plain PyTorch: PyDPF (arXiv:2510.25693) has no
regime-switching component and needs Python 3.12, and the pieces needed are short.

model, per-patient standardised log HRV
    regime s_t in {0..K-1}, Markov with learned transition matrix PI (H2)
    x_t = m_s(t) + phi_s (x_{t-1} - m_s(t-1)) + sigma_s e,   m_s(t) = level_s + a cos + b sin  (H4, H5)
    y_t = x_t + tau n;  blank reading: no evidence (H8)
differentiable: regimes are proposed uniformly and reweighted by PI (so PI gets
gradients), x uses the reparameterisation trick, resampling is soft
(q = ALPHA * w + (1 - ALPHA) / N, weights corrected by w / q). All parameters
are learned by maximising the filter's likelihood of the real readings.

impute  : filter forward and filter on the reversed series, average the two
          estimates at each blank (both sides).
forecast: filter to the end of the history, simulate regimes and levels forward (H9).
"""
import math

import numpy as np
import torch

from .data import from_z, hp, phase, to_z, zstats

K, ALPHA = 2, 0.5
W, BATCH, N_TRAIN, N_RUN = 144, 32, 64, 512


class RSDPF(torch.nn.Module):
    def __init__(self, k=K, alpha=ALPHA):
        super().__init__()
        self.K, self.alpha = k, alpha
        self.level = torch.nn.Parameter(torch.linspace(-0.8, 0.8, k))
        self.phi_raw = torch.nn.Parameter(torch.full((k,), 1.5))
        self.log_sig = torch.nn.Parameter(torch.full((k,), -1.0))
        self.pi_logit = torch.nn.Parameter(torch.eye(k) * 3.0)
        self.log_tau = torch.nn.Parameter(torch.tensor(-1.0))
        self.circ = torch.nn.Parameter(torch.zeros(2))

    def mean(self, s, sc):            # s (B,N) regimes, sc (B,2) sin/cos at this slot
        return self.level[s] + (sc @ self.circ)[:, None]

    def run(self, y, sc, n, forecast=0, sc_future=None, gen=None):
        """y (B,T) with NaN blanks, sc (B,T,2). Returns (loglik (B,), filtered means (B,T), forecast (B,h))."""
        B, T = y.shape
        dev = y.device
        log_pi = torch.log_softmax(self.pi_logit, -1)
        phi, sig, tau = torch.sigmoid(self.phi_raw), torch.exp(self.log_sig), torch.exp(self.log_tau)
        s = torch.randint(0, self.K, (B, n), device=dev, generator=gen)
        x = self.mean(s, sc[:, 0]) + sig[s] * torch.randn(B, n, device=dev, generator=gen)
        logw = torch.full((B, n), -math.log(n), device=dev)
        ll, means = torch.zeros(B, device=dev), []
        for t in range(T):
            if t > 0:
                s_new = torch.randint(0, self.K, (B, n), device=dev, generator=gen)
                logw = logw + log_pi[s, s_new] + math.log(self.K)
                x = (self.mean(s_new, sc[:, t]) + phi[s_new] * (x - self.mean(s_new, sc[:, t - 1]))
                     + sig[s_new] * torch.randn(B, n, device=dev, generator=gen))
                s = s_new
            obs = ~torch.isnan(y[:, t])
            lik = torch.distributions.Normal(x, tau).log_prob(torch.nan_to_num(y[:, t])[:, None])
            lw = logw + torch.where(obs[:, None], lik, torch.zeros_like(lik))
            ll = ll + torch.where(obs, torch.logsumexp(lw, -1) - torch.logsumexp(logw, -1), torch.zeros_like(ll))
            w = torch.softmax(lw, -1)
            means.append((w * x).sum(-1))
            q = self.alpha * w + (1 - self.alpha) / n
            idx = torch.multinomial(q, n, replacement=True, generator=gen)
            logw = torch.log(torch.gather(w, 1, idx) + 1e-12) - torch.log(torch.gather(q, 1, idx))
            logw = logw - torch.logsumexp(logw, -1, keepdim=True)
            x, s = torch.gather(x, 1, idx), torch.gather(s, 1, idx)
        fut = []
        if forecast:
            w = torch.softmax(logw, -1)
            for k in range(forecast):
                s_new = torch.multinomial(torch.softmax(self.pi_logit, -1)[s.reshape(-1)], 1,
                                          generator=gen).reshape(B, n)
                prev_sc = sc[:, -1] if k == 0 else sc_future[:, k - 1]
                x = (self.mean(s_new, sc_future[:, k]) + phi[s_new] * (x - self.mean(s_new, prev_sc))
                     + sig[s_new] * torch.randn(B, n, device=dev, generator=gen))
                s = s_new
                fut.append((w * x).sum(-1))
        return ll, torch.stack(means, 1), (torch.stack(fut, 1) if fut else None)


def _prep(series, period, col, st):
    zs = [to_z(s.x[col].to_numpy(float), s.pid, st) for s in series]
    scs = [np.column_stack(phase(s.x.t, period)) for s in series]
    return zs, scs


def _train(zs, scs, ctx):
    torch.manual_seed(ctx["seed"])
    rng = np.random.default_rng(ctx["seed"])
    dev = ctx["device"]
    m = RSDPF(hp(ctx, "K", K), hp(ctx, "alpha", ALPHA)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=hp(ctx, "lr", 0.01))
    length = min(hp(ctx, "window", W), min(len(z) for z in zs))
    for step in range(10 if ctx["quick"] else hp(ctx, "steps", 1500)):
        ii = rng.integers(0, len(zs), BATCH)
        ys, ss = [], []
        for i in ii:
            a = int(rng.integers(0, len(zs[i]) - length + 1))
            ys.append(zs[i][a:a + length])
            ss.append(scs[i][a:a + length])
        y = torch.as_tensor(np.array(ys), dtype=torch.float32, device=dev)
        sc = torch.as_tensor(np.array(ss), dtype=torch.float32, device=dev)
        ll, _, _ = m.run(y, sc, hp(ctx, "n_train", N_TRAIN))
        loss = -(ll.sum() / (~torch.isnan(y)).sum().clamp(min=1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if step % 100 == 0:
            print(f"    step {step}: -loglik per reading {loss.item():.4f}", flush=True)
    return m.eval()


def _batches(n_series, size=16):
    return [np.arange(i, min(i + size, n_series)) for i in range(0, n_series, size)]


def _pad(arrays, left, fill):
    """Stack variable-length arrays, padding with `fill` at the start (left) or end."""
    L = max(len(a) for a in arrays)
    out = np.full((len(arrays), L) + arrays[0].shape[1:], fill, dtype=np.float32)
    for r, a in enumerate(arrays):
        if left:
            out[r, L - len(a):] = a
        else:
            out[r, :len(a)] = a
    return out


def impute(series, ctx):
    st = zstats(series)
    zs, scs = _prep(series, ctx["period"], "hrv", st)
    m = _train(zs, scs, ctx)
    n, dev = (64 if ctx["quick"] else hp(ctx, "n_run", N_RUN)), ctx["device"]
    gen = torch.Generator(device=dev).manual_seed(ctx["seed"])
    est = [None] * len(zs)
    with torch.no_grad():
        for b in _batches(len(zs)):
            y = torch.as_tensor(_pad([zs[i] for i in b], False, np.nan), device=dev)   # blank padding = no evidence
            sc = torch.as_tensor(_pad([scs[i] for i in b], False, 0.0), device=dev)
            fwd = m.run(y, sc, n, gen=gen)[1]
            bwd = m.run(y.flip(1), sc.flip(1), n, gen=gen)[1].flip(1)
            both = ((fwd + bwd) / 2).cpu().numpy()
            for r, i in enumerate(b):
                est[i] = both[r, :len(zs[i])]
    return [from_z(np.where(np.isnan(z), e, z), s.pid, st) for s, z, e in zip(series, zs, est)]


def forecast(history, horizons, ctx):
    st = zstats(history)
    zs, scs = _prep(history, ctx["period"], "hrv_obs", st)
    m = _train(zs, scs, ctx)
    n, dev, period = (64 if ctx["quick"] else hp(ctx, "n_run", N_RUN)), ctx["device"], ctx["period"]
    gen = torch.Generator(device=dev).manual_seed(ctx["seed"])
    hmax, out = max(horizons), [None] * len(zs)
    with torch.no_grad():
        for b in _batches(len(zs)):
            y = torch.as_tensor(_pad([zs[i] for i in b], True, np.nan), device=dev)   # all end on the same step
            sc = torch.as_tensor(_pad([scs[i] for i in b], True, 0.0), device=dev)
            fsc = np.stack([np.column_stack(phase((history[i].x.t.iat[-1] + np.arange(1, hmax + 1)) % period, period))
                            for i in b]).astype(np.float32)
            f = m.run(y, sc, n, forecast=hmax, sc_future=torch.as_tensor(fsc, device=dev), gen=gen)[2].cpu().numpy()
            for r, i in enumerate(b):
                out[i] = from_z(f[r, :horizons[i]], history[i].pid, st)
    return out
