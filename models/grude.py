"""GRU-ODE-Bayes (De Brouwer et al. 2019), using the authors' model code
(models/vendor/gru_ode_bayes.py, MIT). This file only converts our series to
their sparse format and reads predictions off the model's path.

channels   HRV (standardised log) + the other vitals, each with its own mask:
           a blank or hidden reading is simply unobserved (H7, H8)
between    readings the hidden state evolves by the GRU-ODE (Euler, one slot per step)
at         a reading, the GRU-Bayes cell updates it
prediction at a blank = the model's pre-jump mean at that time

impute  : a forward model and a model trained on reversed time; the two
          predictions at each blank are averaged (both sides).
forecast: run over the end of the history and keep propagating (H9).
"""
import numpy as np
import torch

from .data import COVS, from_z, hp, to_z, zstats
from .vendor.gru_ode_bayes import NNFOwithBayesianJumps

W, BATCH, HIDDEN = 288, 16, 32


def _channels(series, col, st):
    covs = [c for c in COVS if c in series[0].x]
    allc = np.concatenate([s.x[covs].to_numpy(float) for s in series])
    mu, sd = np.nanmean(allc, 0), np.nanstd(allc, 0)
    sd[~(sd > 0)] = 1.0
    return [np.column_stack([to_z(s.x[col].to_numpy(float), s.pid, st),
                             (s.x[covs].to_numpy(float) - mu) / sd]) for s in series]


def _sparse(chunks, offsets, dev):
    """chunks: list of (L_b, C) arrays with NaN blanks, placed at time offsets[b]."""
    ev = []
    for b, (Z, off) in enumerate(zip(chunks, offsets)):
        for t in np.flatnonzero((~np.isnan(Z)).any(1)):
            ev.append((off + t, b, Z[t]))
    ev.sort(key=lambda e: (e[0], e[1]))
    times = np.array(sorted({e[0] for e in ev}), float)
    counts = np.bincount(np.searchsorted(times, [e[0] for e in ev]), minlength=len(times))
    ptr = np.r_[0, np.cumsum(counts)]
    V = np.array([e[2] for e in ev])
    X = torch.as_tensor(np.nan_to_num(V), dtype=torch.float32, device=dev)
    M = torch.as_tensor(~np.isnan(V), dtype=torch.float32, device=dev)
    idx = torch.as_tensor([e[1] for e in ev], dtype=torch.long, device=dev)
    return times, ptr, X, M, idx


def _net(C, dev, ctx):
    return NNFOwithBayesianJumps(input_size=C, hidden_size=hp(ctx, "hidden", HIDDEN),
                                 p_hidden=hp(ctx, "p_hidden", 32), prep_hidden=hp(ctx, "prep_hidden", 8),
                                 cov_size=1, cov_hidden=8, logvar=True, mixing=1e-4, solver="euler").to(dev)


def _train(arrays, ctx):
    torch.manual_seed(ctx["seed"])
    rng = np.random.default_rng(ctx["seed"])
    dev = ctx["device"]
    net = _net(arrays[0].shape[1], dev, ctx)
    opt = torch.optim.Adam(net.parameters(), lr=hp(ctx, "lr", 1e-3))
    lens = np.array([len(a) for a in arrays])
    w, batch = hp(ctx, "window", W), hp(ctx, "batch", BATCH)
    for step in range(5 if ctx["quick"] else hp(ctx, "steps", 1500)):
        chunks = []
        for i in rng.choice(len(arrays), batch, p=lens / lens.sum()):
            a = int(rng.integers(0, max(1, lens[i] - w + 1)))
            chunks.append(arrays[i][a:a + w])
        times, ptr, X, M, idx = _sparse(chunks, [0] * len(chunks), dev)
        T = float(max(len(c) for c in chunks))
        _, loss, _, _ = net(times, ptr, X, M, idx, delta_t=1.0, T=T, cov=torch.ones(len(chunks), 1, device=dev))
        loss = loss / M.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"    step {step}: loss {loss.item():.4f}", flush=True)
    return net.eval()


def _path_means(net, chunks, offsets, T, dev):
    """Pre-jump HRV mean at every integer time 0..T-1, per chunk: (B, T)."""
    times, ptr, X, M, idx = _sparse(chunks, offsets, dev)
    out = net(times, ptr, X, M, idx, delta_t=1.0, T=float(T), cov=torch.ones(len(chunks), 1, device=dev),
              return_path=True)
    path_t, path_p = np.round(out[3]).astype(int), out[4][:, :, 0].cpu().numpy()   # channel 0 = HRV mean
    first = {}
    for k, t in enumerate(path_t):
        first.setdefault(t, k)
    return np.stack([path_p[first[t]] for t in range(T)], axis=1)


def impute(series, ctx):
    st = zstats(series)
    arrays = _channels(series, "hrv", st)
    dev = ctx["device"]
    fwd_net = _train(arrays, ctx)
    bwd_net = _train([a[::-1].copy() for a in arrays], ctx)
    order = np.argsort([len(a) for a in arrays])
    est = [None] * len(arrays)
    with torch.no_grad():
        for b in [order[i:i + BATCH] for i in range(0, len(order), BATCH)]:
            T = max(len(arrays[i]) for i in b)
            f = _path_means(fwd_net, [arrays[i] for i in b], [0] * len(b), T, dev)
            r = _path_means(bwd_net, [arrays[i][::-1] for i in b], [0] * len(b), T, dev)
            for row, i in enumerate(b):
                n = len(arrays[i])
                est[i] = (f[row, :n] + r[row, :n][::-1]) / 2
    return [from_z(np.where(np.isnan(a[:, 0]), e, a[:, 0]), s.pid, st) for s, a, e in zip(series, arrays, est)]


def forecast(history, horizons, ctx):
    st = zstats(history)
    arrays = _channels(history, "hrv_obs", st)
    dev, hmax = ctx["device"], max(horizons)
    net = _train(arrays, ctx)
    out = [None] * len(arrays)
    with torch.no_grad():
        for b in [np.arange(i, min(i + BATCH, len(arrays))) for i in range(0, len(arrays), BATCH)]:
            ctxs = [arrays[i][-hp(ctx, "window", W):] for i in b]
            L = max(len(c) for c in ctxs)
            f = _path_means(net, ctxs, [L - len(c) for c in ctxs], L + hmax, dev)   # all contexts end at L
            for row, i in enumerate(b):
                out[i] = from_z(f[row, L:L + horizons[i]], history[i].pid, st)
    return out
