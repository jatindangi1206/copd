"""Shared training for the neural sequence models (RNN, LSTM, gap-aware SSM
transformer). A model file only supplies the network: (batch, time, features)
-> (batch, time), on per-patient standardised log HRV.

impute  : inputs per slot = value (0 if blank), seen-flag, time of day, slots
          since / until the nearest real reading, standardised vitals. Training
          hides a further HIDE share of the visible readings at random and learns
          to rebuild them (the evaluation mask is never seen). Networks may look
          both ways.
forecast: inputs = gap-filled value, was-real flag, time of day. Trained to
          predict the next slot (loss on real readings only), then rolled forward
          one step at a time. Networks must be causal (H9).
"""
import numpy as np
import torch

from .data import COVS, cov_matrix, from_z, phase, to_z, zstats

W = 288        # training window, slots (2 days of 10-minute data)
HIDE = 0.2     # extra share hidden during imputation training
BATCH = 32


def _gaps(obs):
    """Slots since the last and until the next real reading, per row of a (B, L) mask."""
    B, L = obs.shape
    idx = np.arange(L)
    last = np.where(obs, idx, -10 ** 6)
    since = idx - np.maximum.accumulate(np.pad(last, ((0, 0), (1, 0)), constant_values=-10 ** 6)[:, :-1], axis=1)
    nxt = np.where(obs, idx, 10 ** 6)
    until = np.minimum.accumulate(np.pad(nxt, ((0, 0), (0, 1)), constant_values=10 ** 6)[:, :0:-1], axis=1)[:, ::-1] - idx
    cap = lambda a: np.log1p(np.minimum(a, 1000)) / 7
    return cap(since), cap(until)


def _impute_batch(z, cov, sc, obs):
    since, until = _gaps(obs)
    return np.concatenate([np.nan_to_num(np.where(obs, z, 0))[..., None], obs[..., None], sc,
                           since[..., None], until[..., None], cov], axis=-1).astype(np.float32)


def _windows(arrays, length, rng, n):
    """n random windows of `length` across series; shorter series are padded (pad flag 0)."""
    out = [[] for _ in arrays[0]] + [[]]
    lens = np.array([len(a) for a in arrays[0]])
    for _ in range(n):
        i = int(rng.choice(len(lens), p=lens / lens.sum()))
        a = int(rng.integers(0, max(1, lens[i] - length + 1)))
        for k, arr in enumerate(arrays):
            w = arr[i][a:a + length]
            pad = length - len(w)
            out[k].append(np.concatenate([w, np.zeros((pad,) + w.shape[1:], w.dtype)]) if pad else w)
        out[-1].append(np.r_[np.ones(length - pad), np.zeros(pad)])
    return [np.stack(o) for o in out]


def _fit(net, ctx, make_batch, steps):
    dev = ctx["device"]
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(steps):
        x, y, m = (torch.as_tensor(a, device=dev) for a in make_batch())
        pred = net(x)
        loss = (((pred - y) ** 2) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"    step {step}: loss {loss.item():.4f}", flush=True)
    return net.eval()


def fit_impute(make_net, series, ctx):
    rng = np.random.default_rng(ctx["seed"])
    torch.manual_seed(ctx["seed"])
    st = zstats(series)
    covs = cov_matrix(series, [c for c in COVS if c in series[0].x])
    zs = [to_z(s.x.hrv.to_numpy(float), s.pid, st) for s in series]
    scs = [np.column_stack(phase(s.x.t, ctx["period"])) for s in series]
    length = min(W, max(len(z) for z in zs))
    n_in = 6 + covs[0].shape[1]
    net = make_net(n_in)

    def batch():
        z, c, sc, valid = _windows([zs, covs, scs], length, rng, BATCH)
        seen = ~np.isnan(z) & (valid > 0)
        hide = seen & (rng.random(z.shape) < HIDE)
        x = _impute_batch(z, c, sc, seen & ~hide)
        return x, np.nan_to_num(z).astype(np.float32), hide.astype(np.float32)

    net = _fit(net, ctx, batch, 20 if ctx["quick"] else 3000)
    out = []
    with torch.no_grad():
        for s, z, c, sc in zip(series, zs, covs, scs):
            obs = ~np.isnan(z)
            x = torch.as_tensor(_impute_batch(z[None], c[None], sc[None], obs[None]), device=ctx["device"])
            p = net(x)[0].cpu().numpy()
            out.append(from_z(np.where(obs, z, p), s.pid, st))
    return out


def fit_forecast(make_net, history, horizons, ctx):
    rng = np.random.default_rng(ctx["seed"])
    torch.manual_seed(ctx["seed"])
    st, period = zstats(history), ctx["period"]
    zf = [np.nan_to_num(to_z(s.x.hrv.to_numpy(float), s.pid, st)) for s in history]
    zo = [to_z(s.x.hrv_obs.to_numpy(float), s.pid, st) for s in history]
    feats = [np.column_stack([z, ~np.isnan(o), *phase(s.x.t, period)]).astype(np.float32)
             for s, z, o in zip(history, zf, zo)]
    length = min(W, max(len(f) for f in feats))
    net = make_net(4)

    def batch():
        f, o, valid = _windows([feats, zo], length, rng, BATCH)
        y = o[:, 1:]
        m = (~np.isnan(y)) & (valid[:, 1:] > 0)
        x = f[:, :-1]
        return x, np.where(m, y, 0).astype(np.float32), m.astype(np.float32)   # x[:, :t] -> slot t+1

    net = _fit(net, ctx, batch, 20 if ctx["quick"] else 3000)
    hmax = max(horizons)
    x = np.stack([np.concatenate([np.zeros((length - len(f[-length:]), 4), np.float32), f[-length:]])
                  for f in feats])                      # last `length` slots, left-padded
    tpos = np.array([s.x.t.iat[-1] for s in history])
    preds = []
    with torch.no_grad():
        xt = torch.as_tensor(x, device=ctx["device"])
        for k in range(1, hmax + 1):
            nxt = net(xt)[:, -1]
            preds.append(nxt.cpu().numpy())
            sn, cs = phase((tpos + k) % period, period)
            row = torch.as_tensor(np.column_stack([nxt.cpu().numpy(), np.zeros(len(tpos)), sn, cs]).astype(np.float32),
                                  device=ctx["device"])
            xt = torch.cat([xt[:, 1:], row[:, None]], dim=1)
    P = np.stack(preds, axis=1)
    return [from_z(P[i, :h], s.pid, st) for i, (s, h) in enumerate(zip(history, horizons))]
