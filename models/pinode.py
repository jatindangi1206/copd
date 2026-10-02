"""Physiology-informed neural ODE (torchdiffeq), on per-patient standardised log HRV.

dx/dt = -k (x - c(t))            homeostasis: HRV relaxes to a daily set point c(t) (H4, H5)
        + g(x, time of day, u)   small neural network for what the prior misses;
                                 u = the other vitals (imputation only: not known ahead)
x lives in log space, so exp(x) stays positive (H1). Time is in slots.

Training: every pair of consecutive real readings (up to MAX_GAP slots apart) is
one initial-value problem, x(t_i) -> x(t_j). All pairs are solved together by
rescaling each pair's time to s in [0, 1] (dx/ds = gap * dx/dt).

impute  : integrate forward from the reading before a blank and backward from the
          reading after it, weight the two by closeness (both sides).
forecast: integrate forward from the end of the history (H9).
"""
import numpy as np
import torch

from .data import COVS, cov_matrix, from_z, phase, to_z, zstats

MAX_GAP, HIDDEN, BATCH = 18, 32, 512


class Field(torch.nn.Module):
    def __init__(self, n_u, period):
        super().__init__()
        self.period = period
        self.log_k = torch.nn.Parameter(torch.tensor(-2.0))
        self.c = torch.nn.Parameter(torch.zeros(3))
        self.g = torch.nn.Sequential(torch.nn.Linear(3 + n_u, HIDDEN), torch.nn.Tanh(),
                                     torch.nn.Linear(HIDDEN, 1))
        torch.nn.init.zeros_(self.g[-1].weight)
        self.t0 = self.gap = self.u = None

    def forward(self, s, x):                            # x (B, 1); s scalar in [0, 1]
        t = self.t0 + s * self.gap                        # real time, slots
        a = 2 * np.pi * t / self.period
        sn, cs = torch.sin(a), torch.cos(a)
        set_point = self.c[0] + self.c[1] * cs + self.c[2] * sn
        dx = (-torch.exp(self.log_k) * (x[:, 0] - set_point)
              + self.g(torch.cat([x, sn[:, None], cs[:, None], self.u], 1))[:, 0])
        return (self.gap * dx)[:, None]


def _solve(f, x0, t0, gap, u, grid):
    from torchdiffeq import odeint
    f.t0, f.gap, f.u = t0, gap, u
    # The field is scaled by the gap, so a step of h in s is h * gap slots. Keep it to at most one slot,
    # else an 18-slot gap is crossed in 4-slot steps and rk4 diverges (it did: MAE 1e60).
    step = min(float(grid[1] - grid[0]) if len(grid) > 2 else 1.0, 1.0 / max(float(gap.abs().max()), 1.0))
    return odeint(f, x0[:, None], grid, method="rk4", options={"step_size": step})[..., 0]


def _pairs(zs, us, ts):
    rows = []
    for k, (z, u, t) in enumerate(zip(zs, us, ts)):
        obs = np.flatnonzero(~np.isnan(z))
        for i, j in zip(obs[:-1], obs[1:]):
            if j - i <= MAX_GAP:
                rows.append((z[i], z[j], ts[k][i], j - i, k, i))
    return rows


def _fit(zs, us, ts, n_u, ctx):
    torch.manual_seed(ctx["seed"])
    rng = np.random.default_rng(ctx["seed"])
    dev = ctx["device"]
    f = Field(n_u, ctx["period"]).to(dev)
    P = _pairs(zs, us, ts)
    opt = torch.optim.Adam(f.parameters(), lr=3e-3)
    grid = torch.tensor([0.0, 1.0], device=dev)
    for step in range(20 if ctx["quick"] else 2000):
        b = [P[i] for i in rng.integers(0, len(P), BATCH)]
        x0, x1, t0, gap = (torch.tensor([r[c] for r in b], dtype=torch.float32, device=dev) for c in range(4))
        u = torch.as_tensor(np.array([us[r[4]][r[5]] for r in b]), dtype=torch.float32, device=dev)
        pred = _solve(f, x0, t0, gap, u, grid)[-1]
        loss = ((pred - x1) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"    step {step}: loss {loss.item():.4f}", flush=True)
    return f.eval()


def _inputs(series, ctx, col, use_covs):
    st = zstats(series)
    zs = [to_z(s.x[col].to_numpy(float), s.pid, st) for s in series]
    ts = [s.x.t.to_numpy(float) for s in series]
    if use_covs:
        us = cov_matrix(series, [c for c in COVS if c in series[0].x])
    else:
        us = [np.zeros((len(z), 0)) for z in zs]
    return st, zs, ts, us


def impute(series, ctx):
    st, zs, ts, us = _inputs(series, ctx, "hrv", True)
    f = _fit(zs, us, ts, us[0].shape[1], ctx)
    dev, grid = ctx["device"], torch.tensor([0.0, 1.0], device=ctx["device"])
    jobs = []                                        # (series k, blank slot, from slot, direction)
    for k, z in enumerate(zs):
        obs = np.flatnonzero(~np.isnan(z))
        for t in np.flatnonzero(np.isnan(z)):
            prev, nxt = obs[obs < t], obs[obs > t]
            if len(prev):
                jobs.append((k, t, prev[-1]))
            if len(nxt):
                jobs.append((k, t, nxt[0]))
    est = {}
    with torch.no_grad():
        for a in range(0, len(jobs), 4096):
            J = jobs[a:a + 4096]
            x0 = torch.tensor([zs[k][src] for k, t, src in J], dtype=torch.float32, device=dev)
            t0 = torch.tensor([ts[k][src] for k, t, src in J], dtype=torch.float32, device=dev)
            gap = torch.tensor([t - src for k, t, src in J], dtype=torch.float32, device=dev)   # < 0: backward
            u = torch.as_tensor(np.array([us[k][src] for k, t, src in J]), dtype=torch.float32, device=dev)
            x1 = _solve(f, x0, t0, gap, u, grid)[-1].cpu().numpy()
            for (k, t, src), v in zip(J, x1):
                est.setdefault((k, t), []).append((abs(t - src), v))
    out = []
    for k, (s, z) in enumerate(zip(series, zs)):
        zz = z.copy()
        for t in np.flatnonzero(np.isnan(z)):
            d = est[(k, t)]
            w = np.array([1.0 / dd for dd, _ in d])
            zz[t] = float(np.dot(w, [v for _, v in d]) / w.sum())
        out.append(from_z(zz, s.pid, st))
    return out


def forecast(history, horizons, ctx):
    st, zs, ts, us = _inputs(history, ctx, "hrv_obs", False)
    f = _fit(zs, us, ts, 0, ctx)
    dev, hmax = ctx["device"], max(horizons)
    zf = [to_z(s.x.hrv.to_numpy(float), s.pid, st) for s in history]      # gap-filled: start from the last slot
    with torch.no_grad():
        x0 = torch.tensor([z[-1] for z in zf], dtype=torch.float32, device=dev)
        t0 = torch.tensor([t[-1] for t in ts], dtype=torch.float32, device=dev)
        gap = torch.full((len(zf),), float(hmax), device=dev)
        grid = torch.linspace(0, 1, hmax + 1, device=dev)
        path = _solve(f, x0, t0, gap, torch.zeros(len(zf), 0, device=dev), grid)[1:].T.cpu().numpy()
    return [from_z(path[i, :h], s.pid, st) for i, (s, h) in enumerate(zip(history, horizons))]
