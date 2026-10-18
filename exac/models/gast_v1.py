"""GAST v1: gap-aware state-space transformer for exacerbation detection and onset prediction.

    python -m exac.models.gast_v1   # self-tests of the time logic, masks and padding (needs torch only)

Design, sources and every decision: docs/GAST_DESIGN.md. One token per minute that has a heart-rate
reading, at its real time. Per state channel j:

    h_i = exp(-rate_j * (t_i - t_{i-1})) * h_{i-1} + sigmoid(g_j . u_i) * (w_j . u_i)

Linear in h (a diagonal state space as in S4D/Mamba; the gate depends on the token only), so it is
computed exactly, all at once, by an associative scan. Between tokens the state only fades, by the real
elapsed time; when a token arrives it is written in (GRU-ODE-Bayes: evolve, then correct). The state is
read at the end of every hour of the window, faded from the last token before it (nothing is invented).
The hourly readings pass one causal self-attention layer with a learned time bias per head; the last
one - the end of the classified day - feeds one logistic output per task.

Variants (the ablations of docs/GAST_DESIGN.md section 4):
  full      state space + attention
  ssm       state space only
  attn      attention only: each hour attends to the tokens themselves with learned time embeddings
            (as mTAN), then the same causal self-attention
  nogap     no masks, times since or elapsed times; the state space steps by token count
  maskonly  no values: masks, times since, elapsed times and time of day only
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HR = 1                     # channel order of model_data_exac/minutes.npy: hrv, hr, temp, spo2, steps, sleep_active, steps_active
VARIANTS = ("full", "ssm", "attn", "nogap", "maskonly")
from exac.config import REF_MINUTES   # the state is read at the end of every hour (locked 17 Oct 2026)


def n_features(variant, channels=7):
    return {"nogap": channels + 2, "maskonly": 2 * channels + 3}.get(variant, 3 * channels + 3)


def tokens(v, scale, variant="full"):
    """Window minutes -> tokens: one per minute with a heart-rate reading (pass 1), in time order.

    v: (n, M, C) raw minutes, NaN = no reading; minute 0 is the window's first minute (a midnight).
    scale: (C,) each signal's largest value in the training patients (a change of units only).
    Returns feat (n, L, F) float32, t (n, L) float32 minutes from the window start (padding repeats the
    last real time, so no time passes), valid (n, L) bool. A blank value is 0 with its mask 0."""
    n, M, C = v.shape
    obs = ~np.isnan(v)
    tok = obs[:, :, HR]
    count = tok.sum(1)
    L = max(int(count.max()), 1)
    order = np.argsort(~tok, axis=1, kind="stable")[:, :L]             # the token minutes first, in time order
    valid = np.arange(L)[None, :] < count[:, None]
    t = order.astype(np.float32)
    t = np.where(valid, t, t[np.arange(n), np.maximum(count - 1, 0)][:, None])
    minute = np.arange(M)[None, :, None]
    seen = np.maximum.accumulate(np.where(obs, minute, -1), axis=1)       # last minute <= now with a reading, per signal
    since = np.where(seen >= 0, minute - seen, minute + 1)                # none yet in the window: minutes since its start
    take = lambda x: np.take_along_axis(x, order[:, :, None], 1)
    val = np.nan_to_num(take(v) / scale)
    msk = take(obs).astype(np.float32)
    snc = np.log1p(take(since))
    dt = np.log1p(np.diff(t, axis=1, prepend=-1.0))[..., None]            # minutes since the previous token
    ang = 2 * np.pi * (t % 1440) / 1440
    tod = np.stack([np.sin(ang), np.cos(ang)], -1)                       # time of day: part of the timestamp
    parts = {"nogap": [val, tod], "maskonly": [msk, snc, dt, tod]}.get(variant, [val, msk, snc, dt, tod])
    feat = np.concatenate(parts, -1) * valid[..., None]
    return feat.astype(np.float32), t.astype(np.float32), valid


def scan(a, b):
    """h_i = a_i * h_{i-1} + b_i for every i at once (state before the first token = 0); exact.
    Inclusive associative scan (Hillis-Steele): log2(L) steps over (B, L, N) tensors."""
    off = 1
    while off < a.shape[1]:
        b = b + a * F.pad(b[:, :-off], (0, 0, off, 0))
        a = a * F.pad(a[:, :-off], (0, 0, off, 0), value=1.0)
        off *= 2
    return b


class GAST(nn.Module):
    def __init__(self, n_feat, n_out=1, n_static=0, variant="full", window_minutes=10080, d=32, n_state=32,
                 heads=2, dropout=0.1, fastest_minutes=10.0):
        super().__init__()
        assert variant in VARIANTS, variant
        self.variant, self.K, self.heads = variant, window_minutes // REF_MINUTES, heads
        self.inp = nn.Linear(n_feat, d)
        if variant == "attn":
            self.temb = nn.Linear(1, d)                                      # time embedding [linear, periodic] (mTAN)
            self.start = nn.Parameter(torch.zeros(d))                        # a key each hour may attend to before any token
            self.cross = nn.MultiheadAttention(d, heads, dropout, batch_first=True)
            self.stat = nn.Linear(n_static, d) if n_static else None
        else:
            # rates spread log-evenly from 1/(fastest_minutes) to 1/(the window): memories of minutes to days
            self.log_rate = nn.Parameter(torch.linspace(math.log(1 / fastest_minutes), math.log(1 / window_minutes), n_state))
            self.write, self.gate, self.read = nn.Linear(d, n_state), nn.Linear(d, n_state), nn.Linear(n_state, d)
            self.h0 = nn.Linear(n_static, n_state) if n_static else None    # clinical values set the starting state
        self.norm0 = nn.LayerNorm(d)
        if variant != "ssm":
            self.slope = nn.Parameter(torch.full((heads,), -3.0))           # learned time bias per head (like ALiBi)
            self.att = nn.MultiheadAttention(d, heads, dropout, batch_first=True)
            self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.out = nn.Linear(d, n_out)

    def _ends(self, t):
        return (torch.arange(self.K, device=t.device, dtype=t.dtype) + 1) * REF_MINUTES - 1   # each hour's last minute

    def states(self, u, t, valid, static=None):
        """State-space memory read at the end of every hour: (B, K, n_state)."""
        B = u.shape[0]
        r = self._ends(t)
        rate = self.log_rate.exp()
        tt = (valid.cumsum(1) - 1).clamp(min=0).to(t.dtype) if self.variant == "nogap" else t
        dt = torch.diff(tt, dim=1, prepend=torch.full_like(tt[:, :1], -1.0)).clamp(min=0)
        a = torch.exp(-dt[..., None] * rate)
        b = torch.sigmoid(self.gate(u)) * self.write(u) * valid[..., None]
        h0 = self.h0(static) if self.h0 is not None else None
        if h0 is not None:
            b = torch.cat([b[:, :1] + a[:, :1] * h0[:, None], b[:, 1:]], 1)
        h = scan(a, b)                                                       # state after each token
        pos = torch.searchsorted(t.contiguous(), r.expand(B, -1).contiguous(), right=True) - 1
        p = pos.clamp(min=0)
        hl = torch.gather(h, 1, p[..., None].expand(-1, -1, h.shape[-1]))
        if self.variant != "nogap":                                          # fade by the real time since that token
            gap = (r[None] - torch.gather(t, 1, p)).clamp(min=0)              # hours with no token yet are replaced below;
            hl = torch.exp(-gap[..., None] * rate) * hl                       # clamping keeps that unused branch finite (NaN gradients)
        empty = h0[:, None] * torch.exp(-(r[None, :, None] + 1) * rate) if h0 is not None else torch.zeros_like(hl)
        return torch.where((pos >= 0)[..., None], hl, empty)

    def hourly(self, feat, t, valid, static=None):
        """The hourly readings before attention: (B, K, d)."""
        u = self.inp(feat)
        if self.variant != "attn":
            return self.norm0(F.gelu(self.read(self.states(u, t, valid, static))))
        B, r, span = u.shape[0], self._ends(t), self.K * REF_MINUTES
        emb = lambda x: (lambda e: torch.cat([e[..., :1], torch.sin(e[..., 1:])], -1))(self.temb(x[..., None] / span))
        start = self.start.expand(B, 1, -1) + (self.stat(static)[:, None] if self.stat is not None else 0)
        keys = torch.cat([start, u + emb(t)], 1)
        tk = torch.cat([torch.full_like(t[:, :1], -1.0), t], 1)
        ok = torch.cat([torch.ones_like(valid[:, :1]), valid], 1)
        blocked = (tk[:, None, :] > r[None, :, None]) | ~ok[:, None, :]      # no later token, no padding
        z = self.cross(emb(r).expand(B, -1, -1), keys, torch.cat([start, u], 1),
                       attn_mask=blocked.repeat_interleave(self.heads, 0), need_weights=False)[0]
        return self.norm0(z)

    def forward(self, feat, t, valid, static=None, all_hours=False):
        z = self.hourly(feat, t, valid, static)
        if self.variant != "ssm":
            i = torch.arange(z.shape[1], device=z.device)
            dist = (i[:, None] - i[None, :]).to(z.dtype)                     # hours between readings
            bias = (-F.softplus(self.slope)[:, None, None] * dist.clamp(min=0)).masked_fill(dist < 0, float("-inf"))
            y = self.norm1(z)
            z = z + self.att(y, y, y, attn_mask=bias.repeat(z.shape[0], 1, 1), need_weights=False)[0]
            z = z + self.ffn(self.norm2(z))
        return z if all_hours else self.out(z[:, -1])                        # logits (B, n_out)


def _check():
    torch.manual_seed(0)
    ones = np.ones(7, np.float32)
    as_t = lambda x: [torch.as_tensor(a) for a in x]
    # 1. the scan equals a step-by-step loop
    a, b = torch.rand(2, 37, 3), torch.randn(2, 37, 3)
    h, ref = torch.zeros(2, 3), []
    for i in range(37):
        h = a[:, i] * h + b[:, i]
        ref.append(h)
    assert torch.allclose(scan(a, b), torch.stack(ref, 1), atol=1e-5), "scan != loop"
    # 2. tokens: only minutes with heart rate, in order; masks, times since and elapsed times are exact
    v = np.full((1, 1440, 7), np.nan, np.float32)
    v[0, [0, 5, 6, 100], 1] = 70.0
    v[0, 5, 0] = 40.0
    feat, t, valid = tokens(v, ones)
    assert t.tolist() == [[0, 5, 6, 100]] and valid.all()
    assert feat[0, :, 7].tolist() == [0, 1, 0, 0], "HRV mask"
    assert np.allclose(feat[0, :, 14], np.log1p([1, 0, 1, 95])), "minutes since the last HRV"
    assert np.allclose(feat[0, :, 21], np.log1p([1, 5, 1, 94])), "minutes since the previous token"
    # 3. between tokens the state only fades, by exp(-rate * elapsed minutes)
    m = GAST(n_features("full"), window_minutes=1440).eval()
    fu, tt, va = as_t(tokens(v, ones))
    with torch.no_grad():
        s = m.states(m.inp(fu), tt, va)
        rate = m.log_rate.exp()
        assert torch.allclose(s[0, 3], s[0, 2] * torch.exp(-rate * 60), rtol=1e-4, atol=1e-7), "fade between hours"
    # 4. causal: a token late in the day changes no reading of an earlier hour, before or after attention
    v2 = v.copy()
    v2[0, 1300, [1, 0]] = 99.0, 120.0
    for variant in VARIANTS:
        m = GAST(n_features(variant), window_minutes=1440, variant=variant).eval()
        with torch.no_grad():
            z1, z2 = (m(*as_t(tokens(x, ones, variant)), all_hours=True) for x in (v, v2))
        assert torch.allclose(z1[:, :21], z2[:, :21], atol=1e-5), f"{variant}: a later token leaked into an earlier hour"
        assert not torch.allclose(z1[:, -1], z2[:, -1]), f"{variant}: the last hour ignores the day's tokens"
    # 5. padding: a short window gives the same output alone and batched with a longer one
    long = np.full((1, 1440, 7), np.nan, np.float32)
    long[0, ::3, 1] = 80.0
    for variant in VARIANTS:
        m = GAST(n_features(variant), window_minutes=1440, variant=variant).eval()
        with torch.no_grad():
            alone = m(*as_t(tokens(v, ones, variant)))
            both = m(*as_t(tokens(np.concatenate([v, long]), ones, variant)))
        assert torch.allclose(alone[0], both[0], atol=1e-5), f"{variant}: padding changed the output"
    # 6. clinical starting state and two outputs run
    m = GAST(n_features("full"), n_out=2, n_static=3, window_minutes=1440).eval()
    with torch.no_grad():
        o = m(*as_t(tokens(v, ones)), static=torch.ones(1, 3))
    assert o.shape == (1, 2) and torch.isfinite(o).all()
    # 7. gradients stay finite when the first hours of the window have no token yet
    v3 = np.full((1, 1440, 7), np.nan, np.float32)
    v3[0, [300, 301], 1] = 70.0
    for variant in VARIANTS:
        m = GAST(n_features(variant), window_minutes=1440, variant=variant)
        m(*as_t(tokens(v3, ones, variant))).sum().backward()
        assert all(torch.isfinite(q.grad).all() for q in m.parameters() if q.grad is not None), f"{variant}: NaN gradient"
    print("ok: scan = loop; tokens, masks and times exact; state fades by real time; no later token reaches "
          "an earlier hour; padding changes nothing; clinical start and two outputs run; gradients finite")


# ------------------------------------------------------------------ hooks for exac/run.py (from run_exac.py)
KIND = "net"


def make(n_out, variant):
    from exac import config as C
    from exac import data
    return GAST(n_features(variant), n_out, data.n_static(), variant, window_minutes=data.M, **C.GAST)


def inputs(T, rows, sc, variant):
    from exac import data
    (scale, sscale) = sc
    st = data.static(T, sscale)
    args = list(tokens(data.window(rows), scale, variant)) + ([st] if st is not None else [])
    return [torch.as_tensor(x, device=data.device) for x in args]


if __name__ == "__main__":
    _check()
