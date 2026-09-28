"""Gap-aware state-space transformer - FIRST VERSION, to be replaced by the team's
own design. It is here so the model slot runs end to end; the architecture is
a minimal reading of "state-space layer + transformer, aware of gaps":

  gap-aware SSM layer : per channel, an exponentially decaying average of the
                        embeddings at REAL readings only (blank slots add
                        nothing), divided by the same decay over the seen-flag.
                        The state therefore stays at the last real information
                        and its weight shows how stale that is. Computed as an
                        FFT convolution; both directions for imputation, past
                        only for forecasting (H9).
  transformer layer   : standard self-attention over the sequence (causal mask
                        when forecasting).

Inputs and training loop: seq.py (the same as the RNN and LSTM).
"""
import torch

from .seq import fit_forecast, fit_impute

D, HEADS, BLOCKS = 64, 4, 2


def _decay_conv(u, rate, reverse):
    """out[t] = sum_k a^k u[t-k] along time (dim 1), a = exp(-rate) per channel."""
    L = u.shape[1]
    if reverse:
        u = u.flip(1)
    k = torch.exp(-rate[None, :] * torch.arange(L, device=u.device)[:, None])      # (L, D)
    n = 2 * L
    y = torch.fft.irfft(torch.fft.rfft(u, n=n, dim=1) * torch.fft.rfft(k, n=n, dim=0)[None], n=n, dim=1)[:, :L]
    return y.flip(1) if reverse else y


class GapSSM(torch.nn.Module):
    def __init__(self, d, bidirectional):
        super().__init__()
        self.bi = bidirectional
        self.log_rate = torch.nn.Parameter(torch.linspace(-4, 0, d))    # slow to fast memories
        self.out = torch.nn.Linear(d * (4 if bidirectional else 2), d)

    def forward(self, x, seen):
        rate = torch.nn.functional.softplus(self.log_rate)
        parts = []
        for rev in ([False, True] if self.bi else [False]):
            num = _decay_conv(x * seen, rate, rev)
            den = _decay_conv(seen.expand_as(x), rate, rev)
            parts += [num / (den + 1e-6), torch.log1p(den)]
        return self.out(torch.cat(parts, -1))


class Net(torch.nn.Module):
    def __init__(self, n_in, bidirectional):
        super().__init__()
        self.bi = bidirectional
        self.embed = torch.nn.Linear(n_in, D)
        self.ssm = torch.nn.ModuleList(GapSSM(D, bidirectional) for _ in range(BLOCKS))
        self.att = torch.nn.ModuleList(torch.nn.TransformerEncoderLayer(D, HEADS, 2 * D, 0.1, batch_first=True)
                                       for _ in range(BLOCKS))
        self.head = torch.nn.Linear(D, 1)

    def forward(self, x):
        seen = x[..., 1:2]                       # seen / was-real flag is input column 1 in both tasks
        h = self.embed(x)
        L = x.shape[1]
        mask = None if self.bi else torch.nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        for ssm, att in zip(self.ssm, self.att):
            h = h + ssm(h, seen)
            h = att(h, src_mask=mask, is_causal=mask is not None)
        return self.head(h).squeeze(-1)


def impute(series, ctx):
    return fit_impute(lambda n: Net(n, True), series, ctx)


def forecast(history, horizons, ctx):
    return fit_forecast(lambda n: Net(n, False), history, horizons, ctx)
