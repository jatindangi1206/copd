"""RNN and LSTM (torch.nn.RNN / torch.nn.LSTM). Training loop and inputs: seq.py.

impute  : bidirectional network, reads both sides of a gap.
forecast: one-directional network (H9), rolled forward one slot at a time.
"""
import torch

from .data import hp
from .seq import fit_forecast, fit_impute

HIDDEN, LAYERS = 64, 2


class Net(torch.nn.Module):
    def __init__(self, n_in, cell, bidirectional, hidden=HIDDEN, layers=LAYERS, dropout=0.1):
        super().__init__()
        self.rnn = getattr(torch.nn, cell)(n_in, hidden, layers, batch_first=True,
                                           bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        self.head = torch.nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, x):
        return self.head(self.rnn(x)[0]).squeeze(-1)


def make(cell):
    def net(ctx, bi):
        return lambda n: Net(n, cell, bi, hp(ctx, "hidden", HIDDEN), hp(ctx, "layers", LAYERS), hp(ctx, "dropout", 0.1))

    def impute(series, ctx):
        return fit_impute(net(ctx, True), series, ctx)

    def forecast(history, horizons, ctx):
        return fit_forecast(net(ctx, False), history, horizons, ctx)
    return impute, forecast
