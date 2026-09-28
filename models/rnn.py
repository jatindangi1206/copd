"""RNN and LSTM (torch.nn.RNN / torch.nn.LSTM). Training loop and inputs: seq.py.

impute  : bidirectional network, reads both sides of a gap.
forecast: one-directional network (H9), rolled forward one slot at a time.
"""
import torch

from .seq import fit_forecast, fit_impute

HIDDEN, LAYERS = 64, 2


class Net(torch.nn.Module):
    def __init__(self, n_in, cell, bidirectional):
        super().__init__()
        self.rnn = getattr(torch.nn, cell)(n_in, HIDDEN, LAYERS, batch_first=True,
                                           bidirectional=bidirectional, dropout=0.1)
        self.head = torch.nn.Linear(HIDDEN * (2 if bidirectional else 1), 1)

    def forward(self, x):
        return self.head(self.rnn(x)[0]).squeeze(-1)


def make(cell):
    def impute(series, ctx):
        return fit_impute(lambda n: Net(n, cell, True), series, ctx)

    def forecast(history, horizons, ctx):
        return fit_forecast(lambda n: Net(n, cell, False), history, horizons, ctx)
    return impute, forecast
