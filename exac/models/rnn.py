"""RNN (EXAC_DESIGN 6.4): each minute's 7 scaled readings + 7 "reading present" flags (+ the scaled clinical
values for copd_clinical) -> a learned 1-D convolution over 10-minute blocks (config.BLOCK, CONV) ->
torch.nn.RNN, forward only (config.RNN) -> read at the last step. Training: exac/train.py.
lstm.py reuses this file with the cell changed."""
import numpy as np
import torch

from exac import config as C
from exac import data

KIND = "net"
CELL = "RNN"


class Net(torch.nn.Module):          # copied from models/rnn.py
    def __init__(self, n_in, cell, bidirectional, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.rnn = getattr(torch.nn, cell)(n_in, hidden, layers, batch_first=True,
                                           bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        self.head = torch.nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, x):
        return self.head(self.rnn(x)[0]).squeeze(-1)


class Compressed(torch.nn.Module):   # learned compression of 10-minute blocks, then the RNN / LSTM
    def __init__(self, cell):
        super().__init__()
        self.conv = torch.nn.Conv1d(2 * data.X.shape[2] + data.n_static(), C.CONV, C.BLOCK, stride=C.BLOCK)
        self.rnn = Net(C.CONV, cell, False, **C.RNN)

    def forward(self, x):
        return self.rnn(self.conv(x.transpose(1, 2)).transpose(1, 2))[:, -1:]


def make(n_out, variant=None, cell=CELL):
    return Compressed(cell)


def inputs(T, rows, sc, variant=None):
    v, (scale, sscale) = data.window(rows), sc
    st = data.static(T, sscale)
    x = np.concatenate([np.nan_to_num(v / scale), (~np.isnan(v)).astype(np.float32)], -1)
    if st is not None:
        x = np.concatenate([x, np.repeat(st[:, None], data.M, 1)], -1)
    return [torch.as_tensor(x, device=data.device)]
