"""LSTM (EXAC_DESIGN 6.5): the RNN's design (rnn.py) with only the cell changed to torch.nn.LSTM."""
from exac.models.rnn import KIND, inputs, make as _make  # noqa: F401


def make(n_out, variant=None):
    return _make(n_out, variant, cell="LSTM")
