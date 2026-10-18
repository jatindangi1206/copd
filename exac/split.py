"""Patient split (copied from run_exac.py): no patient in both train and test."""
import numpy as np

from exac import data


def draw(pids, share, rng):
    """share of each stratum (group x has an exacerbation day), drawn at random."""
    out = []
    for _, g in data.P.loc[sorted(pids)].groupby(["group", "pos"]):
        ids = rng.permutation(g.index.to_numpy())
        out += list(ids[:int(len(ids) * share + 0.5)])                     # rounded half up
    return set(out)


def small(T, n, rng, lab):
    """--quick: at most n days, every positive kept up to half of them."""
    if len(T) <= n:
        return T
    pos = T.index[T[lab] == 1]
    keep = rng.choice(pos, min(len(pos), n // 2), replace=False)
    rest = rng.choice(T.index.difference(pos), n - len(keep), replace=False)
    return T.loc[np.sort(np.r_[keep, rest])]
