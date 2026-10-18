"""always_no (EXAC_DESIGN 6.1): says "no exacerbation" for every day. The accuracy of never raising an alarm."""
import numpy as np

KIND = "reference"


def fit(tr, te, ytr, ctx):
    return np.zeros((len(te), 1)), {}
