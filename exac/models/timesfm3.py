"""TimesFM 3 (EXAC_DESIGN 6.8): frozen, + a learned logistic layer. Copied from models/timesfm3.embed and
run_exac.py (tfm_fill, the timesfm3 branches of make_net / net_inputs).

Every minute of the window, each of the 7 signals one variate, values as recorded (TimesFM normalises each
series itself); a blank is masked with TimesFM's own masks, nothing is filled. Each variate's transformer
output at the last patch (causal attention: it has seen the whole window) -> one logistic layer (+ the
scaled clinical values for copd_clinical), trained by exac/train.py. The weights never change, so each
day's output is computed once and cached (config.TIMESFM_CACHE, shared with run_exac.py).
Weights: config.TIMESFM_CHECKPOINT, local only (never downloads; non-commercial licence).
"""
import numpy as np
import torch

from exac import config as C
from exac import data

KIND = "net"
_FROZEN, _CACHE = {}, {}


def embed(v, device, batch=C.TIMESFM_BATCH):
    """v: (n, M, C) raw minutes, NaN = no reading, M a multiple of the patch length (32) -> (n, C * DIM)."""
    from timesfm3.torch.model import TimesFM3Torch
    if device not in _FROZEN:
        _FROZEN[device] = TimesFM3Torch.from_pretrained(C.TIMESFM_CHECKPOINT, local_files_only=True).to(device).eval()
    m = _FROZEN[device]
    n, M, ch = v.shape
    p = m.input_patch_len
    out = []
    with torch.no_grad():
        for k in range(0, n, batch):
            x = torch.as_tensor(np.ascontiguousarray(v[k:k + batch].transpose(0, 2, 1)), device=device)
            x = x.reshape(len(x), ch, M // p, p)
            mask = torch.isnan(x)
            o = m({"values": x, "masks": mask, "patch_is_target": torch.ones(x.shape[:3], dtype=torch.bool, device=device)},
                  return_aux_outputs=True)["__call__:transformer_output"]
            assert o.shape[-1] == C.TIMESFM_DIM, o.shape
            out.append(o[:, :, -1].reshape(len(x), ch * C.TIMESFM_DIM).float().cpu().numpy())
    return np.concatenate(out)


def prepare(rows):
    """Fill the cache for the windows `rows` (n, W), keyed by the classified day's row (NaN = not computed)."""
    f = C.TIMESFM_CACHE
    if not f.exists():
        np.lib.format.open_memmap(f, "w+", np.float32, (len(data.X), data.X.shape[2] * C.TIMESFM_DIM))[:] = np.nan
    T = _CACHE["out"] = np.load(f, mmap_mode="r+")
    rows = np.unique(rows, axis=0)
    todo = rows[np.isnan(T[rows[:, -1], 0])]
    for k in range(0, len(todo), 256):
        T[todo[k:k + 256, -1]] = embed(data.window(todo[k:k + 256]), data.device)
        T.flush()
        print(f"    timesfm3: {min(k + 256, len(todo)):,} of {len(todo):,} new days", flush=True)


def make(n_out, variant=None):
    return torch.nn.Linear(data.X.shape[2] * C.TIMESFM_DIM + data.n_static(), n_out)     # the learned logistic layer


def inputs(T, rows, sc, variant=None):
    x = np.asarray(_CACHE["out"][rows[:, -1]])
    st = data.static(T, sc[1])
    return [torch.as_tensor(np.concatenate([x, st], 1) if st is not None else x, device=data.device)]
