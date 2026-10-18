"""Shared training loop of every network (copied from run_exac.py fit_net; EXAC_DESIGN 6.4).

Half-exacerbation batches, Adam, binary cross-entropy masked per output, gradients clipped; every
C.CHECK_EVERY steps the AUROC on the held-back patients' days; stop after C.PATIENCE checks without
gain or at C.MAX_STEPS; keep the best weights. No exacerbation day held back: train every step.

A network model module provides make(n_out, variant), inputs(T, rows, sc, variant) and, optionally,
prepare(rows) (run once on every window before training: TimesFM's cache).
"""
import numpy as np
import torch

from exac import config as C
from exac import data
from exac.scoring import auroc


def fit_net(mod, variant, tr, held, te, outputs, ctx):
    """outputs: [(label column, validity column or None)] - one per network output."""
    device = data.device
    torch.manual_seed(ctx["seed"])
    rng, sc = np.random.default_rng(ctx["seed"]), data.scales(set(tr.pid))
    net = mod.make(len(outputs), variant).to(device)
    R = {k: data.window_rows(T) for k, T in (("tr", tr), ("held", held), ("te", te))}
    if hasattr(mod, "prepare"):
        mod.prepare(np.concatenate(list(R.values())))
    labels = lambda T: np.stack([np.where(T[v] if v else True, T[c], np.nan) for c, v in outputs], 1).astype(np.float32)
    ytr, yheld = labels(tr), labels(held)
    pos, neg = np.flatnonzero(np.nan_to_num(ytr) .max(1) == 1), np.flatnonzero(np.nan_to_num(ytr).max(1) == 0)
    if not len(pos):
        raise ValueError("no exacerbation day among the training patients")
    w = torch.tensor([1.0] if len(outputs) == 1 else [C.ALPHA, 1.0], device=device)   # gast2: detect, onset

    def predict(T, rows):
        net.eval()
        with torch.no_grad():
            out = [torch.sigmoid(net(*mod.inputs(T.iloc[k:k + 64], rows[k:k + 64], sc, variant))).cpu().numpy()
                   for k in range(0, len(T), 64)]
        net.train()
        return np.concatenate(out) if out else np.zeros((0, len(outputs)))

    def held_auroc():
        if not len(held):
            return None
        ph, s = predict(held, R["held"]), []
        for j in range(len(outputs)):
            ok = ~np.isnan(yheld[:, j])
            if 0 < yheld[ok, j].sum() < ok.sum():
                s.append(auroc(yheld[ok, j], ph[ok, j]))
        return float(np.mean(s)) if s else None

    opt, bce = torch.optim.Adam(net.parameters(), lr=C.LR), torch.nn.BCEWithLogitsLoss(reduction="none")
    best, state, bad, net_steps = -np.inf, None, 0, 0
    net.train()
    for step in range(1, C.MAX_STEPS + 1):
        idx = np.r_[rng.choice(pos, C.BATCH // 2), rng.choice(neg, C.BATCH // 2)]   # half exacerbation days
        logit = net(*mod.inputs(tr.iloc[idx], R["tr"][idx], sc, variant))
        y = torch.as_tensor(ytr[idx], device=device)
        m = ~torch.isnan(y)
        per = (bce(logit, torch.nan_to_num(y)) * m).sum(0) / m.sum(0).clamp(min=1)     # mean per output
        loss = (per * w).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), C.CLIP)
        opt.step()
        net_steps = step
        if step % C.CHECK_EVERY == 0:
            s = held_auroc()
            print(f"    step {step}: loss {loss.item():.4f}, held-back AUROC {s if s is None else round(s, 3)}", flush=True)
            if s is None:                        # no exacerbation day held back: train all steps
                continue
            if s > best:
                best, bad, state = s, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= C.PATIENCE:
                    break
    if state is not None:
        net.load_state_dict(state)
    return predict(te, R["te"]), dict(steps=net_steps, held_auroc=None if state is None else best)
