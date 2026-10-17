"""Exacerbation experiments on every minute as exported: detection (is today an exacerbation day?) and
onset prediction (does a new episode begin in the next 3 days?).

    python run_exac.py --list
    python run_exac.py all --task detect                # every single-task model, 5 ratios x 5 seeds
    python run_exac.py all --task onset --set copd_clinical
    python run_exac.py gast --task onset --days 14      # GAST with 14 days of history (GAST only)
    python run_exac.py gast2                            # GAST with two heads: scores both tasks
    python run_exac.py check                            # every model, tiny training on a small subset

Every decision: docs/EXAC_DESIGN.md and docs/GAST_DESIGN.md. Data: model_data_exac/ (14_exac_data.py).

Samples   detect: days with HRV; label = inside an episode. onset: days with HRV, not inside an episode,
          not among the patient's last 3 days of watch data; label = an episode begins on d+1 ... d+3.
Sets      combined (COPD + AIIMS), copd, copd_clinical (+ the COPD clinical table, as recorded).
Split     by patient: --test % of each stratum (group x has an exacerbation day) at random per --seed;
          the same patients for every model and both tasks. Ratios 80:20, 90:10, 70:30, 60:40, 50:50.
Input     every minute of the day and the 6 before it (--days, GAST only): 7 signals as exported
          (HRV, HR, temperature, SpO2, steps, inside a sleep record, inside a step record), blank = blank.
  trees     XGBoost, CatBoost: one row = every raw value of the window (70,560 at 7 days), blanks as
            blanks; exacerbation days weighted up; 400 trees, depth 4, learning rate 0.05.
  networks  RNN, LSTM: each minute's values (divided by the training maximum) + "reading present"
            flags -> a learned 1-D convolution over 10-minute blocks -> models/rnn.Net, forward only,
            read at the last step.
  GAST v1   models/gast_v1.py: one token per minute with heart rate at its real time; gap-aware state
            space; hourly causal attention; variants for the ablations; gast2 has two heads.
  Networks train on half-exacerbation batches (Adam, 0.001, batches of 32) and stop early on 20% of
  the training patients held back (AUROC every 100 steps, stop after 5 without gain, at most 3,000,
  keep the best; if the held-back days hold no exacerbation day, all 3,000 steps). Flag at 0.5.
Scores    per day: accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, F1, AUROC, AUPRC,
          Brier score; per event (episode, or onset): caught if any of its days is flagged, false
          alarms per 30 days, lead time (onset); 95% intervals by resampling test patients; a
          calibration table; scores by wear coverage, longest gap in the day and days of history.
          Overall and per group (copd, aiims_adult, aiims_paed).
Results   results/exac/<task>/<set>/<model>/test<NN>_seed<S>/ predictions.csv.gz + metrics.json; one
          line per run and group in results/exac/summary_<task>.csv.

Options: --cores N (else it shows what is free and asks; default leaves 2 cores free),
--device cpu|cuda, --quick.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

from models import cores

MODELS = {   # name -> (family, library)
    "always_no": ("reference: never flags", "-"),
    "xgboost": ("trees on raw minutes", "xgboost"),
    "catboost": ("trees on raw minutes", "catboost"),
    "rnn": ("network (10-minute learned compression)", "torch"),
    "lstm": ("network (10-minute learned compression)", "torch"),
    "gast": ("GAST v1", "torch"),
    "gast_ssm": ("GAST v1 ablation: state space only", "torch"),
    "gast_attn": ("GAST v1 ablation: attention only", "torch"),
    "gast_nogap": ("GAST v1 ablation: no gap features", "torch"),
    "gast_maskonly": ("GAST v1 control: masks and times, no values", "torch"),
    "gast2": ("GAST v1, two heads: both tasks", "torch"),
}
GAST = {"gast": "full", "gast2": "full", "gast_ssm": "ssm", "gast_attn": "attn", "gast_nogap": "nogap",
        "gast_maskonly": "maskonly"}
SETS = {"combined": (["copd", "aiims"], False), "copd": (["copd"], False), "copd_clinical": (["copd"], True)}
GROUPS = ["copd", "aiims_adult", "aiims_paed"]
EVENT = {"detect": ("label", "episode"), "onset": ("onset_label", "onset_id")}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("model", nargs="?", help="a model, 'all' (every single-task model) or 'check'")
p.add_argument("--task", default="detect", choices=list(EVENT))
p.add_argument("--set", default="combined", choices=list(SETS))
p.add_argument("--test", type=int, nargs="+", default=[20, 10, 30, 40, 50], help="test share, %% of patients")
p.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--days", type=int, default=7, help="history in days, the classified day included (GAST only)")
p.add_argument("--cores", type=int)
p.add_argument("--device", choices=["cpu", "cuda"])
p.add_argument("--quick", action="store_true", help="tiny training on a small subset: checks the code runs")
p.add_argument("--list", action="store_true")
a = p.parse_args()
if a.list or not a.model:
    for name, (fam, lib) in MODELS.items():
        print(f"{name:14s} {fam:45s} {lib}")
    sys.exit(0)
names = list(MODELS) if a.model == "check" else [n for n in MODELS if n != "gast2"] if a.model == "all" else [a.model]
if any(n not in MODELS for n in names):
    sys.exit(f"unknown model {a.model}; see --list")
if a.days != 7 and any(n not in GAST for n in names):
    sys.exit("--days other than 7 is for the GAST models only (every other model uses 7 days)")

n_cores = cores.choose(a.cores)          # before numpy / torch are imported

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

try:                                     # xgboost before torch: on macOS the reverse order crashes
    import xgboost                       # noqa: F401
except ImportError:
    pass
try:
    import torch
    cores.apply_torch(n_cores)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch, device = None, a.device or "cpu"
print(f"device: {device}")

HERE = Path(__file__).resolve().parent
MD, RES = HERE / "model_data_exac", HERE / "results" / "exac"
W, HR = a.days, 1
M = W * 1440
BLOCK, CONV = 10, 32                     # networks: learned compression of 10-minute blocks into 32 numbers
BATCH, LR = 32, 1e-3
MAX_STEPS, CHECK_EVERY, PATIENCE, HELD_BACK = 3000, 100, 5, 0.20
ALPHA = 1.0                              # gast2: L = L_onset + ALPHA * L_detect
THRESHOLD = 0.5                          # locked per model; classes are balanced in training
BOOT = 1000                              # resamples of test patients for the 95% intervals
if a.quick or a.model == "check":
    MAX_STEPS, CHECK_EVERY, BOOT = 10, 5, 30

cohorts, clinical = SETS[a.set]
D = pd.read_csv(MD / "days.csv", parse_dates=["date"])
X = np.load(MD / "minutes.npy", mmap_mode="r")        # (stored days, 1440, 7), blank = NaN
D["row"] = np.arange(len(D))
S = pd.read_csv(MD / "static.csv").set_index("pid") if clinical else None
D = D[D.cohort.isin(cohorts) & (D.pid.isin(S.index) if clinical else True)].reset_index(drop=True)
POS = {(p_, d_): r for p_, d_, r in zip(D.pid, D.date, D.row)}
with warnings.catch_warnings():                       # a day with no reading of a signal gives NaN here
    warnings.simplefilter("ignore", RuntimeWarning)
    DAYMAX = np.concatenate([np.nanmax(np.abs(X[D.row.to_numpy()[k:k + 2000]]), axis=1) for k in range(0, len(D), 2000)])
P = D[D["sample"]].groupby("pid").agg(group=("group", "first"), pos=("label", "max"))
for g, d in D[D["sample"]].groupby("group"):
    o = D[(D.group == g) & D.onset_valid]
    print(f"{a.set} / {g}: {d.pid.nunique()} patients | detect {len(d):,} days, {d.label.sum()} positive | "
          f"onset {len(o):,} days, {o.onset_label.sum()} positive")


# ------------------------------------------------------------------ inputs
def window_rows(T):
    """(n, W) rows of minutes.npy for each sample's window, oldest day first; -1 = no data that day."""
    return np.array([[POS.get((p_, d_ - pd.Timedelta(days=W - 1 - k)), -1) for k in range(W)]
                     for p_, d_ in zip(T.pid, T.date)], dtype=np.int64).reshape(len(T), W)


def window(rows):
    """(n, W) rows -> (n, M, 7) raw minutes of the window, NaN = no reading."""
    v = np.asarray(X[np.maximum(rows, 0).ravel()], dtype=np.float32).reshape(len(rows), W, 1440, -1)
    v[rows < 0] = np.nan
    return v.reshape(len(rows), M, -1)


def tree_matrix(T):
    rows, k_s = window_rows(T), (S.shape[1] if S is not None else 0)
    out = np.empty((len(T), M * X.shape[2] + k_s), np.float32)
    for k in range(0, len(T), 256):
        out[k:k + 256, :M * X.shape[2]] = window(rows[k:k + 256]).reshape(len(rows[k:k + 256]), -1)
    if k_s:
        out[:, M * X.shape[2]:] = S.reindex(T.pid).to_numpy(np.float32)
    return out


def scales(train):
    """Each signal's (and clinical value's) largest absolute value in the training patients."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = np.nanmax(DAYMAX[D.pid.isin(train).to_numpy()], axis=0)
        st = np.nanmax(np.abs(S[S.index.isin(train)].to_numpy(float)), axis=0) if S is not None else None
    fix = lambda x: None if x is None else np.where(np.isfinite(x) & (x > 0), x, 1).astype(np.float32)
    return fix(s), fix(st)


def net_inputs(name, T, rows, sc):
    v, (scale, sscale) = window(rows), sc
    st = np.nan_to_num(S.reindex(T.pid).to_numpy(np.float32) / sscale) if S is not None else None
    if name in GAST:
        from models import gast_v1
        args = list(gast_v1.tokens(v, scale, GAST[name])) + ([st] if st is not None else [])
        return [torch.as_tensor(x, device=device) for x in args]
    x = np.concatenate([np.nan_to_num(v / scale), (~np.isnan(v)).astype(np.float32)], -1)
    if st is not None:
        x = np.concatenate([x, np.repeat(st[:, None], M, 1)], -1)
    return [torch.as_tensor(x, device=device)]


# ------------------------------------------------------------------ models
def make_net(name, n_out):
    n_static = S.shape[1] if S is not None else 0
    if name in GAST:
        from models import gast_v1
        return gast_v1.GAST(gast_v1.n_features(GAST[name]), n_out, n_static, GAST[name], window_minutes=M)
    from models.rnn import Net

    class Compressed(torch.nn.Module):          # learned compression of 10-minute blocks, then the RNN / LSTM
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(2 * X.shape[2] + n_static, CONV, BLOCK, stride=BLOCK)
            self.rnn = Net(CONV, name.upper(), False)

        def forward(self, x):
            return self.rnn(self.conv(x.transpose(1, 2)).transpose(1, 2))[:, -1:]
    return Compressed()


def fit_tree(name, tr, te, ytr, ctx):
    Xtr, Xte, n, gpu = tree_matrix(tr), tree_matrix(te), (20 if ctx["quick"] else 400), device == "cuda"
    if name == "xgboost":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=n, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1), tree_method="hist",
                              n_jobs=n_cores, device="cuda" if gpu else "cpu", random_state=ctx["seed"])
    else:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=n, depth=4, learning_rate=0.05, auto_class_weights="Balanced",
                               thread_count=n_cores, task_type="GPU" if gpu else "CPU", random_seed=ctx["seed"],
                               verbose=0, allow_writing_files=False)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1:], {}


def fit_net(name, tr, held, te, outputs, ctx):
    """outputs: [(label column, validity column or None)] - one per network output."""
    torch.manual_seed(ctx["seed"])
    rng, sc = np.random.default_rng(ctx["seed"]), scales(set(tr.pid))
    net = make_net(name, len(outputs)).to(device)
    R = {k: window_rows(T) for k, T in (("tr", tr), ("held", held), ("te", te))}
    labels = lambda T: np.stack([np.where(T[v] if v else True, T[c], np.nan) for c, v in outputs], 1).astype(np.float32)
    ytr, yheld = labels(tr), labels(held)
    pos, neg = np.flatnonzero(np.nan_to_num(ytr) .max(1) == 1), np.flatnonzero(np.nan_to_num(ytr).max(1) == 0)
    if not len(pos):
        raise ValueError("no exacerbation day among the training patients")
    w = torch.tensor([1.0] if len(outputs) == 1 else [ALPHA, 1.0], device=device)   # gast2: detect, onset

    def predict(T, rows):
        net.eval()
        with torch.no_grad():
            out = [torch.sigmoid(net(*net_inputs(name, T.iloc[k:k + 64], rows[k:k + 64], sc))).cpu().numpy()
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

    opt, bce = torch.optim.Adam(net.parameters(), lr=LR), torch.nn.BCEWithLogitsLoss(reduction="none")
    best, state, bad, net_steps = -np.inf, None, 0, 0
    net.train()
    for step in range(1, MAX_STEPS + 1):
        idx = np.r_[rng.choice(pos, BATCH // 2), rng.choice(neg, BATCH // 2)]       # half exacerbation days
        logit = net(*net_inputs(name, tr.iloc[idx], R["tr"][idx], sc))
        y = torch.as_tensor(ytr[idx], device=device)
        m = ~torch.isnan(y)
        per = (bce(logit, torch.nan_to_num(y)) * m).sum(0) / m.sum(0).clamp(min=1)     # mean per output
        loss = (per * w).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        net_steps = step
        if step % CHECK_EVERY == 0:
            s = held_auroc()
            print(f"    step {step}: loss {loss.item():.4f}, held-back AUROC {s if s is None else round(s, 3)}", flush=True)
            if s is None:                        # no exacerbation day held back: train all steps
                continue
            if s > best:
                best, bad, state = s, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if state is not None:
        net.load_state_dict(state)
    return predict(te, R["te"]), dict(steps=net_steps, held_auroc=None if state is None else best)


# ------------------------------------------------------------------ scores
def avg_rank(s):
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=r) / cnt)[inv]


def auroc(y, s):
    """Chance that a random positive day scores above a random negative one (ties count half)."""
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    return float((avg_rank(np.asarray(s, float))[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def auprc(y, s):
    """Average precision as scikit-learn computes it: the sum over score thresholds of recall step x precision."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    if not y.sum():
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    last = np.r_[np.flatnonzero(np.diff(s[o])), len(s) - 1]               # last row of each distinct score
    tp = np.cumsum(y[o])[last]
    return float(np.sum(np.diff(np.r_[0, tp / y.sum()]) * tp / (last + 1)))


def core(y, s, ev):
    """The scores that get 95% intervals. ev: event code per day (-1 on negative days)."""
    f = s >= THRESHOLD
    tp, fp, tn, fn = (f & (y == 1)).sum(), (f & (y == 0)).sum(), (~f & (y == 0)).sum(), (~f & (y == 1)).sum()
    div = lambda u, v: u / v if v else float("nan")
    evs = np.unique(ev[y == 1])
    return dict(auroc=auroc(y, s), auprc=auprc(y, s), sensitivity=div(tp, tp + fn), specificity=div(tn, tn + fp),
                ppv=div(tp, tp + fp), brier=float(np.mean((s - y) ** 2)) if len(y) else float("nan"),
                event_sensitivity=div(len(np.unique(ev[(y == 1) & f])), len(evs)),
                false_alarms_per_30_days=div(30 * fp, fp + tn))


def evaluate(task, T, s, rng):
    lab, evc = EVENT[task]
    y, s = T[lab].to_numpy(int), np.asarray(s, float)
    codes = pd.factorize(T[evc].where(y == 1))[0]                          # -1 on negative days
    f = s >= THRESHOLD
    tp, fp, tn, fn = (int(x) for x in ((f & (y == 1)).sum(), (f & (y == 0)).sum(), (~f & (y == 0)).sum(), (~f & (y == 1)).sum()))
    div = lambda u, v: u / v if v else float("nan")
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    m = dict(n_patients=T.pid.nunique(), n_pos_patients=T.pid[y == 1].nunique(), n_days=len(y), n_pos_days=int(y.sum()),
             tp=tp, fp=fp, tn=tn, fn=fn, accuracy=div(tp + tn, len(y)), balanced_accuracy=(sens + spec) / 2,
             npv=div(tn, tn + fn), f1=div(2 * tp, 2 * tp + fp + fn), events=len(np.unique(codes[y == 1])),
             events_caught=len(np.unique(codes[(y == 1) & f])), lead_days_median=float("nan"))
    m.update(core(y, s, codes))
    if task == "onset" and m["events_caught"]:
        hit = T[(y == 1) & f]
        first = hit.groupby(evc).date.min()
        onset = pd.to_datetime(first.index.str.rsplit("_", n=1).str[1], format="%Y%m%d")
        m["lead_days_median"] = float(np.median((onset - pd.DatetimeIndex(first.to_numpy())).days))
    groups = [np.flatnonzero(T.pid.to_numpy() == p_) for p_ in T.pid.unique()]
    boot = []
    for _ in range(BOOT):                                                  # resample patients, keep their days together
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        ev = np.concatenate([np.where(codes[groups[i]] >= 0, codes[groups[i]] + j * (codes.max() + 1), -1)
                             for j, i in enumerate(pick)])                 # a patient drawn twice: separate events
        boot.append(core(y[idx], s[idx], ev))
    B = pd.DataFrame(boot)
    for k in B:
        m[f"{k}_lo"], m[f"{k}_hi"] = (float(x) for x in np.nanpercentile(B[k], [2.5, 97.5])) if B[k].notna().any() else (float("nan"),) * 2
    return m


def extras(task, T, s):
    """Calibration table and scores by wear coverage, longest gap in the day and days of history (thirds)."""
    y, s = T[EVENT[task][0]].to_numpy(int), np.asarray(s, float)
    band = np.minimum((s * 10).astype(int), 9)
    cal = [dict(band=f"{k / 10:.1f}-{(k + 1) / 10:.1f}", n=int((band == k).sum()),
                predicted=float(s[band == k].mean()) if (band == k).any() else None,
                observed=float(y[band == k].mean()) if (band == k).any() else None) for k in range(10)]
    rows = window_rows(T)
    hr_min = D.set_index("row").hr_minutes
    info = {"coverage": np.array([hr_min.reindex(r[r >= 0]).sum() / M for r in rows]),
            "history_days": (rows >= 0).sum(1).astype(float),
            "longest_gap_minutes": np.array([np.diff(np.r_[-1, np.flatnonzero(np.isfinite(X[r, :, HR])), 1440]).max() - 1
                                             for r in rows[:, -1]], float)}
    out = {}
    for k, x in info.items():
        cut = np.digitize(x, np.unique(np.quantile(x, [1 / 3, 2 / 3])))    # thirds of the test days' own values
        out[k] = [dict(part=int(j), low=float(x[cut == j].min()), high=float(x[cut == j].max()), n_days=int((cut == j).sum()),
                       n_pos_days=int(y[cut == j].sum()), auroc=auroc(y[cut == j], s[cut == j]),
                       sensitivity=float(((s >= THRESHOLD) & (y == 1))[cut == j].sum() / max(y[cut == j].sum(), 1)),
                       specificity=float(((s < THRESHOLD) & (y == 0))[cut == j].sum() / max((y[cut == j] == 0).sum(), 1)))
                  for j in np.unique(cut)]
    return dict(calibration=cal, slices=out)


# ------------------------------------------------------------------ split, run, save
def draw(pids, share, rng):
    """share of each stratum (group x has an exacerbation day), drawn at random."""
    out = []
    for _, g in P.loc[sorted(pids)].groupby(["group", "pos"]):
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


def run(name, test_pct, seed, quick):
    test = draw(P.index, test_pct / 100, np.random.default_rng(seed))
    train = set(P.index) - test
    held = draw(train, HELD_BACK, np.random.default_rng([seed, 1])) if name not in ("always_no", "xgboost", "catboost") else set()
    fit = train - held
    tasks = ["detect", "onset"] if name == "gast2" else [a.task]
    base = D[D["sample"]] if name == "gast2" or a.task == "detect" else D[D.onset_valid]
    tr, hd, te = (base[base.pid.isin(s_)] for s_ in (fit, held, test))
    if quick:
        rng = np.random.default_rng(seed)
        lab = "label" if name == "gast2" else EVENT[a.task][0]
        tr, hd, te = small(tr, 600, rng, lab), small(hd, 200, rng, lab), small(te, 300, rng, lab)
    assert not (set(tr.pid) | set(hd.pid)) & set(te.pid) and not set(tr.pid) & set(hd.pid), "patients must not overlap"
    ctx, t0 = dict(quick=quick, seed=seed), time.time()
    if name == "always_no":
        score, info = np.zeros((len(te), 1)), {}
    elif name in ("xgboost", "catboost"):
        score, info = fit_tree(name, tr, te, tr[EVENT[a.task][0]].to_numpy(int), ctx)
    elif name == "gast2":
        score, info = fit_net(name, tr, hd, te, [("label", None), ("onset_label", "onset_valid")], ctx)
    else:
        score, info = fit_net(name, tr, hd, te, [(EVENT[a.task][0], None)], ctx)
    out = []
    for j, task in enumerate(tasks):
        T, s = (te, score[:, j]) if task == "detect" or name != "gast2" else (te[te.onset_valid], score[te.onset_valid.to_numpy(), j])
        rng, by = np.random.default_rng([seed, 2]), {}
        if len(T):
            by["all"] = evaluate(task, T, s, rng)
            for g in GROUPS:
                k = (T.group == g).to_numpy()
                if k.any():
                    by[g] = evaluate(task, T[k], s[k], rng)
        out.append(dict(task=task, T=T, s=s, by=by, extra=extras(task, T, s) if len(T) else {},
                        info=dict(info, seconds=round(time.time() - t0, 1), train_patients=sorted(tr.pid.unique()),
                                  held_back_patients=sorted(hd.pid.unique()), test_patients=sorted(te.pid.unique()))))
    return out


def save(name, test_pct, seed, r):
    lab, evc = EVENT[r["task"]]
    tag = name + (f"_d{W}" if W != 7 else "")
    out = RES / r["task"] / a.set / tag / f"test{test_pct:02d}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    T = r["T"]
    pd.DataFrame({"pid": T.pid, "group": T.group, "date": T.date.dt.date, "label": T[lab], "event": T[evc],
                  "score": r["s"], "flag": (r["s"] >= THRESHOLD).astype(int)}).to_csv(out / "predictions.csv.gz", index=False,
                                                                                    float_format="%.4g")
    head = dict(task=r["task"], set=a.set, model=tag, days=W, test_pct=test_pct, seed=seed, quick=a.quick,
                cores=n_cores, device=device, finished=time.strftime("%Y-%m-%d %H:%M"))
    (out / "metrics.json").write_text(json.dumps(dict(head, groups=r["by"], **r["extra"], training=r["info"]), indent=1,
                                                 default=float))
    rows = [dict(head, group=g, **m) for g, m in r["by"].items()]
    summ = RES / f"summary_{r['task']}.csv"
    pd.DataFrame(rows).to_csv(summ, mode="a", header=not summ.exists(), index=False)
    return out, rows


if a.model == "check":
    if torch is not None:
        from models import gast_v1
        gast_v1._check()
    bad = []
    for name in names:
        try:
            for r in run(name, 20, 0, True):
                assert len(r["s"]) == len(r["T"]) and np.isfinite(r["s"]).all() and ((r["s"] >= 0) & (r["s"] <= 1)).all()
                print(f"ok   {name:14s} {r['task']:6s} AUROC {r['by']['all']['auroc']:.2f} (tiny training)")
        except ImportError as e:
            print(f"skip {name:14s} library missing: {e}")
        except Exception as e:  # report every failure, keep going
            bad.append(name)
            print(f"FAIL {name:14s} {type(e).__name__}: {e}")
    print(f"\n{len(bad)} failed" + (f": {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)

done = []
for test_pct in a.test:
    for seed in a.seed:
        for name in names:
            print(f"\n== {a.set} {name} ({'both tasks' if name == 'gast2' else a.task}) days {W}, "
                  f"train {100 - test_pct}: test {test_pct}, seed {seed}")
            try:
                results = run(name, test_pct, seed, a.quick)
            except ImportError as e:
                print(f"skip {name}: {e}")
                continue
            for r in results:
                out, rows = save(name, test_pct, seed, r)
                done += rows
                for row in rows:
                    print(f"  {row['task']:6s} {row['group']:12s} AUROC {row['auroc']:.3f} [{row['auroc_lo']:.2f}-"
                          f"{row['auroc_hi']:.2f}]  PPV {row['ppv']:.3f}  sens {row['sensitivity']:.3f}  spec "
                          f"{row['specificity']:.3f}  events {row['events_caught']}/{row['events']}  false alarms "
                          f"{row['false_alarms_per_30_days']:.1f}/30 d  Brier {row['brier']:.3f}")
                print(f"  -> {out}")

if done:
    R = pd.DataFrame(done)
    R = R[R.group == "all"]
    for task, Rt in R.groupby("task"):
        for k in ["accuracy", "ppv", "sensitivity", "specificity", "auroc", "auprc", "brier", "event_sensitivity",
                  "false_alarms_per_30_days"]:
            print(f"\n{task} {k}, all groups, by test share (%), mean over seeds {a.seed}")
            print(Rt.pivot_table(index="model", columns="test_pct", values=k, aggfunc="mean").round(3).to_string())
