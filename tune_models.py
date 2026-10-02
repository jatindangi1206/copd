"""Hyperparameter tuning for the imputation and forecasting models (Optuna, TPE).

    python tune_models.py impute xgboost            # tune on a block-style validation set
    python tune_models.py forecast lstm --trials 20
    python tune_models.py daily all                 # every model that does the task, in turn
    python tune_models.py --list

The official scores never see the tuning. Validation is carved out of what a model is allowed to
see anyway:
  impute   the official hidden readings (mask_block) stay hidden; a further 15% of the VISIBLE
           readings is hidden in whole runs (lengths drawn from the real gaps), fixed seed. Models
           are scored on those readings, never on the official ones.
  forecast the last 20% of each series' TRAIN part is held back; the model learns from the first
  daily    80% of it and predicts that. The official test part is never touched.
Trial 0 is always the current defaults, so each study reports default vs best on the same
validation set. A run that gives non-finite, non-positive or >5% above-maximum predictions scores
1000, so tuning cannot pick settings that break the model.

Output: results/tuning/<task>/<model>.json (best params, validation MAE of defaults and best),
<model>_trials.csv, and study_<model>.db (resumable). Apply with `run_models.py <task> <model> --tuned`.
Imputation is tuned on the block validation once and used for both masks.
"""
import argparse
import json
import sys
import time

from models import cores
from models import REGISTRY

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("task", nargs="?", choices=["impute", "forecast", "daily"])
p.add_argument("model", nargs="?", help="model name or 'all'")
p.add_argument("--trials", type=int, help="override the per-model budget")
p.add_argument("--subset", type=int, help="tune on this many series (default: 60 for the slow models)")
p.add_argument("--timeout", type=int, help="stop after this many seconds")
p.add_argument("--cores", type=int)
p.add_argument("--device", choices=["cpu", "cuda"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--quick", action="store_true", help="20-step training: checks the settings are accepted, scores mean nothing")
p.add_argument("--list", action="store_true")
a = p.parse_args()

# budget: cheap models 40 trials, slow ones 8 on a subset, TimesFM a small grid of inference settings
SLOW = {"gru_ode_bayes", "rsdpf", "pf"}
TRIALS = {m: (8 if m in SLOW else 10 if m == "timesfm3" else 40) for m in REGISTRY}
SKIP = {"linear", "last_value", "patient_median"}                      # references have nothing to tune
VAL_PCT, VAL_SEED = 0.15, 12345

if a.list or not a.task:
    for m, (fam, _, tasks, _) in REGISTRY.items():
        if m not in SKIP:
            print(f"{m:16s} {fam:28s} trials {TRIALS[m]:3d}{'  (subset)' if m in SLOW else ''}  {', '.join(tasks)}")
    sys.exit(0)

n_cores = cores.choose(a.cores)                   # before numpy / torch are imported

import numpy as np                                # noqa: E402
import pandas as pd                               # noqa: E402
import optuna                                     # noqa: E402
from models import data, get                      # noqa: E402

try:                                              # xgboost before torch (see run_models.py)
    import xgboost                                # noqa: F401
except ImportError:
    pass
try:
    import torch
    cores.apply_torch(n_cores)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    device = a.device or "cpu"
print(f"device: {device}")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------ search spaces
# Each returns the params for one trial; the first value of every categorical is irrelevant, the
# DEFAULTS below are enqueued as trial 0 (so they must lie inside these ranges).
def trees(t, task, lib):
    d = dict(n_estimators=t.suggest_int("n_estimators", 100, 1500, step=50),
             depth=t.suggest_int("depth", 3, 10), lr=t.suggest_float("lr", 0.01, 0.3, log=True),
             K=t.suggest_int("K", 3, 12))
    if lib == "xgboost":
        d.update(subsample=t.suggest_float("subsample", 0.5, 1.0), colsample=t.suggest_float("colsample", 0.4, 1.0),
                 min_child_weight=t.suggest_int("min_child_weight", 1, 20, log=True),
                 reg_lambda=t.suggest_float("reg_lambda", 0.1, 20, log=True))
    else:
        d.update(l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 20, log=True))
    if task != "impute":
        d["pairs"] = t.suggest_int("pairs", 1000, 6000, step=500)
    return d


def seq(t, task, gast=False):
    d = dict(dropout=t.suggest_float("dropout", 0.0, 0.4), lr=t.suggest_float("lr", 3e-4, 3e-3, log=True),
             window=t.suggest_categorical("window", [96, 144, 288, 432]),
             steps=t.suggest_categorical("steps", [1500, 3000, 5000]),
             batch=t.suggest_categorical("batch", [16, 32, 64]))
    if gast:
        d.update(d=t.suggest_categorical("d", [32, 64, 128]), heads=t.suggest_categorical("heads", [2, 4]),
                 blocks=t.suggest_int("blocks", 1, 3))
    else:
        d.update(hidden=t.suggest_categorical("hidden", [32, 64, 128, 256]), layers=t.suggest_int("layers", 1, 3))
    if task == "impute":
        d["hide"] = t.suggest_float("hide", 0.1, 0.4)
    return d


SPACES = {
    "xgboost": lambda t, k: trees(t, k, "xgboost"),
    "catboost": lambda t, k: trees(t, k, "catboost"),
    "rnn": lambda t, k: seq(t, k),
    "lstm": lambda t, k: seq(t, k),
    "gast": lambda t, k: seq(t, k, gast=True),
    "hmm": lambda t, k: dict(states=t.suggest_int("states", 2, 10), min_covar=t.suggest_float("min_covar", 1e-4, 1e-1, log=True)),
    "nlssm": lambda t, k: dict(q_level=t.suggest_float("q_level", 1e-3, 0.5, log=True),
                               q_cycle=t.suggest_float("q_cycle", 1e-4, 0.05, log=True),
                               r_share=t.suggest_float("r_share", 0.1, 2.0),
                               high_mult=t.suggest_float("high_mult", 1.01, 1.3)),
    "rsdpf": lambda t, k: dict(K=t.suggest_int("K", 2, 4), alpha=t.suggest_float("alpha", 0.2, 0.9),
                               lr=t.suggest_float("lr", 3e-3, 3e-2, log=True),
                               steps=t.suggest_categorical("steps", [500, 1000, 1500]),
                               n_train=t.suggest_categorical("n_train", [32, 64, 128]),
                               window=t.suggest_categorical("window", [72, 144, 288]),
                               n_run=t.suggest_categorical("n_run", [256, 512, 1024])),
    "pf": lambda t, k: dict(n_part=t.suggest_categorical("n_part", [500, 1000, 2000]),
                            sigma_share=t.suggest_float("sigma_share", 0.4, 0.9),
                            **({"n_paths": t.suggest_categorical("n_paths", [50, 100, 200])} if k == "impute" else {})),
    "gru_ode_bayes": lambda t, k: dict(hidden=t.suggest_categorical("hidden", [16, 32, 64]),
                                       p_hidden=t.suggest_categorical("p_hidden", [16, 32, 64]),
                                       prep_hidden=t.suggest_categorical("prep_hidden", [4, 8, 16]),
                                       lr=t.suggest_float("lr", 3e-4, 3e-3, log=True),
                                       steps=t.suggest_categorical("steps", [500, 1000, 1500]),
                                       window=t.suggest_categorical("window", [144, 288]),
                                       batch=t.suggest_categorical("batch", [8, 16, 32])),
    "cd_gamma_dglm": lambda t, k: dict(delta=t.suggest_float("delta", 0.90, 0.995),
                                       nu_scale=t.suggest_float("nu_scale", 0.3, 3.0, log=True)),
    "ossa": lambda t, k: dict(rank=t.suggest_int("rank", 2, 8), embed_frac=t.suggest_float("embed_frac", 0.25, 1.0),
                              nu2_max=t.suggest_float("nu2_max", 0.8, 0.99),
                              **({"iters": t.suggest_categorical("iters", [10, 30, 60])} if k == "impute"
                                 else {"windows": t.suggest_int("windows", 2, 8)})),
    "pinode": lambda t, k: dict(hidden=t.suggest_categorical("hidden", [16, 32, 64]),
                                max_gap=t.suggest_categorical("max_gap", [6, 12, 18]),
                                lr=t.suggest_float("lr", 1e-3, 1e-2, log=True),
                                steps=t.suggest_categorical("steps", [1000, 2000, 3000]),
                                batch=t.suggest_categorical("batch", [256, 512, 1024])),
    "timesfm3": lambda t, k: dict(context=t.suggest_categorical("context", [128, 256, 512, 1024, 2048]),
                                  **({"gap_context": t.suggest_categorical("gap_context", [64, 128, 256, 512, 1024])}
                                     if k == "impute" else {"time_cov": t.suggest_categorical("time_cov", [False, True])})),
}
_TREE = dict(n_estimators=600, depth=6, lr=0.05, K=6)
_SEQ = dict(dropout=0.1, lr=1e-3, window=288, steps=3000, batch=32)
DEFAULTS = {
    "xgboost": dict(_TREE, subsample=0.8, colsample=0.8, min_child_weight=1, reg_lambda=1.0, pairs=3000),
    "catboost": dict(_TREE, l2_leaf_reg=3.0, pairs=3000),
    "rnn": dict(_SEQ, hidden=64, layers=2, hide=0.2),
    "lstm": dict(_SEQ, hidden=64, layers=2, hide=0.2),
    "gast": dict(_SEQ, d=64, heads=4, blocks=2, hide=0.2),
    "hmm": dict(states=3, min_covar=1e-3),
    "nlssm": dict(q_level=0.05, q_cycle=0.002, r_share=0.5, high_mult=1.02),
    "rsdpf": dict(K=2, alpha=0.5, lr=0.01, steps=1500, n_train=64, window=144, n_run=512),
    "pf": dict(n_part=1000, sigma_share=0.7, n_paths=100),
    "gru_ode_bayes": dict(hidden=32, p_hidden=32, prep_hidden=8, lr=1e-3, steps=1500, window=288, batch=16),
    "cd_gamma_dglm": dict(delta=0.98, nu_scale=1.0),
    "ossa": dict(rank=3, embed_frac=1.0, nu2_max=0.95, iters=30, windows=4),
    "pinode": dict(hidden=32, max_gap=18, lr=3e-3, steps=2000, batch=512),
    "timesfm3": dict(context=2048, gap_context=512, time_cov=False),
}


# ------------------------------------------------------------------ validation sets
def val_impute(series, pct=VAL_PCT):
    """Hide a further pct of the visible readings in whole runs (lengths from the real gaps)."""
    rng = np.random.default_rng(VAL_SEED)
    pool = np.array([n for s in series for n in _runs(np.isnan(s.x.hrv.to_numpy(float))) if n <= 18] or [1])
    out = []
    for s in series:
        h = s.x.hrv.to_numpy(float)
        vis = ~np.isnan(h)
        val = np.zeros(len(h), bool)
        target, tries = int(round(vis.sum() * pct)), 0
        while val.sum() < target and tries < 2000:
            L = int(rng.choice(pool))
            i = int(rng.integers(0, max(1, len(h) - L)))
            val[i:i + L] |= vis[i:i + L]
            tries += 1
        x = s.x.copy()
        x["hrv"] = np.where(val, np.nan, h)
        x["hrv_obs"] = x.hrv
        out.append(data.Series(s.id, s.pid, x, h, val))
    return out


def _runs(m):
    a_ = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    return list(a_[1::2] - a_[::2])


def val_forecast(history, frac=0.8, min_train=60, min_val=10):
    """Learn from the first frac of each TRAIN part, predict the rest of it."""
    out = []
    for s in history:
        n = len(s.x)
        cut = int(n * frac)
        obs = s.x.hrv_obs.to_numpy(float)
        if cut < min_train or n - cut < min_val or np.isnan(obs[cut:]).all():
            continue
        x = s.x.iloc[:cut].reset_index(drop=True).copy()
        x["hrv"] = data.lin_fill(x.hrv_obs)                    # fill gaps from the kept part only
        truth = obs[cut:]
        out.append(data.Series(s.id, s.pid, x, truth, ~np.isnan(truth)))
    return out


def build(task):
    ctx = dict(device=device, cores=n_cores, quick=a.quick, seed=a.seed, period=7 if task == "daily" else 144)
    np.random.seed(a.seed)
    if task == "impute":
        return ctx, val_impute(data.load_impute("block"))
    series = data.load_daily() if task == "daily" else data.load_forecast()
    for s, f in zip(series, get("linear", "impute")(series, ctx)):
        s.x["hrv"] = f
    return ctx, val_forecast(series, min_train=10 if task == "daily" else 60, min_val=3 if task == "daily" else 10)


def evaluate(task, name, series, ctx, params):
    c = {**ctx, "params": params}
    fn = get(name, task)
    preds = fn(series, c) if task == "impute" else fn(series, [len(s.truth) for s in series], c)
    _, m = data.score(task, series, preds)
    broken = (m["n_missing_pred"] or m["n_nonfinite"] or m["n_nonpositive"] or m["pct_above_max"] > 5
              or not np.isfinite(m["mae"]))
    if broken:
        print(f"      rejected: missing {m['n_missing_pred']}, non-finite {m['n_nonfinite']}, non-positive "
              f"{m['n_nonpositive']}, {m['pct_above_max']:.1f}% above 129, MAE {m['mae']:.3g}", flush=True)
    return (1e3 if broken else m["mae"]), m


def tune(task, name):
    out = data.RESULTS / "tuning" / task
    out.mkdir(parents=True, exist_ok=True)
    ctx, series = build(task)
    sub = a.subset or (60 if name in SLOW and task != "daily" else None)
    if sub and len(series) > sub:
        keep = np.random.default_rng(VAL_SEED).choice(len(series), sub, replace=False)
        series = [series[i] for i in sorted(keep)]
    n_trials = a.trials or TRIALS[name]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=a.seed),
                                storage=f"sqlite:///{out / ('study_' + name + '.db')}", study_name=name, load_if_exists=True)
    first = not study.trials
    t0 = time.time()

    def objective(trial):
        params = SPACES[name](trial, task)
        t1 = time.time()
        try:
            val, m = evaluate(task, name, series, ctx, params)
        except Exception as e:                                  # a setting that crashes is just a bad setting
            print(f"    trial {trial.number}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            val = 1e3
        print(f"    trial {trial.number:3d}  val MAE {val:8.3f}  ({time.time() - t1:.0f} s)", flush=True)
        return val

    if first:                                                   # trial 0 = the current defaults
        study.enqueue_trial(_in_space(name, task, DEFAULTS[name]))
    done = len([t for t in study.trials if t.state.is_finished()])
    study.optimize(objective, n_trials=max(0, n_trials - done), timeout=a.timeout)
    ok = [t for t in study.trials if t.state.is_finished()]
    best = study.best_trial
    default_val = ok[0].value if ok else float("nan")
    res = dict(model=name, task=task, mask="block" if task == "impute" else None, params=best.params,
               val_mae=best.value, default_val_mae=default_val, trials=len(ok), series=len(series),
               seconds=round(time.time() - t0), validation=("block runs, 15% of visible readings" if task == "impute"
                                                           else "last 20% of the train part"))
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    study.trials_dataframe().to_csv(out / f"{name}_trials.csv", index=False)
    print(f"-> {out / (name + '.json')}  val MAE default {default_val:.3f} -> best {best.value:.3f}")


def _in_space(name, task, defaults):
    """The defaults restricted to the params this task's space actually suggests."""
    class Probe:                                                # run the space once to learn its param names
        names = set()

        def suggest(self, n, lo, *rest, **kw):
            self.names.add(n)
            return lo[0] if isinstance(lo, list) else lo
        suggest_int = suggest_float = suggest_categorical = suggest

    SPACES[name](Probe(), task)
    return {k: v for k, v in defaults.items() if k in Probe.names}


todo = [m for m, v in REGISTRY.items() if a.task in v[2] and m not in SKIP] if a.model == "all" else [a.model]
for m in todo:
    if m not in SPACES or a.task not in REGISTRY[m][2]:
        sys.exit(f"{m} cannot be tuned for {a.task}; see --list")
    print(f"\n== tune {a.task} {m}  ({a.trials or TRIALS[m]} trials)")
    tune(a.task, m)
