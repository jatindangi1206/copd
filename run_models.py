"""Run the imputation and forecasting models, one at a time.

    python run_models.py --list
    python run_models.py impute xgboost                 # hide mask_random, rebuild, score
    python run_models.py impute lstm --mask block       # the harder whole-run mask
    python run_models.py forecast catboost              # train gaps filled by --imputer (default linear)
    python run_models.py forecast lstm --imputer xgboost
    python run_models.py daily cd_gamma_dglm            # day-level forecasting
    python run_models.py impute all                     # every model that does the task, in turn
    python run_models.py check                          # every installed model on 6 series, tiny training

Options: --cores N (else it shows what is free and asks; default leaves 2 cores
free), --device cpu|cuda (default: cuda if available), --seed, --quick.
Results: results/<task>/<run>/predictions.csv.gz and metrics.json, one line
per run in results/summary.csv. Data: model_data/ (built by 11_model_data.py).
"""
import argparse
import sys
import time
import traceback

from models import cores
from models import REGISTRY

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("task", nargs="?", choices=["impute", "forecast", "daily", "check"])
p.add_argument("model", nargs="?", help="model name or 'all'")
p.add_argument("--mask", default="random", choices=["random", "block"])
p.add_argument("--imputer", default="linear", help="imputation model that fills the training gaps first")
p.add_argument("--cores", type=int)
p.add_argument("--device", choices=["cpu", "cuda"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--quick", action="store_true", help="6 series, tiny training: checks the code runs")
p.add_argument("--list", action="store_true")
a = p.parse_args()

if a.list or not a.task:
    print(f"{'model':16s} {'family':28s} {'library':26s} tasks")
    for name, (fam, lib, tasks, _) in REGISTRY.items():
        print(f"{name:16s} {fam:28s} {lib:26s} {', '.join(tasks)}")
    sys.exit(0)

n_cores = cores.choose(a.cores)          # before numpy / torch are imported

import numpy as np                       # noqa: E402
from models import data, get             # noqa: E402

try:                                     # xgboost before torch: on macOS the reverse order crashes
    import xgboost                       # noqa: F401  (two OpenMP runtimes; harmless elsewhere)
except ImportError:
    pass
try:
    import torch
    cores.apply_torch(n_cores)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    device = a.device or "cpu"
print(f"device: {device}")


def run(task, name, quick):
    ctx = dict(device=device, cores=n_cores, quick=quick, seed=a.seed,
               period=7 if task == "daily" else 144)
    np.random.seed(a.seed)
    t0 = time.time()
    if task == "impute":
        series = data.load_impute(a.mask, quick)
        preds = get(name, "impute")(series, ctx)
        run_name, extra = f"{name}__{a.mask}", {"mask": a.mask}
    else:
        series = data.load_daily(quick) if task == "daily" else data.load_forecast(quick)
        filled = get(a.imputer, "impute")(series, ctx)
        for s, f in zip(series, filled):
            s.x["hrv"] = f
        preds = get(name, task)(series, [len(s.truth) for s in series], ctx)
        run_name, extra = f"{name}__imp-{a.imputer}", {"imputer": a.imputer}
    P, m = data.score(task, series, preds)
    extra.update(model=name, seconds=round(time.time() - t0, 1), cores=n_cores, device=device,
                 seed=a.seed, quick=quick)
    return P, m, run_name, extra


if a.task == "check":
    ok, bad = [], []
    for name, (_, _, tasks, _) in REGISTRY.items():
        if name == "timesfm3":
            print("skip timesfm3: needs its weights (run it on the workstation)")
            continue
        for task in tasks:
            try:
                P, m, _, _ = run(task, name, True)
                assert m["n_missing_pred"] == 0, "some scored positions have no prediction"
                assert (P.pred > 0).all(), "non-positive prediction"
                ok.append(f"{name}/{task}")
                print(f"ok   {name:16s} {task:9s} MAE {m['mae']:.1f}")
            except Exception as e:  # report every failure, keep going
                bad.append(f"{name}/{task}")
                print(f"FAIL {name:16s} {task:9s} {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
    print(f"\n{len(ok)} ok, {len(bad)} failed" + (f": {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)

names = [n for n, v in REGISTRY.items() if a.task in v[2]] if a.model == "all" else [a.model]
for name in names:
    if name not in REGISTRY or a.task not in REGISTRY[name][2]:
        sys.exit(f"{name} does not do {a.task}; see --list")
    print(f"\n== {a.task} {name}")
    P, m, run_name, extra = run(a.task, name, a.quick)
    out = data.save(a.task, run_name, P, m, extra)
    print(f"MAE {m['mae']:.2f}  RMSE {m['rmse']:.2f}  n {m['n']:,}  ({extra['seconds']} s) -> {out}")
