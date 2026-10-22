"""Window ablation (EXAC_DESIGN 9): the same models, splits, features and scores; only the window changes.

    python -m exac.window                              # every single-task model, the whole grid, then the tables
    python -m exac.window xgboost catboost --cores 16  # some models only
    python -m exac.window --task onset --days 3 2      # part of the grid
    python -m exac.window --table                      # only rebuild the two tables from the summaries

Detection: input of WINDOW_DAYS days (5, 4, 3, 2). Onset: every WINDOW_DAYS x WINDOW_HORIZON (a new episode
in the next 1-3 days, 1-2 days, next day only). Each setting is one `python -m exac.run` over every ratio and
seed. The 7-day, 3-days-ahead runs already in the summaries are the reference rows of the tables.

Tables: results/exac/window_detect.csv and results/exac/window_onset.csv, one row per set, model, days,
(horizon,) test share and group: the number of seeds, then for every score its mean and SD over seeds and
its 95% interval (<score>_lo, <score>_hi: the mean over seeds of each split's interval, as in the reports).
"""
import argparse
import subprocess
import sys

from exac import config as C
from exac import cores

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("model", nargs="*", default=["all"], help="models as in exac.run (default: all)")
p.add_argument("--task", nargs="+", default=list(C.EVENT), choices=list(C.EVENT))
p.add_argument("--set", default="combined", choices=list(C.SETS))
p.add_argument("--days", type=int, nargs="+", default=C.WINDOW_DAYS)
p.add_argument("--horizon", type=int, nargs="+", default=C.WINDOW_HORIZON, choices=range(1, C.HORIZON + 1))
p.add_argument("--test", type=int, nargs="+")
p.add_argument("--seed", type=int, nargs="+")
p.add_argument("--cores", type=int)
p.add_argument("--device", choices=["cpu", "cuda"])
p.add_argument("--gpu-gb", type=float)
p.add_argument("--quick", action="store_true")
p.add_argument("--table", action="store_true", help="only rebuild the tables")
a = p.parse_args()

HEAD = {"task", "set", "model", "days", "horizon", "test_pct", "seed", "quick", "cores", "device", "finished", "group"}


def table(task):
    import pandas as pd
    f = C.RES / f"summary_{task}.csv"
    if not f.exists():
        return print(f"no {f}")
    R = pd.read_csv(f)
    R = R[R.quick == a.quick].copy()                     # a --quick run gets a table of its quick rows only
    R["model"] = R.model.str.replace(r"_d\d+", "", regex=True)
    keys = ["set", "model", "days", "test_pct", "group"]
    if task == "onset":
        R["horizon"] = R.model.str.extract(r"_h(\d+)$")[0].fillna(C.HORIZON).astype(int)
        R["model"] = R.model.str.replace(r"_h\d+$", "", regex=True)
        keys.insert(3, "horizon")
    R = R.drop_duplicates(keys + ["seed"], keep="last")  # a rerun replaces the earlier one
    S = [c for c in R if c not in HEAD and not c.endswith(("_lo", "_hi"))]
    agg = {}
    for k in S:
        agg[f"{k}_mean"], agg[f"{k}_sd"] = (k, "mean"), (k, "std")
        for b in ("lo", "hi"):
            if f"{k}_{b}" in R:
                agg[f"{k}_{b}"] = (f"{k}_{b}", "mean")   # the 95% interval: mean over seeds of each split's bounds
    T = R.groupby(keys).agg(seeds=("seed", "nunique"), **agg).reset_index()
    out = C.RES / f"window_{task}.csv"
    T.to_csv(out, index=False, float_format="%.4g")
    t0 = T[(T.group == "all") & (T.test_pct == C.TEST_PCT[0]) & (T.set == a.set)]
    cols = ["horizon", "days"] if task == "onset" else ["days"]
    for k in ("auroc", "auprc"):
        print(f"\n{task} {k} [95% interval], all groups, {a.set}, test {C.TEST_PCT[0]}%, mean over seeds")
        t0 = t0.assign(cell=[f"{m:.3f} [{lo:.2f}-{hi:.2f}]" for m, lo, hi in zip(t0[f"{k}_mean"], t0[f"{k}_lo"], t0[f"{k}_hi"])])
        print(t0.pivot_table(index="model", columns=cols, values="cell", aggfunc="first").fillna("").to_string())
    print(f"-> {out}")


if not a.table:
    n = cores.choose(a.cores)                            # ask once, every run gets the same
    extra = ["--set", a.set, "--cores", str(n)]
    for k in ("test", "seed"):
        if getattr(a, k):
            extra += [f"--{k}", *map(str, getattr(a, k))]
    if a.device:
        extra += ["--device", a.device]
    if a.gpu_gb:
        extra += ["--gpu-gb", str(a.gpu_gb)]
    if a.quick:
        extra.append("--quick")
    runs = ([("detect", d, C.HORIZON) for d in a.days] if "detect" in a.task else []) + \
           ([("onset", d, h) for d in a.days for h in a.horizon] if "onset" in a.task else [])
    failed = []
    for k, (task, d, h) in enumerate(runs, 1):
        cmd = [sys.executable, "-m", "exac.run", *a.model, "--task", task, "--days", str(d), "--horizon", str(h), *extra]
        print(f"\n##### {k}/{len(runs)}: {task}, {d} days in" + (f", onset within {h} days" if task == "onset" else ""), flush=True)
        if subprocess.run(cmd, cwd=C.ROOT).returncode:
            failed.append(f"{task} days {d} horizon {h}")
    print("\nfailed: " + ("; ".join(failed) if failed else "none"))
for task in a.task:
    table(task)
