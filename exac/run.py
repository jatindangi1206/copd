"""Exacerbation experiments: detection (is today an exacerbation day?) and onset prediction (does a new
episode begin in the next 3 days?). The same commands as run_exac.py, from the repo root:

    python -m exac.run --list
    python -m exac.run check                          # every model, tiny training on a small subset
    python -m exac.run all --task detect              # every single-task model, 5 ratios x 5 seeds
    python -m exac.run xgboost catboost --task onset --set copd
    python -m exac.run gast --days 14                 # GAST with 14 days of history (GAST only)
    python -m exac.run gast2                          # GAST with two heads: scores both tasks

Settings: exac/config.py. Models: exac/models/. Design: docs/EXAC_DESIGN.md, docs/GAST_DESIGN.md.
Results: results/exac/<task>/<set>/<model>/test<NN>_seed<S>/ predictions.csv.gz + metrics.json; one line
per run and group in results/exac/summary_<task>.csv.
Options: --cores N (else it shows what is free and asks; default leaves 2 free), --device cpu|cuda,
--gpu-gb G (cap this process's GPU memory, for parallel runs), --quick.
"""
import argparse
import json
import sys
import time

from exac import config as C
from exac import cores
from exac.models import ANY_DAYS, MODELS, TWO_TASKS

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("model", nargs="*", help="one or more models, 'all' (every single-task model) or 'check'")
p.add_argument("--task", default="detect", choices=list(C.EVENT))
p.add_argument("--set", default="combined", choices=list(C.SETS))
p.add_argument("--test", type=int, nargs="+", default=C.TEST_PCT, help="test share, %% of patients")
p.add_argument("--seed", type=int, nargs="+", default=C.SEEDS)
p.add_argument("--days", type=int, default=C.DAYS, help="history in days, the classified day included (GAST only)")
p.add_argument("--cores", type=int)
p.add_argument("--device", choices=["cpu", "cuda"])
p.add_argument("--gpu-gb", type=float, help="cap this process's GPU memory, GB (torch and CatBoost)")
p.add_argument("--quick", action="store_true", help="tiny training on a small subset: checks the code runs")
p.add_argument("--list", action="store_true")
a = p.parse_args()
if a.list or not a.model:
    for name, (fam, f, _) in MODELS.items():
        print(f"{name:14s} {fam:47s} exac/models/{f}.py")
    sys.exit(0)
check = a.model == ["check"]
names = (list(MODELS) if check else [n for n in MODELS if n not in TWO_TASKS] if a.model == ["all"] else a.model)
if any(n not in MODELS for n in names):
    sys.exit(f"unknown model in {a.model}; see --list")
if a.days != C.DAYS and any(n not in ANY_DAYS for n in names):
    sys.exit(f"--days other than {C.DAYS} is for the GAST models only (every other model uses {C.DAYS} days)")
if a.quick or check:
    C.MAX_STEPS, C.CHECK_EVERY, C.BOOT = C.QUICK["MAX_STEPS"], C.QUICK["CHECK_EVERY"], C.QUICK["BOOT"]

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
    if device == "cuda" and a.gpu_gb:
        torch.cuda.set_per_process_memory_fraction(a.gpu_gb * 1e9 / torch.cuda.get_device_properties(0).total_memory)
except ImportError:
    torch, device = None, a.device or "cpu"
print(f"device: {device}")

from exac import data, models, scoring, split   # noqa: E402
from exac.train import fit_net                  # noqa: E402

data.load(a.set, a.days, device)
D, W = data.D, data.W


def run(name, test_pct, seed, quick):
    mod, variant = models.load(name)
    test = split.draw(data.P.index, test_pct / 100, np.random.default_rng(seed))
    train = set(data.P.index) - test
    held = split.draw(train, C.HELD_BACK, np.random.default_rng([seed, 1])) if mod.KIND == "net" else set()
    fit = train - held
    tasks = ["detect", "onset"] if name in TWO_TASKS else [a.task]
    base = D[D["sample"]] if name in TWO_TASKS or a.task == "detect" else D[D.onset_valid]
    tr, hd, te = (base[base.pid.isin(s_)] for s_ in (fit, held, test))
    if quick:
        rng = np.random.default_rng(seed)
        lab = "label" if name in TWO_TASKS else C.EVENT[a.task][0]
        n_tr, n_hd, n_te = C.QUICK["days"]
        tr, hd, te = split.small(tr, n_tr, rng, lab), split.small(hd, n_hd, rng, lab), split.small(te, n_te, rng, lab)
    assert not (set(tr.pid) | set(hd.pid)) & set(te.pid) and not set(tr.pid) & set(hd.pid), "patients must not overlap"
    ctx, t0 = dict(quick=quick, seed=seed, cores=n_cores, gpu_gb=a.gpu_gb), time.time()
    if mod.KIND != "net":
        score, info = mod.fit(tr, te, tr[C.EVENT[a.task][0]].to_numpy(int), ctx)
    elif name in TWO_TASKS:
        score, info = fit_net(mod, variant, tr, hd, te, [("label", None), ("onset_label", "onset_valid")], ctx)
    else:
        score, info = fit_net(mod, variant, tr, hd, te, [(C.EVENT[a.task][0], None)], ctx)
    out = []
    for j, task in enumerate(tasks):
        T, s = (te, score[:, j]) if task == "detect" or name not in TWO_TASKS else (te[te.onset_valid], score[te.onset_valid.to_numpy(), j])
        rng, by = np.random.default_rng([seed, 2]), {}
        if len(T):
            by["all"] = scoring.evaluate(task, T, s, rng)
            for g in C.GROUPS:
                k = (T.group == g).to_numpy()
                if k.any():
                    by[g] = scoring.evaluate(task, T[k], s[k], rng)
        out.append(dict(task=task, T=T, s=s, by=by, extra=scoring.extras(task, T, s) if len(T) else {},
                        info=dict(info, seconds=round(time.time() - t0, 1), train_patients=sorted(tr.pid.unique()),
                                  held_back_patients=sorted(hd.pid.unique()), test_patients=sorted(te.pid.unique()))))
    return out


def save(name, test_pct, seed, r):
    lab, evc = C.EVENT[r["task"]]
    tag = name + (f"_d{W}" if W != C.DAYS else "")
    out = C.RES / r["task"] / a.set / tag / f"test{test_pct:02d}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    T = r["T"]
    pd.DataFrame({"pid": T.pid, "group": T.group, "date": T.date.dt.date, "label": T[lab], "event": T[evc],
                  "score": r["s"], "flag": (r["s"] >= C.THRESHOLD).astype(int)}).to_csv(out / "predictions.csv.gz", index=False,
                                                                                      float_format="%.4g")
    head = dict(task=r["task"], set=a.set, model=tag, days=W, test_pct=test_pct, seed=seed, quick=a.quick,
                cores=n_cores, device=device, finished=time.strftime("%Y-%m-%d %H:%M"))
    (out / "metrics.json").write_text(json.dumps(dict(head, groups=r["by"], **r["extra"], training=r["info"]), indent=1,
                                                 default=float))
    rows = [dict(head, group=g, **m) for g, m in r["by"].items()]
    summ = C.RES / f"summary_{r['task']}.csv"
    pd.DataFrame(rows).to_csv(summ, mode="a", header=not summ.exists(), index=False)
    return out, rows


if check:
    if torch is not None:
        from exac.models import gast_v1
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
            print(f"\n== {a.set} {name} ({'both tasks' if name in TWO_TASKS else a.task}) days {W}, "
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
