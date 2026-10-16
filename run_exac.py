"""Exacerbation classification: is this patient-day an exacerbation day?

    python run_exac.py --list
    python run_exac.py xgboost                             # combined set; test 20, 10, 30, 40, 50 %; seed 0
    python run_exac.py all --set copd_clinical --seed 0 1 2 3 4
    python run_exac.py check                               # every model, tiny training, one split: the code runs

Data: model_data_exac/ (14_exac_data.py). A sample is one labelled patient-day with HRV.
Sets:  combined       COPD + AIIMS (a-series adults + p-series children), watch data only
       copd           COPD, watch data only
       copd_clinical  COPD, watch data + the clinical table (tests, labs, history; as recorded)
Groups, also scored on their own: copd, aiims_adult, aiims_paed.

How the condition shapes the design. Asthma and COPD are chronic; an exacerbation is an episode of one
day or a few that comes back in the same patient, and the sheets mention night symptoms and disturbed
sleep. So:
  split    by patient: a patient's recurring episodes never sit on both sides. --test % of each stratum
           (group x has an exacerbation day) is drawn at random with --seed; every model gets the same
           split. 80:20 is --test 20.
  inputs   only data up to the end of the day being classified (H9), with 7 days of context, and also
           as change from the patient's own median of the 28 days before (H10): children, adults and
           COPD differ in absolute HR and HRV.
    tabular  (logistic, xgboost, catboost) per vital that day: mean, sd, min, max, readings; the 7-day
             mean; the change from the 28-day baseline; HRV and HR asleep and awake (sleep as
             exported); how often sleep broke that day.
    sequence (rnn, lstm, gast) the last 7 days of 10-minute slots: each vital standardised with the
             training patients' statistics (HRV and steps logged), HRV / HR / temperature also as change
             from the 28-day baseline; blank = 0 plus a seen-flag; time of day. The networks are the
             ones in models/ (rnn.Net, gast.Net), read at the last slot.
    copd_clinical adds the clinical values: as columns (tabular) or on every slot (sequence).
  scores   per day: accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, F1 at 0.5; AUROC,
           AUPRC. Per episode: caught if any of its days is flagged; false alarms = flagged days per
           30 days without an exacerbation.
Classes are balanced in training (class weights; half-positive batches), so the threshold is 0.5.
Results: results/exac/<set>/test<NN>_seed<S>/<model>/ predictions.csv.gz + metrics.json; one line per
run and group (all, copd, aiims_adult, aiims_paed) in results/exac/summary.csv.

Options: --cores N (else it shows what is free and asks; default leaves 2 cores free),
--device cpu|cuda, --quick.
Not here: hmm, nlssm, pf, rsdpf, gru_ode_bayes, cd_gamma_dglm, ossa, pinode and timesfm3 rebuild or
forecast HRV values (run_models.py); they give no exacerbation output to score.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from models import cores

MODELS = {   # name -> (family, library, inputs)
    "always_no": ("reference", "-", "tabular"),
    "logistic":  ("reference", "scikit-learn", "tabular"),
    "xgboost":   ("general ML", "xgboost", "tabular"),
    "catboost":  ("general ML", "catboost", "tabular"),
    "rnn":       ("neural sequence", "torch", "sequence"),
    "lstm":      ("neural sequence", "torch", "sequence"),
    "gast":      ("hybrid / research (first version)", "torch", "sequence"),
}
SETS = {"combined": (["copd", "aiims"], False), "copd": (["copd"], False), "copd_clinical": (["copd"], True)}
GROUPS = ["copd", "aiims_adult", "aiims_paed"]

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("model", nargs="?", help="a model, 'all', or 'check'")
p.add_argument("--set", default="combined", choices=list(SETS))
p.add_argument("--test", type=int, nargs="+", default=[20, 10, 30, 40, 50], help="test share, %% of patients")
p.add_argument("--seed", type=int, nargs="+", default=[0])
p.add_argument("--cores", type=int)
p.add_argument("--device", choices=["cpu", "cuda"])
p.add_argument("--quick", action="store_true", help="tiny training: checks the code runs")
p.add_argument("--list", action="store_true")
a = p.parse_args()
if a.list or not a.model:
    for name, (fam, lib, kind) in MODELS.items():
        print(f"{name:10s} {fam:34s} {lib:13s} {kind}")
    sys.exit(0)
if a.model not in (*MODELS, "all", "check"):
    sys.exit(f"unknown model {a.model}; see --list")

n_cores = cores.choose(a.cores)          # before numpy / torch are imported

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score   # noqa: E402

try:                                     # xgboost before torch: on macOS the reverse order crashes
    import xgboost                       # noqa: F401
except ImportError:
    pass
try:
    import torch
    cores.apply_torch(n_cores)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    device = a.device or "cpu"
print(f"device: {device}")

HERE = Path(__file__).resolve().parent
MD, RES = HERE / "model_data_exac", HERE / "results" / "exac"
CH = ["hrv", "hr", "temp", "steps", "sleep_frac", "steps_active_frac"]
DEV = ["hrv", "hr", "temp"]              # also given to the sequence models as change from the 28-day baseline
W_DAYS, SLOTS, BATCH = 7, 144, 32
THRESHOLD = 0.5                          # ponytail: fixed; classes are balanced in training

cohorts, clinical = SETS[a.set]
D = pd.read_csv(MD / "days.csv", parse_dates=["date"])
D = D[D.cohort.isin(cohorts)]
S = pd.read_csv(MD / "static.csv").set_index("pid") if clinical else None
if clinical:
    D = D[D.pid.isin(S.index)]
D = D.reset_index(drop=True)
G = pd.read_csv(MD / "grid.csv.gz", parse_dates=["time"])
G = G[G.pid.isin(set(D.pid))].reset_index(drop=True)
for g, d in D.groupby("group"):
    print(f"{a.set} / {g}: {d.pid.nunique()} patients, {len(d):,} days, {int(d.label.sum())} exacerbation days, "
          f"{d.episode.nunique()} episodes in {d[d.label == 1].pid.nunique()} patients")


# ------------------------------------------------------------------ inputs
def baseline(m):
    """Each day's baseline: the patient's median of the 28 days before it (past days only)."""
    return m.rolling("28D", closed="left", min_periods=3).median()


def tab_features():
    """One row per sample day; see the module docstring."""
    T = G.assign(date=G.time.dt.normalize(), asleep=G.sleep_frac >= 0.5)
    g = T.groupby(["pid", "date"])[CH].agg(["mean", "std", "min", "max", "count"])
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    st = T.groupby(["pid", "date", "asleep"])[["hrv", "hr"]].mean().unstack("asleep")
    st.columns = [f"{c}_{'asleep' if s else 'awake'}" for c, s in st.columns]
    woke = T.asleep.shift(fill_value=False) & ~T.asleep & T.pid.eq(T.pid.shift())    # asleep -> awake
    g = g.join(st).join(woke.groupby([T.pid, T.date]).sum().rename("sleep_breaks"))
    out = []
    for _, f in g.reset_index().groupby("pid"):
        f = f.set_index("date").sort_index()
        for c in CH:
            m = f[f"{c}_mean"]
            f[f"{c}_mean7"] = m.rolling("7D", min_periods=1).mean()
            f[f"{c}_dev28"] = m - baseline(m)
        out.append(f.reset_index())
    X = D[["pid", "date"]].merge(pd.concat(out), on=["pid", "date"], how="left").drop(columns=["pid", "date"])
    if S is not None:
        X = pd.concat([X, S.reindex(D.pid).reset_index(drop=True).add_prefix("c_")], axis=1)
    return X.to_numpy(float)


def seq_arrays(train):
    """Per patient: (first slot time, slots x inputs), standardised with the training patients only."""
    X = G.assign(hrv=np.log(G.hrv.where(G.hrv > 0)), steps=np.log1p(G.steps))
    tr = X[X.pid.isin(train)]
    mu, sd = tr[CH].mean(), tr[CH].std().replace(0, 1).fillna(1)
    if S is not None:
        st = S[S.index.isin(train)]
        smu, ssd = st.mean(), st.std().replace(0, 1).fillna(1)
    out = {}
    for pid, g in X.groupby("pid"):
        z = ((g[CH] - mu) / sd).to_numpy(np.float32)
        seen = (~np.isnan(z[:, :4])).astype(np.float32)          # hrv, hr, temp, steps can be blank
        day = g.time.dt.normalize()
        dev = np.column_stack([(g[c].to_numpy() - baseline(g[c].groupby(day).median()).reindex(day).to_numpy()) / sd[c]
                               for c in DEV])
        ang = 2 * np.pi * (g.time.dt.hour * 6 + g.time.dt.minute // 10).to_numpy() / SLOTS
        cols = [np.nan_to_num(z[:, :1]), seen[:, :1], np.nan_to_num(z[:, 1:]), seen[:, 1:], np.nan_to_num(dev),
                np.sin(ang)[:, None], np.cos(ang)[:, None]]       # HRV's seen-flag is column 1: gast reads it
        if S is not None:
            v = np.nan_to_num(((S.loc[pid] - smu) / ssd).to_numpy(np.float32))
            cols.append(np.repeat(v[None], len(g), axis=0))
        out[pid] = (g.time.iat[0], np.concatenate(cols, axis=1).astype(np.float32))
    return out


def windows(arrs, rows):
    """For each (pid, date): the W_DAYS days of slots up to that day's last slot, zero-padded on the left."""
    L, out = W_DAYS * SLOTS, []
    for pid, date in rows:
        t0, F = arrs[pid]
        end = min((date + pd.Timedelta(days=1) - t0) // pd.Timedelta(minutes=10) - 1, len(F) - 1)
        w = F[max(0, end - L + 1):end + 1]
        out.append(np.concatenate([np.zeros((L - len(w), F.shape[1]), np.float32), w]))
    return np.stack(out)


# ------------------------------------------------------------------ models
def fit_tab(name, Xtr, ytr, Xte, ctx):
    if name == "always_no":
        return np.zeros(len(Xte))
    n, gpu = (20 if ctx["quick"] else 400), ctx["device"] == "cuda"
    if name == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=2000))
    elif name == "xgboost":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=n, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1), tree_method="hist",
                              n_jobs=ctx["cores"], device="cuda" if gpu else "cpu", random_state=ctx["seed"])
    else:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=n, depth=4, learning_rate=0.05, auto_class_weights="Balanced",
                               thread_count=ctx["cores"], task_type="GPU" if gpu else "CPU",
                               random_seed=ctx["seed"], verbose=0, allow_writing_files=False)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def fit_seq(name, tr, te, arrs, ctx):
    from models.gast import Net as GastNet
    from models.rnn import Net as RnnNet
    dev, rng = ctx["device"], np.random.default_rng(ctx["seed"])
    torch.manual_seed(ctx["seed"])
    n_in = next(iter(arrs.values()))[1].shape[1]
    net = (GastNet(n_in, False) if name == "gast" else RnnNet(n_in, name.upper(), False)).to(dev)
    rows, y = list(zip(tr.pid, tr.date)), tr.label.to_numpy()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    opt, loss_f = torch.optim.Adam(net.parameters(), lr=1e-3), torch.nn.BCEWithLogitsLoss()
    net.train()
    for step in range(10 if ctx["quick"] else 1500):
        idx = np.r_[rng.choice(pos, BATCH // 2), rng.choice(neg, BATCH // 2)]        # half positive
        x = torch.as_tensor(windows(arrs, [rows[i] for i in idx]), device=dev)
        loss = loss_f(net(x)[:, -1], torch.as_tensor(y[idx], dtype=torch.float32, device=dev))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"    step {step}: loss {loss.item():.4f}", flush=True)
    net.eval()
    te_rows, out = list(zip(te.pid, te.date)), []
    with torch.no_grad():
        for k in range(0, len(te_rows), 128):
            x = torch.as_tensor(windows(arrs, te_rows[k:k + 128]), device=dev)
            out.append(torch.sigmoid(net(x)[:, -1]).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ split, score, save
def split(test_pct, seed):
    """Test patients: test_pct % of each stratum (group x any exacerbation day), drawn at random."""
    P = D.groupby("pid").agg(group=("group", "first"), pos=("label", "max"))
    rng, test = np.random.default_rng(seed), []
    for _, g in P.groupby(["group", "pos"]):
        ids = rng.permutation(g.index.to_numpy())
        test += list(ids[:int(len(ids) * test_pct / 100 + 0.5)])    # rounded half up
    return set(test)


def metrics(y, s, episode):
    y, s, episode = np.asarray(y, int), np.asarray(s, float), np.asarray(episode, object)
    pr = (s >= THRESHOLD).astype(int)
    tp, fp = int(((pr == 1) & (y == 1)).sum()), int(((pr == 1) & (y == 0)).sum())
    tn, fn = int(((pr == 0) & (y == 0)).sum()), int(((pr == 0) & (y == 1)).sum())
    div = lambda u, v: u / v if v else float("nan")
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    caught = pd.Series(pr[y == 1]).groupby(episode[y == 1]).max()      # caught: any of its days flagged
    return dict(threshold=THRESHOLD, tp=tp, fp=fp, tn=tn, fn=fn, accuracy=div(tp + tn, len(y)),
                balanced_accuracy=(sens + spec) / 2, sensitivity=sens, specificity=spec, ppv=div(tp, tp + fp),
                npv=div(tn, tn + fn), f1=div(2 * tp, 2 * tp + fp + fn),
                auroc=roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan"),
                auprc=average_precision_score(y, s) if y.sum() else float("nan"),
                episodes=len(caught), episodes_caught=int(caught.sum()),
                episode_sensitivity=div(int(caught.sum()), len(caught)),
                false_alarms_per_30_days=div(30 * fp, fp + tn))


ARRS = {}            # sequence inputs of the current split, shared by rnn / lstm / gast


def run(name, test_pct, seed, quick):
    test = split(test_pct, seed)
    tr, te = D[~D.pid.isin(test)], D[D.pid.isin(test)]
    assert len(tr) and len(te) and not set(tr.pid) & set(te.pid), "train and test must be disjoint patients"
    ctx = dict(device=device, cores=n_cores, quick=quick, seed=seed)
    t0 = time.time()
    if MODELS[name][2] == "tabular":
        score = fit_tab(name, XT[tr.index], tr.label.to_numpy(), XT[te.index], ctx)
    else:
        if (test_pct, seed) not in ARRS:
            ARRS.clear()
            ARRS[(test_pct, seed)] = seq_arrays(set(tr.pid))
        score = fit_seq(name, tr, te, ARRS[(test_pct, seed)], ctx)
    score = np.asarray(score, float)
    head = dict(set=a.set, model=name, test_pct=test_pct, seed=seed, quick=quick, n_train_patients=tr.pid.nunique(),
                n_train_days=len(tr), n_train_pos_days=int(tr.label.sum()))
    rows = []
    for grp in ["all"] + [g for g in GROUPS if (te.group == g).any()]:
        k = np.ones(len(te), bool) if grp == "all" else (te.group == grp).to_numpy()
        t = te[k]
        rows.append({**head, "group": grp, "n_test_patients": t.pid.nunique(),
                     "n_test_pos_patients": t[t.label == 1].pid.nunique(), "n_test_days": len(t),
                     "n_test_pos_days": int(t.label.sum()), **metrics(t.label, score[k], t.episode),
                     "seconds": round(time.time() - t0, 1), "cores": n_cores, "device": device})
    P = te[["pid", "group", "date", "label", "episode"]].assign(score=score, pred=(score >= THRESHOLD).astype(int))
    return P, rows, sorted(tr.pid.unique()), sorted(test)


def save(P, rows, train, test):
    m = rows[0]
    out = RES / a.set / f"test{m['test_pct']:02d}_seed{m['seed']}" / m["model"]
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "predictions.csv.gz", index=False, float_format="%.4g")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    rows = [{**r, "finished": stamp} for r in rows]
    (out / "metrics.json").write_text(json.dumps({"by_group": rows, "train_patients": train, "test_patients": test},
                                                 indent=1))
    summ = RES / "summary.csv"
    pd.DataFrame(rows).to_csv(summ, mode="a", header=not summ.exists(), index=False)
    return out


XT = tab_features()

if a.model == "check":
    bad = []
    for name in MODELS:
        try:
            P, rows, _, _ = run(name, 20, 0, True)
            assert len(P) == rows[0]["n_test_days"] and np.isfinite(P.score).all() and P.score.between(0, 1).all()
            print(f"ok   {name:10s} AUROC {rows[0]['auroc']:.2f} (tiny training)")
        except Exception as e:  # report every failure, keep going
            bad.append(name)
            print(f"FAIL {name:10s} {type(e).__name__}: {e}")
    print(f"\n{len(MODELS) - len(bad)} ok, {len(bad)} failed" + (f": {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)

done = []
for test_pct in a.test:
    for seed in a.seed:
        for name in (MODELS if a.model == "all" else [a.model]):
            print(f"\n== {a.set} {name} train {100 - test_pct}: test {test_pct}, seed {seed}")
            try:
                P, rows, train, test = run(name, test_pct, seed, a.quick)
            except ImportError as e:
                print(f"skip {name}: {e}")
                continue
            out = save(P, rows, train, test)
            done += rows
            for r in rows:
                print(f"  {r['group']:12s} accuracy {r['accuracy']:.3f}  PPV {r['ppv']:.3f}  sensitivity "
                      f"{r['sensitivity']:.3f}  specificity {r['specificity']:.3f}  AUROC {r['auroc']:.3f}  "
                      f"episodes caught {r['episodes_caught']}/{r['episodes']}  false alarms "
                      f"{r['false_alarms_per_30_days']:.1f} per 30 days  ({r['n_test_pos_days']} of "
                      f"{r['n_test_days']:,} days positive)")
            print(f"  -> {out}")

if done:
    R = pd.DataFrame(done)
    R = R[R.group == "all"]
    for k in ["accuracy", "ppv", "sensitivity", "specificity", "auroc", "auprc", "episode_sensitivity",
              "false_alarms_per_30_days"]:
        print(f"\n{k}, by test share (%), all groups, mean over seeds {a.seed}")
        print(R.pivot_table(index="model", columns="test_pct", values=k, aggfunc="mean").round(3).to_string())
