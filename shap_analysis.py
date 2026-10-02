"""What drives the predictions: SHAP for the tree models and the classifiers, ablation for the rest.

    python shap_analysis.py trees            # XGBoost, CatBoost: imputation, 10-minute and daily forecasting
    python shap_analysis.py ablation         # drop each vital in turn: RNN, LSTM, gast, neural ODE, trees
    python shap_analysis.py classification   # the exacerbation classifiers, per enrolment test
    python shap_analysis.py all              # all three, then the figures
    add --tuned to use the settings from tune_models.py (else the defaults)

trees           exact TreeSHAP, on the readings the official test scores (the hidden block-mask readings;
                the held-back forecast steps; the held-back days). Values are in standardised log HRV, so
                compare bars with each other, not with HRV units.
ablation        SHAP is not defined for the neural and state-space models, so the question "do the other
                vitals help?" is answered by refitting without each vital and scoring the block test again.
                Only RNN, LSTM, gast, the neural ODE and the trees use vitals; the other models use HRV only.
classification  each of the best three methods (by AUC, from 12_exac_classify.py) is refitted on all patients,
                settings picked by 3-fold inner CV, and explained. This describes what the fitted method
                leans on; with 37 patients it is not evidence that a test matters.

Writes numbers/shap_importance.csv, numbers/ablation.csv, results/shap/*, figures 14a-14e.
"""
import argparse
import importlib
import json
import sys

from models import cores

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("part", choices=["trees", "ablation", "classification", "figures", "all"])
p.add_argument("--tuned", action="store_true")
p.add_argument("--cores", type=int)
p.add_argument("--device", choices=["cpu", "cuda"])
p.add_argument("--max-rows", type=int, default=4000, help="rows explained per model")
a = p.parse_args()
n_cores = cores.choose(a.cores)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import matplotlib.pyplot as plt                       # noqa: E402

from common import BLUE, ORANGE, GREY, INK2, HERE, NUMBERS, save   # noqa: E402
from models import data, get, gbm                     # noqa: E402

try:
    import xgboost                                    # noqa: F401  (before torch, see run_models.py)
except ImportError:
    pass
import torch                                          # noqa: E402
cores.apply_torch(n_cores)
device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
SHAP = HERE / "results" / "shap"
SHAP.mkdir(parents=True, exist_ok=True)
LIBS = ["xgboost", "catboost"]
TASKS = [("impute", "Filling hidden HRV (block test)"), ("forecast", "Forecasting every 10 minutes"), ("daily", "Forecasting daily")]


def params_for(task, model):
    f = HERE / "results" / "tuning" / task / f"{model}.json"
    return json.loads(f.read_text())["params"] if a.tuned and f.exists() else {}


def ctx_for(task, model, **extra):
    return dict(device=device, cores=n_cores, quick=False, seed=0, period=7 if task == "daily" else 144,
                params={**params_for(task, model), **extra})


def group(name, task):
    if name.startswith("lag") and task == "impute":
        return "readings before"
    if name.startswith("lead"):
        return "readings after"
    if name.startswith("lag"):
        return "recent readings"
    return {"prev": "nearest reading before/after", "next": "nearest reading before/after",
            "dprev": "distance to nearest reading", "dnext": "distance to nearest reading",
            "sin": "time of day", "cos": "time of day", "sin_now": "time of day now", "cos_now": "time of day now",
            "sin_target": "time of day of the target", "cos_target": "time of day of the target",
            "mean_day": "last day's mean and spread", "sd_day": "last day's mean and spread",
            "h": "how far ahead", "hr": "heart rate", "temp": "temperature", "steps": "steps",
            "sleep_frac": "sleep", "steps_active_frac": "step activity"}[name]


def plain(name):
    """Input names as a reader would say them."""
    import re
    fixed = {"prev": "nearest reading before", "next": "nearest reading after", "dprev": "slots back to that reading",
             "dnext": "slots on to that reading", "sin": "time of day (sine)", "cos": "time of day (cosine)",
             "sin_now": "time of day now (sine)", "cos_now": "time of day now (cosine)",
             "sin_target": "time of day of the target (sine)", "cos_target": "time of day of the target (cosine)",
             "mean_day": "last day's mean", "sd_day": "last day's spread", "h": "how far ahead", "hr": "heart rate",
             "temp": "temperature", "steps": "steps", "sleep_frac": "asleep", "steps_active_frac": "step activity",
             "lag0": "the last reading"}
    if name in fixed:
        return fixed[name]
    k = re.fullmatch(r"(lag|lead)(\d+)", name)
    return f"reading {k[2]} {'before' if k[1] == 'lag' else 'after'}" if k else name


MODELNAME = {"xgboost": "XGBoost", "catboost": "CatBoost", "rnn": "RNN", "lstm": "LSTM", "gast": "our transformer",
             "pinode": "neural ODE", "gaussian_nb": "naive Bayes", "random_forest": "random forest",
             "grad_boost": "gradient boosting", "lda": "linear discriminant analysis", "decision_tree": "decision tree",
             "logreg_l2": "logistic regression", "extra_trees": "extra trees", "knn": "nearest neighbours",
             "logreg_l1": "sparse logistic regression", "svm_linear": "straight SVM", "svm_rbf": "curved SVM"}


# ------------------------------------------------------------------ trees
def scored_rows(task, lib):
    """(model, X at the scored positions, feature names) for one task."""
    import shap  # noqa: F401
    ctx = ctx_for(task, lib)
    if task == "impute":
        series = data.load_impute("block")
        m, F, zs, st = gbm.fit_impute(lib, series, ctx)
        X = pd.concat([f.iloc[np.flatnonzero(s.score)] for s, f in zip(series, F)], ignore_index=True)
        return m, X
    series = data.load_daily() if task == "daily" else data.load_forecast()
    for s, f in zip(series, get("linear", "impute")(series, ctx)):
        s.x["hrv"] = f
    m, Xs, st, names = gbm.fit_forecast(lib, series, [len(s.truth) for s in series], ctx)
    X = np.concatenate([x[np.flatnonzero(s.score)] for s, x in zip(series, Xs)])
    return m, pd.DataFrame(X, columns=names)


def trees():
    import shap
    rows, store = [], {}
    for task, _ in TASKS:
        for lib in LIBS:
            print(f"== SHAP {task} {lib}", flush=True)
            m, X = scored_rows(task, lib)
            X = X.sample(min(a.max_rows, len(X)), random_state=0)
            sv = shap.TreeExplainer(m).shap_values(X)
            sv = np.asarray(sv)
            store[(task, lib)] = (sv, X)
            for j, c in enumerate(X.columns):
                rows.append(dict(task=task, model=lib, feature=c, group=group(c, task),
                                 mean_abs_shap=float(np.abs(sv[:, j]).mean()), rows=len(X)))
    d = pd.DataFrame(rows)
    d.to_csv(NUMBERS / "shap_importance.csv", index=False)
    pd.to_pickle(store, SHAP / "trees.pkl")
    print("-> numbers/shap_importance.csv")


# ------------------------------------------------------------------ ablation
VITAL_GROUPS = {"heart rate": ["hr"], "temperature": ["temp"], "steps": ["steps", "steps_active_frac"],
                "sleep": ["sleep_frac"], "all vitals": ["hr", "temp", "steps", "steps_active_frac", "sleep_frac"]}
ABL_MODELS = ["xgboost", "catboost", "rnn", "lstm", "gast", "pinode"]


def ablation():
    rows = []
    series = data.load_impute("block")
    for model in ABL_MODELS:
        base = None
        for name, drop in [("none", [])] + list(VITAL_GROUPS.items()):
            drop = drop if isinstance(drop, list) else drop
            print(f"== ablation {model}: without {name}", flush=True)
            ctx = ctx_for("impute", model, drop=drop)
            preds = get(model, "impute")(series, ctx)
            _, m = data.score("impute", series, preds)
            base = m["mae"] if name == "none" else base
            rows.append(dict(model=model, dropped=name, mae=m["mae"], change=m["mae"] - base))
    pd.DataFrame(rows).to_csv(NUMBERS / "ablation.csv", index=False)
    print("-> numbers/ablation.csv")


# ------------------------------------------------------------------ classification
def classification():
    import shap
    from sklearn.model_selection import GridSearchCV
    ex = importlib.import_module("12_exac_classify")
    X, _ = ex.feature_table()
    y, fu = ex.dated_target(X)
    X = X.copy()
    X["followup_days"] = fu.values
    lab = y.dropna().index
    X, y = X.loc[lab], y.loc[lab].astype(int)
    M = pd.read_csv(HERE / "results" / "exac_monitoring" / "exac_models.csv")
    rows, store = [], {}
    for fs, XS in {"all": X, "reduced": ex.reduced(X)}.items():
        best = M[(M.feature_set == fs) & (M.model != "dummy")].sort_values("auc", ascending=False).head(3)
        for r in best.itertuples():
            print(f"== SHAP classification {fs} {r.model}", flush=True)
            model, grid, scale = ex.grids(0, False)[r.model]
            pipe = ex.pipe(model, scale)
            fit = (GridSearchCV(pipe, grid, scoring="roc_auc", cv=3).fit(XS, y).best_estimator_ if grid
                   else pipe.fit(XS, y))
            Xt = pd.DataFrame(fit[:-1].transform(XS), columns=XS.columns, index=XS.index)
            clf = fit[-1]
            tree = r.model in ("decision_tree", "random_forest", "extra_trees", "grad_boost", "xgboost")
            if tree:
                sv = shap.TreeExplainer(clf).shap_values(Xt.values)      # arrays: xgboost rejects names with [ ]
                sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
            else:
                f = lambda z: clf.predict_proba(np.asarray(z))[:, 1]
                sv = shap.KernelExplainer(f, shap.kmeans(Xt.values, 8)).shap_values(Xt.values, nsamples=300, silent=True)
            sv = np.asarray(sv)
            store[(fs, r.model)] = (sv, Xt)
            for j, c in enumerate(XS.columns):
                rows.append(dict(task="classification", model=r.model, feature_set=fs, feature=c, group=_cgroup(c),
                                 mean_abs_shap=float(np.abs(sv[:, j]).mean()), rows=len(Xt)))
    pd.DataFrame(rows).to_csv(NUMBERS / "shap_classification.csv", index=False)
    pd.to_pickle(store, SHAP / "classification.pkl")
    print("-> numbers/shap_classification.csv")


def _cgroup(c):
    if c.startswith("spiro"):
        return "spirometry"
    if c.startswith("impul"):
        return "oscillometry"
    if c.startswith("six_m"):
        return "six-minute walk"
    return {"age": "age", "sex_male": "sex", "bode": "BODE", "cat": "CAT", "followup_days": "days monitored"}[c]


# ------------------------------------------------------------------ figures
def figures():
    import shap
    d = pd.read_csv(NUMBERS / "shap_importance.csv")
    store = pd.read_pickle(SHAP / "trees.pkl")
    for k, (task, title) in enumerate(TASKS):
        g = d[d.task == task].groupby(["group", "model"]).mean_abs_shap.sum().unstack()
        g = g.loc[g.xgboost.sort_values().index]
        fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw=dict(width_ratios=[1, 1.15]))
        y = np.arange(len(g))
        ax[0].barh(y - .2, g.xgboost, height=.38, color=BLUE, label="XGBoost")
        ax[0].barh(y + .2, g.catboost, height=.38, color=ORANGE, label="CatBoost")
        ax[0].set_yticks(y, g.index)
        ax[0].set(xlabel="average effect on the prediction (standardised log HRV)")
        ax[0].set_title("Which kinds of input matter", fontsize=10)
        ax[0].legend(loc="lower right")
        ax[0].grid(axis="y", visible=False)
        sv, X = store[(task, "xgboost")]
        shap.plots.beeswarm(shap.Explanation(values=sv, data=X.values, feature_names=[plain(c) for c in X.columns]),
                            max_display=12, ax=ax[1], show=False, plot_size=None)
        ax[1].set_xlabel("push on the prediction (standardised log HRV)")
        ax[1].set_title("XGBoost, one dot per prediction: right = pushes HRV up, left = pushes it down", fontsize=10)
        fig.suptitle(f"What drives the prediction: {title}")
        fig.tight_layout()
        save(fig, f"14{'abc'[k]}-shap-{task}")
    ab = pd.read_csv(NUMBERS / "ablation.csv")
    ab = ab[ab.dropped != "none"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    groups = list(VITAL_GROUPS)
    w = .8 / len(ABL_MODELS)
    cols = [BLUE, ORANGE, GREY, "#9cc3cf", "#6b4c9a", "#8a6d3b"]
    for i, mname in enumerate(ABL_MODELS):
        v = ab[ab.model == mname].set_index("dropped").change.reindex(groups)
        ax.bar(np.arange(len(groups)) + i * w - .4 + w / 2, v, width=w, color=cols[i], label=MODELNAME[mname])
    ax.axhline(0, color=INK2, lw=.8)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set(ylabel="change in average error when the vital is removed\n(above 0 = the vital was helping)")
    ax.legend(ncol=3, fontsize=9)
    ax.set_title("Does each vital help fill hidden HRV? Refit without it and score the block test again", fontsize=11)
    fig.tight_layout()
    save(fig, "14d-ablation-vitals")
    c = pd.read_csv(NUMBERS / "shap_classification.csv")
    cs = pd.read_pickle(SHAP / "classification.pkl")
    cr = c[c.feature_set == "reduced"]
    top = cr.groupby("model").mean_abs_shap.sum().sort_values(ascending=False).index[:3]
    g = cr.groupby(["group", "model"]).mean_abs_shap.sum().unstack().fillna(0)[list(top)]
    g = 100 * g / g.sum()                          # shares: the methods' outputs are on different scales
    g = g.loc[g.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw=dict(width_ratios=[1, 1.2]))
    y = np.arange(len(g))
    for i, mname in enumerate(top):
        ax[0].barh(y + (i - 1) * .26, g[mname], height=.26, color=[BLUE, ORANGE, GREY][i], label=MODELNAME[mname])
    ax[0].set_yticks(y, g.index)
    ax[0].set(xlabel="share of the method's total push on the answer (%)")
    ax[0].set_title("Which kinds of test the fitted methods lean on (reduced set)", fontsize=10)
    ax[0].legend(loc="lower right")
    ax[0].grid(axis="y", visible=False)
    sv, Xt = cs[("reduced", top[0])]
    shap.plots.beeswarm(shap.Explanation(values=sv, data=Xt.values, feature_names=[n.split(":")[-1][:28] for n in Xt.columns]),
                        max_display=10, ax=ax[1], show=False, plot_size=None)
    ax[1].set_xlabel("push on the answer (right = towards yes)")
    ax[1].set_title(f"{MODELNAME[top[0]]}, one dot per patient and test (names as in the datasheet)", fontsize=10)
    fig.suptitle("What the exacerbation classifiers lean on (fitted on all patients; descriptive, not evidence of a link)")
    fig.tight_layout()
    save(fig, "14e-shap-classification")


if a.part in ("trees", "all"):
    trees()
if a.part in ("ablation", "all"):
    ablation()
if a.part in ("classification", "all"):
    classification()
if a.part in ("figures", "all"):
    figures()
