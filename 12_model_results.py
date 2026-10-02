"""Model results from the GPU run, as a table and two figures for the report.

    python 12_model_results.py

Reads results/summary.csv (one row per model run, written by run_models.py). Writes numbers/model_results.csv (one row per model and test, with
plain names and whether the run is usable) and figures 12a (imputation), 12b (forecasting).
Exacerbation classification (EXAC: target = dated exacerbation during monitoring): numbers/exac_models.csv
with plain method names, numbers/exac_permutation.csv, numbers/exac_inputs.csv (inputs per group) and
figure 12c. The target counts are checked against the datasheet.
A run is not usable if it never finished, its error is not a finite, plausible number, or its
predictions are impossible: at or below 0, or (more than 5%) above the watch's maximum.
Those counts are recorded by run_models.py in summary.csv, so nothing is flagged by hand.
"""
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import BLUE, ORANGE, GREY, INK2, HERE, NUMBERS, save
from models import REGISTRY
from codings import load_exacerbations

RUN = HERE / "results"                        # written by run_models.py and 12_exac_classify.py
EXAC = RUN / "exac_monitoring"
NAMES = {
    "linear": "Straight line (reference)", "last_value": "Last reading (reference)",
    "patient_median": "Patient's median (reference)", "xgboost": "XGBoost", "catboost": "CatBoost",
    "rnn": "RNN", "lstm": "LSTM", "hmm": "Hidden Markov model", "nlssm": "Nonlinear state-space model",
    "rsdpf": "RS-DPF", "pf": "Particle filter and smoother", "gru_ode_bayes": "GRU-ODE-Bayes",
    "cd_gamma_dglm": "CD-Gamma-DGLM", "ossa": "OSSA", "pinode": "Physiology-informed neural ODE",
    "gast": "Gap-aware state-space transformer (our design)", "timesfm3": "TimesFM 3",
}
REFERENCE = {"impute": "linear", "forecast": "patient_median", "daily": "patient_median"}
TESTS = [("impute", "random", "Random test"), ("impute", "block", "Block test"),
         ("forecast", None, "Every 10 minutes"), ("daily", None, "Daily")]

CLF = {   # classification methods: plain name, what it does
    "logreg_l2": ("Logistic regression", "Adds up the inputs, each with a weight, and turns the sum into a chance of yes."),
    "logreg_l1": ("Logistic regression, sparse", "The same, but it can set some weights to zero, dropping those inputs."),
    "lda": ("Linear discriminant analysis", "Finds the weighted sum of the inputs that best separates the yes and no groups."),
    "gaussian_nb": ("Naive Bayes", "Looks at each input on its own, asks how typical the value is for yes and for no patients, and combines the answers."),
    "knn": ("Nearest neighbours", "Gives a patient the answer most common among the most similar patients."),
    "decision_tree": ("Decision tree", "A chain of questions of the form 'is this value above a cut-off?' that ends in yes or no."),
    "random_forest": ("Random forest", "Many decision trees, each built on a random part of the patients and inputs, voting together."),
    "extra_trees": ("Extra trees", "Like a random forest, but the cut-offs are chosen at random."),
    "grad_boost": ("Gradient boosting", "Small decision trees added one after another, each correcting the mistakes of those before."),
    "xgboost": ("XGBoost", "A widely used version of gradient boosting."),
    "svm_linear": ("Support vector machine, straight", "Draws the straight dividing line between yes and no that leaves the widest gap."),
    "svm_rbf": ("Support vector machine, curved", "The same, but the dividing line can curve."),
    "dummy": ("Always {} (check)", "Not a real method: always gives the most common answer, {}. It scores exactly 0.5 and checks the scoring."),
}
X = pd.read_csv(EXAC / "exac_models.csv")
common = "no" if X[X.model == "dummy"].sens.iat[0] == 0 else "yes"       # the check always says the majority answer
X["name"], X["what"] = (X.model.map(lambda m: CLF[m][i].format(common, common)) for i in (0, 1))
X["order"] = X.model.map(list(CLF).index)          # listing order for the methods table
X.to_csv(NUMBERS / "exac_models.csv", index=False)
P = pd.read_csv(EXAC / "exac_permutation.csv")
P.to_csv(NUMBERS / "exac_permutation.csv", index=False)
inputs = pd.read_csv(EXAC / "exac_inputs.csv")
inputs.to_csv(NUMBERS / "exac_inputs.csv", index=False)
assert inputs.n.sum() == X[X.feature_set == "all"].n_features.iat[0], "input list does not match the 'all' set"
E = load_exacerbations()
assert P.n_yes.iat[0] == E.pid.nunique(), "yes = patients with a dated exacerbation in the datasheet"
assert P.n_yes.iat[0] + P.n_no.iat[0] == P.n_patients.iat[0]
S = pd.read_csv(RUN / "summary.csv")
rows = []
for task, mask, label in TESTS:
    kind = "impute" if task == "impute" else task
    for model, (_, _, tasks, _) in REGISTRY.items():
        if kind not in tasks:
            continue
        r = S[(S.task == task) & (S.model == model) & ((S["mask"] == mask) if mask else True)]
        mae = rmse = np.nan
        note = ""
        if r.empty:
            note = "did not finish"
        else:
            r = r.iloc[-1]
            mae, rmse = r.mae, r.rmse
            if not np.isfinite(mae) or mae > 1000 or r.n_nonfinite > 0:
                note = "gave impossible values"
            elif r.n_nonpositive > 0:
                note = "gave non-positive HRV values"
            elif r.pct_above_max > 5:
                note = "gave values above the watch's maximum"
        rows.append(dict(task=task, test=label, model=model, name=NAMES[model],
                         reference=model == REFERENCE[task], usable=not note, note=note,
                         mae=mae if not note else np.nan, rmse=rmse if not note else np.nan,
                         mae_below_120=r.mae_below_120 if not note else np.nan,
                         mae_120_up=r.mae_120_up if not note else np.nan,
                         n=int(r.n) if not r.empty else 0))
R = pd.DataFrame(rows)
R.to_csv(NUMBERS / "model_results.csv", index=False)
assert R.groupby("test").reference.sum().eq(1).all(), "one reference per test"


def panel(ax, test, extra_ref=None):
    d = R[(R.test == test) & R.usable].sort_values("mae", ascending=False)
    colors = [ORANGE if (m == REFERENCE[d.task.iat[0]] or m == extra_ref) else BLUE for m in d.model]
    ax.barh(d.name, d.mae, color=colors, height=.65)
    for y, v in enumerate(d.mae):
        ax.text(v + .3, y, f"{v:.1f}", va="center", fontsize=8, color=INK2)
    ref = d.mae[d.reference].iat[0]
    ax.axvline(ref, color=ORANGE, ls="--", lw=1)
    bad = R[(R.test == test) & ~R.usable]
    title = test + (f"\nno usable result: {', '.join(bad.name)}" if len(bad) else "")
    ax.set_title(title, fontsize=10)
    ax.set(xlim=(0, d.mae.max() * 1.15), xlabel="average error (MAE), lower is better")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", visible=False)


fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
panel(ax[0], "Random test")
panel(ax[1], "Block test")
fig.suptitle("Filling hidden HRV readings (orange: straight line, the reference; dashed: its error)")
fig.tight_layout()
save(fig, "12a-imputation-results")

fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
panel(ax[0], "Every 10 minutes", extra_ref="last_value")
panel(ax[1], "Daily", extra_ref="last_value")
fig.suptitle("Forecasting HRV (orange: references, the patient's median and the last reading; "
             "dashed: patient's median)")
fig.tight_layout()
save(fig, "12b-forecasting-results")

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
for a, (fs, label) in zip(ax, [("all", "All {} values"), ("reduced", "Reduced set, {} values")]):
    d = X[X.feature_set == fs].sort_values("auc")
    a.barh(d.name, d.auc, color=[GREY if m == "dummy" else BLUE for m in d.model], height=.65)
    for y, v in enumerate(d.auc):
        a.text(v + .01, y, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    a.axvline(.5, color=ORANGE, ls="--", lw=1)
    a.set(xlim=(0, 1), xlabel="AUC")
    a.set_title(label.format(d.n_features.iat[0]), fontsize=10)
    a.tick_params(axis="y", labelsize=8.5)
    a.grid(axis="y", visible=False)
fig.suptitle("Sorting patients by exacerbation during monitoring (yes / no) from enrolment tests and follow-up days\n"
             f"AUC: chance a patient with an exacerbation is ranked above one without; 0.5 (dashed) = coin toss; grey: always-{common} check")
fig.tight_layout()
save(fig, "12c-exacerbation-results")

# ------------------------------------------------------------------ tuned runs (tune_models.py, run_models.py --tuned)
TUNED = RUN / "summary_tuned.csv"
if TUNED.exists():
    import json
    St = pd.read_csv(TUNED)
    mcol = "mask" if "mask" in St else "imputer"           # the column that holds the mask (impute) or imputer
    rows = []
    for task, mask, label in TESTS:
        for model in REGISTRY:
            if model in REFERENCE.values() or model in ("linear", "last_value"):
                continue
            r = St[(St.task == task) & (St.model == model) & ((St[mcol] == mask) if mask else True)]
            if r.empty:
                continue
            r, base = r.iloc[-1], R[(R.test == label) & (R.model == model)].iloc[0]
            ok = np.isfinite(r.mae) and r.mae < 1000 and r.n_nonfinite == 0 and r.n_nonpositive == 0 and r.pct_above_max <= 5
            rows.append(dict(task=task, test=label, model=model, name=NAMES[model], default=base.mae,
                             tuned=r.mae if ok else np.nan, change=(r.mae - base.mae) if ok else np.nan,
                             reference=R[(R.test == label) & R.reference].mae.iat[0]))
    Tn = pd.DataFrame(rows)
    Tn.to_csv(NUMBERS / "model_results_tuned.csv", index=False)
    tun = []
    for f in sorted((RUN / "tuning").glob("*/*.json")):
        j = json.loads(f.read_text())
        tun.append(dict(task=j["task"], model=j["model"], name=NAMES[j["model"]], trials=j["trials"], series=j["series"],
                        default_val_mae=j["default_val_mae"], val_mae=j["val_mae"],
                        params="; ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in j["params"].items())))
    pd.DataFrame(tun).to_csv(NUMBERS / "tuning.csv", index=False)

    fig, ax = plt.subplots(1, 4, figsize=(17, 6.5), sharey=False)
    for a_, (_, _, label) in zip(ax, TESTS):
        d = Tn[Tn.test == label].sort_values("tuned", ascending=False).reset_index(drop=True)
        a_.hlines(d.index, d[["default", "tuned"]].min(axis=1), d[["default", "tuned"]].max(axis=1), color=GREY, lw=1.5)
        a_.plot(d.default, d.index, "o", color=GREY, ms=6, label="default settings")
        a_.plot(d.tuned, d.index, "o", color=BLUE, ms=6, label="tuned settings")
        a_.axvline(d.reference.iat[0], color=ORANGE, ls="--", lw=1)
        a_.set_yticks(d.index, [n.split(" (")[0] for n in d.name], fontsize=8)
        a_.set_title(label, fontsize=10)
        a_.set_xlabel("average error (MAE)")
        a_.grid(axis="y", visible=False)
    ax[3].legend(fontsize=8, loc="upper right")
    fig.suptitle("Default against tuned settings, on the official tests (dashed: the reference)")
    fig.tight_layout()
    save(fig, "12d-tuned-results")
    for label, d in Tn.groupby("test", sort=False):
        print(f"tuned {label:17s} better {int((d.change < -0.05).sum())}, same {int((d.change.abs() <= 0.05).sum())}, "
              f"worse {int((d.change > 0.05).sum())}; best tuned {d.sort_values('tuned').name.iat[0]} {d.tuned.min():.2f}")

for test, d in R.groupby("test", sort=False):
    ok = d[d.usable].sort_values("mae")
    ref = ok[ok.reference].iloc[0]
    print(f"{test:17s} n {d.n.max():>6,}  best {ok.name.iat[0]} {ok.mae.iat[0]:.2f}  "
          f"reference {ref['name']} {ref.mae:.2f}  ({100 * (1 - ok.mae.iat[0] / ref.mae):.1f}% better)  "
          f"not usable: {', '.join(d.name[~d.usable]) or 'none'}")
