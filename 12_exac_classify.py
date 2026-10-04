"""Can enrolment tests classify which patients had a COPD exacerbation?

    python 12_exac_classify.py                 # --target dated (default)
    python 12_exac_classify.py --target history
    python 12_exac_classify.py --quick         # fewer repeats/permutations

TARGETS. Two different questions, and they are not interchangeable.

  dated (default) - did this patient have a DATED exacerbation during monitoring?
    Positives are the patients with an episode in the `exacerbation` sheet.
    Negatives are monitored patients with none. EXCLUDED: patients with an event
    recorded with no date (c005, c032, c037: status genuinely unknown), and any
    patient with no wearable readings at all (c034) - they were never observed
    for events, so scoring them "no" would invent negatives. Every other
    datasheet patient is included.
    Monitoring length varies from about a week to eleven months, so `followup_days`
    is added as a feature: a patient watched briefly had less opportunity to have
    an event recorded. It is a proxy (wearable recording span, from
    numbers/data_presence.csv), not a study follow-up date, which the sheet lacks.

  history - EXC_HPNED_YES_NO, exacerbation in the 12 months BEFORE enrolment.
    This is a recorded field, not an observed outcome, and
    the features come from the same visit, so it is a cross-sectional association.
    The two targets disagree for many patients: they measure different things.

WHY IT IS SET UP THIS WAY.
  * Few patients, many features, and only some patients have every test, so
    imputation happens INSIDE each fold, never before splitting.
  * Classes are unbalanced, so accuracy is not a headline: with the `dated` target
    always answering "no" scores 61%. ROC-AUC and balanced accuracy are reported
    instead, always beside a DummyClassifier that must land at 0.500.
  * Tuning and scoring on the same split inflates everything, so the estimate comes
    from NESTED cross-validation: inner folds pick hyperparameters, outer folds score.
  * AUC is computed per fold and averaged, never pooled across folds - pooling ranks
    patients partly by which fold they fell in, which scores a constant model 0.43.
  * At this sample size a good-looking AUC can be luck, so labels are also shuffled
    PERM times and the whole thing rerun. An AUC that does not clear that null is
    not a result.

FEATURE SETS. "all" is every numeric column of the four requested sections that is
>30% populated, plus age, sex, BODE and CAT. "reduced" drops the oscillometry *_REF
predicted norms (they describe the reference population, not the patient) and the
%PRED columns that are algebraically 100*ACT/REF, and drops post-bronchodilator
spirometry (sparsely populated). Both are reported so the gap shows what the extra
columns cost.
"""
import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
from codings import MISSING, PID, SHEET, clean, load_baseline, load_exacerbations, yes_no

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)

HERE = Path(__file__).resolve().parent
NUMBERS = HERE / "numbers"
RESULTS = HERE / "results"
EXAC = RESULTS / "exac_monitoring"      # what 12_model_results.py and 10_report.py read
HISTORY_COL = "EXC_HPNED_YES_NO"
SECTIONS = ["IMPULSE_OSCILLOMETRY", "SPIROMETRY_PRE", "SPIROMETRY_POST", "SIX_MINUTE_WALK_TEST"]
MIN_FILL = 0.30           # a column must be this populated to be a candidate at all
OUTER, INNER = 5, 3       # nested CV folds
N_JOBS = max(1, (os.cpu_count() or 4) - 2)     # leave 2 cores for the UI (project rule); --cores N overrides


# ------------------------------------------------------------------ features
def feature_table():
    """Numeric features from the requested sections, read by POSITION.

    The sheet repeats column names (FEV/FVC %PRED_PRE_0 appears twice - once as the
    ratio, once as % predicted), so names alone cannot address a column.
    """
    raw = pd.ExcelFile(SHEET).parse("Baseline", header=None)
    section = raw.iloc[1].ffill().astype(str).str.strip().str.rstrip(":").str.strip()
    names = raw.iloc[2].astype(str).str.strip()
    d = raw.iloc[3:]
    pid = d[names[names == "PARTCPNT_ID"].index[0]].astype(str).str.strip().str.lower()
    keep = pid.str.match(PID)
    d, pid = d[keep], pid[keep]

    out, seen = {}, {}
    for c in d.columns:
        sec, nm = section[c], names[c]
        if sec not in SECTIONS or nm.startswith("DATE") or nm == "PARTCPNT_ID":
            continue
        t = d[c].astype(str).str.strip().str.upper()
        v = pd.to_numeric(d[c].where(~t.isin(MISSING) & (t != "NAN")), errors="coerce")
        if v.notna().mean() < MIN_FILL:
            continue
        seen[nm] = seen.get(nm, 0) + 1                      # disambiguate repeated names
        label = f"{sec[:5].lower()}:{nm}" + (f"#{seen[nm]}" if seen[nm] > 1 else "")
        out[label] = v.values
    X = pd.DataFrame(out, index=pid.values)

    B = load_baseline().set_index("pid")
    X["age"] = clean(B["AGE"]).reindex(X.index).values
    X["sex_male"] = (clean(B["GENDER"]).reindex(X.index) == 1).astype(float).values
    X["bode"] = clean(B["BODE_INDXTOT_SCORE_0"]).reindex(X.index).values     # see docstring caveat
    X["cat"] = clean(B["CAAT_TOT_SCORE_0"]).reindex(X.index).values
    return X, yes_no(B[HISTORY_COL]).reindex(X.index)


def dated_target(X):
    """Did the patient have a DATED exacerbation while being monitored?

    Returns (y, followup_days). Patients whose status cannot be observed are left
    NaN in y and dropped by the caller: an undated event (we know something happened
    but not when) and anyone with no wearable data (never watched, so a "no" would
    be invented rather than measured).
    """
    E = load_exacerbations()
    dated = set(E.pid)
    unknown = set(E.attrs["undated_patients"])
    pres = pd.read_csv(HERE / "numbers" / "data_presence.csv")       # built by 03_data_presence.py
    pres = pres[pres.minutes > 0]                                    # no readings at all = never observed
    watched = dict(zip(pres.patient, pres.days))                     # recording span, the follow-up proxy
    y, days = {}, {}
    for pid in X.index:
        days[pid] = watched.get(pid, np.nan)
        if pid in dated:
            y[pid] = 1.0                                             # observed an episode, with a date
        elif pid in unknown or pid not in watched:
            y[pid] = np.nan                                          # status not observable
        else:
            y[pid] = 0.0                                             # monitored, no episode recorded
    return pd.Series(y).reindex(X.index), pd.Series(days).reindex(X.index)


def reduced(X):
    """Drop the reference norms, the algebraically-derived %PRED twins, and post-BD."""
    drop = [c for c in X.columns
            if "_REF_" in c                                  # predicted norm, not the patient
            or c.startswith("spiro:") and "POST" in c.upper() # 11 of 33 populated
            or any(k in c for k in ("R5_%PRED", "R20_%PRED")) # ~ 100*ACT/REF
            or "_ACT_VAL" in c and "R5-20" in c]             # R5 - R20, a difference of two kept columns
    return X.drop(columns=drop)


# ------------------------------------------------------------------ models
def grids(seed, quick):
    n = 100 if quick else 300
    return {
        "dummy":        (DummyClassifier(strategy="prior"), {}, False),
        "logreg_l2":    (LogisticRegression(max_iter=5000, class_weight="balanced"),
                         {"m__C": [0.01, 0.1, 1, 10]}, True),
        "logreg_l1":    (LogisticRegression(max_iter=5000, penalty="l1", solver="liblinear",
                                            class_weight="balanced"),
                         {"m__C": [0.01, 0.1, 1, 10]}, True),
        "lda":          (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), {}, True),
        "gaussian_nb":  (GaussianNB(), {"m__var_smoothing": [1e-9, 1e-6, 1e-3]}, True),
        "knn":          (KNeighborsClassifier(), {"m__n_neighbors": [3, 5, 7],
                                                   "m__weights": ["uniform", "distance"]}, True),
        "svm_rbf":      (SVC(kernel="rbf", probability=True, class_weight="balanced",
                             random_state=seed),
                         {"m__C": [0.1, 1, 10], "m__gamma": ["scale", 0.01, 0.1]}, True),
        "svm_linear":   (SVC(kernel="linear", probability=True, class_weight="balanced",
                             random_state=seed), {"m__C": [0.01, 0.1, 1, 10]}, True),
        "decision_tree": (DecisionTreeClassifier(class_weight="balanced", random_state=seed),
                          {"m__max_depth": [2, 3, 4], "m__min_samples_leaf": [2, 4]}, False),
        "random_forest": (RandomForestClassifier(n_estimators=n, class_weight="balanced",
                                                 random_state=seed, n_jobs=1),
                          {"m__max_depth": [2, 3, None], "m__min_samples_leaf": [1, 2, 4]}, False),
        "extra_trees":  (ExtraTreesClassifier(n_estimators=n, class_weight="balanced",
                                              random_state=seed, n_jobs=1),
                         {"m__max_depth": [2, 3, None], "m__min_samples_leaf": [1, 2, 4]}, False),
        "grad_boost":   (GradientBoostingClassifier(random_state=seed),
                         {"m__n_estimators": [50, 150], "m__max_depth": [1, 2],
                          "m__learning_rate": [0.05, 0.1]}, False),
        "xgboost":      (_xgb(seed), {"m__n_estimators": [50, 150], "m__max_depth": [1, 2],
                                       "m__learning_rate": [0.05, 0.1]}, False),
    }


def _xgb(seed):
    from xgboost import XGBClassifier
    return XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=1,
                         random_state=seed, reg_lambda=1.0)


def pipe(model, scale):
    steps = [("imp", SimpleImputer(strategy="median"))]      # inside the fold, never before
    if scale:
        steps.append(("sc", StandardScaler()))
    return Pipeline(steps + [("m", model)])


# ------------------------------------------------------------------ evaluation
def nested_scores(X, y, model, grid, scale, seed, repeats, jobs=1):
    """Outer folds score, inner folds tune. Returns one row per outer repeat.

    jobs parallelises the inner grid search. The permutation loop instead runs whole
    permutations in parallel with jobs=1 here, so the two never oversubscribe.
    """
    rows = []
    for r in range(repeats):
        folds = list(RepeatedStratifiedKFold(n_splits=OUTER, n_repeats=1,
                                             random_state=seed + r).split(X, y))
        per_fold = []
        for tr, te in folds:
            s = GridSearchCV(pipe(model, scale), grid, scoring="roc_auc", cv=INNER,
                             n_jobs=jobs, refit=True) if grid else pipe(model, scale)
            s.fit(X.iloc[tr], y.iloc[tr])
            pr = s.predict_proba(X.iloc[te])[:, 1]
            yt = y.iloc[te]
            assert yt.nunique() == 2, "an outer fold has only one class; OUTER is too large"
            # AUC per fold, never pooled across folds: each fold's model has its own
            # calibration, so pooling ranks patients partly by which fold they landed in
            # (a constant classifier scores 0.43 rather than 0.50 that way).
            pred = (pr >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
            per_fold.append(dict(auc=roc_auc_score(yt, pr),
                                 bal_acc=balanced_accuracy_score(yt, pred),
                                 f1=f1_score(yt, pred, zero_division=0),
                                 sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1)))
        rows.append(pd.DataFrame(per_fold).mean().to_dict())
    return pd.DataFrame(rows)


def main(quick, target):
    X, y = feature_table()
    if target == "dated":
        y, followup = dated_target(X)
        X = X.copy()
        X["followup_days"] = followup.values      # unequal observation time is a real confounder
    lab = y.dropna().index
    fu = followup.loc[lab] if target == "dated" else None
    X, y = X.loc[lab], y.loc[lab].astype(int)
    repeats = 3 if quick else 20
    perms = 30 if quick else 200
    print(f"target '{target}': {len(y)} patients, {int(y.sum())} positive / "
          f"{int((y == 0).sum())} negative; majority-class rate {max(y.mean(), 1 - y.mean()):.1%}")

    sets = {"all": X, "reduced": reduced(X)}
    rows, perm_rows = [], []
    for sname, XS in sets.items():
        print(f"\n== feature set '{sname}': {XS.shape[1]} features, "
              f"{XS.dropna().shape[0]} of {len(XS)} rows complete")
        for mname, (model, grid, scale) in grids(0, quick).items():
            s = nested_scores(XS, y, model, grid, scale, 0, repeats, jobs=N_JOBS)
            r = dict(feature_set=sname, model=mname, n_features=XS.shape[1],
                     auc=s.auc.mean(), auc_lo=s.auc.quantile(.025), auc_hi=s.auc.quantile(.975),
                     bal_acc=s.bal_acc.mean(), sens=s.sens.mean(), spec=s.spec.mean(),
                     f1=s.f1.mean())
            rows.append(r)
            print(f"   {mname:15s} AUC {r['auc']:.3f} [{r['auc_lo']:.2f}-{r['auc_hi']:.2f}]  "
                  f"bal.acc {r['bal_acc']:.3f}  sens {r['sens']:.2f}  spec {r['spec']:.2f}")

        # null distribution: same pipeline, labels shuffled
        best = max([r for r in rows if r["feature_set"] == sname], key=lambda r: r["auc"])
        model, grid, scale = grids(0, quick)[best["model"]]
        rng = np.random.default_rng(0)
        shuffled = [pd.Series(rng.permutation(y.values), index=y.index) for _ in range(perms)]
        null = np.array(Parallel(n_jobs=N_JOBS)(
            delayed(lambda ys, sd: nested_scores(XS, ys, model, grid, scale, sd, 1).auc.iloc[0])(ys, 100 + i)
            for i, ys in enumerate(shuffled)))
        p = float((null >= best["auc"]).mean())
        perm_rows.append(dict(feature_set=sname, model=best["model"], auc=best["auc"],
                              null_mean=null.mean(), null_p95=np.percentile(null, 95),
                              p_value=p, permutations=perms, n_patients=len(y), n_yes=int(y.sum()),
                              n_no=int((y == 0).sum()), n_complete=int(XS.dropna().shape[0]),
                              followup_min=int(fu.min()) if fu is not None else np.nan,
                              followup_max=int(fu.max()) if fu is not None else np.nan))
        print(f"   permutation test on '{best['model']}': observed AUC {best['auc']:.3f}, "
              f"null mean {null.mean():.3f}, null 95th pct {np.percentile(null, 95):.3f}, "
              f"p = {p:.3f}")

    R = pd.DataFrame(rows).sort_values(["feature_set", "auc"], ascending=[True, False])
    P = pd.DataFrame(perm_rows)
    EXAC.mkdir(parents=True, exist_ok=True)
    out = EXAC if target == "dated" else RESULTS / f"exac_{target}"
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "exac_models.csv", index=False)
    P.to_csv(out / "exac_permutation.csv", index=False)
    inputs = pd.Series([c.split(":")[0] if ":" in c else c for c in X.columns]).value_counts(sort=False)
    inputs.rename_axis("group").rename("n").reset_index().to_csv(out / "exac_inputs.csv", index=False)
    write_md(R, P, X, y, sets, repeats, perms, target)
    print(f"\n-> {out}/exac_models.csv, exac_permutation.csv, exac_inputs.csv, "
          f"results/exacerbation_models_{target}.md")

    # self-check: the dummy must sit at chance, and nothing may beat the permutation null silently
    d = R[R.model == "dummy"]
    assert (d.auc.between(0.45, 0.55)).all(), \
        f"a constant classifier must score ~0.50; got {list(d.auc.round(3))} - AUC is being pooled across folds"
    print("ok: dummy sits at chance; permutation p-values recorded next to every headline AUC")


def write_md(R, P, X, y, sets, repeats, perms, target):
    desc = ("a DATED exacerbation recorded during monitoring" if target == "dated"
            else f"`{HISTORY_COL}` - exacerbation in the 12 months BEFORE enrolment")
    note = ("- **excluded**: patients with an undated event (status unknown) and patients with no "
            "wearable data (never observed). `followup_days` is included as a feature because "
            "monitoring length varies from about a week to eleven months."
            if target == "dated" else
            "- **cross-sectional association, not prediction**: features and label come from the "
            "same enrolment visit.")
    L = [f"# Exacerbation classification (target: {target})", "",
         f"- **target**: {desc}. "
         f"{int(y.sum())} positive / {int((y == 0).sum())} negative, n = {len(y)}.",
         note,
         f"- **majority-class rate {max(y.mean(), 1 - y.mean()):.1%}** - so accuracy is not a "
         f"useful metric here; ROC-AUC and balanced accuracy are reported instead.",
         f"- **nested CV**: {OUTER} outer folds x {INNER} inner folds, {repeats} repeats. "
         f"Median imputation and scaling happen inside each fold.",
         f"- **permutation test**: {perms} label shuffles through the identical pipeline.", ""]
    for s, XS in sets.items():
        L += [f"## Feature set `{s}` ({XS.shape[1]} features, "
              f"{XS.dropna().shape[0]}/{len(XS)} rows complete)", "",
              "| model | AUC | 95% CI | balanced acc | sensitivity | specificity | F1 |",
              "|---|---|---|---|---|---|---|"]
        for r in R[R.feature_set == s].itertuples():
            L.append(f"| {r.model} | {r.auc:.3f} | {r.auc_lo:.2f}-{r.auc_hi:.2f} | "
                     f"{r.bal_acc:.3f} | {r.sens:.2f} | {r.spec:.2f} | {r.f1:.2f} |")
        p = P[P.feature_set == s].iloc[0]
        L += ["", f"Permutation test on the best model (`{p.model}`): observed AUC "
                  f"**{p.auc:.3f}**, null mean {p.null_mean:.3f}, null 95th percentile "
                  f"{p.null_p95:.3f}, **p = {p.p_value:.3f}** over {int(p.permutations)} shuffles.", ""]
    L += ["## Features used", "", "```", *[f"{c}" for c in X.columns], "```", ""]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"exacerbation_models_{target}.md").write_text("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--target", choices=["dated", "history"], default="dated")
    ap.add_argument("--cores", type=int, help="parallel jobs (default: all but 2)")
    a = ap.parse_args()
    if a.cores:
        N_JOBS = a.cores
    main(a.quick, a.target)
