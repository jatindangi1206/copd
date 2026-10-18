"""XGBoost (EXAC_DESIGN 6.2) on every raw value of the window (data.tree_matrix), blanks left blank.
Settings: config.XGBOOST. Exacerbation days weighted up: scale_pos_weight = normal / exacerbation days."""
from exac import config as C
from exac import data

KIND = "tree"


def fit(tr, te, ytr, ctx):
    import xgboost as xgb
    Xtr, Xte, gpu = data.tree_matrix(tr), data.tree_matrix(te), data.device == "cuda"
    kw = dict(C.XGBOOST, n_estimators=C.QUICK["trees"]) if ctx["quick"] else C.XGBOOST
    m = xgb.XGBClassifier(**kw, scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1),
                          n_jobs=ctx["cores"], device="cuda" if gpu else "cpu", random_state=ctx["seed"])
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1:], {}
