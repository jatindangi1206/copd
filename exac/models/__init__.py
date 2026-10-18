"""Every model of the exacerbation experiments: name -> (what it is, file in this folder, variant).

To add a model: write a file here with KIND = "reference" | "tree" (fit(tr, te, ytr, ctx)) or "net"
(make(n_out, variant), inputs(T, rows, sc, variant), optional prepare(rows)), and add a line below.
"""
import importlib

MODELS = {
    "always_no": ("reference: never flags", "always_no", None),
    "xgboost": ("trees on raw minutes", "xgboost", None),
    "catboost": ("trees on raw minutes", "catboost", None),
    "rnn": ("network (10-minute learned compression)", "rnn", None),
    "lstm": ("network (10-minute learned compression)", "lstm", None),
    "gast": ("GAST v1", "gast_v1", "full"),
    "gast_ssm": ("GAST v1 ablation: state space only", "gast_v1", "ssm"),
    "gast_attn": ("GAST v1 ablation: attention only", "gast_v1", "attn"),
    "gast_nogap": ("GAST v1 ablation: no gap features", "gast_v1", "nogap"),
    "gast_maskonly": ("GAST v1 control: masks and times, no values", "gast_v1", "maskonly"),
    "gast2": ("GAST v1, two heads: both tasks", "gast_v1", "full"),
    "timesfm3": ("TimesFM 3, frozen, + a learned logistic layer", "timesfm3", None),
}
TWO_TASKS = {"gast2"}                         # trained on detection and onset at once
ANY_DAYS = {n for n, (_, f, _) in MODELS.items() if f == "gast_v1"}     # --days other than 7: GAST only


def load(name):
    """-> (module, variant)"""
    _, f, variant = MODELS[name]
    return importlib.import_module(f"exac.models.{f}"), variant
