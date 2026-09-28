"""Model registry. Models are imported only when run, so a missing library only
affects the models that need it.

name -> (family, library it rests on, tasks, loader(kind) -> function)
kind is "impute" or "forecast" (daily uses the forecast function).
"""
import importlib


def _mod(name):
    return importlib.import_module(f".{name}", __name__)


def _simple(module):
    return lambda kind: getattr(_mod(module), kind)


ALL = ("impute", "forecast", "daily")
FORE = ("forecast", "daily")

REGISTRY = {
    # references
    "linear":         ("reference", "numpy", ("impute",), lambda k: _mod("baselines").impute),
    "last_value":     ("reference", "numpy", FORE, lambda k: _mod("baselines").forecast_last),
    "patient_median": ("reference", "numpy", FORE, lambda k: _mod("baselines").forecast_median),
    # general machine learning
    "xgboost":  ("general ML", "xgboost", ALL, lambda k: getattr(_mod("gbm"), f"make_{k}")("xgboost")),
    "catboost": ("general ML", "catboost", ALL, lambda k: getattr(_mod("gbm"), f"make_{k}")("catboost")),
    # neural sequence
    "rnn":  ("neural sequence", "torch", ALL, lambda k: _mod("rnn").make("RNN")[k == "forecast"]),
    "lstm": ("neural sequence", "torch", ALL, lambda k: _mod("rnn").make("LSTM")[k == "forecast"]),
    # latent / state-space
    "hmm":   ("latent / state-space", "hmmlearn", ALL, _simple("hmm")),
    "nlssm": ("latent / state-space", "filterpy (UKF + RTS)", ALL, _simple("nlssm")),
    "rsdpf": ("latent / state-space", "torch", ALL, _simple("rsdpf")),
    "pf":    ("latent / state-space", "particles", ALL, _simple("pf")),
    # Bayesian / irregular time
    "gru_ode_bayes": ("Bayesian / irregular time", "torch + authors' code", ALL, _simple("grude")),
    "cd_gamma_dglm": ("Bayesian / irregular time", "numpy + scipy", ALL, _simple("dglm")),
    # signal decomposition
    "ossa": ("signal decomposition", "numpy", ALL, _simple("ossa")),
    # hybrid / research
    "pinode": ("hybrid / research", "torchdiffeq", ALL, _simple("pinode")),
    "gast":   ("hybrid / research", "torch (first version)", ALL, _simple("gast")),
    # foundation model
    "timesfm3": ("foundation model", "timesfm", ALL, _simple("timesfm3")),
}


def get(name, task):
    kind = "impute" if task == "impute" else "forecast"
    return REGISTRY[name][3](kind)
