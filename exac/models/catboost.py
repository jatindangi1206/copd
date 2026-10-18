"""CatBoost (EXAC_DESIGN 6.3) on every raw value of the window (data.tree_matrix), blanks left blank.
Settings: config.CATBOOST (auto_class_weights="Balanced"). On the GPU it takes 95% of the memory unless
--gpu-gb caps it (gpu_ram_part)."""
from exac import config as C
from exac import data

KIND = "tree"


def fit(tr, te, ytr, ctx):
    from catboost import CatBoostClassifier
    Xtr, Xte, gpu = data.tree_matrix(tr), data.tree_matrix(te), data.device == "cuda"
    kw = dict(C.CATBOOST, iterations=C.QUICK["trees"]) if ctx["quick"] else C.CATBOOST
    if gpu and ctx.get("gpu_gb"):
        import torch
        kw = dict(kw, gpu_ram_part=ctx["gpu_gb"] * 1e9 / torch.cuda.get_device_properties(0).total_memory)
    m = CatBoostClassifier(**kw, thread_count=ctx["cores"], task_type="GPU" if gpu else "CPU",
                           random_seed=ctx["seed"], verbose=0, allow_writing_files=False)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1:], {}
