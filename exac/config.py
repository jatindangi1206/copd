"""Every setting of the exacerbation experiments, in one place. Edit here, nowhere else.

Each value is a decision from docs/EXAC_DESIGN.md / docs/GAST_DESIGN.md (the section is given). Changing
one changes the experiment: record why in the design log. Stdlib only (imported before numpy / torch).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # the repo
KEEP_FREE = 4                                           # CPU cores always left for the GUI (exac/cores.py)
MD = ROOT / "model_data_exac"                           # input, built by 14_exac_data.py
RES = ROOT / "results" / "exac"                         # output: <task>/<set>/<model>/test<NN>_seed<S>/

# ---------------------------------------------------------------- data, sets, tasks (EXAC_DESIGN 2, 3, 5, 6.7)
SETS = {"combined": (["copd", "aiims"], False),         # set -> (cohorts, add the COPD clinical table)
        "copd": (["copd"], False),
        "copd_clinical": (["copd"], True)}
GROUPS = ["copd", "aiims_adult", "aiims_paed"]          # scores are also reported per group
EVENT = {"detect": ("label", "episode"),                # task -> (label column, event id column) in days.csv
         "onset": ("onset_label", "onset_id")}
HR = 1                                                  # channel of heart rate in minutes.npy
DAYS = 7                                                # window: the classified day + the 6 before (3)

# ---------------------------------------------------------------- split and scoring (5)
TEST_PCT = [20, 10, 30, 40, 50]                         # test share, % of patients in each stratum
SEEDS = [0, 1, 2, 3, 4]
THRESHOLD = 0.5                                         # a day is flagged if its probability >= this
BOOT = 1000                                             # resamples of test patients for the 95% intervals

# ---------------------------------------------------------------- tree models (6.2, 6.3)
XGBOOST = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
               tree_method="hist")                      # + scale_pos_weight = normal / exacerbation days
CATBOOST = dict(iterations=400, depth=4, learning_rate=0.05, auto_class_weights="Balanced")

# ---------------------------------------------------------------- networks (6.4, 6.5)
BLOCK, CONV = 10, 32                                    # learned compression: 10-minute blocks -> 32 numbers
RNN = dict(hidden=64, layers=2, dropout=0.1)            # RNN and LSTM size
BATCH, LR = 32, 1e-3                                    # batches half exacerbation days; Adam
MAX_STEPS, CHECK_EVERY, PATIENCE = 3000, 100, 5         # early stopping on held-back patients (AUROC)
HELD_BACK = 0.20                                        # share of training patients held back
CLIP = 1.0                                              # gradient clipping

# ---------------------------------------------------------------- GAST v1 (GAST_DESIGN 3, 6, 8)
GAST = dict(d=32, n_state=32, heads=2, dropout=0.1, fastest_minutes=10.0)
REF_MINUTES = 60                                        # the state is read at the end of every hour
ALPHA = 1.0                                             # gast2: L = L_onset + ALPHA * L_detect

# ---------------------------------------------------------------- TimesFM 3 (6.8)
TIMESFM_CHECKPOINT = os.environ.get("TIMESFM_CHECKPOINT", "/home/kcdha/timesfm3/timesfm-3.0-pytorch")
TIMESFM_DIM = 1280                                      # transformer width (its config.json)
TIMESFM_BATCH = 4                                       # windows per forward pass (GPU memory)
TIMESFM_CACHE = MD / "timesfm3_out.npy"                 # frozen outputs per day (shared with run_exac.py)

# ---------------------------------------------------------------- --quick / check: tiny training, code check only
QUICK = dict(MAX_STEPS=10, CHECK_EVERY=5, BOOT=30, trees=20, days=(600, 200, 300))   # train, held-back, test days
