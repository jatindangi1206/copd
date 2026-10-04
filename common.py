"""Paths and plot style for the COPD clinical repo. Self-contained: everything
this folder needs is under data/.

THEME=dark in the environment draws every figure for a dark page and writes it
to figs_dark/ instead of figs/. Same data, same numbers.
"""
import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
# Where the cohort's data lives. Another cohort: set these, or keep the same names under its own data/.
DATA = Path(os.environ.get("COHORT_DATA", HERE / "data"))
RAW = DATA / os.environ.get("COHORT_EXPORT", "copd")                       # per-vital export tree
WEAR = DATA / "wearable"                                                    # 1-minute tables (01_build_wearable.py)
SHEET = DATA / os.environ.get("COHORT_SHEET", "COPDAI_DATASHEET_01.xls")   # clinical datasheet
PID = r"^c\d{3}$"                                                          # what a patient ID looks like
THEME = os.environ.get("THEME", "light")
DARK = THEME == "dark"
FIGS = HERE / ("figs_dark" if DARK else "figs")
NUMBERS = HERE / "numbers"
for d in (FIGS, NUMBERS):
    d.mkdir(exist_ok=True)

VITALS = ["hr", "hrv", "temp", "spo2", "steps"]
if DARK:
    BLUE, ORANGE, GREY = "#6fb3c8", "#f07a45", "#9a9a95"
    BG, INK, INK2, LINE = "#1c1c1b", "#ecebe6", "#b9b8b0", "#3a3a37"
    CMAP = "Blues_r"                     # near-zero recedes into the dark page
else:
    BLUE, ORANGE, GREY = "#3b6978", "#c1440e", "#8a8a8a"
    BG, INK, INK2, LINE = "#ffffff", "#1b1b1a", "#555450", "#e2e2e2"
    CMAP = "Blues"
STAGE_COLORS = {"deep": BLUE, "light": "#9cc3cf" if not DARK else "#2f5f6d", "awake": ORANGE}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": LINE, "grid.linestyle": "-",
    "grid.linewidth": .6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "font.size": 11, "axes.titlesize": 12, "figure.titlesize": 13,
})


def patients():
    return sorted(f.name.split(".")[0] for f in WEAR.glob("*.csv.gz"))


def master(pid, cols=None):
    """One patient's 1-minute master. One row per recorded minute."""
    return pd.read_csv(WEAR / f"{pid}.csv.gz", usecols=cols,
                       parse_dates=["time"] if cols is None or "time" in cols else None,
                       low_memory=False)


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"  {FIGS.name}/{name}.png")
