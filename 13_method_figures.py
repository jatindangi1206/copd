"""Diagrams that explain how the imputation, forecasting and classification tests work.

    THEME=light python 13_method_figures.py ; THEME=dark python 13_method_figures.py

13a-imputation-flow      one real stretch of HRV: hide chunks, fill them, compare
13b-forecasting-flow     one real segment (and one patient's days): learn, hold back, predict, compare
13c-classification-flow  the exacerbation test, step by step, with the run's own numbers
13d-model-families       every model family: input -> how it works -> output
13e-classification-methods  the same for the classification methods

The data panels use the best usable model of each test, read from results/summary.csv, so
rerun this after the models (it needs results/ and results/exac_monitoring/).
"""
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

from common import BLUE, ORANGE, GREY, INK, INK2, LINE, DARK, HERE, save

MD, RES = HERE / "model_data", HERE / "results"
FILL = "#b48ee0" if DARK else "#6b4c9a"          # what a model produced
REFS = ["linear", "last_value", "patient_median"]
NAMES = {"xgboost": "XGBoost", "catboost": "CatBoost", "rnn": "RNN", "lstm": "LSTM",
         "hmm": "Hidden Markov model", "nlssm": "Nonlinear state-space model", "rsdpf": "RS-DPF",
         "pf": "Particle filter", "gru_ode_bayes": "GRU-ODE-Bayes", "cd_gamma_dglm": "CD-Gamma-DGLM",
         "ossa": "OSSA", "pinode": "Physiology-informed neural ODE",
         "gast": "Gap-aware state-space transformer", "timesfm3": "TimesFM 3"}

S = pd.read_csv(RES / "summary.csv").drop_duplicates(["task", "run"], keep="last")
S = S[np.isfinite(S.mae) & (S.mae < 1000) & (S.n_nonpositive == 0) & (S.pct_above_max <= 5) & ~S.model.isin(REFS)]


def best(task, mask=None):
    d = S[S.task == task]
    if mask:
        d = d[d["mask"] == mask]
    return d.sort_values("mae").iloc[0]


def runs(m):
    out, c = [], 0
    for v in m:
        if v:
            c += 1
        elif c:
            out.append(c)
            c = 0
    return out + ([c] if c else [])


def spans(m):
    """(start, length) of each run of True."""
    a = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    return list(zip(a[::2], a[1::2] - a[::2]))


def style(ax, ylab="HRV"):
    ax.set_ylim(10, 150)
    ax.set_ylabel(ylab)
    ax.grid(axis="x", visible=False)


def bar_title(ax, text):
    ax.set_title(text, loc="left", fontsize=11, pad=6)


# ============================================================ 13a imputation
T = pd.read_csv(MD / "model_10min.csv.gz", parse_dates=["time"])
T = T[T.segment.notna()]
W = 288                                                    # two days of 10-minute slots

pick = None
for sid, g in T.groupby("segment"):
    for a in range(0, len(g) - W + 1, 72):
        w = g.iloc[a:a + W]
        k = (len([r for r in runs(w.mask_block.values == 1) if r >= 3]), int(w.hrv.notna().sum()))
        if pick is None or k > pick[0]:
            pick = (k, sid, a)
_, sid, a0 = pick
G = T[T.segment == sid].reset_index(drop=True)
w = G.iloc[a0:a0 + W].reset_index(drop=True)
bi = best("impute", "block")
P = pd.read_csv(RES / "impute" / bi.run / "predictions.csv.gz")
P = P[P.series == sid]
P = P[(P.step >= a0) & (P.step < a0 + W)].assign(x=lambda d: (d.step - a0) / 6)
x = np.arange(W) / 6
real = w.hrv.notna().to_numpy()
hid = w.mask_block.to_numpy() == 1
kept = real & ~hid
hid_runs = [(s, n) for s, n in spans(hid) if n >= 3]
show_run = max(hid_runs, key=lambda r: r[1])
err = (P.pred - P.truth).abs().mean()

fig, ax = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
for a_ in ax:
    style(a_)
# 1 the data
a_ = ax[0]
a_.plot(x[real], w.hrv[real], ".", color=BLUE, ms=4)
a_.plot(x[~real], np.full((~real).sum(), 12), "|", color=GREY, ms=5)
a_.text(0.3, 21, "grey ticks at the bottom: blank slots, the watch gave no reading", color=INK2, va="center", fontsize=9)
bar_title(a_, "1   The data: about one reading every 10 minutes, with blanks where the watch gave nothing")
# 2 hide chunks
a_ = ax[1]
for s, n in spans(hid):
    a_.axvspan(x[s] - .08, x[s + n - 1] + .08, color=ORANGE, alpha=.14, lw=0)
a_.plot(x[kept], w.hrv[kept], ".", color=BLUE, ms=4)
a_.plot(x[hid], w.hrv[hid], "o", mfc="none", mec=ORANGE, ms=5, mew=1.1)
a_.annotate("hidden chunk: removed from what the model sees,\nkept aside as the answer key", (x[show_run[0]], 128),
            (x[show_run[0]] + 3, 128), color=ORANGE, fontsize=9, va="center",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=.8))
bar_title(a_, "2   Hide chunks of real readings (orange). The model never sees them")
# 3 the model fills from both sides
a_ = ax[2]
s0, n0 = show_run
for s, n in spans(hid):
    a_.axvspan(x[s] - .08, x[s + n - 1] + .08, color=ORANGE, alpha=.14, lw=0)
a_.plot(x[kept], w.hrv[kept], ".", color=BLUE, ms=4)
a_.plot(P.x, P.pred, "s", color=FILL, ms=3.5)
l_i = max(i for i in range(s0) if kept[i])
r_i = min(i for i in range(s0 + n0, W) if kept[i])
for src, dst, txt, ha in [(l_i, s0, "reads the left side", "right"), (r_i, s0 + n0 - 1, "and the right side", "left")]:
    a_.annotate("", (x[dst], 143), (x[src], 143), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    a_.text(x[dst] + (-.6 if ha == "right" else .6), 143, txt, ha=ha, va="center", fontsize=9, color=INK2)
a_.plot([], [], "s", color=FILL, ms=4, label=f"filled by the model ({NAMES[bi.model]})")
a_.plot([], [], ".", color=BLUE, label="readings the model could see")
a_.legend(loc="lower right", fontsize=9, ncol=2)
bar_title(a_, "3   The model fills every hidden reading from the readings on both sides of the gap")
# 4 compare
a_ = ax[3]
for s, n in spans(hid):
    a_.axvspan(x[s] - .08, x[s + n - 1] + .08, color=ORANGE, alpha=.14, lw=0)
a_.vlines(P.x, P.truth, P.pred, color=GREY, lw=1)
a_.plot(P.x, P.truth, "o", mfc="none", mec=ORANGE, ms=5, mew=1.1, label="real value (the answer key)")
a_.plot(P.x, P.pred, "s", color=FILL, ms=3.5, label="filled value")
a_.legend(loc="lower right", fontsize=9, ncol=2)
a_.text(0.5, 20, f"grey line = how far the fill is from the real value.  Average length in this window: {err:.1f}  "
        f"(this is the score, MAE)", color=INK2, fontsize=9)
bar_title(a_, "4   Each filled value is checked against its real value: the average distance is the score")
ax[3].set_xlabel("hours from the start of the stretch")
fig.suptitle(f"How the imputation test works, on real data (patient {sid.split('_')[0]}, two days)", y=.995)
fig.tight_layout()
save(fig, "13a-imputation-flow")

# ============================================================ 13b forecasting
bf = best("forecast")
cand = [(sid_, g) for sid_, g in T.groupby("segment") if 90 <= len(g) / 6 <= 200]
sid2, g2 = max(cand, key=lambda c: c[1].hrv.notna().sum())
g2 = g2.reset_index(drop=True)
cut = int((g2.split == "train").sum())
Pf = pd.read_csv(RES / "forecast" / bf.run / "predictions.csv.gz")
Pf = Pf[Pf.series == sid2]
x2 = np.arange(len(g2)) / 6
xt = (cut + Pf.step.to_numpy() - 1) / 6
tr, te = g2.iloc[:cut], g2.iloc[cut:]
errf = (Pf.pred - Pf.truth).abs().mean()

fig, ax = plt.subplots(5, 1, figsize=(13, 16), gridspec_kw=dict(height_ratios=[1, 1, 1, 1, 1]))
for a_ in ax[:4]:
    style(a_)
for a_ in ax[1:3]:
    a_.sharex(ax[0])
a_ = ax[0]
a_.plot(x2[g2.hrv.notna()], g2.hrv.dropna(), ".", color=BLUE, ms=3.5)
bar_title(a_, "1   The data: one stretch of readings (a segment)")
a_ = ax[1]
a_.plot(x2[:cut][tr.hrv.notna()], tr.hrv.dropna(), ".", color=BLUE, ms=3.5)
a_.plot(x2[cut:][te.hrv.notna()], te.hrv.dropna(), ".", color=ORANGE, ms=3.5)
a_.axvline(x2[cut], color=INK2, ls="--", lw=1)
a_.text(x2[cut] / 2, 142, "first 80%: the model learns from this", ha="center", color=BLUE, fontsize=9)
a_.text((x2[cut] + x2[-1]) / 2, 142, "last 20%: held back, the answer key", ha="center", color=ORANGE, fontsize=9, va="center")
bar_title(a_, "2   Keep the first 80% to learn from. Hold back the last 20% (never from the middle)")
a_ = ax[2]
a_.plot(x2[:cut][tr.hrv.notna()], tr.hrv.dropna(), ".", color=BLUE, ms=3.5)
a_.axvline(x2[cut], color=INK2, ls="--", lw=1)
a_.annotate("now", (x2[cut], 12), (x2[cut], 12), ha="center", va="bottom", color=INK2, fontsize=9)
a_.plot(xt, Pf.pred, "-", color=FILL, lw=1.4, label=f"prediction ({NAMES[bf.model]})")
a_.annotate("", (x2[cut] - .1, 140), (x2[cut] / 2, 140), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
a_.text(x2[cut] / 2, 143, "uses only what came before 'now'", ha="center", va="bottom", fontsize=9, color=INK2)
a_.legend(loc="lower right", fontsize=9)
bar_title(a_, "3   The model predicts the held-back part from the past only. It never looks ahead")
a_ = ax[3]
a_.vlines(xt, Pf.truth, Pf.pred, color=GREY, lw=.8)
a_.plot(xt, Pf.truth, ".", color=ORANGE, ms=4, label="real value (the answer key)")
a_.plot(xt, Pf.pred, "-", color=FILL, lw=1.4, label="predicted value")
a_.set_xlim(x2[cut] - 2, x2[-1] + 2)
a_.set_ylim(10, 158)
a_.legend(loc="upper left", fontsize=9, ncol=2)
a_.text(x2[cut], 16, f"grey line = distance between prediction and real value.  Average over this segment: {errf:.1f} (MAE)",
        color=INK2, fontsize=9)
bar_title(a_, "4   Predictions are checked against the real readings with the same score (MAE)")
ax[3].set_xlabel("hours from the start of the segment (zoomed on the held-back part)")
# daily
D = pd.read_csv(MD / "model_daily.csv", parse_dates=["date"])
bd = best("daily")
Pd = pd.read_csv(RES / "daily" / bd.run / "predictions.csv.gz")
pid = Pd.groupby("series").size().idxmax()
d = D[D.pid == pid]
full = d.set_index("date").reindex(pd.date_range(d.date.min(), d.date.max()))
last_tr = d.date[d.split == "train"].max()
trd, ted = full[full.index <= last_tr], full[full.index > last_tr]
pp = Pd[Pd.series == pid]
a_ = ax[4]
a_.plot(np.arange(len(trd)), trd.hrv_median, "o-", color=BLUE, ms=3.5, lw=.8, label="days the model learns from")
a_.plot(len(trd) + np.arange(len(ted)), ted.hrv_median, "o", color=ORANGE, ms=4, label="held-back days (answer key)")
a_.plot(len(trd) + pp.step.to_numpy() - 1, pp.pred, "s", color=FILL, ms=4, label=f"predicted ({NAMES[bd.model]})")
a_.axvline(len(trd) - .5, color=INK2, ls="--", lw=1)
a_.set_ylim(10, 150)
a_.set_ylabel("daily median HRV")
a_.set_xlabel("days from the patient's first HRV day")
a_.legend(loc="lower left", fontsize=9, ncol=3)
a_.grid(axis="x", visible=False)
bar_title(a_, f"Daily forecasting works the same way, with one value per day (the day's median HRV; patient {pid})")
fig.suptitle(f"How the forecasting test works, on real data (segment {sid2})", y=.995)
fig.tight_layout()
save(fig, "13b-forecasting-flow")


# ============================================================ drawing helpers
def canvas(w, h, rows=1, **kw):
    fig, ax = plt.subplots(rows, 1, figsize=(w, h), **kw)
    for a_ in np.atleast_1d(ax):
        a_.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", ec=None, fc="none", fs=9, color=None, ha="center", lw=1.1, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec or INK2, lw=lw))
    if text:
        ax.text(x + (w / 2 if ha == "center" else 1.2), y + h / 2, text, ha=ha, va="center", fontsize=fs,
                color=color or INK, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=None):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color or INK2, lw=1.2))


rng = np.random.default_rng(1)


def wave(n, amp=1, ph=0, noise=0):
    t = np.linspace(0, 1, n)
    return amp * np.sin(2 * np.pi * (t + ph)) + noise * rng.standard_normal(n)


# ============================================================ 13c classification flow
XP = pd.read_csv(RES / "exac_monitoring" / "exac_permutation.csv").set_index("feature_set")
XI = pd.read_csv(RES / "exac_monitoring" / "exac_inputs.csv").set_index("group").n
n_p, n_y, n_n = int(XP.n_patients.iat[0]), int(XP.n_yes.iat[0]), int(XP.n_no.iat[0])
perm = int(XP.permutations.iat[0])
fig, ax = canvas(14, 11)
ax.set_xlim(0, 100)
ax.set_ylim(-1, 100)
ax.text(0, 98, "How the exacerbation test works", fontsize=13, fontweight="bold", va="top")
# step 1: table
ax.text(0, 92, f"1   One row per patient ({n_p} patients): the tests taken at enrolment, and the answer",
        fontsize=11, va="center")
cols = [("lung function", f"{XI['spiro'] + XI['impul']} values"), ("walk test", f"{XI['six_m']} values"),
        ("age", ""), ("sex", ""), ("BODE", ""), ("CAT", ""), ("days\nmonitored", "")]
x0 = 2
for i, (c, sub) in enumerate(cols):
    wd = 14 if i < 2 else 7.5
    box(ax, x0, 80, wd - .8, 7, c, fs=8.5)
    for r in range(3):
        ax.add_patch(Rectangle((x0, 77 - r * 2.6), wd - .8, 2.1, fc=LINE, ec="none"))
    x0 += wd
ax.text(x0 - .2, 83.5, "→", fontsize=14, ha="left", va="center", color=INK2)
box(ax, x0 + 3, 80, 16, 7, f"answer\nyes {n_y} / no {n_n}", ec=ORANGE, fs=8.5, color=ORANGE)
for r, c in enumerate([ORANGE, BLUE, BLUE]):
    ax.add_patch(Rectangle((x0 + 3, 77 - r * 2.6), 16, 2.1, fc=c, ec="none", alpha=.55))
ax.text(2, 69, "yes = the patient has a dated exacerbation in the datasheet during monitoring.   "
        "Left out: patients whose event has no date, and anyone with no watch data.", fontsize=8.5, color=INK2)
# step 2: groups
ax.text(0, 62, f"2   Split the patients into 5 groups. Each group takes one turn as the test group; the other 4 are for learning",
        fontsize=11, va="center")
for r in range(5):
    for c in range(5):
        ax.add_patch(Rectangle((2 + c * 9, 55 - r * 3.4), 8.4, 2.8, fc=ORANGE if r == c else BLUE,
                               alpha=.85 if r == c else .35, ec="none"))
    ax.text(48, 56.4 - r * 3.4, f"turn {r + 1}", fontsize=8.5, va="center", color=INK2)
ax.text(60, 55, "orange = test group (never used for learning\nor for choosing settings).\nBlue = learning patients.",
        fontsize=9, va="center", color=INK2)
# step 3: one turn
ax.text(0, 37, "3   In each turn: the method learns from the blue patients, then gives each orange patient a chance of yes",
        fontsize=11, va="center")
box(ax, 2, 26, 17, 7, "learning patients\n(inputs + answers)", fs=9)
arrow(ax, 19.5, 29.5, 25.5, 29.5)
box(ax, 26, 26, 17, 7, "method\n(12 kinds, next figures)", fs=9, ec=FILL, color=FILL)
arrow(ax, 43.5, 29.5, 49.5, 29.5)
box(ax, 50, 26, 17, 7, "test patient\n(inputs only)", fs=9, ec=ORANGE, color=ORANGE)
arrow(ax, 67.5, 29.5, 73.5, 29.5)
box(ax, 74, 26, 24, 7, "chance of yes, 0 to 1", fs=9, bold=True)
ax.text(2, 22, "A missing test value is filled with the middle value of the learning patients. "
        "A further 3-way split inside the learning patients picks the method's settings.", fontsize=8.5, color=INK2)
# step 4: auc
ax.text(0, 17, "4   Score: AUC. Take one patient with a yes and one with a no. How often does the method give the yes patient the higher chance?",
        fontsize=11, va="center")
ax.text(2, 13.2, "0.5 = a coin toss.  1 = always right.  Below 0.5 = wrong more often than right.  "
        "Scored per turn and averaged over 20 repeats of the grouping.", fontsize=9, color=INK2)
# step 5: shuffle
ax.text(0, 9, f"5   Honesty check: shuffle the answers between patients {perm} times and run the same test", fontsize=11, va="center")
bs = XP.loc["reduced"]
lo, hi = 0.3, 0.9
sx = lambda v: 2 + (v - lo) / (hi - lo) * 70
ax.plot([sx(lo), sx(hi)], [2, 2], color=INK2, lw=1)
for v in (0.3, 0.5, 0.7, 0.9):
    ax.plot([sx(v)] * 2, [1.4, 2.6], color=INK2, lw=1)
    ax.text(sx(v), .2, f"{v:.1f}", ha="center", va="top", fontsize=8, color=INK2)
ax.plot([sx(bs.null_mean)] * 2, [1.2, 3], color=GREY, lw=3)
ax.plot([sx(bs.null_p95)] * 2, [1.2, 3], color=GREY, lw=3)
ax.plot([sx(bs.auc)] * 2, [.8, 3.6], color=ORANGE, lw=3)
ax.text(sx(bs.auc), 4.6, f"real best: {bs.auc:.2f}", color=ORANGE, ha="center", fontsize=8.5)
ax.text(sx(bs.null_mean), 4.6, f"shuffled, typical: {bs.null_mean:.2f}", color=INK2, ha="right", fontsize=8.5)
ax.text(sx(bs.null_p95) + 1.2, 4.6, f"95% of shuffles stay below {bs.null_p95:.2f}", color=INK2, ha="left", fontsize=8.5)
ax.text(77, 2.2, f"{int(round(bs.p_value * perm))} of {perm} shuffles\nscored as high or higher.\nA real signal needs few.", fontsize=9,
        va="center", color=INK2)
save(fig, "13c-classification-flow")


# ============================================================ 13d model families
def tree(ax, cx, top, size=1.0, leaves=True):
    """A small decision tree: root, two children, four leaves."""
    pts = [(cx, top), (cx - 6 * size, top - 7 * size), (cx + 6 * size, top - 7 * size)]
    for q in pts[1:]:
        ax.plot([cx, q[0]], [top, q[1]], color=FILL, lw=1.2)
    for i, dx in enumerate((-9, -3, 3, 9)):
        par = pts[1] if dx < 0 else pts[2]
        ax.plot([par[0], cx + dx * size], [par[1], top - 15 * size], color=FILL, lw=1.2)
        ax.add_patch(Rectangle((cx + dx * size - 1.1 * size, top - 17.2 * size), 2.2 * size, 2.2 * size,
                               fc=BLUE if i % 2 == 0 else ORANGE, ec="none"))
    for q in pts:
        ax.add_patch(Circle(q, 1.1 * size, fc=FILL, ec="none"))


def sk_gbm(ax, x0, x1):
    xs = np.linspace(x0 + 1, x0 + 14, 6)
    ys = 14 + 5 * np.sin(np.linspace(0, 3, 6))
    ax.plot(xs, ys, "o", color=BLUE, ms=4)
    ax.plot(x1 - 14 + np.linspace(0, 13, 6), 14 + 5 * np.sin(np.linspace(3, 6, 6)), "o", color=BLUE, ms=4)
    ax.text(x0 + 7.5, 24.5, "6 before", ha="center", fontsize=8, color=INK2)
    ax.text(x1 - 7.5, 24.5, "6 after", ha="center", fontsize=8, color=INK2)
    cx = (x0 + x1) / 2
    tree(ax, cx, 26, .85)
    ax.text(cx, 7, "many small trees, each a chain\nof yes/no questions", ha="center", fontsize=8, color=INK2)
    ax.text(cx, 2.8, "+ time of day, heart rate, steps, sleep", ha="center", fontsize=8, color=INK2)


def sk_seq(ax, x0, x1):
    xs = np.linspace(x0 + 3, x1 - 3, 6)
    for i, xx in enumerate(xs):
        ax.add_patch(Circle((xx, 15), 2.2, fc="none", ec=FILL, lw=1.3))
        ax.plot(xx, 6, "o" if i != 3 else "x", color=BLUE if i != 3 else ORANGE, ms=5)
        ax.plot([xx, xx], [7, 12.6], color=GREY, lw=.8)
        if i < 5:
            ax.annotate("", (xs[i + 1] - 2.4, 16), (xx + 2.4, 16), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
            ax.annotate("", (xx + 2.4, 14), (xs[i + 1] - 2.4, 14), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1, ls="--"))
    ax.text((x0 + x1) / 2, 24, "a running memory is passed along the readings", ha="center", fontsize=8, color=INK2)
    ax.text((x0 + x1) / 2, 20.6, "solid: forwards   dashed: backwards (imputation only)", ha="center", fontsize=7.5, color=INK2)
    ax.text((x0 + x1) / 2, 1, "x = the blank to fill", ha="center", fontsize=8, color=ORANGE)


def sk_state(ax, x0, x1):
    t = np.linspace(x0 + 1, x1 - 1, 40)
    lvl = 15 + 4 * np.sin(np.linspace(0, 5, 40))
    ax.plot(t, lvl, color=FILL, lw=1.6)
    obs = np.ones(40, bool)
    obs[16:26] = False
    ax.plot(t[obs], lvl[obs] + 2.2 * rng.standard_normal(obs.sum()), ".", color=BLUE, ms=5)
    ax.plot(t[~obs], lvl[~obs], ":", color=ORANGE, lw=2)
    ax.text((x0 + x1) / 2, 24.5, "hidden true level (purple) drifts; each reading is a noisy look at it", ha="center", fontsize=8, color=INK2)
    ax.text(t[21], 6, "blank: no new look,\nthe level keeps drifting", ha="center", fontsize=8, color=ORANGE)
    ax.text((x0 + x1) / 2, 1.6, "HMM: a few hidden states.  Particle filter: many candidate paths", ha="center", fontsize=7.5, color=INK2)


def sk_irregular(ax, x0, x1):
    xs = np.array([3, 9, 14, 30, 36, 41]) * (x1 - x0) / 44 + x0
    ys = [12, 17, 13, 18, 12, 16]
    for i in range(len(xs) - 1):
        tt = np.linspace(xs[i], xs[i + 1], 12)
        ax.plot(tt, ys[i] + (14.5 - ys[i]) * (1 - np.exp(-(tt - xs[i]) / 6)) * .6, color=FILL, lw=1.5)
        ax.plot([xs[i + 1]] * 2, [ys[i] + (14.5 - ys[i]) * (1 - np.exp(-(xs[i + 1] - xs[i]) / 6)) * .6, ys[i + 1]],
                color=ORANGE, lw=1.2)
    ax.plot(xs, ys, "o", color=BLUE, ms=5)
    ax.text((x0 + x1) / 2, 24, "smooth drift between readings, a jump at each new reading", ha="center", fontsize=8, color=INK2)
    ax.text((x0 + x1) / 2, 6, "the time between readings is part of the model,\nso a long gap simply means a long drift", ha="center", fontsize=8, color=INK2)


def sk_ssa(ax, x0, x1):
    w = (x1 - x0 - 8) / 3
    parts = [("daily wave", wave(30, 4)), ("slow drift", np.linspace(-3, 3, 30)), ("rest (dropped)", wave(30, .8, .3, .9))]
    for i, (name, y) in enumerate(parts):
        xs = np.linspace(x0 + i * (w + 4), x0 + i * (w + 4) + w, 30)
        ax.plot(xs, 15 + y, color=FILL if i < 2 else GREY, lw=1.3)
        ax.text(xs.mean(), 7.5, name, ha="center", fontsize=8, color=INK2)
    ax.text(x0 + w + 2, 15, "+", fontsize=14, ha="center", va="center", color=INK2)
    ax.text(x0 + 2 * w + 6, 15, "+", fontsize=14, ha="center", va="center", color=INK2)
    ax.text((x0 + x1) / 2, 24, "split the series into repeating parts, keep the strongest", ha="center", fontsize=8, color=INK2)
    ax.text((x0 + x1) / 2, 1.2, "a blank is filled with the kept parts", ha="center", fontsize=8, color=INK2)


def sk_hybrid(ax, x0, x1):
    m = (x0 + x1) / 2
    t = np.linspace(x0 + 1, m - 3, 30)
    sp = 15 + 5 * np.sin(np.linspace(0, 5, 30))
    ax.plot(t, sp, "--", color=GREY, lw=1.2)
    path = 15 + 7 * np.cos(np.linspace(0, 4, 30)) * np.exp(-np.linspace(0, 1.2, 30)) + 3 * np.sin(np.linspace(0, 5, 30))
    ax.plot(t, path, color=FILL, lw=1.6)
    ax.text((x0 + m) / 2 - 1, 24, "neural ODE: pulled toward\nthe daily set point (dashed)", ha="center", fontsize=8, color=INK2, va="center")
    ax.text((x0 + m) / 2 - 1, 4, "+ a small network for what is left", ha="center", fontsize=8, color=INK2)
    xs = np.linspace(m + 3, x1 - 2, 7)
    ys = 15 + 3 * np.sin(np.linspace(0, 6, 7))
    for i in range(7):
        for j in range(i + 1, 7):
            if (i + j) % 3 == 0:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=FILL, lw=.7, alpha=.7)
    ax.plot(xs, ys, "o", color=BLUE, ms=5)
    ax.text((m + x1) / 2 + 1, 24, "our transformer: each reading\nlooks at the others (lines),\nweighted by how stale it is", ha="center", fontsize=8, color=INK2, va="center")
    ax.text((m + x1) / 2 + 1, 4, "", fontsize=8)


def sk_found(ax, x0, x1):
    box(ax, x0 + 1, 8, (x1 - x0) * .52, 13, "pretrained on a huge set of\nother time series\n(not on this cohort)", ec=FILL, fs=8.5, color=FILL)
    xs = np.linspace(x0 + (x1 - x0) * .62, x1 - 1, 12)
    ax.plot(xs[:7], 14 + wave(7, 3, 0, .5), "o", color=BLUE, ms=4)
    ax.plot(xs[6:], 14 + wave(6, 3, .3), ":", color=FILL, lw=2)
    arrow(ax, x0 + (x1 - x0) * .54, 14.5, x0 + (x1 - x0) * .6, 14.5)
    ax.text((x0 + x1) / 2, 3.2, "used as released. A gap: predict forwards from the left,\nbackwards from the right, blend the two", ha="center", fontsize=8, color=INK2)


def sk_refs(ax, x0, x1):
    w = (x1 - x0 - 4) / 3
    for i, name in enumerate(["straight line between\nthe two sides", "repeat the last\nreading", "the patient's\nmiddle value"]):
        xs = np.linspace(x0 + i * (w + 2), x0 + i * (w + 2) + w, 8)
        y = 14 + wave(8, 4, .1 * i, 0)
        if i == 0:
            ax.plot(xs[[0, -1]], y[[0, -1]], color=GREY, lw=1.6)
            ax.plot(xs[[0, -1]], y[[0, -1]], "o", color=BLUE, ms=4)
        elif i == 1:
            ax.plot(xs[:3], y[:3], "o", color=BLUE, ms=4)
            ax.plot(xs[2:], np.full(6, y[2]), color=GREY, lw=1.6)
        else:
            ax.plot(xs, y, "o", color=BLUE, ms=3, alpha=.5)
            ax.plot(xs, np.full(8, np.median(y)), color=GREY, lw=1.6)
        ax.text(xs.mean(), 5, name, ha="center", fontsize=8, color=INK2)
    ax.text((x0 + x1) / 2, 24, "a model is only useful if it beats these", ha="center", fontsize=8, color=INK2)


def frame(a_, title, inp, out, sk, note):
    """One row: title, input box -> sketch -> output box, note underneath."""
    a_.set_xlim(0, 100)
    a_.set_ylim(-9, 36)
    a_.text(0, 34, title, fontsize=11, fontweight="bold", va="center")
    wrap = lambda t, n: "\n".join(textwrap.fill(line, n) for line in t.split("\n"))
    box(a_, 0, 8, 29, 18, "")
    a_.text(1.2, 24, "INPUT", fontsize=8, color=INK2, fontweight="bold", va="center")
    a_.text(1.2, 15.5, wrap(inp, 42), fontsize=8.3, va="center", linespacing=1.4)
    arrow(a_, 29.5, 17, 33.5, 17)
    sk(a_, 35, 72)
    arrow(a_, 73, 17, 77, 17)
    box(a_, 77.5, 8, 22.5, 18, "")
    a_.text(78.7, 24, "OUTPUT", fontsize=8, color=INK2, fontweight="bold", va="center")
    a_.text(78.7, 15.5, wrap(out, 30), fontsize=8.3, va="center", linespacing=1.4)
    a_.text(0, -4, note, fontsize=8.3, color=INK2, va="center", linespacing=1.4)


FAMILIES = [
    ("General machine learning: XGBoost, CatBoost", sk_gbm,
     "the readings before and after the blank, the nearest real\nreading and how far away it is, time of day, other vitals",
     "a value for every blank",
     "Imputation: looks both ways.   Forecast: from the last known point, predicts h steps later.\nBoth libraries accept blanks as they are."),
    ("Neural sequence: RNN, LSTM", sk_seq,
     "the readings in order (blank = 0 plus a 'blank' flag),\ntime of day, other vitals",
     "a value for every slot",
     "Imputation: reads the sequence in both directions.   Forecast: reads the past only, then fills\nthe held-back part in one pass, without feeding its own answers back."),
    ("Latent / state-space: hidden Markov model, nonlinear state-space model, RS-DPF, particle filter", sk_state,
     "the readings; a blank is simply a step with no reading",
     "an estimated level for every slot",
     "Imputation: filter forward, then smooth backward.   Forecast: filter to 'now', then let the level drift on.\nNonlinear state-space keeps values between a floor and a ceiling; the particle filter keeps them positive."),
    ("Bayesian / irregular time: GRU-ODE-Bayes, CD-Gamma-DGLM", sk_irregular,
     "the readings and the exact time of each",
     "an estimate (and its uncertainty)\nfor every slot",
     "Imputation: forwards and backwards.   Forecast: forwards only.\nCD-Gamma-DGLM can only produce positive values."),
    ("Signal decomposition: OSSA", sk_ssa,
     "the log of the readings, with blanks",
     "a value for every blank",
     "Imputation: refill the blanks, rebuild, repeat until they settle.   Forecast: carry the repeating parts forward.\nWorks on the log of HRV, so results stay positive."),
    ("Hybrid / research: physiology-informed neural ODE, gap-aware state-space transformer (our design)", sk_hybrid,
     "the readings, time of day, other vitals\n(vitals for imputation only)",
     "a value for every slot",
     "Neural ODE: integrates forwards and backwards from the readings either side.   Our transformer: remembers\nonly real readings and how long ago they were. Imputation both ways, forecast past only."),
    ("Foundation model: TimesFM 3", sk_found,
     "the readings before the gap (imputation also uses the readings after it)",
     "a value for every slot",
     "Not trained on this cohort. Imputation: forecast across each gap from both sides and blend.\nForecast: the ordinary forecast from the past."),
    ("References (not models): the bar every model must beat", sk_refs,
     "the readings either side (straight line),\nthe last reading, the patient's median",
     "a value for every slot",
     "Imputation: straight line.   Forecast: last reading repeated, and the patient's median."),
]
fig, ax = canvas(14, 3.7 * len(FAMILIES), len(FAMILIES))
for a_, (title, sk, inp, out, note) in zip(ax, FAMILIES):
    frame(a_, title, inp, out, sk, note)
fig.suptitle("What goes in, how each family of models works, and what comes out. The score is the same for all: "
             "average distance from the real value (MAE)", y=.998, fontsize=11.5)
fig.tight_layout(h_pad=0.6)
save(fig, "13d-model-families")


# ============================================================ 13e classification methods
def scatter2(ax, x0, x1, boundary=None, point=False, circle=False):
    r = np.random.default_rng(3)
    ny = np.clip(r.normal(0, 1, (14, 2)) * [3, 3] + [x0 + 11, 19], [x0, 6], [x0 + 36, 27])
    nn = np.clip(r.normal(0, 1, (14, 2)) * [3, 3] + [x0 + 25, 11], [x0, 6], [x0 + 36, 27])
    ax.plot(nn[:, 0], nn[:, 1], "o", color=BLUE, ms=4)
    ax.plot(ny[:, 0], ny[:, 1], "o", color=ORANGE, ms=4)
    if boundary == "line":
        ax.plot([x0 + 14, x0 + 22], [5, 26], color=FILL, lw=1.6)
    if boundary == "curve":
        t = np.linspace(0, 1, 30)
        ax.plot(x0 + 14 + 9 * t + 3 * np.sin(5 * t), 5 + 21 * t, color=FILL, lw=1.6)
    if point:
        ax.plot(x0 + 17, 15, "*", color=INK, ms=11)
    if circle:
        ax.add_patch(Circle((x0 + 17, 15), 6.5, fc="none", ec=FILL, lw=1.3, ls="--"))


def sk_weights(ax, x0, x1):
    ws = [3, -2, 4, 1, -3]
    for i, wv in enumerate(ws):
        ax.add_patch(Rectangle((x0 + 2 + i * 4, 15), 3, wv * 1.6, fc=ORANGE if wv > 0 else BLUE, alpha=.8, ec="none"))
    ax.plot([x0 + 1, x0 + 22], [15, 15], color=INK2, lw=.8)
    ax.text(x0 + 11, 4, "each test value gets a weight;\nadd them up", ha="center", fontsize=8, color=INK2)
    arrow(ax, x0 + 24, 15, x0 + 28, 15)
    t = np.linspace(-6, 6, 30)
    ax.plot(x0 + 28 + (t + 6) * .8, 6 + 18 / (1 + np.exp(-t)), color=FILL, lw=1.6)
    ax.text(x0 + 33, 2, "sum -> chance of yes", ha="center", fontsize=8, color=INK2)


def sk_trees(ax, x0, x1):
    tree(ax, x0 + 10, 26, .9)
    ax.text(x0 + 10, 7, "'is this value above a cut-off?'\nchained to a yes or no", ha="center", fontsize=8, color=INK2)
    for k, cx in enumerate((x0 + 26, x0 + 33, x0 + 40)):
        tree(ax, cx, 22, .3)
    ax.text(x0 + 33, 13, "many trees vote", ha="center", fontsize=8, color=INK2)


def sk_nb(ax, x0, x1):
    t = np.linspace(-3, 3, 40)
    ax.plot(x0 + 11 + t * 3.2 - 5, 5 + 17 * np.exp(-t ** 2 / 1.3), color=BLUE, lw=1.6)
    ax.plot(x0 + 11 + t * 3.2 + 5, 5 + 17 * np.exp(-t ** 2 / 1.3), color=ORANGE, lw=1.6)
    ax.plot([x0 + 14] * 2, [3, 24], color=INK, lw=1, ls="--")
    ax.text(x0 + 14, 26, "this patient's value", ha="center", fontsize=8, color=INK2)
    ax.text(x0 + 32, 14, "for each test: how typical is\nthis value for yes patients,\nand for no patients?\nCombine the answers", ha="center",
            fontsize=8, color=INK2, va="center")


def sk_check(ax, x0, x1):
    ax.plot([x0 + 2, x0 + 40], [14, 14], color=GREY, lw=2)
    ax.text(x0 + 21, 8, "ignores every input: always the\nmost common answer (no)", ha="center", fontsize=8, color=INK2)
    ax.text(x0 + 21, 21, "AUC = 0.5 exactly", ha="center", fontsize=8.5, color=INK2)


MEETH = [
    ("Add up weighted tests: logistic regression, sparse logistic regression, linear discriminant analysis, straight support vector machine",
     sk_weights, "Draws a straight dividing line (or weighted sum) between yes and no patients. 'Sparse' sets some weights to zero."),
    ("Chains of questions: decision tree, random forest, extra trees, gradient boosting, XGBoost",
     sk_trees, "A tree asks 'above a cut-off?' repeatedly. Forests average many trees; boosting adds small trees that fix earlier mistakes."),
    ("Nearest neighbours",
     lambda a, x0, x1: scatter2(a, x0, x1, point=True, circle=True),
     "Finds the most similar patients (circle) and gives the answer most of them had."),
    ("Curved support vector machine",
     lambda a, x0, x1: scatter2(a, x0, x1, boundary="curve"),
     "Like the straight version, but the dividing line is allowed to bend."),
    ("Naive Bayes", sk_nb, "Treats each test on its own and multiplies the evidence."),
    ("Always no: the check", sk_check, "Not a real method. It confirms the scoring: it must land on 0.5."),
]
fig, ax = canvas(14, 3.5 * len(MEETH), len(MEETH))
for a_, (title, sk, note) in zip(ax, MEETH):
    frame(a_, title, "one patient's enrolment tests: lung function, walk test, age, sex, BODE, CAT, days monitored",
          "a chance of yes, 0 to 1 (yes = a dated exacerbation during monitoring)", sk, note)
fig.suptitle("How each kind of classification method turns enrolment tests into a chance of yes. "
             "The score is the same for all: AUC", y=.998, fontsize=11.5)
fig.tight_layout(h_pad=0.6)
save(fig, "13e-classification-methods")


# ============================================================ 13f tuning flow
fig, ax = canvas(14, 7.5)
ax.set_xlim(0, 100)
ax.set_ylim(-3, 60)
ax.text(0, 58, "How the settings are tuned without touching the official test", fontsize=13, fontweight="bold", va="top")
ax.text(0, 50.5, "1   A practice set is carved out of what the model is allowed to see anyway", fontsize=11, va="center")
r2 = np.random.default_rng(7)
ax.text(2, 45, "Imputation", fontsize=9.5, va="center")
ax.add_patch(Rectangle((16, 43), 60, 4, fc=BLUE, alpha=.45, ec="none"))
for x0_, w_, c_ in [(21, 2.5, ORANGE), (33, 4, ORANGE), (52, 2, ORANGE), (66, 3, ORANGE), (27, 2, FILL), (41, 3.5, FILL), (59, 2.5, FILL), (71, 2, FILL)]:
    ax.add_patch(Rectangle((x0_, 43), w_, 4, fc=c_, ec="none"))
ax.text(2, 37.5, "Forecasting", fontsize=9.5, va="center")
ax.add_patch(Rectangle((16, 35.5), 38.4, 4, fc=BLUE, alpha=.45, ec="none"))
ax.add_patch(Rectangle((54.4, 35.5), 9.6, 4, fc=FILL, ec="none"))
ax.add_patch(Rectangle((64, 35.5), 12, 4, fc=ORANGE, ec="none"))
for yy, c_, txt in [(46.5, BLUE, "readings the model learns from"), (42, FILL, "practice set: used to compare settings"),
                    (37.5, ORANGE, "official test: never seen while tuning")]:
    ax.add_patch(Rectangle((79, yy - 1), 2.5, 2, fc=c_, alpha=.45 if c_ == BLUE else 1, ec="none"))
    ax.text(82.5, yy, txt, fontsize=8.5, va="center", color=INK2)
ax.text(16, 32.3, "Imputation: a further 15% of the visible readings is hidden in whole runs.   "
        "Forecasting: the last 20% of the learning part is held back.", fontsize=8.5, color=INK2)
ax.text(0, 26, "2   Many settings are tried; each is scored on the practice set only", fontsize=11, va="center")
box(ax, 2, 12, 17, 9, "the current\ndefault settings\n(always try 1)", fs=9)
arrow(ax, 19.5, 16.5, 24.5, 16.5)
box(ax, 25, 12, 19, 9, "try a setting:\nfit the model,\nscore on the practice set", fs=9, ec=FILL, color=FILL)
arrow(ax, 44.5, 16.5, 49.5, 16.5)
box(ax, 50, 12, 19, 9, "the search suggests\nthe next setting from\nthe scores so far", fs=9)
ax.annotate("", (34.5, 11.7), (59.5, 11.7), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, connectionstyle="arc3,rad=-.3"))
ax.text(47, 6.6, "repeat: 40 tries (8 for the slow models)", ha="center", fontsize=8.5, color=INK2)
arrow(ax, 69.5, 16.5, 74.5, 16.5)
box(ax, 75, 12, 23, 9, "keep the setting with the\nlowest practice error", fs=9, bold=True)
ax.text(0, 2.5, "3   The model is run once more on the official test with its best setting, and compared with its default result",
        fontsize=11, va="center")
ax.text(2, -1.5, "A setting that gives impossible values (not a number, zero or below, or more than 5% above the watch's maximum) is thrown out.",
        fontsize=8.5, color=INK2)
save(fig, "13f-tuning-flow")
