# GAST v1: Gap-Aware State-Space Transformer - design notes

From the user's brief of 17 Oct 2026 (GAST is the user's proposed architecture, not an existing model).
**Locked** = the user answered; **proposed** = Claude's suggestion, not yet confirmed. The shared task,
labels, splits and scoring are in `EXAC_DESIGN.md`; this file covers only GAST.

Nothing here claims a new method or better clinical performance. Every mechanism below comes from the
cited papers; what is ours is the combination, and whether it helps is what the ablations test.

## 0. What the data looks like (checked on 17 Oct 2026, not assumed)

- One row per recorded minute. When the watch is worn, HR arrives every minute. HRV arrives about every
  10 minutes: in one AIIMS patient, 80% of the gaps between HRV minutes are 10 minutes, then 20, 30, 40 ... Temperature,
  SpO2, steps and the sleep and step records come at their own times.
- **Repeated HRV values.** Each HRV value sits once, at its own timestamp. In the raw COPD export only 8 of
  106,618 readings share a timestamp. 4.1% equal the reading before (mostly 10 minutes apart; 76% below
  120; 2% at the device cap of 129), which fits two equal whole-number readings, not one measurement copied
  across rows. Nothing is removed. The minute data never carries an HRV value forward to later minutes.
- Pass 1 (minutes with HR 0-300), pass 2 (sample days with HRV) and the day labels: `EXAC_DESIGN.md`.
- Exacerbation days are about 1% of sample days (around 60 episodes): any model must be small.

## 1. The five papers: what each gives us

| paper | mechanism | taken | not taken |
|---|---|---|---|
| S4 (Gu et al. 2021) | continuous-time state space x'(t) = A x(t) + B u(t), y = C x + D u; exact (zero-order-hold) step: x_k = exp(Δ A) x_{k-1} + A^-1 (exp(Δ A) - I) B u_k; diagonal variants (S4D) stay stable when Re(A) < 0 | a **diagonal A with negative entries** and the **exact step exp(Δt A) for any elapsed time Δt** | the convolution mode (it needs equal steps), the HiPPO normal-plus-low-rank machinery |
| Mamba (Gu & Dao 2023) | selective SSM: the step Δ_t and B_t, C_t depend on the input; h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t x_t, y_t = C_t h_t; parallel scan | **input-dependent write and read** (a gate computed from the token), with the recurrence kept **linear in h** so it runs as a parallel scan | the CUDA kernels and the dependency (a plain PyTorch scan is enough at our sizes); a learned Δ - we use the **real elapsed time** |
| GRU-D (Che et al. 2018) | per variable: mask m, time since its last observation δ; hidden decay γ = exp(-max(0, W δ + b)); input decay of the last value towards the empirical mean | **masks and time-since-last-observation per variable as inputs** (missingness can carry information); memory that fades with elapsed time | the **input decay towards the mean**: it invents values |
| GRU-ODE-Bayes (De Brouwer et al. 2019) | hidden state evolves between observations by an ODE dh/dt = (1 - z)(g - h); at an observation a GRU-Bayes update from the observed dimensions; Gaussian outputs | the split **evolve between observations, correct when they arrive** | the ODE solver (our evolution is linear, so exp(Δt A) is exact and cheap); the Gaussian likelihood (left for an uncertainty version) |
| mTAN (Shukla & Marlin 2021) | learned time embedding φ(t) = [ω_0 t + α_0, sin(ω_k t + α_k)]; attention from fixed reference times to irregular observations, masked per variable | **attention that uses real times**, and **reference (query) times** so attention runs on a fixed, small number of tokens | the VAE / interpolation decoder |

## 2. Critique of the sketch

- **Sound:** state evolution by exp(A Δt) with a stable diagonal A; masks and elapsed times as inputs;
  causal, time-aware attention; a shared encoder with separate heads and masked losses.
- **Change - the update.** The sketch's h_t = h_t^- + g(x, m, Δt, h_t^-) adds an unbounded term that also
  depends on h. Over thousands of minutes it can drift without bound, and because it depends on h it
  forces a step-by-step loop. v1 keeps the recurrence linear in h, with a gate computed from the token
  only (as Mamba does): stable, and computable all at once by a parallel scan. Non-linearity goes between
  layers, as in S4 and Mamba.
- **Possibly redundant - attention.** The state-space memory and attention both reach far back. With about
  60 episodes, attention may only add weights to fit. v1 keeps it to one small layer, and the ablations
  test whether it helps.
- **Decay is not uncertainty.** Agreed: v1 has no uncertainty output. A later version could carry a
  variance per state (Kalman-like), as GRU-ODE-Bayes does.
- **Patient baseline.** No hand-made baseline features (the user's rule). The state that runs through the
  window is a learned, causal memory of the patient's own recent signals; whether a longer history helps
  is an experiment (history length).
- **Missingness artifacts.** The model could learn "watch off = exacerbation". Checks: a version without
  gap features, a control that sees only the masks and times, and scores by wear coverage.
- **Overfitting.** Small widths, few layers, dropout, and the early stopping already locked for networks.

## 3. GAST v1 (locked 17 Oct 2026, see section 6)

**Tokens.** One token per minute that has a heart-rate reading inside the window (pass 1), at
its real timestamp. No grid, no blocks: elapsed time between tokens is real time, so wear gaps of any
length are handled without a threshold. Per token:
- for each of the 7 signals: the value (divided by its training maximum; 0 when not observed), the mask,
  and log(1 + minutes since that signal's last observation in the window);
- Δt = minutes since the previous token;
- time of day (sine and cosine of the clock time; it is part of the timestamp; can be switched off);
- `copd_clinical`: the clinical values set the starting state.

**Layer A - gap-aware selective state space** (per state channel j, N channels):
h_{i,j} = exp(-λ_j Δt_i) h_{i-1,j} + σ(g_j · u_i) (w_j · u_i), then y_i = C h_i + D u_i, a GELU, a
residual connection and layer norm. λ_j > 0 are learned rates (a stable diagonal A = -λ), started spread
log-evenly between 1/(10 minutes) and 1/(window length), so some channels remember minutes and others
days. Linear in h: computed by a parallel scan.

**Layer B - time-aware causal attention.** The state is read at reference times, **every hour** (168 for
7 days; evolved from the last
token before each by exp(-λ (r - t_last)), so no observation is invented). Each reference token attends
only to earlier ones, with a learned bias per head that depends on the real time between them (like
ALiBi). One layer, a few heads.

**Heads.** The reference token at the end of the classified day feeds a logistic head: p_active
(detection) and/or p_onset (prediction, next 3 days). Losses are masked binary cross-entropy: a day
counts for a task only where that task's label is valid (onset: not inside an episode, not the last 3
days of data). With two heads: L = L_onset + α L_detection, α configurable (default 1).

**Training.** As the other networks (`EXAC_DESIGN.md` 6.4): half-positive batches, Adam, early stopping on
held-back training patients (AUROC), flag at 0.5. Plain PyTorch; padding and masks for variable token
counts.

**Configurable, nothing hidden.** History length, state channels, layers, attention on/off, reference
spacing, heads, width, dropout, α, time of day on/off, gap features on/off, the λ starting range. Every run
writes its full configuration next to its results.

## 4. Ablations (from the brief)

| run | question |
|---|---|
| XGBoost / CatBoost (raw minutes, locked) | how far do trees on every raw value go? |
| RNN / LSTM (locked) | an ordinary recurrent memory |
| GAST, state space only | does the gap-aware memory help? |
| GAST, attention only | is the state space needed? |
| GAST without gap features (no masks, times since, Δt) | does knowing the gaps help? |
| missingness-only control (masks and times, no values) | is the model only reading wear patterns? |
| full GAST | does the combination help? |

## 5. Validation added by the brief (applies to every model)

- Lead time for onset prediction: days between the first flag and the onset.
- Uncertainty intervals: 95% intervals by resampling patients (bootstrap), since episodes are few.
- Scores by wear coverage, after long gaps, by how much history a day has, per group and per fold.

## 6. Decisions (locked 17 Oct 2026)

| question | answer |
|---|---|
| what is one token | every minute with a heart-rate reading, at its real time; memory evolves by the real elapsed time |
| one or two heads | both: single-task GAST (detection, prediction) for the comparison with the other models, and a shared two-head GAST (L = L_onset + alpha L_detection) as its own experiment |
| attention reference times | every hour (168 tokens for 7 days) |
| history length | 7 days is the main setting for every model; GAST is also run with 3 and 14 days |

GAST v1 replaces the placeholder `models/gast.py` in the exacerbation experiments.

## 7. Build order (the brief: start simple, test, then add)

1. Data: minute arrays and both label sets (detection, prediction), with self-checks.
2. Runner with always_no and the tree models; check splits, labels and scoring on them first.
3. RNN and LSTM with the learned 10-minute compression.
4. GAST, one piece at a time, each with a small test of its time logic and masks before it is used:
   a. tokens (values, masks, times since, elapsed time) - test: nothing from after the day's end, no
      value where the mask is 0;
   b. the state-space layer alone - test: the scan equals a plain step-by-step loop; a longer gap
      shrinks the old state by exactly exp(-lambda dt);
   c. add hourly reference tokens and causal attention - test: no reference token sees a later one;
   d. the ablations (section 4), then the two-head version, then 3 and 14 days.

## 8. Built (17 Oct 2026)

`models/gast_v1.py` (`python -m models.gast_v1` runs the self-tests) and `run_exac.py` (models `gast`,
`gast_ssm`, `gast_attn`, `gast_nogap`, `gast_maskonly`, `gast2`; `--days 3` / `--days 14`). Defaults:
width 32, 32 state channels, 2 attention heads, dropout 0.1, rates started between 1/(10 minutes) and
1/(the window), attention time bias started at softplus(-3) per hour. All are arguments of `GAST`.

Self-tests: the scan equals a step-by-step loop; token times, masks, times since and elapsed times are
exact; between tokens the state fades by exactly exp(-rate x elapsed minutes); a token late in the day
changes no reading of an earlier hour, before or after attention (every variant); padding changes
nothing; the clinical start and two outputs run; gradients stay finite when the first hours have no
token. That last test was added after `run_exac.py check` found NaN scores: for hours before the first
token the fade was computed from a negative gap, exp(+large) = inf in a branch `torch.where` discards,
whose gradient is still 0 x inf = NaN. The gap is now clamped at 0.

`run_exac.py check` on the COPD set, both tasks: every GAST variant, RNN and LSTM train and score (tiny
training; the scores mean nothing). XGBoost and CatBoost were not run on the Mac (no working library
there); the workstation's `check` is their first run.
