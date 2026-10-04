# Models and the maths behind them

Every equation here is what the code does, not a textbook version. File names point to the code.
Section 9 lists every setting you can change and how. Section 10 has the pipeline-level settings.

Contents: [1 Notation](#1-notation) · [2 From export to modelling data](#2-from-export-to-modelling-data) ·
[3 Shared preprocessing](#3-shared-preprocessing) · [4 Scores](#4-scores) ·
[5 References](#5-references) · [6 The 14 models](#6-the-14-models) ·
[7 Exacerbation classification](#7-exacerbation-classification) · [8 Tuning, SHAP, ablation](#8-tuning-shap-and-ablation) ·
[9 Changing a setting](#9-changing-a-setting) · [10 Pipeline settings](#10-pipeline-settings)

---

## 1 Notation

| symbol | meaning |
|---|---|
| $p$ | patient |
| $t$ | 10-minute slot index; $P$ slots per cycle: $P=144$ (10-minute data, one day) or $P=7$ (daily data, one week) |
| $y_t$ | HRV in slot $t$ as the watch reports it (SDNN, no unit); blank if missing |
| $o_t\in\{0,1\}$ | 1 if $y_t$ is a real reading the model is allowed to see |
| $\varphi_t = 2\pi\,(t \bmod P)/P$ | time of day (or day of week) as an angle; models get $(\sin\varphi_t,\cos\varphi_t)$ |
| $u_t$ | the other vitals in slot $t$: heart rate, temperature, steps, share of the slot asleep, share in a steps interval |
| $z_t$ | standardised log HRV (section 3) |
| $\hat y_t$ | a model's estimate |

---

## 2 From export to modelling data

### 2.1 The 1-minute table (`01_build_wearable.py`)
Each timestamp is floored to the minute, $\lfloor\tau\rfloor_{\text{min}}$. For a point signal (heart rate, HRV,
temperature, SpO2) the value in minute $m$ is the mean of the readings that fall in it. An interval signal
(a sleep block or a steps count, start $a$, end $b$) puts its values on minute $\lfloor a\rfloor$ and sets a
flag $f_m = 1$ for $\lfloor a\rfloor \le m \le \lfloor b\rfloor$. If the export has the same interval twice
(same start minute), sleep minutes are added and steps keep the maximum. The table has a row for every
minute from the patient's first to last reading. Readings before `EARLIEST` are dropped (device-clock errors).

### 2.2 The 10-minute grid (`11_model_data.py`)
The 1-minute table is cut into 10-minute slots, from each patient's first to last HRV reading:

$$y_t = \operatorname{mean}\{\text{HRV readings in slot } t\},\qquad \text{hr}_t = \operatorname{mean}\{\text{heart rate}\},\qquad \text{sleep\_frac}_t = \tfrac{1}{10}\textstyle\sum_{m\in t} f^{\text{sleep}}_m$$

and likewise for temperature, SpO2 (means), steps (sum) and steps_active_frac. Nothing is filled in.

### 2.3 Missing runs and segments
For a blank slot, $r_t$ = length of the run of consecutive blank slots it sits in ($r_t=0$ if observed).
A **segment** is a maximal stretch with no run longer than $X$:

$$\text{cut at every } t \text{ with } r_t > X,\qquad \text{keep pieces trimmed to their first and last reading, with length} \ge 6\cdot\texttt{MIN\_SEG\_HOURS}\ \text{slots.}$$

$X = 18$ (180 minutes). Models only ever see one segment at a time.

### 2.4 Imputation masks (the official test)
Inside each segment with $n$ observed readings, $m = \operatorname{round}(0.2\,n)$ are hidden:

* **random**: $m$ observed slots drawn uniformly without replacement (seed 0).
* **block**: repeat until $m$ are hidden: draw a run length $L$ from the pool of real blank-run lengths inside
  segments, a start $a$ uniformly, and hide every observed slot in $[a, a+L)$ (at most 1000 draws).

The hidden readings are blanked in the model's input and are the only positions scored.

### 2.5 Forecast split
In each segment of $n$ slots, slots $t < \lfloor 0.8\,n \rfloor$ are **train** and the rest are **test**.
Blanks in the train part are filled with a straight line before forecasting (`--imputer linear`); models that
handle blanks themselves read the unfilled copy. For **daily** forecasting, each patient-day with at least one
HRV reading gets $y_d = \operatorname{median}$ of that day's slots, and the last 20% of a patient's days are test.

---

## 3 Shared preprocessing
**Per-patient log standardisation** (each patient has their own baseline, H10). From the readings a model
may see:

$$z_t = \frac{\log y_t - \mu_p}{\sigma_p},\qquad \mu_p = \operatorname{mean}(\log y),\ \sigma_p = \operatorname{sd}(\log y)\ \ (\ge 10^{-3}),\qquad \hat y_t = \exp(\hat z_t\,\sigma_p + \mu_p).$$

Working in log space keeps every back-transformed value positive (H1). Used by the trees, RNN, LSTM, gast,
HMM, RS-DPF, GRU-ODE-Bayes and the neural ODE (`models/data.py: zstats, to_z, from_z`).

**Vitals** are standardised over all series, $\tilde u = (u-\bar u)/s_u$; a blank becomes 0 and a 0/1
"seen" flag is added (`cov_matrix`).

---

## 4 Scores
Over the scored positions $S$ (hidden readings for imputation, real readings of the test part for forecasting):

$$\text{MAE} = \frac{1}{|S|}\sum_{t\in S}|\hat y_t - y_t|,\qquad \text{RMSE} = \sqrt{\frac{1}{|S|}\sum_{t\in S}(\hat y_t - y_t)^2}.$$

Also reported: MAE split at $y<120$ and $y\ge 120$ (the watch tops out at 129), and the median of the
per-patient MAEs. A run is **not usable** if any prediction is non-finite or $\le 0$, or more than 5% are
above `HRV_MAX` = 129 (`run_models.py check`, `12_model_results.py`).

---

## 5 References
The bars a model must beat (`models/baselines.py`).

* **Straight line** (imputation): between the nearest readings $y_a$ before and $y_b$ after a gap,
  $\hat y_t = y_a + \frac{t-a}{b-a}(y_b - y_a)$; flat at the ends.
* **Last reading** (forecast): $\hat y_{T+h} = y_{T'}$, the last real reading.
* **Patient's median** (forecast): $\hat y_{T+h} = \operatorname{median}$ of the patient's train readings.

---

## 6 The 14 models
Each model implements `impute(series, ctx)` (fill every blank; readings on both sides of a gap may be used) and
`forecast(history, horizons, ctx)` (predict the test part from the train part only, H9). Settings are read with
`hp(ctx, name, default)`; the tables give the name, default, tuning range and meaning.

### 6.1 Gradient-boosted trees: XGBoost, CatBoost (`models/gbm.py`)
An additive model of $M$ regression trees $f_m$, each fitted to the remaining error of the ones before:

$$F(x) = \sum_{m=1}^{M}\eta\, f_m(x),\qquad \min_{f_m}\sum_i\big(z_i - F_{m-1}(x_i) - f_m(x_i)\big)^2 + \Omega(f_m).$$

XGBoost uses $\Omega(f) = \tfrac{\lambda}{2}\lVert w\rVert^2$ over leaf weights $w$ (plus the
`min_child_weight` limit); CatBoost uses symmetric (oblivious) trees with L2 leaf regularisation. One global
model is fitted across all patients, on standardised log HRV $z$.

**Imputation.** For every slot, the features are
$x_t = (z_{t-1..t-K},\ z_{t+1..t+K},\ z^{\text{prev}}, d^{\text{prev}}, z^{\text{next}}, d^{\text{next}},\ \sin\varphi_t, \cos\varphi_t,\ u_t)$,
where $z^{\text{prev}}$ is the nearest real reading before $t$ and $d^{\text{prev}}$ its distance in slots
(likewise after). Blanks stay blank: both libraries route missing values natively. The model is fitted on the
visible readings and applied to the blanks.

**Forecast (direct multi-step).** Training pairs (origin $o$, step $h$) are drawn at random inside the train part
(`pairs` per series); the target is the real reading $z_{o+h}$. Features use only data up to $o$:

$$x_{o,h} = \big(z_o, \dots, z_{o-k+1},\ \bar z_{\text{last day}},\ \operatorname{sd}(z_{\text{last day}}),\ h,\ \sin\varphi_o, \cos\varphi_o,\ \sin\varphi_{o+h}, \cos\varphi_{o+h},\ u_o\big),\quad k=\min(2K, P).$$

At test time $o$ = the last train slot and $h = 1..H$. No prediction is fed back.

| setting | default | tuned over | meaning |
|---|---|---|---|
| `n_estimators` | 600 | 100–1500 | number of trees $M$ |
| `depth` | 6 | 3–10 | tree depth |
| `lr` | 0.05 | 0.01–0.3 (log) | learning rate $\eta$ |
| `K` | 6 | 3–12 | neighbours each side (imputation); $k=2K$ lags (forecast) |
| `subsample`, `colsample` | 0.8, 0.8 | 0.5–1, 0.4–1 | XGBoost: share of rows / features per tree |
| `min_child_weight`, `reg_lambda` | 1, 1.0 | 1–20, 0.1–20 (log) | XGBoost: leaf size limit, $\lambda$ |
| `l2_leaf_reg` | 3.0 | 1–20 (log) | CatBoost: leaf L2 penalty |
| `pairs` | 3000 | 1000–6000 | forecast: training (origin, step) pairs per series |
| `catboost_gpu` | False | – | CatBoost on GPU (stalls on a shared GPU) |

### 6.2 Neural sequence: RNN, LSTM (`models/rnn.py`, `models/seq.py`)
A recurrent network reads a window of slots in order and carries a hidden state $h_t$:

$$\text{RNN:}\quad h_t = \tanh(W x_t + U h_{t-1} + b).$$
$$\text{LSTM:}\quad \begin{aligned} i_t&=\sigma(W_i x_t + U_i h_{t-1}+b_i),\ \ f_t=\sigma(W_f x_t + U_f h_{t-1}+b_f),\ \ o_t=\sigma(W_o x_t + U_o h_{t-1}+b_o)\\ \tilde c_t&=\tanh(W_c x_t + U_c h_{t-1}+b_c),\ \ c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t,\ \ h_t = o_t\odot\tanh(c_t)\end{aligned}$$

with `layers` stacked layers (dropout between them) and output $\hat z_t = w^\top h_t + b$.

**Imputation (bidirectional).** Input per slot:
$x_t = (z_t o_t,\ o_t,\ \sin\varphi_t,\cos\varphi_t,\ s^{\leftarrow}_t,\ s^{\rightarrow}_t,\ \tilde u_t,\ \text{seen}(u_t))$,
where $s^{\leftarrow}_t = \log(1+\min(\text{slots since the last reading},1000))/7$ and $s^{\rightarrow}_t$ the same
to the next reading. Forward and backward passes are concatenated, $h_t=[\overrightarrow{h}_t;\overleftarrow{h}_t]$.
Training hides a further share `hide` of the visible readings at random and minimises

$$\mathcal L = \frac{\sum_t m_t(\hat z_t - z_t)^2}{\sum_t m_t},\qquad m_t = 1 \text{ if } t \text{ was hidden for training}.$$

The official hidden readings are never seen. Adam, gradient norm clipped at 1.

**Forecast (direct multi-step, causal).** Input at slot $t$ is the previous slot's value and real-reading flag,
plus this slot's time of day: $x_t = (\bar z_{t-1}\,\mathbb 1[t-1\le o],\ o_{t-1}\,\mathbb 1[t-1\le o],\ \sin\varphi_t, \cos\varphi_t)$.
In training, an origin $o$ is drawn in each window and every slot after it is blanked, so the network learns to
predict the whole stretch after $o$ from the past and the clock. At test time the history is followed by $H$
blank slots and read in one pass. The loss is the masked squared error on real readings. The network never
sees its own output.

| setting | default | tuned over | meaning |
|---|---|---|---|
| `hidden` | 64 | 32, 64, 128, 256 | size of $h_t$ |
| `layers` | 2 | 1–3 | stacked layers |
| `dropout` | 0.1 | 0–0.4 | between layers |
| `lr` | 1e-3 | 3e-4–3e-3 (log) | Adam learning rate |
| `window` | 288 | 96, 144, 288, 432 | training window in slots (288 = 2 days) |
| `steps` | 3000 | 1500, 3000, 5000 | training steps |
| `batch` | 32 | 16, 32, 64 | windows per step |
| `hide` | 0.2 | 0.1–0.4 | imputation: extra share hidden while training |

### 6.3 Gap-aware state-space transformer, our design (`models/gast.py`)
A placeholder for the team's own architecture, built from the same training loop as 6.2. The input $x_t$ is
embedded, $e_t = W_e x_t$, then `blocks` times: a gap-aware state-space layer, then a self-attention layer.

**Gap-aware state-space layer.** Per channel $j$, with a learned decay $a_j = \exp(-\operatorname{softplus}(\rho_j))$,
an average that only real readings feed:

$$N_t = \sum_{k\ge 0} a^k\, s_{t-k}\, e_{t-k},\qquad D_t = \sum_{k\ge 0} a^k\, s_{t-k},\qquad \text{state}_t = \Big[\frac{N_t}{\max(D_t,0)+10^{-6}},\ \log(1+\max(D_t,0))\Big],$$

where $s_t$ is the real-reading flag. The first part holds the last real information; the second says how stale it
is. Both sums are computed as an FFT convolution; the $\max(\cdot,0)$ removes FFT rounding below zero, which used to
make the division blow up after long gaps. Imputation adds the same sums run backwards in time. The layer output
$W_o\,\text{state}_t$ is added to $e_t$.

**Self-attention** (a standard transformer encoder layer): $\operatorname{softmax}(QK^\top/\sqrt{d/\text{heads}})V$ per
head, then a 2-layer feed-forward block, each with a residual connection and layer norm; a causal mask when
forecasting. Output $\hat z_t = w^\top h_t$. Training, imputation and forecasting as in 6.2.

| setting | default | tuned over | meaning |
|---|---|---|---|
| `d` | 64 | 32, 64, 128 | channels |
| `heads` | 4 | 2, 4 | attention heads |
| `blocks` | 2 | 1–3 | (state-space + attention) blocks |
| `dropout`, `lr`, `window`, `steps`, `batch`, `hide` | as 6.2 | as 6.2 | as 6.2 |

### 6.4 Hidden Markov model (`models/hmm.py`)
$S$ hidden states with transition matrix $A_{ij} = P(q_t=j\mid q_{t-1}=i)$ and Gaussian emissions on standardised log HRV:

$$z_t \mid q_t=j \sim \mathcal N(\mu_j, \sigma_j^2).$$

One global HMM is fitted with EM (Baum–Welch, `hmmlearn`, up to 200 iterations) on the unbroken runs of real readings.
Inference uses those parameters in a forward–backward pass where a blank contributes no evidence
($\log p(z_t\mid q_t)=0$):

$$\alpha_t(j) = p(z_t\mid j)\sum_i \alpha_{t-1}(i)A_{ij},\quad \beta_t(i) = \sum_j A_{ij}\,p(z_{t+1}\mid j)\,\beta_{t+1}(j),\quad \gamma_t(j)\propto\alpha_t(j)\beta_t(j).$$

**Imputation:** $\hat z_t = \sum_j \gamma_t(j)\,\mu_j$. **Forecast:** from the filtered state $\pi_T \propto \alpha_T$,
$\pi_{T+h} = \pi_T A^h$ and $\hat z_{T+h} = \pi_{T+h}^\top \mu$.

| setting | default | tuned over | meaning |
|---|---|---|---|
| `states` | 3 | 2–10 | number of hidden states $S$ |
| `min_covar` | 1e-3 | 1e-4–1e-1 (log) | floor on $\sigma_j^2$ |

### 6.5 Nonlinear state-space model (`models/nlssm.py`)
State $x_t = (\ell_t, c_{1,t}, c_{2,t})$: a drifting level and a rotating daily rhythm.

$$x_t = G x_{t-1} + w_t,\quad G = \begin{pmatrix}1&0&0\\0&\cos\omega&-\sin\omega\\0&\sin\omega&\cos\omega\end{pmatrix},\ \omega = \tfrac{2\pi}{P},\quad w_t\sim\mathcal N(0,Q)$$
$$y_t = \text{HIGH}\cdot\frac{1}{1+e^{-(\ell_t + c_{1,t})}} + v_t,\qquad v_t\sim\mathcal N(0,R),\qquad \text{HIGH} = \texttt{high\_mult}\cdot\max y.$$

The sigmoid keeps every reading between 0 and HIGH (H1). With $v$ = variance of the step-to-step change of
$\operatorname{logit}(y/\text{HIGH})$: $Q = \operatorname{diag}(q_\ell v,\ q_c v,\ q_c v)$, $R = r\cdot\operatorname{var}(\Delta y)$.
The initial level and rhythm come from a least-squares fit of $a + b\cos\varphi + c\sin\varphi$.

Solved with the unscented Kalman filter (scaled sigma points, $\alpha=1,\ \beta=2,\ \kappa=0$, so all mean weights are
$\ge 0$ and the mean stays in range); a blank skips the update step. **Imputation:** Rauch–Tung–Striebel smoother,
then the unscented mean of the reading. **Forecast:** predict steps only.

| setting | default | tuned over | meaning |
|---|---|---|---|
| `q_level` | 0.05 | 1e-3–0.5 (log) | level noise, share of $v$ |
| `q_cycle` | 0.002 | 1e-4–0.05 (log) | rhythm noise, share of $v$ |
| `r_share` | 0.5 | 0.1–2 | reading noise, share of $\operatorname{var}(\Delta y)$ |
| `high_mult` | 1.02 | 1.01–1.3 | ceiling of the sigmoid, as a multiple of the highest reading |

### 6.6 Particle filter and smoother (`models/pf.py`)
On log HRV $x$, fitted per series:

$$\mu_t = a + b\cos\varphi_t + c\sin\varphi_t,\qquad x_t - \mu_t = \rho\,(x_{t-1}-\mu_{t-1}) + \sigma\varepsilon_t,\qquad \log y_t = x_t + \tau\eta_t.$$

$a,b,c$ by least squares on the readings; $\rho$ = lag-1 correlation of the residual (clipped to $[0, 0.99]$);
with $v$ the residual variance and $s$ = `sigma_share`: $\sigma^2 = s\,v\,(1-\rho^2)$, $\tau^2 = (1-s)\,v$.
A bootstrap particle filter with $N$ particles (`particles` library); a blank has likelihood 1.
**Imputation:** backward sampling of $M$ smoothed paths, $\hat y_t = \frac{1}{M}\sum_m \exp(x^{(m)}_t + \tau^2/2)$.
**Forecast:** the final particles are pushed through the model, $\hat y_{T+h} = \sum_i w_i\exp(x^{(i)}_{T+h}+\tau^2/2)$.

| setting | default | tuned over | meaning |
|---|---|---|---|
| `n_part` | 1000 | 500, 1000, 2000 | particles $N$ |
| `n_paths` | 100 | 50, 100, 200 | imputation: smoothed paths $M$ |
| `sigma_share` | 0.7 | 0.4–0.9 | share of residual variance given to the level, $s$ |

### 6.7 RS-DPF: regime-switching differentiable particle filter (`models/rsdpf.py`)
After Li et al. 2023. On standardised log HRV, with $K$ regimes $s_t$ (a Markov chain with $\Pi = \operatorname{softmax}$ of learned logits):

$$x_t = m_{s_t}(t) + \phi_{s_t}\big(x_{t-1} - m_{s_t}(t-1)\big) + \sigma_{s_t}\varepsilon_t,\quad m_s(t) = \ell_s + a\cos\varphi_t + b\sin\varphi_t,\quad z_t = x_t + \tau\eta_t,$$

with $\phi_s = \operatorname{sigmoid}(\cdot)$, $\sigma_s, \tau = \exp(\cdot)$. Filtering with $N$ particles: regimes are proposed uniformly
and reweighted by $K\,\Pi_{s_{t-1}s_t}$; $x$ is drawn with the reparameterisation trick; resampling is soft,
$q = \alpha w + (1-\alpha)/N$ with weights corrected by $w/q$, so gradients flow through it. All parameters are learned by
maximising the filter's log-likelihood estimate $\sum_t \log\big(\sum_i w^{(i)}_t\,p(z_t\mid x^{(i)}_t)\big)$ with Adam.
**Imputation:** the mean of the filter run forwards and on the reversed series. **Forecast:** regimes and levels simulated forward.

| setting | default | tuned over | meaning |
|---|---|---|---|
| `K` | 2 | 2–4 | regimes |
| `alpha` | 0.5 | 0.2–0.9 | soft-resampling mix $\alpha$ |
| `lr` | 0.01 | 3e-3–3e-2 (log) | Adam learning rate |
| `steps` | 1500 | 500, 1000, 1500 | training steps |
| `n_train`, `n_run` | 64, 512 | 32–128, 256–1024 | particles in training / at prediction |
| `window` | 144 | 72, 144, 288 | training window in slots |

### 6.8 GRU-ODE-Bayes (`models/grude.py`, authors' code in `models/vendor/`)
De Brouwer et al. 2019. A hidden state $h(t)$ evolves continuously between readings and jumps at each reading.
The model's belief about the readings is $p(t) = f_\theta(h(t)) = (\mu(t), \log\sigma^2(t))$ for every channel
(HRV $z$ and the standardised vitals).

$$\text{between readings:}\quad \frac{dh}{dt} = (1-z)\odot(n - h),\quad z = \sigma(W_z p + U_z h),\quad n = \tanh(W_n p + U_n (z\odot h)),$$

integrated with Euler steps of one slot. At a reading $x$ (with mask $M$ for which channels are present):

$$\mathcal L_1 = \tfrac12\sum M\Big[\tfrac{(x-\mu)^2}{\sigma^2} + \log\sigma^2 + \log 2\pi\Big],\qquad h \leftarrow \operatorname{GRUCell}\big(\operatorname{ReLU}(W_{\text{prep}}[x, \mu, \log\sigma^2, (x-\mu)/\sigma]),\ h\big),$$

and $\mathcal L_2$ = KL from $\mathcal N(\mu,\sigma^2)$ to $\mathcal N(x, 0.01^2)$; the loss is $\mathcal L_1 + 10^{-4}\mathcal L_2$
(Adam). The prediction at a blank is $\mu(t)$ before any jump. **Imputation:** one model on forward time and one on
reversed time, averaged. **Forecast:** run over the end of the history and keep integrating.

| setting | default | tuned over | meaning |
|---|---|---|---|
| `hidden` | 32 | 16, 32, 64 | size of $h$ |
| `p_hidden` | 32 | 16, 32, 64 | hidden size of $f_\theta$ |
| `prep_hidden` | 8 | 4, 8, 16 | size of the per-channel input layer |
| `lr` | 1e-3 | 3e-4–3e-3 (log) | Adam learning rate |
| `steps` | 1500 | 500, 1000, 1500 | training steps |
| `window` | 288 | 144, 288 | training window in slots |
| `batch` | 16 | 8, 16, 32 | windows per step |

### 6.9 CD-Gamma-DGLM (`models/dglm.py`)
A Bayesian dynamic generalised linear model (West & Harrison) with a Gamma reading and log link, so the mean is always positive.

$$\theta_t = G\theta_{t-1} + \omega_t,\quad \eta_t = F^\top\theta_t = \ell_t + c_{1,t},\quad \mu_t = e^{\eta_t},\quad y_t \sim \operatorname{Gamma}(\text{shape }\nu,\ \text{mean }\mu_t),$$

$G$ as in 6.5, $F = (1,1,0)$. **Filter** (blank: evolve only):

$$a_t = G m_{t-1},\quad R_t = G C_{t-1} G^\top/\delta,\quad f_t = F^\top a_t,\quad q_t = F^\top R_t F.$$

The prior on $\lambda = 1/\mu$ is matched to $(f,q)$: $\lambda\sim\operatorname{Gamma}(\alpha,\beta)$ with $\psi_1(\alpha) = q$ and $\beta = e^{\psi(\alpha)+f}$
($\psi$ digamma, $\psi_1$ trigamma, solved by Newton). The conjugate update with $y_t$ is $\alpha^* = \alpha+\nu$, $\beta^* = \beta + \nu y_t$, giving
$f^* = \log\beta^* - \psi(\alpha^*)$, $q^* = \psi_1(\alpha^*)$, mapped back by linear Bayes:

$$m_t = a_t + R_tF\,\frac{f^*-f_t}{q_t},\qquad C_t = R_t - R_tFF^\top R_t\,\frac{1 - q^*/q_t}{q_t}.$$

**Imputation:** a Rauch–Tung–Striebel smoother (pseudo-inverse of $R_{t+1}$), then $\hat y_t = \exp(f_t + q_t/2)$.
**Forecast:** evolution variance held at $W = \frac{1-\delta}{\delta}GCG^\top$ from the last filtered step, then $a\leftarrow Ga$, $R\leftarrow GRG^\top + W$.
$\nu = \operatorname{clip}\big(\texttt{nu\_scale}/(\tfrac12\operatorname{var}(\Delta\log y)),\ 1,\ 500\big)$ per series.

| setting | default | tuned over | meaning |
|---|---|---|---|
| `delta` | 0.98 | 0.90–0.995 | discount: how fast old data is forgotten |
| `nu_scale` | 1.0 | 0.3–3 (log) | multiplies the Gamma shape $\nu$ (reading noise) |

### 6.10 OSSA, singular spectrum analysis (`models/ossa.py`)
On log HRV $x_1..x_n$ with window $L = \lfloor\min(P\cdot\texttt{embed\_frac},\ n/2)\rfloor$, the trajectory (Hankel) matrix
$\mathbf X_{ij} = x_{i+j-1}$ ($L\times(n-L+1)$) is decomposed, $\mathbf X = \sum_i s_i u_i v_i^\top$. Keeping the leading $r$ terms and averaging
along anti-diagonals gives the reconstruction $\tilde x$.

**Imputation** (Kondrashov & Ghil): start from a straight-line fill and iterate
$x^{(k+1)}_t = \operatorname{clip}(\tilde x^{(k)}_t,\ \min x, \max x)$ at the blanks, readings left as they are, `iters` times.
**Forecast:** on the last `windows` cycles, with $\pi$ = last row of $[u_1..u_r]$ and $\nu^2 = \lVert\pi\rVert^2$, the linear recurrence

$$x_{n+1} = \sum_{j=1}^{L-1} R_j\, x_{n+1-L+j},\qquad R = \frac{1}{1-\nu^2}\,U^{\triangledown}\pi,$$

($U^{\triangledown}$ = the first $L-1$ rows), each step clipped to the observed range. If $\nu^2 \ge$ `nu2_max` the recurrence is unstable
and the forecast is the window's mean level instead.

| setting | default | tuned over | meaning |
|---|---|---|---|
| `rank` | 3 | 2–8 | components kept $r$ |
| `embed_frac` | 1.0 | 0.25–1 | window $L$ as a share of one cycle |
| `iters` | 30 | 10, 30, 60 | imputation iterations |
| `windows` | 4 | 2–8 | forecast: cycles of history used |
| `nu2_max` | 0.95 | 0.8–0.99 | above this, fall back to the mean level |

### 6.11 Physiology-informed neural ODE (`models/pinode.py`)
On standardised log HRV $x$, a homeostatic pull towards a daily set point plus a small network for what that misses:

$$\frac{dx}{dt} = -k\,\big(x - c(t)\big) + g_\theta\big(x, \sin\varphi_t, \cos\varphi_t, u\big),\qquad c(t) = c_0 + c_1\cos\varphi_t + c_2\sin\varphi_t,\quad k = e^{\kappa},$$

with $g_\theta$ a one-hidden-layer tanh network (last layer starts at zero) and $u$ the vitals at the starting reading (imputation only).
**Training:** every pair of consecutive real readings at most `max_gap` slots apart is an initial-value problem
$x(t_i)\to x(t_j)$, solved with RK4 in rescaled time $s\in[0,1]$, $\frac{dx}{ds} = |t_j-t_i|\,\frac{dx}{dt}$, steps of at most one slot;
loss = mean squared error of $x(t_j)$, Adam.
**Imputation:** from the reading before a blank (forwards in time) and the one after it (backwards in time, where the pull still
relaxes towards $c(t)$, the time-reversed version of the same process), combined with weights $1/\lvert t - t_{\text{source}}\rvert$.
**Forecast:** forwards from the last slot.

| setting | default | tuned over | meaning |
|---|---|---|---|
| `hidden` | 32 | 16, 32, 64 | width of $g_\theta$ |
| `max_gap` | 18 | 6, 12, 18 | longest pair used in training, slots |
| `lr` | 3e-3 | 1e-3–1e-2 (log) | Adam learning rate |
| `steps` | 2000 | 1000, 2000, 3000 | training steps |
| `batch` | 512 | 256, 512, 1024 | pairs per step |

### 6.12 TimesFM 3 (`models/timesfm3.py`)
Google's pretrained time-series foundation model (a transformer forecaster, weights `google/timesfm-3.0-pytorch`, non-commercial
licence), used as released: never trained on this cohort, and treated here as a black box $\hat y_{T+1..T+H} = \operatorname{TFM}(y_{T-C+1..T})$
that sees the last $C$ = `context` values (it normalises its input itself; `make_positive` is on).
**Forecast:** the gap-filled history. **Imputation:** for a gap of length $L$ starting at $a$: a forward forecast $f$ from the
`gap_context` values before it and a backward forecast $b$ from the reversed values after it, blended by position:

$$\hat y_{a+i} = (1-w_i)\,f_i + w_i\,b_i,\qquad w_i = \frac{i+1}{L+1},\ i=0..L-1.$$

| setting | default | tuned over | meaning |
|---|---|---|---|
| `context` | 2048 | 128–2048 | forecast: history length read |
| `gap_context` | 512 | 64–1024 | imputation: history read either side of a gap |
| `time_cov` | False | False, True | forecast: give time of day as a covariate |

---

## 7 Exacerbation classification (`12_exac_classify.py`)
**Target.** $y_p = 1$ if patient $p$ has a dated exacerbation in the datasheet; patients with an undated event or no watch data are left out.
**Inputs.** Spirometry, oscillometry and walk-test columns at least 30% filled, plus age, sex, BODE, CAT and days monitored
(the recording span). A "reduced" set drops the oscillometry reference values, the %-predicted columns that are
$100\times$actual/reference, and post-bronchodilator spirometry. Inside every training fold, missing values are filled with the training
median and (for the scale-sensitive methods) standardised.

**Methods**, each giving $\hat p = P(y=1\mid x)$:

| method | rule | settings searched (inner CV) |
|---|---|---|
| logistic regression | $\hat p = \sigma(w^\top x + b)$, minimise log-loss $+\ \lVert w\rVert_2^2/C$ (class-balanced) | $C\in\{0.01,0.1,1,10\}$ |
| sparse logistic | the same with $\lVert w\rVert_1/C$ | same |
| linear discriminant analysis | Gaussian classes with a shared, shrunk covariance; $\hat p$ from Bayes' rule | – |
| naive Bayes | $p(x\mid y) = \prod_j\mathcal N(x_j;\mu_{jy},\sigma^2_{jy})$, Bayes' rule | variance smoothing $10^{-9..-3}$ |
| nearest neighbours | share of yes among the $k$ nearest training patients | $k\in\{3,5,7\}$, uniform/distance |
| decision tree | recursive splits minimising Gini impurity | depth 2–4, leaf 2–4 |
| random forest / extra trees | average of trees on bootstrap samples / with random cut-offs | depth 2, 3, none; leaf 1–4 |
| gradient boosting, XGBoost | additive trees on the log-loss (6.1) | 50/150 trees, depth 1–2, rate 0.05/0.1 |
| SVM, straight / curved | maximum-margin boundary, linear / RBF kernel, Platt-scaled to $\hat p$ | $C$, and $\gamma$ for RBF |
| always no (check) | constant prior | – |

**Nested cross-validation.** 5 outer folds (stratified) score; inside each training part, 3 folds choose the settings.
Repeated 20 times. **AUC** per outer fold, averaged (never pooled across folds):

$$\text{AUC} = \frac{1}{n_1 n_0}\sum_{i:y_i=1}\ \sum_{j:y_j=0}\Big(\mathbb 1[\hat p_i > \hat p_j] + \tfrac12\mathbb 1[\hat p_i = \hat p_j]\Big).$$

PPV = TP/(TP+FP). **Permutation test:** the labels are shuffled $B=200$ times and the whole procedure rerun for the best method;
$p = \#\{\text{AUC}_{\text{shuffled}} \ge \text{AUC}_{\text{observed}}\}/B$.

---

## 8 Tuning, SHAP and ablation

### 8.1 Tuning (`tune_models.py`)
For model $m$ and task, find $\theta^* = \arg\min_\theta \text{MAE}_{\text{practice}}(m,\theta)$ over the ranges in section 6.
The **practice set** never touches the official test:

* imputation: the official hidden readings stay hidden, and a further 15% of the *visible* readings is hidden in whole runs
  (lengths drawn from the real gaps, seed 12345);
* forecast and daily: the last 20% of each series' *train* part is held back (gaps refilled from the kept part only).

The search is TPE (Optuna, seed 0): after a few tries it splits the tries so far at a quantile $\gamma$ of the score into good and bad,
fits densities $\ell(\theta)$ (good) and $g(\theta)$ (bad), and tries next the $\theta$ that maximises $\ell(\theta)/g(\theta)$.
Try 0 is always the defaults. A setting whose predictions are non-finite, $\le 0$ or more than 5% above 129 scores 1000.
Budget: 40 tries; 8 for RS-DPF, the particle filter and GRU-ODE-Bayes (on 60 segments); 10 for TimesFM 3. Imputation is tuned on the
block-style practice set and the same setting is used for both imputation tests. `numbers/tuning.csv` has every chosen setting.

### 8.2 SHAP (`shap_analysis.py`)
For a prediction $f(x)$ the Shapley value of input $j$ is its average contribution over all orders of adding inputs:

$$\phi_j = \sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\Big(v(S\cup\{j\}) - v(S)\Big),\qquad f(x) = \phi_0 + \sum_j\phi_j.$$

For the trees $v$ is computed exactly from the tree structure (TreeSHAP), on the readings the official test scores (up to 4000 rows),
in standardised log HRV units. For the non-tree classifiers it is estimated with KernelSHAP (8 k-means background points, 300 samples).
Reported: $\overline{|\phi_j|}$ per input and summed over groups of inputs.

### 8.3 Ablation
For the models that use the other vitals, with $V$ a vital (or all of them):
$\Delta_V = \text{MAE}_{\text{block}}(\text{fitted without } V) - \text{MAE}_{\text{block}}(\text{all inputs})$. Above zero, $V$ was helping.

---

## 9 Changing a setting
Three ways, from quickest to most permanent:

1. **One run:** `python run_models.py impute xgboost --mask block --params '{"K": 8, "lr": 0.1}'`.
   Goes to `results/summary_custom.csv`, so it never replaces an official result.
2. **Tuned:** `python tune_models.py <task> <model>` writes `results/tuning/<task>/<model>.json`; `--tuned` uses it.
   To try a different range, edit that model's line in `SPACES` in `tune_models.py` (and its `DEFAULTS` entry if the default moves).
3. **New default:** change the constant at the top of the model's file (e.g. `K = 6` in `models/gbm.py`) and the matching
   `DEFAULTS` entry in `tune_models.py`, then rerun the `models` stage.

Every name in the tables above is accepted by `--params`. Settings a model does not know are ignored.

## 10 Pipeline settings

| setting | where | value | meaning |
|---|---|---|---|
| data location | `common.py` (`COHORT_DATA`, `COHORT_EXPORT`, `COHORT_SHEET`) | `data/`, `copd`, `COPDAI_DATASHEET_01.xls` | where the cohort's files are |
| `PID` | `common.py` | `^c\d{3}$` | what a patient ID looks like |
| `EARLIEST` | `01_build_wearable.py` | 2025-01-01 | readings before this are clock errors |
| `X` | `11_model_data.py` | 18 slots | longest blank run inside a segment |
| `MIN_SEG_HOURS` | `11_model_data.py` | 24 | shortest segment |
| `MASK_PCT`, `TRAIN_PCT`, `SEED` | `11_model_data.py` | 20, 80, 0 | hidden share, train share, random seed |
| `HRV_MAX` | `models/data.py` | 129 | the watch's ceiling, used by the plausibility checks |
| `COVS` | `models/data.py` | hr, temp, steps, sleep_frac, steps_active_frac | the other vitals models may use |
| `VAL_PCT`, `VAL_SEED`, `TRIALS` | `tune_models.py` | 0.15, 12345, 40/8/10 | practice set and budget |
| `OUTER`, `INNER`, repeats, permutations | `12_exac_classify.py` | 5, 3, 20, 200 | classification design |
