"""Particle filter and particle smoother (bootstrap filter + backward-sampling
smoother from Chopin's `particles` library).

model (log HRV, so readings stay positive - H1)
    mu_t = a + b cos + c sin                    daily rhythm (H5), fitted per series
    x_t - mu_t = RHO (x_{t-1} - mu_{t-1}) + s e  level pulled back to the rhythm (H4)
    log y_t = x_t + tau n                        blank reading: no evidence (H8)
rho, s, tau are set per series from its own readings.

impute  : filter + backward-sampling smoother (both sides): mean of exp(x) at blanks.
forecast: filter to the end of the history, push the particles forward (H9).
"""
import numpy as np

N_PART, N_PATHS = 1000, 100


def _model_class():
    from particles import distributions as dists
    from particles import state_space_models as ssm

    class NanNormal(dists.Normal):
        def logpdf(self, x):
            if np.isnan(x):
                return np.zeros(np.shape(self.loc))
            return super().logpdf(x)

    class LogAR(ssm.StateSpaceModel):
        default_params = {"mu": None, "rho": 0.5, "sigma": 0.2, "tau": 0.1, "sd0": 0.3}

        def PX0(self):
            return dists.Normal(loc=self.mu[0], scale=self.sd0)

        def PX(self, t, xp):
            return dists.Normal(loc=self.mu[t] + self.rho * (xp - self.mu[t - 1]), scale=self.sigma)

        def PY(self, t, xp, x):
            return NanNormal(loc=x, scale=self.tau)
    return LogAR, ssm


def _params(ly, t, period, h=0):
    ok = ~np.isnan(ly)
    a = 2 * np.pi * t / period
    X = np.column_stack([np.ones(len(t)), np.cos(a), np.sin(a)])
    coef = np.linalg.lstsq(X[ok], ly[ok], rcond=None)[0]
    tt = np.r_[t, (t[-1] + np.arange(1, h + 1)) % period]
    af = 2 * np.pi * tt / period
    mu = np.column_stack([np.ones(len(tt)), np.cos(af), np.sin(af)]) @ coef
    r = ly - mu[:len(t)]
    both = ok[1:] & ok[:-1]
    rho = float(np.clip(np.corrcoef(r[:-1][both], r[1:][both])[0, 1], 0.0, 0.99)) if both.sum() > 3 else 0.5
    v = max(np.nanvar(r), 1e-4)
    return dict(mu=mu, rho=rho, sigma=np.sqrt(0.7 * v * (1 - rho ** 2) + 1e-6), tau=np.sqrt(0.3 * v),
                sd0=np.sqrt(v))


def _run(ly, p, ctx, smooth):
    import particles
    LogAR, ssm = _model_class()
    n = 100 if ctx["quick"] else N_PART
    model = LogAR(**p)
    np.random.seed(ctx["seed"])                      # particles draws from numpy's global generator
    pf = particles.SMC(fk=ssm.Bootstrap(ssm=model, data=ly), N=n, store_history=smooth, verbose=False)
    pf.run()
    return pf


def impute(series, ctx):
    out = []
    for s in series:
        ly = np.log(s.x.hrv.to_numpy(float))
        p = _params(ly, s.x.t.to_numpy(), ctx["period"])
        pf = _run(ly, p, ctx, smooth=True)
        paths = np.array(pf.hist.backward_sampling(20 if ctx["quick"] else N_PATHS))   # (T, M)
        pred = np.exp(paths + p["tau"] ** 2 / 2).mean(axis=1)
        y = s.x.hrv.to_numpy(float)
        out.append(np.where(np.isnan(y), pred, y))
    return out


def forecast(history, horizons, ctx):
    rng = np.random.default_rng(ctx["seed"])
    out = []
    for s, h in zip(history, horizons):
        ly = np.log(s.x.hrv_obs.to_numpy(float))
        n = len(ly)
        p = _params(ly, s.x.t.to_numpy(), ctx["period"], h)
        pf = _run(ly, {**p, "mu": p["mu"][:n]}, ctx, smooth=False)
        x, w, mu = pf.X.copy(), pf.W, p["mu"]
        preds = []
        for k in range(h):
            x = mu[n + k] + p["rho"] * (x - mu[n + k - 1]) + p["sigma"] * rng.standard_normal(len(x))
            preds.append(float(w @ np.exp(x + p["tau"] ** 2 / 2)))
        out.append(np.array(preds))
    return out
