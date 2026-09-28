"""Nonlinear state-space model, solved with the unscented Kalman filter and RTS
smoother from filterpy.

state      x = [level, c1, c2]: a random-walk level (H4) plus a daily rhythm,
           (c1, c2) rotating by 2*pi/period each slot (H5)
reading    y = LOW + (HIGH - LOW) * sigmoid(level + c1) + noise
           The sigmoid keeps every prediction inside (LOW, HIGH): positive and
           bounded (H1), and it flattens near the device's ceiling.
blanks     the filter predicts but skips the update (H8).

impute  : filter forward, RTS smoother back (both sides), mean reading by the
          unscented transform.
forecast: filter to the end of the history, then predict forward only (H9).
Noise levels are set per series from the spread of its readings (Q_SHARE etc.).
"""
import numpy as np

LOW = 0.0
Q_LEVEL, Q_CYCLE, R_SHARE = 0.05, 0.002, 0.5    # shares of the series' step-to-step variance


def _setup(y, t, period, high):
    from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
    w = 2 * np.pi / period
    rot = np.array([[1, 0, 0], [0, np.cos(w), -np.sin(w)], [0, np.sin(w), np.cos(w)]])
    hx = lambda x: np.array([LOW + (high - LOW) / (1 + np.exp(-(x[0] + x[1])))])
    pts = MerweScaledSigmaPoints(3, alpha=0.3, beta=2.0, kappa=0.0)
    ukf = UnscentedKalmanFilter(3, 1, 1.0, hx=hx, fx=lambda x, dt: rot @ x, points=pts)
    ok = ~np.isnan(y)
    p = np.clip((y[ok] - LOW) / (high - LOW), 1e-3, 1 - 1e-3)
    zl = np.log(p / (1 - p))
    dz = np.diff(zl) if ok.sum() > 2 else np.array([0.1])
    a = 2 * np.pi * t[ok] / period
    c = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), np.cos(a), np.sin(a)]), zl, rcond=None)[0]
    a0 = 2 * np.pi * t[0] / period                    # rhythm at the first slot
    amp = np.hypot(c[1], c[2])
    ph = np.arctan2(c[2], c[1])
    ukf.x = np.array([c[0], amp * np.cos(a0 - ph), amp * np.sin(a0 - ph)])
    v = max(np.var(dz), 1e-3)
    ukf.P = np.diag([v, amp ** 2 + 1e-3, amp ** 2 + 1e-3])
    ukf.Q = np.diag([Q_LEVEL * v, Q_CYCLE * v, Q_CYCLE * v])
    ukf.R = np.array([[R_SHARE * max(np.var(np.diff(y[ok])) if ok.sum() > 2 else 1.0, 1.0)]])
    return ukf, hx, pts


def _mean_reading(hx, pts, x, P):
    s = pts.sigma_points(x, P)
    return float(pts.Wm @ np.array([hx(v)[0] for v in s]))


def _high(series):
    return 1.02 * max(np.nanmax(s.x.hrv_obs) for s in series)


def impute(series, ctx):
    high, out = _high(series), []
    for s in series:
        y = s.x.hrv.to_numpy(float)
        ukf, hx, pts = _setup(y, s.x.t.to_numpy(), ctx["period"], high)
        mu, cov = ukf.batch_filter([None if np.isnan(v) else np.array([v]) for v in y])
        xs, ps, _ = ukf.rts_smoother(mu, cov)
        pred = np.array([_mean_reading(hx, pts, xs[i], ps[i]) for i in range(len(y))])
        out.append(np.where(np.isnan(y), pred, y))
    return out


def forecast(history, horizons, ctx):
    high, out = _high(history), []
    for s, h in zip(history, horizons):
        y = s.x.hrv_obs.to_numpy(float)
        ukf, hx, pts = _setup(y, s.x.t.to_numpy(), ctx["period"], high)
        mu, cov = ukf.batch_filter([None if np.isnan(v) else np.array([v]) for v in y])
        ukf.x, ukf.P = mu[-1], cov[-1]
        preds = []
        for _ in range(h):
            ukf.predict()
            preds.append(_mean_reading(hx, pts, ukf.x, ukf.P))
        out.append(np.array(preds))
    return out
