"""Step 1: the raw watch export -> one 1-minute table per patient (data/wearable/<pid>.csv.gz).

    python 01_build_wearable.py            # build every patient folder in the export
    python 01_build_wearable.py --check    # rebuild in memory and compare with the files on disk

Export layout read: <export>/<pid>/<vital>/<pid>_<vital>.csv (common.RAW). For another export, edit
POINT and INTERVAL below: which file, which time column, which value column.

Rules (each one checked against the original build, all 40 patients identical):
  - every timestamp is floored to the minute; the table has one row for EVERY minute from the
    patient's first to last reading (blank where nothing was recorded)
  - point readings (heart rate, HRV, temperature, SpO2) in the same minute are averaged
  - an interval (a sleep block, a steps count) puts its values on its start minute and sets a
    0/1 flag on every minute from its start to its end, both included
  - the same interval exported twice (same start minute): sleep minutes are added, steps keep the larger
  - readings before EARLIEST are dropped as device-clock errors
"""
import argparse

import numpy as np
import pandas as pd

from common import RAW, WEAR

EARLIEST = pd.Timestamp("2025-01-01")   # c016 has 48 readings dated 2022-07-20; the earliest real one is 2025-08
POINT = {         # file: (time column, value column, table column)
    "heartrate": ("logDateTime", "lastRate", "hr"),
    "hrv": ("createdTime", "hrvValue", "hrv"),
    "temperature": ("createdTime", "temperature", "temp"),
    "spo2": ("createdTime", "spo2Value", "spo2"),
}
INTERVAL = {      # file: (start column, end column, {value column: table column}, duplicates, flag column)
    "sleep": ("logDateTime", "logEndTime",
              {"deepSleep": "deep_sleep", "lightSleep": "light_sleep", "almostAwake": "almost_awake"}, "sum", "sleep_active"),
    "steps": ("logDateTime", "logEndTime", {"steps": "steps"}, "max", "steps_active"),
}
COLUMNS = ["p_id", "time", "age", "sex", "hr", "hrv", "temp", "deep_sleep", "light_sleep", "almost_awake",
           "sleep_active", "spo2", "steps", "steps_active"]      # age, sex: blank, kept so the format matches


def _read(pid, vital):
    f = RAW / pid / vital / f"{pid}_{vital}.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def _minute(s):
    return pd.to_datetime(s, format="ISO8601").dt.floor("min")


def build(pid):
    parts, flags = [], {}
    for vital, (tc, vc, col) in POINT.items():
        d = _read(pid, vital)
        if d.empty:
            continue
        t = _minute(d[tc])
        keep = t >= EARLIEST
        parts.append(pd.DataFrame({"time": t[keep], col: pd.to_numeric(d[vc][keep], errors="coerce")})
                     .groupby("time").mean())
    for vital, (sc, ec, cols, how, flag) in INTERVAL.items():
        d = _read(pid, vital)
        if d.empty:
            continue
        s, e = _minute(d[sc]), _minute(d[ec])
        keep = s >= EARLIEST
        d, s, e = d[keep], s[keep], e[keep]
        parts.append(pd.DataFrame({"time": s, **{c: pd.to_numeric(d[v], errors="coerce") for v, c in cols.items()}})
                     .groupby("time").agg(how))
        flags[flag] = pd.DatetimeIndex(np.concatenate([pd.date_range(a, b, freq="min").values for a, b in zip(s, e)])
                                       if len(s) else [])
    M = pd.concat(parts, axis=1)
    span = M.index.append([f for f in flags.values()])
    M = M.reindex(pd.date_range(span.min(), span.max(), freq="min"))
    for flag in ("sleep_active", "steps_active"):
        M[flag] = M.index.isin(flags.get(flag, pd.DatetimeIndex([]))).astype(int)
    M = M.rename_axis("time").reset_index().assign(p_id=pid, age=np.nan, sex=np.nan)
    return M.reindex(columns=COLUMNS)


def same(a, b):
    """True if two tables hold the same minutes and values (floats to 1e-4: the CSV rounds them)."""
    if len(a) != len(b) or (a.time.values != b.time.values).any():
        return False
    for c in COLUMNS[2:]:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        if not (np.isclose(x, y, rtol=0, atol=1e-4) | (np.isnan(x) & np.isnan(y))).all():
            return False
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with data/wearable instead of writing")
    a = ap.parse_args()
    pids = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    WEAR.mkdir(parents=True, exist_ok=True)
    bad = []
    for pid in pids:
        M = build(pid)
        if a.check:
            f = WEAR / f"{pid}.csv.gz"
            if not f.exists() or not same(pd.read_csv(f, parse_dates=["time"], low_memory=False), M):
                bad.append(pid)
        else:
            M.to_csv(WEAR / f"{pid}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})   # mtime 0: same bytes every run
        print(f"{pid}: {len(M):,} minutes", flush=True)
    if a.check:
        print(f"{len(pids) - len(bad)} of {len(pids)} identical" + (f"; differ: {', '.join(bad)}" if bad else ""))
        raise SystemExit(1 if bad else 0)
