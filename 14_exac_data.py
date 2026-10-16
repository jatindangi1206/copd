"""Exacerbation classification data: one row per labelled patient-day, and each patient's 10-minute grid.

    python 14_exac_data.py --check     # date-parser self-test on literal cells, no data
    python 14_exac_data.py             # writes model_data_exac/ (read by run_exac.py)

Cohorts: COPD (data/wearable, this repo) and AIIMS = a-series (adults) + p-series (children), 1-minute
masters in ../final_data/aiims (or $FINAL_DATA/aiims). ACHE is left out: it has no exacerbation record.
An exacerbation is an episode of one or more days that can come back in the same patient, so every
positive day carries its episode id.

Labels, per calendar day as exported (no clock change):
  COPD   datasheet "exacerbation" sheet (codings.load_exacerbations): each dated day = 1.
  AIIMS  adult sheet (a-series): every day of each episode (one date, or a range) = 1.
         paediatric sheet (p-series): the onset day ("Exacerbation From") and the day after = 1
         (PAED_DAYS; the user, 16 Oct 2026, for now). The visit date only checks the reading.
         An a- or p-series patient in neither sheet had no exacerbation recorded: every day = 0.
  Every other day = 0. A patient with an event whose date cannot be read (COPD: undated in the
  datasheet; AIIMS: a sheet entry in dropped.csv) is left out: some of their 0s would be wrong.
A patient-day is a sample if it has at least one HRV reading.

Sheet traps (adult sheet of 14 Sep 2026, paediatric of 23 May 2026):
  adult text    dd-mm-yyyy; episodes split by ";" or ","; a range is two dates ("a to b", "a - b",
                "a- b") or "19 to 20-11-2025".
  adult Excel   date cells are read as stored: that sheet was typed in a dd-mm locale (its cells with a
                day above 12, which cannot be swapped dates, read correctly).
  paed. Excel   date cells have day and month swapped (read as stored, onsets fall after their own
                visits; swapped back, days before). Text is dd/mm/yy or dd/mm/yyyy. A blank id is the
                patient above. A row is kept only if onset <= visit <= onset + MAX_VISIT_DAYS.

model_data_exac/
  grid.csv.gz    pid, cohort, time (10-min slot), hrv, hr, temp, steps, sleep_frac, steps_active_frac
  days.csv       pid, cohort, group (copd | aiims_adult | aiims_paed), date, hrv_n (HRV readings that
                 day), label, episode (pid_startdate on exacerbation days)
  static.csv     COPD clinical table as recorded (clinical_table.csv) minus exac_dated (the target);
                 sex as 1 = male, GOLD one-hot (not ordinal)
  episodes.csv   every parsed episode with its sheet text, to check against the sheets
  dropped.csv    sheet entries not used, and why
"""
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from codings import load_exacerbations, wearable_pids

HERE = Path(__file__).resolve().parent
SHEETS = HERE / "drive-download-20261016T072544Z-1-001"
ADULT = SHEETS / "Adult_Exacerbation_LastUpdated(14_09_26).xlsx"
PAED = SHEETS / "Peadiatric_Exacerbation_LastUpdated(23052026).xlsx"
AIIMS = Path(os.environ.get("FINAL_DATA", HERE.parent / "final_data")) / "aiims"
COPD = HERE / "data" / "wearable"
OUT = HERE / "model_data_exac"
MAX_VISIT_DAYS = 45               # paediatric: a visit later than this after the onset is not trusted
PAED_DAYS = 2                     # paediatric episode: the onset day and the next day (user, 16 Oct 2026)
YEARS = (2025, 2026)              # a date outside these years is a typo, not an event
COLS = ["time", "hr", "hrv", "temp", "steps", "sleep_active", "steps_active"]
DMY = re.compile(r"(\d{1,2})-(\d{1,2})-(\d{4})")


def ts(day, month, year):
    try:
        t = pd.Timestamp(int(year), int(month), int(day))
    except ValueError:
        return pd.NaT
    return t if YEARS[0] <= t.year <= YEARS[1] else pd.NaT


def adult_cell(cell):
    """One adult 'Exacerbation duration' cell -> ([(start, end), ...], [text not understood])."""
    if isinstance(cell, datetime):                        # Excel date cell, dd-mm locale: as stored
        t = pd.Timestamp(cell).normalize()
        return [(t, t)], []
    eps, bad = [], []
    for part in re.split(r"[;,]", str(cell)):
        part = part.strip()
        if not part:
            continue
        d = [ts(*x) for x in DMY.findall(part)]
        short = re.fullmatch(r"(\d{1,2})\s*to\s*\d{1,2}-\d{1,2}-\d{4}", part, re.I)
        if len(d) == 2:
            s, e = d
        elif len(d) == 1 and short:                       # "19 to 20-11-2025"
            e = d[0]
            s = ts(short.group(1), e.month, e.year) if pd.notna(e) else pd.NaT
        elif len(d) == 1:
            s = e = d[0]
        else:
            s = e = pd.NaT
        if pd.isna(s) or pd.isna(e) or s > e:
            bad.append(part)
        else:
            eps.append((s, e))
    return eps, bad


def paed_date(v):
    """Paediatric date: Excel cells come day/month swapped; text is dd/mm/yy or dd/mm/yyyy."""
    if isinstance(v, datetime):
        return ts(v.month, v.day, v.year)                 # the stored month is the day
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", str(v).strip())
    if not m:
        return pd.NaT
    d, mo, y = (int(g) for g in m.groups())
    return ts(d, mo, y + 2000 if y < 100 else y)


def _check():
    T = pd.Timestamp
    # made-up dates in the sheets' formats
    e, b = adult_cell("05-12-2025; 21-01-2026; 04-02-2026; 27-08-2026 - 11-09-2026")
    assert len(e) == 4 and not b and e[3] == (T("2026-08-27"), T("2026-09-11")), e
    e, b = adult_cell("28-11-2025, 10-02-2026 to 03-03-2026; 14-04-2026- 16-04-2026; 12-06-2026")
    assert len(e) == 4 and e[1] == (T("2026-02-10"), T("2026-03-03")) and e[2] == (T("2026-04-14"), T("2026-04-16")), e
    assert adult_cell("17 to 18-11-2025") == ([(T("2025-11-17"), T("2025-11-18"))], [])
    assert adult_cell("04-01-2026 To 09-01-2026 ") == ([(T("2026-01-04"), T("2026-01-09"))], [])
    assert adult_cell("10-09-2026   ") == ([(T("2026-09-10"), T("2026-09-10"))], [])
    assert adult_cell(datetime(2026, 1, 21)) == ([(T("2026-01-21"), T("2026-01-21"))], [])
    assert adult_cell("No symptoms") == ([], ["No symptoms"])
    assert adult_cell("31-02-2026") == ([], ["31-02-2026"])            # not a date
    assert adult_cell("20-04-2026 to 18-04-2026") == ([], ["20-04-2026 to 18-04-2026"])   # ends before it starts
    assert paed_date(datetime(2026, 10, 3)) == T("2026-03-10")
    assert paed_date("17/03/26") == T("2026-03-17") and paed_date("27/12/2025") == T("2025-12-27")
    assert pd.isna(paed_date("unknown")) and pd.isna(paed_date(datetime(2026, 1, 13)))   # day 13 cannot be a month
    print("ok: adult and paediatric date cells")


def adult_sheet():
    d = pd.read_excel(ADULT, header=None).iloc[1:]
    eps, drop = [], []
    for r in d.itertuples(index=False):                   # 1 id, 2 episodes per the sheet, 3 dates, 5 remark
        if pd.isna(r[1]):
            continue
        pid = str(r[1]).strip().lower()
        got, bad = adult_cell(r[3])
        drop += [dict(source="adult", pid=pid, text=t, why="not a date or a range") for t in bad]
        if pd.notna(r[2]) and len(got) + len(bad) != int(r[2]):
            print(f"  note: {pid}: the sheet says {int(r[2])} episodes, the cell holds {len(got) + len(bad)}")
        eps += [dict(source="adult", pid=pid, start=s, end=e, text=str(r[3]).strip(),
                     remark="" if pd.isna(r[5]) else str(r[5]).strip()) for s, e in got]
    return eps, drop


def paed_sheet():
    d = pd.read_excel(PAED, header=None).iloc[1:]
    d[1] = d[1].ffill()                                   # a blank id is the patient above
    eps, drop = [], []
    for r in d.itertuples(index=False):                   # 1 id, 2 visit, 3 onset, 4 score, 5 diagnosis
        if pd.isna(r[2]) and pd.isna(r[3]):
            continue
        pid = str(r[1]).strip().lower()
        visit, onset = paed_date(r[2]), paed_date(r[3])
        text = f"visit {r[2]}; onset {r[3]}"
        if pd.isna(visit) or pd.isna(onset) or not onset <= visit <= onset + pd.Timedelta(days=MAX_VISIT_DAYS):
            drop.append(dict(source="paediatric", pid=pid, text=text,
                             why=f"read as onset {onset.date() if pd.notna(onset) else '?'}, visit "
                                 f"{visit.date() if pd.notna(visit) else '?'}: not onset <= visit <= onset + {MAX_VISIT_DAYS} d"))
            continue
        eps.append(dict(source="paediatric", pid=pid, start=onset, end=onset + pd.Timedelta(days=PAED_DAYS - 1), text=text,
                        remark="" if pd.isna(r[5]) else str(r[5]).strip()))
    return eps, drop


def grid(path, pid, cohort):
    """10-minute slots from the first to the last HRV reading, as 11_model_data.py does."""
    m = pd.read_csv(path, usecols=COLS)
    m["time"] = pd.to_datetime(m.time, format="ISO8601")  # the exports mix ":59" and ":59.000"
    m.loc[~m.hr.between(0, 300), "hr"] = np.nan          # heart rate kept if 0-300 bpm
    h = m.time[m.hrv.notna()]
    if h.empty:
        return None
    m = m[(m.time >= h.min().floor("10min")) & (m.time <= h.max())].set_index("time")
    r = m.resample("10min")
    g = pd.DataFrame({"hrv": r.hrv.mean(), "hr": r.hr.mean(), "temp": r.temp.mean(),
                      "steps": r.steps.sum(min_count=1), "sleep_frac": r.sleep_active.mean().fillna(0),
                      "steps_active_frac": r.steps_active.mean().fillna(0)})
    return g.reset_index().assign(pid=pid, cohort=cohort)


if __name__ == "__main__":
    _check()
    if "--check" in sys.argv:
        sys.exit()
    OUT.mkdir(exist_ok=True)
    ad, ad_drop = adult_sheet()
    pa, pa_drop = paed_sheet()
    E = load_exacerbations()
    EP = pd.concat([pd.DataFrame(ad + pa),
                    pd.DataFrame({"source": "copd datasheet", "pid": E.pid, "start": E.date.dt.normalize(),
                                  "end": E.date.dt.normalize(), "text": "", "remark": ""})],
                   ignore_index=True)
    DROP = pd.DataFrame(ad_drop + pa_drop, columns=["source", "pid", "text", "why"])

    EP["episode"] = EP.pid + "_" + EP.start.dt.strftime("%Y%m%d")
    episode = {}
    for p_, s_, e_, k_ in zip(EP.pid, EP.start, EP.end, EP.episode):
        for d_ in pd.date_range(s_, e_):
            episode.setdefault((p_, d_), k_)              # overlapping episodes: the first one
    pos = set(episode)
    left_out = set(E.attrs["undated_patients"]) | set(DROP.pid)
    listed = set(EP.pid[EP.source != "copd datasheet"]) | set(DROP.pid)
    files = {f.name.split(".")[0]: f for f in AIIMS.glob("*.csv")}
    aiims = set(files) - left_out                         # every a- and p-series patient
    copd = wearable_pids() - left_out
    DROP = pd.concat([DROP, pd.DataFrame([dict(source="adult/paediatric", pid=p, text="", why="no AIIMS watch file")
                                          for p in sorted(listed - set(files))])], ignore_index=True)

    G = [grid(COPD / f"{p}.csv.gz", p, "copd") for p in sorted(copd)] + \
        [grid(files[p], p, "aiims") for p in sorted(aiims)]
    G = pd.concat([g for g in G if g is not None], ignore_index=True)
    D = G.assign(date=G.time.dt.normalize()).groupby(["pid", "cohort", "date"]).hrv.count().rename("hrv_n").reset_index()
    D = D[D.hrv_n > 0].copy()
    key = list(zip(D.pid, D.date))
    D["label"] = [int(k in pos) for k in key]
    D["episode"] = [episode.get(k, "") for k in key]
    D["group"] = np.where(D.cohort == "copd", "copd", np.where(D.pid.str.startswith("a"), "aiims_adult", "aiims_paed"))
    D = D[["pid", "cohort", "group", "date", "hrv_n", "label", "episode"]]

    S = pd.read_csv(HERE / "clinical_table.csv").drop(columns=["exac_dated", "mmrc_text"], errors="ignore")
    S["sex"] = S.sex.map({"male": 1.0, "female": 0.0})
    S = pd.get_dummies(S, columns=["gold"], dtype=float).set_index("pid").apply(pd.to_numeric, errors="coerce")

    G.to_csv(OUT / "grid.csv.gz", index=False, float_format="%.4g")
    D.to_csv(OUT / "days.csv", index=False)
    S.to_csv(OUT / "static.csv")
    EP.to_csv(OUT / "episodes.csv", index=False)
    DROP.to_csv(OUT / "dropped.csv", index=False)

    print(f"left out, an event without a readable date: {', '.join(sorted(left_out)) or 'nobody'}")
    for r in DROP.itertuples():
        print(f"  dropped {r.source} {r.pid}: {r.text} - {r.why}")
    for c, g in D.groupby("group"):
        pids, have = set(g.pid), set(zip(g.pid, g.date))
        lost = sum(p in pids and (p, d) not in have for p, d in pos)
        print(f"{c}: {g.pid.nunique()} patients, {len(g):,} days with HRV, {g.label.sum()} exacerbation days "
              f"in {g[g.label == 1].episode.nunique()} episodes, {g[g.label == 1].pid.nunique()} patients "
              f"({lost} more exacerbation days have no HRV)")
    print(f"-> {OUT}")

    assert not D.duplicated(["pid", "date"]).any() and not G.duplicated(["pid", "time"]).any()
    assert D[D.label == 1].pid.isin(listed | set(E.pid)).all(), "a positive day for a patient in no sheet"
    assert (D.episode != "").eq(D.label == 1).all(), "every exacerbation day, and only those, has an episode"
    assert not D.pid.isin(left_out).any(), "a left-out patient is in the data"
    print("ok: one row per patient-day and slot; positives only from the sheets; no left-out patient")
