"""Exacerbation data for both experiments: every minute as exported, and the labels per patient-day.

    python 14_exac_data.py --check     # date-parser self-test on made-up cells, no data
    python 14_exac_data.py             # writes model_data_exac/ (read by run_exac.py)

Every decision behind this file: docs/EXAC_DESIGN.md.

Cohorts: COPD (data/wearable, this repo) and AIIMS = a-series (adults) + p-series (children), 1-minute
masters in ../final_data/aiims (or $FINAL_DATA/aiims). ACHE is left out: it has no exacerbation record.

Minutes (pass 1): only minutes with a heart-rate reading of 0-300; every other signal is kept only on
those minutes, as exported - nothing averaged, filled or moved. Every day with at least one such minute
is stored, so days without HRV still serve as context for the days after them.

Episodes, per calendar day as exported:
  COPD      each dated exacerbation in the datasheet (codings.load_exacerbations) is one day.
  adults    every day of each episode in the adult sheet (one date, or a range).
  children  the onset day ("Exacerbation From") only (PAED_DAYS). The visit date only checks the reading.
  An episode with no recorded end is that one day. An a- or p-series patient in neither sheet had no
  exacerbation. A patient with an event whose date cannot be read is left out (some of their normal
  days would be wrong).
Detection:  sample = a day with at least one HRV reading (pass 2); label = the day is inside an episode.
Prediction: sample = a detection sample that is not inside an episode and not among the last HORIZON
            days of the patient's watch data; label = an episode begins on d+1 ... d+HORIZON (the onset
            is each episode's first recorded day).

Sheet traps (adult sheet of 14 Sep 2026, paediatric of 23 May 2026):
  adult text    dd-mm-yyyy; episodes split by ";" or ","; a range is two dates ("a to b", "a - b",
                "a- b") or "19 to 20-11-2025".
  adult Excel   date cells are read as stored: that sheet was typed in a dd-mm locale (its cells with a
                day above 12, which cannot be swapped dates, read correctly).
  paed. Excel   date cells have day and month swapped (read as stored, onsets fall after their own
                visits; swapped back, days before). Text is dd/mm/yy or dd/mm/yyyy. A blank id is the
                patient above. A row is kept only if onset <= visit <= onset + MAX_VISIT_DAYS.

model_data_exac/
  minutes.npy    float32 (stored days, 1440 minutes, 7 signals: CH), blank = NaN
  days.csv       one row per stored day, in the order of minutes.npy: pid, cohort, group, date,
                 hr_minutes, hrv_minutes, sample, label, episode, onset_valid, onset_label, onset_id
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
CH = ["hrv", "hr", "temp", "spo2", "steps", "sleep_active", "steps_active"]   # per minute, as exported
HORIZON = 3                       # prediction: an onset on d+1 ... d+3 (locked 17 Oct 2026)
PAED_DAYS = 1                     # children: the onset day only (locked 17 Oct 2026)
MAX_VISIT_DAYS = 45               # paediatric: a visit later than this after the onset is not trusted
YEARS = (2025, 2026)              # a date outside these years is a typo, not an event
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
        eps.append(dict(source="paediatric", pid=pid, start=onset, end=onset + pd.Timedelta(days=PAED_DAYS - 1),
                        text=text, remark="" if pd.isna(r[5]) else str(r[5]).strip()))
    return eps, drop


def patient_minutes(path):
    """-> (days, minutes (days, 1440, CH), HR minutes per day, HRV minutes per day), or None.
    Only minutes with heart rate 0-300 (pass 1); the other signals only on those minutes; as exported."""
    m = pd.read_csv(path, usecols=["time"] + CH)
    m["time"] = pd.to_datetime(m.time, format="ISO8601")  # the exports mix ":59" and ":59.000"
    m = m[m.hr.between(0, 300)]
    if m.empty:
        return None
    day = m.time.dt.normalize()
    days = pd.DatetimeIndex(day.unique()).sort_values()
    i = days.get_indexer(day)
    a = np.full((len(days), 1440, len(CH)), np.nan, np.float32)
    a[i, (m.time.dt.hour * 60 + m.time.dt.minute).to_numpy()] = m[CH].to_numpy(np.float32)
    return days, a, np.bincount(i, minlength=len(days)), np.bincount(i[m.hrv.notna().to_numpy()], minlength=len(days))


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
    EP["episode"] = EP.pid + "_" + EP.start.dt.strftime("%Y%m%d")
    DROP = pd.DataFrame(ad_drop + pa_drop, columns=["source", "pid", "text", "why"])
    episode = {}
    for p_, s_, e_, k_ in zip(EP.pid, EP.start, EP.end, EP.episode):
        for d_ in pd.date_range(s_, e_):
            episode.setdefault((p_, d_), k_)              # overlapping episodes: the first one
    onsets = {p_: g.sort_values("start")[["start", "episode"]].drop_duplicates("start") for p_, g in EP.groupby("pid")}

    left_out = set(E.attrs["undated_patients"]) | set(DROP.pid)
    listed = set(EP.pid[EP.source != "copd datasheet"]) | set(DROP.pid)
    files = {f.name.split(".")[0]: f for f in AIIMS.glob("*.csv")}
    DROP = pd.concat([DROP, pd.DataFrame([dict(source="adult/paediatric", pid=p, text="", why="no AIIMS watch file")
                                          for p in sorted(listed - set(files))])], ignore_index=True)
    who = [("copd", p, COPD / f"{p}.csv.gz") for p in sorted(wearable_pids() - left_out)] + \
          [("aiims", p, files[p]) for p in sorted(set(files) - left_out)]       # every a- and p-series patient

    rows, arrays = [], []
    for cohort, p, path in who:
        r = patient_minutes(path)
        if r is not None:
            days, a, hr_n, hrv_n = r
            rows.append(pd.DataFrame({"pid": p, "cohort": cohort, "date": days, "hr_minutes": hr_n, "hrv_minutes": hrv_n}))
            arrays.append(a)
    D = pd.concat(rows, ignore_index=True)
    X = np.lib.format.open_memmap(OUT / "minutes.npy", mode="w+", dtype=np.float32, shape=(len(D), 1440, len(CH)))
    k = 0
    for a in arrays:                                      # written one patient at a time: no second copy in memory
        X[k:k + len(a)] = a
        k += len(a)
    X.flush()

    key = list(zip(D.pid, D.date))
    D["group"] = np.where(D.cohort == "copd", "copd", np.where(D.pid.str.startswith("a"), "aiims_adult", "aiims_paed"))
    D["sample"] = D.hrv_minutes > 0
    inside = np.array([k_ in episode for k_ in key])
    D["label"] = (inside & D["sample"]).astype(int)
    D["episode"] = [episode[k_] if (k_ in episode and s_) else "" for k_, s_ in zip(key, D["sample"])]
    last_day = D.groupby("pid").date.transform("max")
    D["onset_valid"] = D["sample"] & ~inside & (D.date <= last_day - pd.Timedelta(days=HORIZON))
    onset_label, onset_id = np.zeros(len(D), int), np.full(len(D), "", object)
    for i, (p_, d_, ok) in enumerate(zip(D.pid, D.date, D.onset_valid)):
        if ok and p_ in onsets:
            o = onsets[p_]
            nxt = o[(o.start > d_) & (o.start <= d_ + pd.Timedelta(days=HORIZON))]
            if len(nxt):                                  # ponytail: the nearest onset; a second one within 3 days is rare
                onset_label[i], onset_id[i] = 1, nxt.episode.iat[0]
    D["onset_label"], D["onset_id"] = onset_label, onset_id
    D = D[["pid", "cohort", "group", "date", "hr_minutes", "hrv_minutes", "sample", "label", "episode",
           "onset_valid", "onset_label", "onset_id"]]

    S = pd.read_csv(HERE / "clinical_table.csv").drop(columns=["exac_dated", "mmrc_text"], errors="ignore")
    S["sex"] = S.sex.map({"male": 1.0, "female": 0.0})
    S = pd.get_dummies(S, columns=[c for c in ["gold"] if c in S], dtype=float).set_index("pid").apply(pd.to_numeric, errors="coerce")

    D.to_csv(OUT / "days.csv", index=False)
    S.to_csv(OUT / "static.csv")
    EP.to_csv(OUT / "episodes.csv", index=False)
    DROP.to_csv(OUT / "dropped.csv", index=False)
    (OUT / "grid.csv.gz").unlink(missing_ok=True)         # the earlier 10-minute grid (replaced by minutes.npy)

    print(f"left out, an event without a readable date: {', '.join(sorted(left_out)) or 'nobody'}")
    for r in DROP.itertuples():
        print(f"  dropped {r.source} {r.pid}: {r.text} - {r.why}")
    print(f"{len(D):,} stored days ({X.nbytes / 1e9:.2f} GB of minutes), {D['sample'].sum():,} with HRV")
    for g, d in D.groupby("group"):
        s, o = d[d["sample"]], d[d.onset_valid]
        print(f"{g}: {d.pid.nunique()} patients | detection: {len(s):,} days, {s.label.sum()} exacerbation days, "
              f"{s[s.label == 1].episode.nunique()} episodes | prediction: {len(o):,} days, {o.onset_label.sum()} "
              f"before an onset, {o[o.onset_label == 1].onset_id.nunique()} onsets")
    print(f"-> {OUT}")

    assert X.shape[0] == len(D) and not D.duplicated(["pid", "date"]).any()
    assert D[D.label == 1].pid.isin(listed | set(E.pid)).all(), "a positive day for a patient in no sheet"
    assert (D.episode != "").eq(D.label == 1).all(), "every exacerbation day, and only those, has an episode"
    assert not (D.onset_valid & (D.label == 1)).any() and not ((D.onset_label == 1) & ~D.onset_valid).any()
    assert not D.pid.isin(left_out).any(), "a left-out patient is in the data"
    hrv_ok = [np.isfinite(X[i, :, 0]).any() for i in np.flatnonzero(D["sample"].to_numpy())[::97]]
    assert all(hrv_ok), "a sample day without HRV in minutes.npy"
    print("ok: one row per patient-day; labels only from the sheets; prediction days outside episodes; "
          "no left-out patient; sample days hold HRV")
