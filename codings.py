"""The COPD datasheet's coding scheme, in one place.

    from codings import load_baseline, load_exacerbations, YES_NO, DECODE

The sheet is numeric but NOT model-ready, and the traps are not obvious:

  * Yes=1, No=2, None=0 - so raw values make "No" twice "Yes". Every yes/no
    column has to be remapped to 1/0 or a model reads a false ordering.
  * ND / NK / NA are real strings meaning Not Done / Not Known / Not Available.
    They are missing, not categories.
  * BP_mmHg_SYS and BP_mmHg_DIA are SWAPPED in the sheet - the column labelled
    systolic holds 64-98 and the one labelled diastolic holds 101-153, in all
    32 rows that have both. load_baseline() un-swaps them.
  * mMRC is roman text with half-grades: "0-I", "I", "I-II", "II", "II-III",
    "III". Mapped to 0.5/1/1.5/2/2.5/3 - a range is taken at its midpoint.
  * GOLD A/B/E is NOT ordinal. A and B are both low-exacerbation (they differ
    by symptom burden); E is the high-exacerbation group. Encoded as one-hot
    plus a gold_E risk flag, never as A<B<E.
  * AGE has one impossible value: c025 is recorded as 7, with an adult's body
    (63 kg, 160 cm, BMI 24.6). load_baseline() sets ages outside 18-100 to NaN
    and records them in .attrs["implausible_age"].
  * BODE_INDXTOT_SCORE_0 does not reconcile with its own four components. Scored
    from BMI / FEV1%pred / mMRC / 6MWD it agrees with the sheet in only 4 of the
    18 rows where all four are present, and the sheet's BODE correlates +0.01
    with 6-minute walk distance when 6MWD is one of its inputs. Treat the
    recorded BODE as unverified; bode_reconstructed() recomputes it.
  * AVG_DUR_EACH_EP_DAYS mixes units - 13 rows say "0", others say "5MIN",
    "2-4 MINT", "8-10DAYS". Minute-valued entries are episode durations in
    minutes in a column declared in days; they are dropped as uninterpretable.

Run `python codings.py --check` to self-test the decoders.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import PID, SHEET, WEAR   # noqa: E402  (paths and the patient-ID pattern live in common.py)


def wearable_pids():
    """Patients with a 1-minute watch file: the analysed cohort."""
    return {f.name.split(".")[0] for f in WEAR.glob("*.csv.gz")}
MISSING = {"ND", "NK", "NA", "NIL", "-", ""}          # per the sheet's universal coding

# raw code -> meaning, for the columns whose codes are not self-evident
DECODE = {
    "GENDER": {1: "male", 2: "female", 3: "other"},
    "CRNT_STAT_SYM_CNTRL": {1: "controlled", 2: "partly", 3: "uncontrolled"},
    "FRM_OF_SMKNG": {1: "bidi", 2: "cigarette", 3: "both", 4: "other"},
    "CURNT_STATUS": {1: "continue", 2: "cessation", 3: "modified", 4: "replacement"},
    "BRTH_SOUNDS": {1: "normal", 2: "reduced", 3: "absent"},
    "NOR_ABN": {1: "normal", 2: "abnormal"},
    "TRTMNT_PLAN": {1: "step_up", 2: "step_down", 3: "same"},
}
MMRC = {"0": 0.0, "0-I": 0.5, "I": 1.0, "I-II": 1.5, "II": 2.0, "II-III": 2.5,
        "III": 3.0, "III-IV": 3.5, "IV": 4.0}
# yes/no columns in the baseline sheet (Yes=1, No=2, None=0)
YES_NO = ["COUGH_0", "BRTHLSNS_0", "WHZNG_0", "SPUTUM_0", "FEVER_0", "PALPITATION_0",
          "PEDAL_EDEMA_0", "EXC_HPNED_YES_NO", "SMOKE_HX", "H/O_BMFE", "OCC_EXP",
          "H/O_COMORB", "SLP_DIST", "H/O_TB", "H/O_FAM_COPD", "SMART_WATCH_PRV_0"]


def clean(s):
    """Strip the sheet's missing-value tokens, then coerce to number."""
    t = s.astype(str).str.strip().str.upper()
    return pd.to_numeric(s.where(~t.isin(MISSING) & s.notna()), errors="coerce")


def yes_no(s):
    """Yes=1, No=2, None=0  ->  1 / 0 / 0, anything else missing."""
    v = clean(s)
    return v.map({1: 1.0, 2: 0.0, 0: 0.0})


def mmrc(s):
    """Roman mMRC text, ranges at their midpoint."""
    return s.astype(str).str.strip().str.upper().map(MMRC)


def duration_days(s):
    """AVG_DUR_EACH_EP_DAYS: keep day-valued entries, drop minute-valued ones."""
    t = s.astype(str).str.strip().str.upper()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    for i, v in t.items():
        if "MIN" in v:                       # minutes in a column declared in days
            continue
        nums = re.findall(r"\d+(?:\.\d+)?", v)
        if nums:
            out[i] = np.mean([float(n) for n in nums])   # "8 TO 10" -> 9
    return out


def load_baseline(sheet=SHEET):
    """The Baseline sheet, with a lowercase `pid` and the known errors repaired."""
    B = pd.ExcelFile(sheet).parse("Baseline", header=None)
    d = B.iloc[3:].reset_index(drop=True)
    d.columns = [str(c).strip() for c in B.iloc[2]]
    d = d[d["PARTCPNT_ID"].notna()].copy()
    d["pid"] = d["PARTCPNT_ID"].astype(str).str.strip().str.lower()
    # the sheet's systolic/diastolic headers are swapped; verify, then repair
    lo, hi = clean(d["BP_mmHg_SYS"]), clean(d["BP_mmHg_DIA"])
    both = lo.notna() & hi.notna()
    if both.any() and (lo[both] < hi[both]).mean() > 0.9:
        d["bp_systolic"], d["bp_diastolic"] = hi, lo
        d.attrs["bp_swapped"] = True
    else:
        d["bp_systolic"], d["bp_diastolic"] = lo, hi
        d.attrs["bp_swapped"] = False
    a = clean(d["AGE"])
    bad = d.pid[(a < 18) | (a > 100)].tolist()
    d["age_clean"] = a.where((a >= 18) & (a <= 100))
    d.attrs["implausible_age"] = bad
    return d


def load_exacerbations(sheet=SHEET):
    """-> one row per dated episode: pid, date. Undated events are reported, not kept."""
    E = pd.ExcelFile(sheet).parse("exacerbation", header=0)
    E = E[E["Subject ID"].notna()].copy()
    E["pid"] = E["Subject ID"].astype(str).str.strip().str.lower()
    dcols = [c for c in E.columns if str(c).startswith("Date-")]
    long = E.melt(id_vars="pid", value_vars=dcols, value_name="date")
    long = long.dropna(subset=["date"])[["pid", "date"]].sort_values(["pid", "date"])
    long.attrs["undated_patients"] = sorted(set(E.pid) - set(long.pid))
    return long.reset_index(drop=True)


def load_enrolment(sheet=SHEET):
    """-> pid, enrolment, source. The sheet has no enrolment-date field, so enrolment is
    the date the smart watch was provided (the DATE after SMART_WATCH_PRV_0). One entry
    is not a date (c022: "01-06-20260"); there the treatment-plan date beside it is used."""
    d = load_baseline(sheet)
    cols = list(d.columns)

    def date_after(name):
        return pd.to_datetime(d.iloc[:, cols.index(name) + 1], dayfirst=True,
                              format="mixed", errors="coerce")

    watch, plan = date_after("SMART_WATCH_PRV_0"), date_after("TRTMNT_PLAN_0")
    return pd.DataFrame({"pid": d.pid.values, "enrolment": watch.fillna(plan).values,
                         "source": np.where(watch.notna(), "smart watch date", "treatment plan date")})


def bode_reconstructed(d):
    """Recompute BODE from its four published components, for cross-checking.

    BMI <=21 -> 1 else 0;  FEV1%pred >=65/50-64/36-49/<=35 -> 0/1/2/3;
    mMRC 0-1/2/3/4 -> 0/1/2/3;  6MWD >=350/250-349/150-249/<=149 -> 0/1/2/3.
    """
    bmi = clean(d["BMI_kg/M2_0"])
    fev = clean(d["FEV 1 [L]_%POSTD_POST_0"])
    mm = mmrc(d["mMRC_GRD_0"])
    walk = clean(d["DIS_COVRD_0"])
    return ((bmi <= 21).astype(float)
            + pd.cut(fev, [-1, 34, 49, 64, 1e4], labels=[3, 2, 1, 0]).astype(float)
            + pd.cut(mm, [-1, 1, 2, 3, 10], labels=[0, 1, 2, 3]).astype(float)
            + pd.cut(walk, [-1, 149, 249, 349, 1e5], labels=[3, 2, 1, 0]).astype(float))


def _check():
    s = pd.Series(["1", "2", "0", "ND", "NK", None])
    assert yes_no(s).tolist()[:3] == [1.0, 0.0, 0.0]
    assert yes_no(s)[3:].isna().all(), "ND/NK must be missing, not a category"
    assert mmrc(pd.Series(["0-I", "II-III", "III"])).tolist() == [0.5, 2.5, 3.0]
    d = duration_days(pd.Series(["8 TO 10", "5MIN", "2-4 MINT", "0", "5 DAYS"]))
    assert d.tolist()[0] == 9.0 and np.isnan(d[1]) and np.isnan(d[2])
    assert d.tolist()[3] == 0.0 and d.tolist()[4] == 5.0
    assert clean(pd.Series(["NA", "12"])).tolist()[1] == 12.0
    b = pd.DataFrame({"BMI_kg/M2_0": [20.0], "FEV 1 [L]_%POSTD_POST_0": [40.0],
                      "mMRC_GRD_0": ["II"], "DIS_COVRD_0": [200.0]})
    assert bode_reconstructed(b).iloc[0] == 1 + 2 + 1 + 2
    print("ok: yes/no remap, ND/NK as missing, mMRC midpoints, mixed-unit durations, "
          "BODE reconstruction")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    else:
        print(__doc__)
