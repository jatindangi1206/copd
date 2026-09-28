"""Build the report twice: COPD_EDA_Report.html (light) and COPD_EDA_Report_dark.html.

    python 10_report.py      (after 02-09, run for both themes)

The light report embeds figs/, the dark one figs_dark/ (same scripts, THEME=dark).
Every figure in the report is a PNG in that folder, and every PNG in the folder
is in the report - the script stops if one is left out. Numbers in the text are
read from numbers/ and the data, not typed in. Sections 11-12 are empty slots
for the imputation and forecasting results.
"""
import base64
import html
import json

import pandas as pd

from common import HERE, NUMBERS, master, patients
from codings import load_baseline, load_exacerbations, clean

CADENCE, MIN_READINGS, WINDOW = 10, 200, 14

# ------------------------------------------------------------------ numbers
X = pd.read_csv(HERE / "clinical_table.csv")
P = pd.read_csv(NUMBERS / "data_presence.csv")
M = pd.read_csv(NUMBERS / "hrv_missingness.csv")
S = pd.read_csv(NUMBERS / "hrv_sleep.csv")
SB = pd.read_csv(NUMBERS / "sleep_blocks.csv")
SD = pd.read_csv(NUMBERS / "sleep_stage_hrv.csv")
SEC = pd.read_csv(NUMBERS / "datasheet_sections.csv")
COM = pd.read_csv(NUMBERS / "comorbidities.csv")
IMG = pd.read_csv(NUMBERS / "imaging.csv")
TR = pd.read_csv(NUMBERS / "treatment.csv")
TRN = pd.read_csv(NUMBERS / "treatment_count.csv")
E = load_exacerbations()
B = load_baseline()

H = pd.concat([master(p, ["time", "hrv"]).assign(pid=p) for p in patients()])
H = H[H.hrv.notna()].sort_values(["pid", "time"])
gap = H.groupby("pid").time.diff().dt.total_seconds().div(60).dropna()
per = H.groupby("pid").size()
rich = per[per >= MIN_READINGS].index
med = H[H.pid.isin(rich)].groupby("pid").hrv.median()

ep_ok = 0
for _, e in E.iterrows():
    h = H[H.pid == e.pid]
    near = h[(h.time >= e.date - pd.Timedelta(days=WINDOW)) & (h.time < e.date + pd.Timedelta(days=WINDOW + 1))]
    ep_ok += int(len(h) >= MIN_READINGS and len(near) > 0)

cov = M.coverage.dropna()
stage = 100 * SB[["deep", "light", "awake"]].sum() / SB.scored.sum()
pdeep = SB.groupby("patient").deep.sum() / SB.groupby("patient").scored.sum() * 100
lung = {c: clean(B[c]) for c in ("FEV 1 [L]_%PRED_PRE_0", "FVC [L]_%PRED_PRE_0", "R5_%PRED_0",
                                 "X5_%PRED_0", "DIS_COVRD_0", "MIN_SPO2_0")}
n = dict(
    patients=len(X), men=int((X.sex == "male").sum()), women=int((X.sex == "female").sum()),
    age_lo=X.age.min(), age_hi=X.age.max(), age_med=X.age.median(),
    gold={g: int((X.gold == g).sum()) for g in "ABE"},
    mmrc25=int((X.mmrc == 2.5).sum()), no_exac=int((X.exac_dated == 0).sum()),
    hr=int(P.hr.sum()), hrv=int(P.hrv.sum()), temp=int(P.temp.sum()), spo2=int(P.spo2.sum()),
    steps=int(P.steps.sum()), hrv_pts=int((P.hrv > 0).sum()),
    no_hrv=", ".join(P.patient[P.hrv == 0]), rich=len(rich),
    gap10=100 * (gap == CADENCE).mean(), gap_hour=100 * (gap > 60).mean(),
    cov_med=cov.median(), cov50=int((cov >= 50).sum()), cov_n=len(cov),
    top=100 * (H.hrv >= 120).mean(), hrv_max=H.hrv.max(),
    med_lo=med.min(), med_hi=med.max(),
    sleep_lower=int((S["diff"] < 0).sum()), sleep_n=len(S),
    blocks=len(SB), stage_pts=SB.patient.nunique(),
    deep_lo=pdeep.min(), deep_hi=pdeep.max(),
    deep=stage["deep"], light=stage["light"], awake=stage["awake"],
    scored=100 * (SB.scored / SB.minutes).median(),
    deep_lower=int((SD["diff"] < 0).sum()), deep_n=len(SD),
    ep=len(E), ep_pts=E.pid.nunique(), ep_ok=ep_ok,
    undated=", ".join(E.attrs["undated_patients"]),
    sec_full="; ".join(SEC.section[SEC.pct >= 90]),
    sec_part="; ".join(SEC.section[(SEC.pct >= 30) & (SEC.pct < 70)]),
    sec_empty="; ".join(SEC.section[SEC.pct < 10]),
    fev_med=lung["FEV 1 [L]_%PRED_PRE_0"].median(), fvc_med=lung["FVC [L]_%PRED_PRE_0"].median(),
    r5_lo=lung["R5_%PRED_0"].min(), r5_hi=lung["R5_%PRED_0"].max(),
    x5_lo=lung["X5_%PRED_0"].min(), x5_hi=lung["X5_%PRED_0"].max(),
    walk_med=lung["DIS_COVRD_0"].median(), spo2_min=lung["MIN_SPO2_0"].min(),
    spo2_med=lung["MIN_SPO2_0"].median(),
)
age7 = X.pid[X.age < 18].tolist()
gp = {t: 100 * (gap <= t).mean() for t in (10, 20, 60, 180, 360)}

# ------------------------------------------------------------------ modelling data
MD = HERE / "model_data"
CFG = json.loads((MD / "config.json").read_text())
T = pd.read_csv(MD / "model_10min.csv.gz", parse_dates=["time"])
SG = pd.read_csv(MD / "segments.csv")
PT = pd.read_csv(MD / "patients.csv")
DY = pd.read_csv(MD / "model_daily.csv")
inseg = T[T.segment.notna()]
m = dict(
    rows=len(T), pts=T.pid.nunique(), segs=len(SG), seg_pts=SG.pid.nunique(),
    kept=100 * inseg.hrv.notna().sum() / T.hrv.notna().sum(),
    miss_in=100 * inseg.hrv.isna().mean(),
    seg_h_med=SG.hours.median(), seg_h_min=SG.hours.min(), seg_h_max=SG.hours.max(),
    obs_in=int(inseg.hrv.notna().sum()), m_rand=int(T.mask_random.sum()), m_block=int(T.mask_block.sum()),
    train=int((T.split == "train").sum()), test=int((T.split == "test").sum()),
    days=len(DY), d_train=int((DY.split == "train").sum()), d_test=int((DY.split == "test").sum()),
    no_seg=", ".join(PT.pid[PT.segments == 0]),
)


# ------------------------------------------------------------------ helpers
def slot(label, height=180):
    return f'<div class="slot" style="min-height:{height}px"><span>{label}</span></div>'


def table(rows, head):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def page(figdir):
    used = []

    def fig(name, title, shows, takeaway):
        used.append(name)
        b64 = base64.b64encode((figdir / f"{name}.png").read_bytes()).decode()
        return f"""<figure>
<img src="data:image/png;base64,{b64}" alt="{html.escape(title)}">
<figcaption><span class="fn">Figure {len(used)}.</span> <i>{title}.</i> {shows} {takeaway}</figcaption></figure>"""

    ex = inseg[inseg.pid == inseg.pid.iat[0]].iloc[:8]
    sample = table([[r.pid, f"{r.time:%d %b %H:%M}", "" if pd.isna(r.hrv) else f"{r.hrv:.0f}",
                     "" if pd.isna(r.hr) else f"{r.hr:.0f}", f"{r.sleep_frac:.1f}", r.missing_run, r.segment,
                     r.split, r.mask_random, r.mask_block] for r in ex.itertuples()],
                   ["pid", "time", "hrv", "hr", "sleep_frac", "missing_run", "segment", "split",
                    "mask_random", "mask_block"])

    body = f"""
<header>
<h1>COPD pilot cohort: clinical and wearable data</h1>
</header>
<p class="abstract"><span class="lead">Summary.</span> {n['patients']} patients with COPD have both a clinical record from enrolment and smartwatch data. The watch gave {n['hrv']:,} HRV readings from {n['hrv_pts']} patients and {n['hr']:,} heart-rate readings; the clinical team dated {n['ep']} exacerbations in {n['ep_pts']} patients. This report describes what the cohort holds and what the data look like, then sets out the modelling data: HRV cut into segments at gaps longer than {CFG['x_minutes']} minutes, used first for imputation and then for forecasting.</p>

<h2>1&ensp;The data</h2>
<p>The data come from two sources joined by patient ID. The clinical datasheet has one row per patient at enrolment (symptoms, history, examination, blood tests, imaging, lung tests, walk test, questionnaires and treatment) and a separate list of exacerbation dates. It covers 41 patients; the 33 who also have watch data (c001&ndash;c033) are analysed here. The smartwatch export holds time-stamped readings for each patient, which we combined into one row per recorded minute. Nothing is filled in, so a missing reading stays missing.</p>
{table([
    ("Heart rate", f"{n['hr']:,}", "about every minute", "beats per minute"),
    ("HRV", f"{n['hrv']:,}", "about every 10 minutes", "not given in the export"),
    ("Temperature", f"{n['temp']:,}", "irregular", "not given in the export"),
    ("Steps", f"{n['steps']:,}", "reported in intervals", "count"),
    ("SpO2 (blood oxygen)", f"{n['spo2']:,}", "only when measured", "%"),
    ("Sleep", f"{n['blocks']:,} sleep blocks", "reported in blocks, with minutes of deep, light and almost-awake sleep", "minutes"),
], ["Watch signal", "Readings", "How often", "Unit"])}
<p>HRV (heart-rate variability) is how much the time between heartbeats varies. The watch reports it as SDNN, one value about every 10 minutes. The export gives no unit, so values are shown as reported.</p>
<p>Values are used as recorded. The only changes are coding conversions from the datasheet's own key:
Yes = 1 / No = 2 read as yes / no; ND, NK, NA (not done / not known / not available) read as blank.
The sheet's systolic and diastolic blood-pressure labels are swapped (the "systolic" column is lower in every row), so the higher value is used as systolic.
{f"Patient {', '.join(age7)} has age recorded as 7; it is shown as recorded." if age7 else ""}</p>

<h2>2&ensp;What the datasheet holds</h2>
{fig("09a-datasheet-completeness", "Completeness of each datasheet section",
     "For each section of the clinical datasheet, the share of its cells that hold a value across the 33 patients.",
     f"Nearly complete: {n['sec_full']}. Partly filled: {n['sec_part']}. Almost empty: {n['sec_empty']}.")}

<h2>3&ensp;Who the patients are</h2>
{fig("clin_cohort", "Cohort at enrolment",
     "Age, sex, body-mass index, GOLD group, breathlessness grade (mMRC) and smoking index for the 33 patients.",
     f"{n['men']} men and {n['women']} women. Ages {n['age_lo']:.0f}–{n['age_hi']:.0f} as recorded, median {n['age_med']:.0f}. GOLD groups A {n['gold']['A']}, B {n['gold']['B']}, E {n['gold']['E']}. The most common breathlessness grade is II–III ({n['mmrc25']} of 33).")}
{fig("09b-symptoms-and-history", "Symptoms, history and examination",
     "Left: patients recorded as yes for each symptom, history item and examination finding. Right: etiotype (the cause recorded for the COPD), current smoking, and symptom control.",
     "Cough and breathlessness are the most common current symptoms, but neither is universal. Most patients smoked and have since stopped; cigarette smoking is the recorded cause for most, pollution for the rest. Most examination findings are rare.")}
<h3>Comorbidities</h3>
<p>Conditions named in the datasheet's diagnosis and comorbidity text, counted per patient (a patient can have several):</p>
{table([(r.condition, r.patients) for r in COM.itertuples()], ["Condition", "Patients"])}
{fig("clin_severity", "Severity and exacerbations",
     "Symptom score (CAT) against BODE score; breathlessness grade against walk distance; how many dated exacerbations each patient had.",
     f"Neither pair moves together (r close to 0). {n['no_exac']} of 33 patients have no dated exacerbation.")}

<h2>4&ensp;Lung function, tests and treatment</h2>
{fig("09c-lung-and-walk", "Lung function and walk test",
     "Spirometry (FEV1, FVC) and oscillometry (R5, X5) as % of the predicted value, before bronchodilator; six-minute walk distance and the lowest SpO2 during the walk. Orange line: median.",
     f"Median FEV1 is {n['fev_med']:.0f}% and FVC {n['fvc_med']:.0f}% of predicted. Oscillometry varies widely (R5 {n['r5_lo']:.0f}–{n['r5_hi']:.0f}%, X5 {n['x5_lo']:.0f}–{n['x5_hi']:.0f}%). Median walk distance {n['walk_med']:.0f} m; the lowest SpO2 on the walk has median {n['spo2_med']:.0f}% and goes down to {n['spo2_min']:.0f}%.")}
<h3>Imaging</h3>
<p>For every test, the patients whose result is coded 0 are exactly those marked "not available", so 0 means the test was not done.</p>
{table([(r.test, r.available, r.normal, r.abnormal, r.blank) for r in IMG.itertuples()],
       ["Test", "Available", "Normal", "Abnormal", "Result blank"])}
<h3>Treatment</h3>
<p>Treatments listed per patient, and the first-listed treatment as recorded:</p>
<div class="two">
{table([(r.treatments_listed, r.patients) for r in TRN.itertuples()], ["Treatments listed", "Patients"])}
{table([(r.first_listed, r.patients) for r in TR.head(6).itertuples()], ["First listed (top 6)", "Patients"])}
</div>

<h2>5&ensp;How clinical measures relate</h2>
{fig("clin_corr", "Clinical measures against each other",
     "Correlation between each pair of baseline measures. Red means both rise together, blue means one rises as the other falls, white means no link. Blank cells have fewer than 10 patients with both values.",
     "Most pairs are weak. The clearest: resting SpO2 with the lowest SpO2 during the walk test (+0.61), lung function (FEV1) with the lowest walk-test SpO2 (+0.60), and lung function with BODE (−0.63). These describe this group only and do not show cause.")}
{fig("clin_vs_wearable", "Clinic readings against the watch",
     "Each dot is a patient: a value measured in clinic (horizontal) against the median of that patient's watch readings (vertical). On the dashed line the two are equal.",
     "Clinic pulse and watch heart rate agree only loosely (r +0.36), and the watch is usually lower, as expected for weeks of daily life against one clinic visit. The SpO2 panel rests on very few watch readings.")}

<h2>6&ensp;How much watch data there is</h2>
{fig("03b-totals-and-days", "Total readings and recording span",
     "Left: total readings per signal. Right: days from each patient's first to last reading of any signal.",
     f"Heart rate dominates ({n['hr']:,}); SpO2 is almost absent ({n['spo2']:,}). Spans run from about a week to eleven months.")}
{fig("03a-readings-per-patient", "Readings per patient, per signal",
     "Readings per patient for each signal, on a log scale (each step is ten times the one below).",
     f"Volume differs widely between patients. {n['no_hrv']} have no HRV.")}
{fig("05c-readings-per-day", "HRV readings per day",
     "Each row is a patient, each dot a day; the colour scale on the right gives that day's readings (144 is one every 10 minutes).",
     "Recording is dense from mid-May to early July, then thinner or absent for many patients.")}

<h2>7&ensp;Gaps in HRV</h2>
{fig("04-hrv-over-time-per-patient", "Every HRV reading over time",
     "Each dot is one HRV reading, on the patient's own timeline. Gaps on screen are gaps in the data.",
     "For almost every patient, readings form a low band and a top band near 120–129, with long breaks between recording periods.")}
{fig("05a-gap-between-readings", "Time between consecutive HRV readings",
     "How long after one reading the next one arrives. Left: gaps up to 2 hours. Right: all gaps.",
     f"{n['gap10']:.1f}% of gaps are exactly 10 minutes: when the watch records HRV, it keeps to schedule. Peaks at 20, 30, 40 minutes are missed readings; {n['gap_hour']:.1f}% of gaps are over an hour.")}
<p>The minute difference is worked out like this: keep only the times at which an HRV value arrived, and subtract the previous arrival time from each one, per patient.</p>
{table([("12:00", "45", ""), ("12:10", "56", "10"), ("12:20", "65", "10"), ("12:30", "", "(nothing arrived, row not used)"), ("12:40", "55", "20")],
       ["Time", "HRV", "Minute difference"])}
{fig("05e-minute-difference", "Minute difference between consecutive HRV readings",
     "Histogram of the minute difference at three ranges: up to 3 hours (5-minute bins), 6 hours and 12 hours (10-minute bins). Log scale on the counts.",
     f"{gp[10]:.1f}% of gaps are 10 minutes or less, {gp[20]:.1f}% 20 or less, {gp[60]:.1f}% an hour or less, {gp[180]:.1f}% three hours or less and {gp[360]:.1f}% six hours or less. After the 10-minute peak the counts fall off smoothly, with small peaks at each missed 10-minute reading.")}
{fig("05b-coverage-per-patient", "Coverage per patient",
     "Left: share of 10-minute slots holding a reading, between each patient's first and last HRV reading. Right: how many patients remain at each minimum coverage.",
     f"Median coverage {n['cov_med']:.0f}%. {n['cov50']} of {n['cov_n']} patients reach 50%.")}
{fig("05d-daily-coverage", "Coverage per day",
     "For every patient-day, the share of the day's 144 slots with a reading.",
     "Days are either nearly empty or partly covered; almost no day is complete.")}

<h2>8&ensp;HRV values</h2>
{fig("06b-hrv-histogram", "All HRV readings",
     "How often each HRV value occurs, all patients together.",
     f"Two groups: a peak in the low 30s, and a block from 120 to {n['hrv_max']:.0f} (the highest value) holding {n['top']:.0f}% of readings. The count jumps sharply at 120.")}
{fig("06a-hrv-curves-per-patient", "HRV distribution, one curve per patient",
     f"A smoothed histogram for each of the {n['rich']} patients with at least {MIN_READINGS} readings, overlaid.",
     "Every patient shows the same two-group shape; only the balance between the groups differs.")}
{fig("06c-hrv-curve-each-patient", "HRV distribution for each patient",
     "The same curves, one panel per patient.",
     "Some patients sit mostly in the low group (e.g. c010, c024), others mostly in the top group (e.g. c004, c020).")}
{fig("06d-median-hrv-per-patient", "Median HRV per patient",
     f"The middle value of each patient's HRV readings ({n['rich']} patients with at least {MIN_READINGS} readings).",
     f"Medians range from {n['med_lo']:.0f} to {n['med_hi']:.0f}, so each patient needs their own baseline.")}

<h2>9&ensp;Sleep</h2>
<p>The watch reports sleep as blocks. For each block it gives how many minutes were deep, light and almost awake, but not which minutes, so stage results are per block. Stage minutes cover a median {n['scored']:.0f}% of a block's length.</p>
{fig("07a-hrv-sleep", "HRV during sleep",
     "For each patient, median HRV inside and outside the sleep blocks the watch reports.",
     f"HRV is lower during recorded sleep for {n['sleep_lower']} of {n['sleep_n']} patients.")}
{fig("08a-sleep-stage-split", "Sleep stages per patient",
     f"For each patient, how their scored sleep minutes split into deep, light and almost awake ({n['stage_pts']} patients, {n['blocks']:,} blocks). Top bar: all patients together.",
     f"Overall {n['deep']:.0f}% deep, {n['light']:.0f}% light, {n['awake']:.0f}% almost awake. Light sleep is the largest share overall; the deep share ranges from {n['deep_lo']:.0f}% to {n['deep_hi']:.0f}%.")}
{fig("08b-hrv-deep-vs-light", "HRV in deep vs light sleep",
     "For each patient, the median HRV of blocks that are mostly (70% or more) deep sleep and of blocks that are mostly light sleep. Only blocks with at least 3 HRV readings.",
     f"HRV is lower in deep-sleep blocks for {n['deep_lower']} of {n['deep_n']} patients, about half. HRV differs between sleep and waking hours (chart above), but not consistently between deep and light sleep.")}

<h2>10&ensp;Exacerbations</h2>
<p>The datasheet lists {n['ep']} dated exacerbations in {n['ep_pts']} patients (date only, no time or duration). {n['undated']} have an event with no date.
{n['ep_ok']} of the {n['ep']} dates have HRV readings within two weeks, from a patient with at least {MIN_READINGS} readings.</p>
{fig("07b-hrv-around-exacerbations", "HRV around each exacerbation",
     "HRV from two weeks before to two weeks after each dated exacerbation. Grey: readings; blue: daily median; dashed: the patient's usual median.",
     "Several dates have no HRV nearby. Where there is data, the daily median swings from day to day, and no single change around the date repeats across events.")}

<h2>11&ensp;Modelling data</h2>
<p>The modelling plan has two steps, in this order: <b>first imputation</b> (fill in missing HRV), <b>then forecasting</b> (predict later HRV). Both run on one prepared table, built by <code>11_model_data.py</code>:</p>
{table([
    ("model_data/model_10min.csv.gz", f"{m['rows']:,} rows, {m['pts']} patients", "One row per patient per 10-minute slot (the HRV rhythm), from each patient's first to last HRV reading. The main modelling file."),
    ("model_data/model_daily.csv", f"{m['days']:,} rows", "One row per patient-day with at least one HRV reading, for day-level forecasting."),
    ("model_data/segments.csv", f"{m['segs']} rows", "One row per segment: patient, start, end, length, missing share, train and test size."),
    ("model_data/patients.csv", f"{len(PT)} rows", "Per patient: coverage, number of segments, share of HRV kept, and the patient's own threshold."),
    ("model_data/config.json", "", "The settings used: threshold, minimum segment length, hidden share, split, random seed."),
], ["File", "Size", "What it is"])}
<h3>What each row of the main file holds</h3>
{table([
    ("pid, time, hour", "Patient, start of the 10-minute slot, hour of day."),
    ("hrv, hrv_n", "HRV reading in the slot (blank if none) and how many readings fell in it (almost always 0 or 1)."),
    ("hr, hr_n", "Mean heart rate over the slot's minutes, and how many minutes had a reading."),
    ("temp, spo2, steps", "Mean temperature, mean SpO2 and steps reported in the slot, as recorded (blank if none)."),
    ("sleep_frac, steps_active_frac", "Share of the slot's minutes inside a watch sleep block / a steps interval."),
    ("exac_day", "1 on a dated exacerbation day."),
    ("missing_run", "If HRV is missing, how many slots in a row are missing around it; 0 if present."),
    ("segment, split", "Segment ID (blank if the slot is not in a segment) and train / test for forecasting."),
    ("mask_random, mask_block", "1 = this HRV reading is hidden for the imputation test (random / whole-run pattern)."),
    ("c_age … c_exac_12m", "Clinical values from enrolment, repeated on every row: age, sex, BMI, mMRC, GOLD, CAT, FEV1 % (before bronchodilator), walk distance, clinic SpO2, exacerbations in the past year."),
], ["Columns", "Meaning"])}
<p>The first rows of the first segment. Nothing is filled in: a blank HRV is a missing reading.</p>
{sample}

<h3>Segments</h3>
<p>Models fail when too much is missing, so they run inside <b>segments</b>: stretches where HRV is never missing for more than <b>x</b> readings in a row. Where a longer gap occurs, the segment ends. For this pilot, <b>x = {CFG['x_slots']} readings ({CFG['x_minutes']} minutes)</b>, set after looking at the minute differences (section 7) and the scan below. Segments must be at least {CFG['min_segment_hours']} hours long, so every segment covers a full day.</p>
{fig("11a-segment-threshold", "What each choice of x keeps",
     "Left: for each candidate x, the share of all HRV readings kept inside segments and the share of segment slots with no reading. Right: the share kept for each patient (grey) and the cohort.",
     f"Small x keeps almost nothing; large x keeps everything but fills segments with gaps. At x = {CFG['x_minutes']} minutes, {m['kept']:.0f}% of HRV is kept and {m['miss_in']:.0f}% of segment slots are missing. Patients differ a lot in how much a given x keeps.")}
{fig("11b-segments", "Segments for each patient",
     f"Each row is a patient. Grey: the span of their HRV data. Coloured: segments, split into the first {CFG['train_pct']}% (train) and last {100 - CFG['train_pct']}% (test).",
     f"{m['segs']} segments in {m['seg_pts']} patients, {m['seg_h_min']:.0f} to {m['seg_h_max']:.0f} hours long (median {m['seg_h_med']:.0f}). {m['no_seg']} have no segment. Segments cluster where recording was dense (mid-May to early July).")}
<p>This is a pilot threshold for a sparse cohort. It is one setting in <code>11_model_data.py</code>, to be reviewed with physiology experts and, as more data arrives, set per patient rather than for the whole cohort.</p>

<h2>12&ensp;Models</h2>
<p>The models were not picked at random. We first drew up 10 characteristics of HRV from the literature, such as that it is positive and bounded, has a day and a night mode, and follows a daily (circadian) rhythm, and tested each one on data. Eight held (table below). We then built a corpus of imputation and forecasting methods with their assumptions, strengths and weaknesses, and kept only the methods that do not violate any of the eight, either natively or through a published modification. The gap-aware state-space transformer is our own hybrid design. The list will be refined as the models are tested.</p>
<h3>The ten characteristics</h3>
<p>Each characteristic was taken from the literature and then checked against data, not assumed. The full definitions, tests and references are in the <a href="https://docs.google.com/document/d/1V8rGaJFxRt5ccTJUKq4JcynNGbu9lqcs7h0J0KMzWDA/edit?usp=sharing">research document</a> (first tab).</p>
{table([
    ("H1", "Bounded and positive: values stay above zero and below physiological limits", "kept"),
    ("H2", "Two modes, a day state and a night state", "kept"),
    ("H3", "Strong short-term memory: each 10-minute value strongly predicts the next", "dropped: lag-1 autocorrelation about 0.41, moderate rather than strong"),
    ("H4", "Drifting baseline: the patient's average moves over the day", "kept"),
    ("H5", "24-hour (circadian) rhythm", "kept"),
    ("H6", "Heavy-tailed spikes from sudden events or sensor glitches", "dropped: the watch drops a bad reading instead of recording an extreme value, so spikes show up as missing data"),
    ("H7", "Irregular timing: readings do not land exactly on the 10-minute grid", "kept"),
    ("H8", "Two kinds of missingness: short random gaps and long gaps when the watch is off", "kept"),
    ("H9", "Real time: a forecast made at time t may not use data after t", "kept (a constraint we set, for forecasting; imputation of this pilot data may use readings on both sides of a gap)"),
    ("H10", "Each patient has their own baseline", "kept"),
], ["", "Characteristic", "Result"])}
<p>The same research found long gaps must not be bridged: its baseline-drift analysis works inside segments that never cross a gap of more than 180 minutes, the same x used in section 11.</p>
<h3>Model set</h3>
{table([
    ("General machine learning", "XGBoost; CatBoost"),
    ("Neural sequence", "RNN; LSTM"),
    ("Latent / state-space", "Hidden Markov model (HMM); nonlinear state-space model; RS-DPF; particle filter and particle smoother"),
    ("Bayesian / irregular time", "GRU-ODE-Bayes; CD-Gamma-DGLM"),
    ("Signal decomposition", "OSSA"),
    ("Hybrid / research", "Physiology-informed neural ODE; gap-aware state-space transformer (our design)"),
    ("Foundation model", "TimesFM 3"),
], ["Family", "Models"])}

<h2>13&ensp;Imputation</h2>
<h3>Design</h3>
<p>In every segment, {CFG['mask_pct']}% of the HRV readings that exist are hidden at random ({m['m_rand']:,} of {m['obs_in']:,}; column <code>mask_random</code>). Models then reconstruct the hidden readings from the HRV before and after them (bidirectional) and from the other signals in the same slots: heart rate, temperature, steps and sleep. Each filled value is scored against the real value that was hidden.</p>
<p>The hidden readings are fixed by a random seed ({CFG['seed']}), so every method is scored on exactly the same readings. Real gaps come as runs rather than single readings, so a second, harder test (<code>mask_block</code>) hides {m['m_block']:,} readings in whole runs whose lengths are drawn from the real gaps inside segments. Straight-line filling between neighbouring readings is the reference that any model should beat.</p>
<h3>Method</h3>{slot("Method", 90)}
<h3>Results</h3>
{table([("", "", "", "", "")] * 8, ["Method", "Random: MAE", "Random: RMSE", "Block: MAE", "Notes"])}
<p class="small">MAE is the average distance between filled and real value. RMSE: the same, but large misses count more. Both in HRV as reported by the watch; lower is better.</p>
{slot("Figure")}
<h3>Takeaway</h3>{slot("Takeaway", 70)}

<h2>14&ensp;Forecasting</h2>
<h3>Design</h3>
<p>Forecasting always predicts the later part of the data, never the middle. In every segment the first {CFG['train_pct']}% is used for training and the last {100 - CFG['train_pct']}% for testing ({m['train']:,} and {m['test']:,} slots; column <code>split</code>). The day-level table <code>model_daily.csv</code> is split the same way per patient: the first {CFG['train_pct']}% of days train and the last {100 - CFG['train_pct']}% test ({m['d_train']:,} and {m['d_test']:,} days).</p>
<p>Gaps in the training part are filled by the chosen imputation model first; the test part is scored only on real readings. Two references set the bar any model should beat: the last observed value, and the patient's own median.</p>
<h3>Method</h3>{slot("Method", 90)}
<h3>Results</h3>
{table([("", "", "", "", "")] * 8, ["Model", "Level", "Horizon", "MAE", "Notes"])}
{slot("Figure")}
<h3>Takeaway</h3>{slot("Takeaway", 70)}

<h2>15&ensp;Conclusions</h2>
<ul>
<li>{n['patients']} patients, almost all men, mostly GOLD groups A and B, with a detailed clinical record at enrolment and weeks to months of watch data.</li>
<li>The datasheet is nearly complete for symptoms, history, examination, imaging, walk test and spirometry before bronchodilator; blood tests and post-bronchodilator spirometry are about half filled.</li>
<li>The watch records HRV on a steady 10-minute schedule when it records, but coverage is incomplete (median {n['cov_med']:.0f}%).</li>
<li>HRV readings fall into a low group and a top group at 120–{n['hrv_max']:.0f} for every patient; levels differ between patients and drop during sleep, but not between deep and light sleep.</li>
<li>{n['ep_ok']} of {n['ep']} dated exacerbations have HRV data around them.</li>
<li>For modelling, HRV is cut into {m['segs']} segments ({m['seg_pts']} patients) wherever more than {CFG['x_minutes']} minutes are missing; imputation is tested on {CFG['mask_pct']}% hidden readings, forecasting on the last {100 - CFG['train_pct']}% of each segment.</li>
</ul>
{slot("Model results", 90)}

<h2>Terms</h2>
{table([
    ("HRV", "Heart-rate variability: how much the time between heartbeats varies. Reported by the watch as SDNN."),
    ("SpO2", "Blood oxygen level, %."),
    ("CAT", "COPD Assessment Test: 8-question symptom score, 0–40, higher is worse."),
    ("mMRC", "Breathlessness grade, 0–4, higher is worse."),
    ("GOLD group", "A and B: 0–1 moderate flare-ups a year (B has more symptoms). E: 2 or more, or 1 needing hospital."),
    ("Etiotype", "The cause recorded for the COPD. C: cigarette smoking. P: pollution (air, biomass, occupational)."),
    ("FEV1 / FVC", "Air blown out in the first second / in total, as % of the expected value. Lower is worse."),
    ("R5 / X5", "Oscillometry: airway resistance and reactance at 5 Hz, as % of the expected value."),
    ("BODE", "0–10 score combining BMI, FEV1, breathlessness and walk distance, as recorded in the sheet."),
    ("6MWT", "Six-minute walk test: distance walked in six minutes."),
    ("r", "Correlation, from −1 to +1. 0 means no link; closer to ±1 means the two move together more consistently."),
    ("Coverage", "Share of expected 10-minute HRV slots that hold a reading."),
    ("Segment", "A stretch of HRV with no run of missing readings longer than x; models run inside segments."),
    ("MAE / RMSE", "Average error between predicted and real HRV; RMSE weighs large misses more. Lower is better."),
    ("Sleep block", "One stretch of sleep reported by the watch, with its minutes of deep, light and almost-awake sleep."),
], ["Term", "Meaning"])}
"""
    missing = {f.stem for f in figdir.glob("*.png")} - set(used)
    assert not missing, f"figures in {figdir.name}/ not in the report: {sorted(missing)}"
    return body, len(used)


TOKENS = {
    "light": "--bg:#ffffff;--ink:#222;--ink2:#5f5d58;--rule:#cfccc4;--link:#1f5f7a;--slot:#f6f5f1",
    "dark": "--bg:#1c1c1b;--ink:#e6e4de;--ink2:#a9a79f;--rule:#44433f;--link:#8cc0d6;--slot:#1d1d1b",
}
CSS = """
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.62 Charter,"Iowan Old Style","Palatino Linotype",Georgia,serif}
main{max-width:860px;margin:0 auto;padding:56px 20px 96px}
header{border-bottom:1px solid var(--rule);padding-bottom:18px;margin-bottom:22px}
h1{font-size:30px;font-weight:normal;line-height:1.25;margin:0 0 8px}
.byline{margin:0;color:var(--ink2);font-size:15px}
.abstract{font-size:16.5px}
.lead{font-variant:small-caps;letter-spacing:.03em}
h2{font-size:22px;font-weight:normal;margin:48px 0 10px}
h3{font-size:17px;font-style:italic;font-weight:normal;margin:24px 0 6px}
p{margin:.7em 0}
a{color:var(--link)}
code{font:14px ui-monospace,Menlo,Consolas,monospace}
figure{margin:26px 0 30px}
figure img{display:block;width:100%;height:auto}
figcaption{font-size:15px;line-height:1.5;color:var(--ink2);margin-top:8px}
.fn{color:var(--ink);font-weight:bold}
.tw{overflow-x:auto;margin:14px 0 18px}
table{border-collapse:collapse;width:100%;font-size:15px;border-top:1.5px solid var(--ink);border-bottom:1.5px solid var(--ink)}
th,td{text-align:left;padding:5px 10px 5px 0;vertical-align:top}
th{font-weight:normal;font-style:italic;border-bottom:1px solid var(--ink2)}
td{font-variant-numeric:tabular-nums}
td:empty{height:30px}
.two{display:grid;grid-template-columns:1fr 1.4fr;gap:28px;align-items:start}
@media (max-width:700px){.two{grid-template-columns:1fr}}
.slot{border:1px solid var(--rule);background:var(--slot);margin:10px 0;padding:6px 10px}
.slot span{font-size:13px;color:var(--ink2);font-style:italic}
.small{font-size:14.5px;color:var(--ink2)}
footer{margin-top:56px;padding-top:12px;border-top:1px solid var(--rule);color:var(--ink2);font-size:13.5px}
"""

for theme, folder, out in (("light", "figs", "COPD_EDA_Report.html"),
                           ("dark", "figs_dark", "COPD_EDA_Report_dark.html")):
    body, k = page(HERE / folder)
    path = HERE / out
    path.write_text(f"""<!doctype html><html lang="en" data-theme="{theme}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="{theme}">
<title>COPD Cohort Report</title><style>:root{{{TOKENS[theme]}}}{CSS}</style></head>
<body><main>{body}</main></body></html>""")
    print(f"wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB), {k} figures from {folder}/")
