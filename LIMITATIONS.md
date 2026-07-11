# Limitations

An honest inventory of what this project can and cannot claim, backed by the
rigor analyses run on the current dataset (2020–2024, 460 weekly rows × 2
regions). Reviewers, judges, and downstream users should read this document
before drawing conclusions from the model or the pipeline.

---

## L1 — The harvest-loss calibration hypothesis was not empirically tested

**Claim we cannot make yet:** *"Outcome-anchored calibration on CMFRI harvest
closure events beats standard sigmoid cross-validation."*

**What the data supports:** we proposed the method, built the pipeline
(`chatbot`/`orchestrator`/`predict` are all closure-aware), and ran a **power
analysis** to determine the calibration-set size required for the hypothesis
to be testable. Results:

| N (calibration events) | Platt Brier (mean) | 95% CI | Beats raw XGB (0.087)? |
|---|---|---|---|
| **4  (CMFRI reality)** | **0.193** | [0.087, 0.437] | ❌ No |
| 8   | 0.151 | [0.086, 0.358] | ❌ |
| 16  | 0.126 | [0.079, 0.261] | ❌ |
| 32  | 0.119 | [0.079, 0.263] | ❌ |
| 64  | 0.101 | [0.080, 0.161] | ❌ (marginal) |
| 92 (max data available) | 0.098 | [0.079, 0.138] | ❌ (still worse) |

**Two conclusions from the power analysis:**

1. At the CMFRI-observed rate of ≈ 4 documented closure events per 5-year
   study window, Platt calibration produces a Brier score 2.2× worse than
   uncalibrated XGBoost, with a bootstrap 95% CI so wide that individual
   forecasts would be unusable.

2. Even at N=92 (the theoretical maximum available in the current dataset,
   using `bloom_or_documented` as an outcome-label stand-in), harvest-loss
   Platt scaling *never* beats raw XGBoost. This suggests the value of
   outcome-anchored calibration may be constrained to cases where the base
   model is meaningfully miscalibrated — which XGBoost on this data is not.

**How the paper must frame this:** *"We proposed harvest-loss calibration and
built the pipeline to test it. Data limitations (n=4 closure events across
2020–2024) prevented a meaningful empirical test. Our power analysis suggests
dozens of documented closure events would be needed for the method to become
even potentially viable — and that at all sample sizes we could simulate on the
current data, the raw XGBoost baseline was not beaten. The hypothesis remains
open for future work when a richer closure log becomes available (Kerala/
Karnataka State Fisheries Department bulletins, CMFRI HAB unit internal
register, or 10+ additional years of CMFRI Annual Reports)."*

**See:** `results/power_analysis.png`, `code/notebooks/BloomWatchAI_Calibration.ipynb`.

---

## L2 — The bloom definition is a proxy and has a species-specific blind spot

**Claim we cannot make:** *"BloomWatch predicts confirmed toxic Harmful Algal
Bloom (HAB) events."*

**What the data supports:** *"BloomWatch predicts elevated-chlorophyll weeks
operationally defined as satellite-derived chl-a > 2 mg/m³ per 8-day window,
combined with CMFRI-documented events."*

**Cross-validation of the proxy against the 4 real CMFRI events:**

| Date | Region | chl_a_mean | Fires chl > 2? | Notes |
|---|---|---|---|---|
| 2020-08-12 | Kerala | 7.16 | ✅ yes | Proxy validates real event |
| 2022-05-09 | Karnataka | NaN | (missing) | Cloud cover masked satellite retrieval |
| 2023-04-15 | Kerala | 0.19 | ❌ **no** | Trichodesmium bloom off Cochin — species mismatch |
| 2024-03-13 | Kerala | 0.13 | ❌ **no** | Trichodesmium bloom off Kochi — species mismatch |

Only **1 of 4 (25%)** confirmed CMFRI bloom events cross the chl-a > 2 mg/m³
threshold. Two of the four (2023, 2024) are Trichodesmium blooms — a
nitrogen-fixing cyanobacterium whose photopigments differ from standard
chlorophyll-a satellite retrievals. The remaining event was masked by cloud
cover.

**Practical consequence:** the model has a **species-specific blind spot**
for Trichodesmium blooms — which are exactly the blooms most damaging to
Kerala's shellfish industry (they cause hypoxia-driven mass mortality at
bloom collapse). Farmers relying on BloomWatch for Trichodesmium detection
would receive false-negative warnings.

**Threshold sensitivity analysis** (chl > 1.5 / 2 / 3 / 5):

| Threshold | Positive rate | AUC | Brier | ECE |
|---|---|---|---|---|
| chl > 1.5 | 26.1% | 0.77 | 0.15 | 0.13 |
| **chl > 2.0** | **20.4%** | **0.83** | **0.09** | **0.07** |
| chl > 3.0 | 12.2% | 0.76 | 0.09 | 0.04 |
| chl > 5.0 | 6.1% | 0.59 | 0.07 | 0.05 |

The 2 mg/m³ threshold peaks on AUC — not just a literature convention but
empirically the best signal-to-noise trade-off for this dataset.

**How the paper must frame this:** *"The forecast target is elevated-
chlorophyll weeks as a proxy for bloom risk, not confirmed toxic HAB events.
Threshold sensitivity supports 2 mg/m³ as the operational bar. Cross-
validation against the 4 documented CMFRI events reveals that only 1 crosses
the threshold — the other 3 are missed by cloud cover or species mismatch,
notably Trichodesmium. Future work should incorporate species-specific
indices such as NDCI (Normalized Difference Chlorophyll Index) and CDOM
absorption at 443 nm (which CMFRI's own 2024 report identified as elevated
during Trichodesmium blooms off Kochi)."*

**See:** `code/notebooks/BloomWatchAI_Calibration.ipynb` (threshold sensitivity cell).

---

## L3 — The reliability diagram shows bidirectional calibration errors

**Claim we cannot make:** *"When the model predicts 60% risk, the farmer
should expect a 60% chance of bloom."*

**What the data supports:** pooled reliability across 3 rolling splits (N=276,
5 quantile bins with ≥54 samples each):

| Bin | Mean predicted | Empirical rate | Error direction |
|---|---|---|---|
| 1 | 0.001 | 3.6% | Slight over |
| 2 | 0.004 | 3.6% | Well calibrated |
| 3 | 0.020 | 7.1% | Under-predicts |
| 4 | **0.153** | **27.8%** | **UNDER-predicts moderate risk (bin 4)** |
| 5 | **0.655** | **42.9%** | **OVER-predicts high risk (bin 5)** |

**Two independent findings:**

1. **Moderate-risk under-prediction (bin 4):** predictions of 0.15 correspond
   to empirical bloom rates of 0.28. The model calls "moderate" what is
   empirically "elevated." Farmers relying on the raw probability at this
   band would under-react.

2. **High-risk over-confidence (bin 5):** predictions of 0.65 correspond to
   empirical bloom rates of 0.43. The model calls "high" what is empirically
   "moderate–elevated." Farmers relying on the raw probability at this band
   would over-react.

**Operational consequence:** raw model probabilities in the 0.15–0.35 band
should be presented as *"Elevated"* (not "Low–Medium"), and probabilities
above 0.5 should be presented as *"Elevated caution — model less reliable in
this range"* (not "Very High"). This is implemented in
`bloomwatch-app/pipeline/predict.py::qualitative_risk`.

**Why the flattening happens:** at these sample sizes, the tail of the
prediction distribution (few "confident-high" predictions per test set) is
noisy. Pooled reliability (N=276) is far more stable than single-split (N=92)
— we recommend pooled reporting for any future publication.

**How the paper must frame this:** *"Reliability analysis reveals a
bidirectional calibration error: the model under-predicts empirical bloom
rates in the moderate-signal band (0.15 → 0.28 observed) and over-predicts
at the high-signal band (0.65 → 0.43 observed). We recalibrate the user-
facing risk-band labels accordingly and recommend that agent chains
consuming BloomWatch treat any raw probability above 0.15 as an elevated-
caution flag rather than a graded percentage."*

**See:** `results/BloomWatch_reliability_v2.png`.

---

## L4 — 5 years of data, not 20

**Claim we cannot make:** *"BloomWatch generalises across seasonal cycles,
ENSO variability, and multi-decadal monsoon trends."*

**What the data supports:** the study window is 2020–2024 — five full annual
cycles, 460 weekly rows across two regions. Model performance on longer
horizons is untested.

**Why the window is what it is:**

- MODIS-Aqua retired in Feb 2025, forcing a mid-pipeline sensor pivot to
  VIIRS-SNPP. Pre-2020 data would require additional sensor-comparability
  work.
- IMD district-level cumulative rainfall snapshots are only reliably scraped
  from 2020 onward via the dashboard endpoint. Earlier records exist as
  PDFs.
- CMFRI Annual Reports for 2010–2019 exist on the project's file system but
  yield ≈ 5 additional dated bloom events across those 9 years — not enough
  to resolve L1.

**Extending to 2015–2019 is feasible but expensive** (~2 days of pipeline
work). The main scientific gain would be an additional monsoon-season cycle
diversity, not a resolution of the calibration sample-size floor identified
in L1.

**How the paper must frame this:** *"The study window is 5 years. This
limits seasonal generalisation testing. Extending to the full VIIRS +
MODIS-Aqua record (2015–2024, ~10 years) is achievable future work; the
pipeline is fully automated and requires only bounding-box re-runs. It does
not, however, resolve the sample-size floor for outcome-anchored
calibration identified in L1."*

---

## Summary — what the project honestly demonstrates

- **A working weekly bloom-risk forecast pipeline** for Kerala + Karnataka,
  refreshed automatically each Friday from public NASA VIIRS + IMD sources.
- **A well-calibrated base learner** (raw XGBoost, ECE 0.07 on 2024
  hold-out, AUC 0.83) — on the elevated-chlorophyll target.
- **A rigorous, publishable negative result:** post-hoc calibration
  (including a novel outcome-anchored variant) does not improve on the raw
  learner at any calibration-set size we could simulate, with a power
  analysis quantifying the empirical sample-size floor.
- **A methodology template** (harvest-loss Platt scaling) that transfers
  directly to flood damage records, wildfire loss claims, and crop
  insurance data — but which requires far more outcome-labelled events than
  our 4 CMFRI closures provide.
- **A calibrated user-facing risk-band mapping** that acknowledges the
  bidirectional error pattern rather than pretending percentages are precise.
- **An honest species-specific blind spot** (Trichodesmium) that should be
  named in every user-facing communication.

The project should not be pitched as *"HAB prediction"* or *"harvest-loss
calibration validated."* It should be pitched as *"elevated bloom risk
forecasting with an honest sample-size floor analysis for outcome-anchored
calibration."*
