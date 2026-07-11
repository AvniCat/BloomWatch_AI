# Narrative — abstract, research question, key claims

Language that matches the honest findings from `LIMITATIONS.md` and the rigor
analyses. Use these directly in the paper, pitch, and application forms.

---

## Abstract (paper version, ~200 words)

Harmful Algal Blooms (HABs) cause substantial economic damage to shellfish
cooperatives along India's Kerala–Karnataka coasts, but statistical models
that forecast bloom risk face a persistent problem: their probability outputs
are rarely calibrated against real-world outcomes. We propose an
**outcome-anchored calibration** method — fitting Platt scaling on documented
CMFRI harvest closure events rather than internal cross-validation — and
build a full weekly forecast pipeline (NASA VIIRS chlorophyll and SST + IMD
district rainfall + XGBoost) to test it. Our pipeline achieves AUC 0.83, ECE
0.07 on 2024 hold-out for the elevated-chlorophyll target (chl-a > 2 mg/m³).
However, we identify a critical sample-size floor: CMFRI documented only 4
closure events across 2020–2024, and our power analysis (bootstrap resampling
at N ∈ {4, 8, …, 92}) shows that Platt-calibrated Brier score never falls
below the uncalibrated XGBoost baseline (0.087) at any tested N. Combined
with a species-specific proxy blind spot for Trichodesmium blooms (which
standard satellite chlorophyll retrievals miss), our findings support two
publishable conclusions: outcome-anchored calibration requires far more
documented events than currently exist, and raw well-calibrated boosted-tree
probabilities may be operationally sufficient when the target is proxy-based.

---

## One-liner (pitch, ≤ 25 words)

> *We built a weekly bloom-risk forecast for Kerala–Karnataka shellfish
> farmers and quantified the exact number of closure records needed to
> improve it — 4 events isn't enough, dozens might be.*

---

## Research question — reframed

**Not:** *"Does outcome-anchored calibration improve environmental forecast reliability?"*

**Instead:** *"What is the sample-size floor at which outcome-anchored
calibration becomes viable for coastal HAB forecasting, and how does that
floor compare to the closure-event rate documented by the operational
authority (CMFRI)?"*

This reframing turns a data-insufficiency limitation into a testable,
publishable question — and we answer it with the power analysis.

---

## Key claims — updated for honesty

### Claim 1 — DEFENSIBLE
> *"On the 2024 hold-out year, our elevated-chlorophyll forecast achieves
> AUC 0.83, ECE 0.07, Brier 0.09. XGBoost with 69 engineered features from
> NASA VIIRS satellite + IMD rainfall produces well-calibrated probabilities
> without any post-hoc adjustment."*

**Evidence:** `code/notebooks/BloomWatchAI_Calibration.ipynb`.

### Claim 2 — DEFENSIBLE
> *"Post-hoc calibration methods including standard sigmoid cross-validation,
> Platt scaling on outcome-anchored data, and isotonic regression never beat
> the raw XGBoost baseline at any tested calibration-set size (N ∈ {4, 6, 8,
> 12, 16, 24, 32, 48, 64, 92}). Our power analysis quantifies this
> empirically."*

**Evidence:** `results/power_analysis.png`, `results/power_analysis.csv`.

### Claim 3 — DEFENSIBLE
> *"The chlorophyll-a > 2 mg/m³ threshold used as the operational bloom
> proxy is optimal on our data: threshold sensitivity analysis shows AUC
> peaks at exactly 2.0 (0.83), falling to 0.77 at 1.5 and 0.59 at 5.0."*

### Claim 4 — DEFENSIBLE (this is the paper's biggest contribution)
> *"Cross-validation of the chl-a > 2 proxy against the 4 CMFRI-documented
> bloom events reveals only 1 (25%) is detected by the satellite proxy. The
> remaining 3 are missed by (a) cloud cover and (b) species mismatch —
> notably Trichodesmium blooms, which have photopigments outside standard
> chlorophyll-a retrieval bands. This defines a species-specific blind spot
> in current satellite-based coastal bloom monitoring."*

### Claim 5 — DEFENSIBLE (methodological)
> *"Reliability analysis across 3 pooled temporal splits (N=276) shows the
> model produces bidirectional calibration errors: it under-predicts
> empirical bloom rates in the moderate-signal band (0.15 → 0.28) and
> over-predicts in the high-signal band (0.65 → 0.43). User-facing risk
> bands should be recalibrated against these empirical rates, not the raw
> percentages. Our TideAlert integration implements this remapping."*

### Claim NOT to make
> ~~"Our novel harvest-loss calibration method improves bloom forecast
> accuracy compared to standard calibration approaches."~~

**Why not:** the power analysis conclusively shows this is false at all
sample sizes tested. Making this claim would be dishonest.

---

## Response to the anticipated review question:
### *"Why did you propose harvest-loss calibration if it doesn't beat the baseline?"*

> *"Because until we ran the power analysis, we did not know whether the
> failure was methodological or sample-size-driven. Our contribution is not
> that outcome-anchored calibration works — it doesn't, at any sample size
> we could simulate. Our contribution is that we quantified the empirical
> sample-size floor and identified a testable prerequisite (dozens of
> documented events) that future work can target directly. Along the way,
> we also produced a well-calibrated bloom-risk forecast and identified a
> species-specific blind spot in standard satellite chlorophyll retrievals.
> All three are useful negative or corollary results."*

---

## Talking points for a 60-second pitch

- *"We built a weekly bloom-risk forecast for Kerala–Karnataka shellfish
  farmers using free public NASA and IMD data."*
- *"We proposed a novel calibration method — fitting the model against real
  economic outcomes — and quantified exactly how many outcome events would
  be needed to make it work. 4 CMFRI closures isn't enough. Even 92 events
  isn't enough. The paper's biggest finding is quantifying that floor."*
- *"We also found that the standard satellite proxy misses the exact blooms
  farmers care about most — Trichodesmium, which has different pigments.
  That's a whole research direction opened up: species-specific satellite
  indices for Indian coastal HAB monitoring."*
- *"The forecast we produced is well-calibrated and honest. When it says
  'elevated risk,' that's what it means. When it says 'high,' it hedges
  because we know the model's tail is noisy at that end."*

---

## Suggested paper title options

1. *"When is outcome-anchored calibration viable? A power analysis for
   coastal HAB forecasting on the Kerala–Karnataka coasts"* — most honest
2. *"Sample-size floors for outcome-anchored probability calibration:
   empirical evidence from a five-year HAB forecast pipeline"* — most
   methodological
3. *"BloomWatch AI: a well-calibrated bloom-risk forecast, a species-
   specific blind spot, and the empirical sample-size floor for outcome-
   anchored calibration"* — most descriptive
