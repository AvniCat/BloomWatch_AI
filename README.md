# BloomWatch AI

A weekly harmful-algal-bloom risk forecast for shellfish cooperatives on
India's Kerala–Karnataka coast, built entirely on free public data (NASA
VIIRS satellite chlorophyll-a + sea surface temperature, IMD district
rainfall, CMFRI documented bloom events).

**Important scope note:** the model forecasts *elevated-chlorophyll weeks*
(satellite-derived chl-a > 2 mg/m³), not confirmed toxic Harmful Algal Bloom
(HAB) events. See [`LIMITATIONS.md`](LIMITATIONS.md) for the distinction,
the calibration sample-size floor, and a documented species-specific blind
spot (standard chlorophyll retrieval misses Trichodesmium blooms — the ones
most damaging to Kerala's shellfish industry).

## Final reported results (pooled across 3 rolling temporal validation splits, test years 2022 & 2024, N=276)

| Metric | Value |
|---|---|
| Model | XGBoost, 300 trees, depth 4, 69 engineered features |
| AUC | 0.809 pooled (0.787–0.831 per-split) |
| Brier score | 0.131 pooled (0.087 best-split) |
| Expected Calibration Error | 0.094 pooled (0.066 best-split) |
| Precision / Recall | 0.487 / 0.404 pooled |
| Dataset | 460 weekly rows × 2 regions, 2020–2024 |

The best individual split (train 2020–2022, test 2024) reaches AUC 0.831 —
report the pooled figures above as the headline number, not the best split
alone; the paper does the same (see `LIMITATIONS.md`).

Source of truth for these numbers: [`results/per_split_accuracy.csv`](results/per_split_accuracy.csv),
[`results/power_analysis.csv`](results/power_analysis.csv),
[`results/reliability_pooled.csv`](results/reliability_pooled.csv) — these
are what the paper's Table 4 and Figures 4–5 are built from.

The paper's central finding is a **negative result**: a proposed
outcome-anchored calibration method (Platt scaling fit on documented CMFRI
harvest-closure events instead of internal cross-validation) never beats the
raw XGBoost baseline at any tested calibration-set size (N = 4 to 92,
bootstrapped). The power analysis quantifies the sample-size floor at which
it *would* become viable — see `LIMITATIONS.md` L1.

## Live deployment

- Frontend: https://bloom-watch-appai.lovable.app
- Backend (FastAPI): https://bloomwatch-ai.onrender.com — `/health`, `/forecast`, `/model_status`, `/chat`
- The forecast refreshes automatically every Friday via GitHub Actions (`app/.github/workflows`), with a 5% AUC drift guard before any retrained model replaces the deployed one.

> An earlier deployment at `bloom-watch-ai-coastal.lovable.app` is retired —
> use the link above everywhere (paper, poster, deck, portal submission).

## Repo layout

```
BloomWatch_AI/
├── app/                    Deployable system: FastAPI + XGBoost + Gemini RAG chatbot + Next.js frontend
│   ├── pipeline/               Weekly ingestion (MODIS/VIIRS, IMD) + feature engineering + prediction
│   ├── api/                    FastAPI service (/health, /forecast, /model_status, /chat)
│   ├── chatbot/                RAG orchestrator, ChromaDB vectorstore, Gemini + Ollama fallback
│   ├── frontend/               Next.js UI
│   └── scripts/                train_and_save.py, refresh_weekly.py, retrain_weekly.py
├── code/
│   ├── notebooks/              Colab-ready analysis, incl. BloomWatchAI_Calibration.ipynb (power analysis)
│   ├── pipelines/               Data-collection scripts (CMFRI PDF extraction, MODIS, IMD, dataset merges)
│   └── src/                    Early monthly-resolution baseline models (see "Baselines" below)
├── data/                    Weekly modelling-ready CSVs + sources/schema docs
├── results/                 Trained models, predictions, feature importance, rigor artefacts
├── docs/                    Research paper, poster, pitch deck
├── LIMITATIONS.md           Five honest, quantified limitations — read this before citing any number above
├── NARRATIVE.md             Abstract, defensible claims, reviewer Q&A prep
└── LICENSE                  MIT
```

### Baselines — `code/src/` and `results/model_metrics.csv`

`code/src/train_*.py` and the metrics in `results/model_metrics.csv` /
`results/weekly_forecast_metrics.md` are from earlier exploratory passes
(a monthly-resolution 2002–2018/2019–2024 split, and an early weekly split
with only 24 hold-out weeks). They predate the final study design and are
kept for reproducibility of the model-comparison process, **not** as
reported results — `results/weekly_forecast_metrics.md` in particular
reports a suspiciously perfect ROC-AUC of 1.00 on a 3-positive-sample
hold-out, which is a small-sample artifact, not a real result. **The
numbers in the table above, from the three rolling temporal splits, are the
only ones the paper reports and the only ones that should be cited.**

## Quickstart — reviewer

```bash
pip install -r requirements.txt
jupyter notebook code/notebooks/BloomWatchAI_Calibration.ipynb
```

## Quickstart — run the live app locally

```bash
cd app
pip install -r requirements.txt
cp .env.example .env    # paste your own GEMINI_API_KEY
python scripts/train_and_save.py
python scripts/simulate_current_week.py
python -m pipeline.build_features
python -m pipeline.predict
python api/main.py       # http://localhost:8000
```

Full step-by-step: [`app/README.md`](app/README.md) and [`app/DEPLOYMENT.md`](app/DEPLOYMENT.md).

## Data provenance

- NASA VIIRS-SNPP / MODIS-Aqua L3 chlorophyll-a + SST — NASA OceanColor
- IMD 0.25° gridded daily rainfall — India Meteorological Department
- CMFRI Annual Reports 2016–2024 — documented HAB events

See [`data/README.md`](data/README.md) for full source URLs, schemas, and
regeneration scripts.

## License

MIT — see [`LICENSE`](LICENSE).
