# BloomWatch AI — Live App

Live weekly Harmful Algal Bloom forecast + multilingual chatbot for shellfish cooperatives along Kerala and Karnataka coasts.

## Layout

```
bloomwatch-app/
├── pipeline/           Weekly data ingestion + prediction
│   ├── pull_modis.py       Fetch newest MODIS 8-day SST + Chl-a
│   ├── pull_imd.py         Fetch newest IMD daily rainfall
│   ├── build_features.py   Engineer 69 features for current week
│   └── predict.py          Load model, produce forecast JSON
├── api/                FastAPI service (Phase 2)
├── chatbot/            LLM orchestrator (Phase 2)
├── data/
│   ├── historical/         Historical CSV (2020–2024)
│   ├── live/               Latest MODIS + IMD downloads
│   └── current_forecast.json  Generated each week
├── models/             Pickled XGBoost + metadata
├── scripts/
│   ├── train_and_save.py   Train and pickle the model
│   └── refresh_weekly.py   Orchestrate full pipeline
└── .github/workflows/  GitHub Actions weekly cron
```

## Phase status

- [x] Phase 1 — Live forecast pipeline
- [ ] Phase 2 — Inference API + RAG backend
- [ ] Phase 3 — Frontend
- [ ] Phase 4 — Field readiness (multilingual, monitoring)

## Setup

```bash
cd bloomwatch-app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# One-time: train the model on the historical CSV
python scripts/train_and_save.py

# Weekly refresh (also runs via GitHub Actions on Fridays)
python scripts/refresh_weekly.py
```

## LLM provider (Phase 2)

Two providers, one interface — configured via `.env`:

- **Gemini 2.5 Flash** (free tier, primary for user chat, native Malayalam + Kannada)
- **Ollama** (local Llama 3.2, embeddings + background jobs + Gemini fallback)

## Cost

Zero. Every layer runs on a free tier:
- MODIS + IMD: free public data
- GitHub Actions: free tier
- Gemini: 1M tokens/day free
- Ollama: fully local
- Hosting: Vercel free tier

## Data provenance

- MODIS-Aqua L3m 8-day SST + Chl-a — NASA OceanColor
- IMD 0.25° gridded daily rainfall — India Meteorological Department
- CMFRI Annual Reports — HAB event context
