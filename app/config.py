"""Shared configuration — regions, paths, feature engineering constants."""
from pathlib import Path
import os

APP_ROOT = Path(__file__).resolve().parent

# Load .env from app root so all scripts see EARTHDATA_USER, EARTHDATA_PASS, etc.
try:
    from dotenv import load_dotenv
    load_dotenv(APP_ROOT / ".env")
except ImportError:
    pass  # dotenv is optional; env vars can still be set directly
DATA_DIR = APP_ROOT / "data"
MODEL_DIR = APP_ROOT / "models"

HISTORICAL_CSV = Path(os.getenv("HISTORICAL_CSV",
                                DATA_DIR / "historical/revised_master_dts_weekly.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "xgb_bloom.pkl"))
MODEL_META_PATH = MODEL_PATH.with_suffix(".meta.json")
CURRENT_FORECAST_PATH = Path(os.getenv("CURRENT_FORECAST_PATH",
                                       DATA_DIR / "current_forecast.json"))

LIVE_MODIS_DIR = DATA_DIR / "live/modis"
LIVE_IMD_DIR = DATA_DIR / "live/imd"

# Regional bounding boxes (approx coastal strips, WGS84)
REGIONS = {
    "Kerala":    {"lat_min":  8.0, "lat_max": 12.8, "lon_min": 74.5, "lon_max": 77.5},
    "Karnataka": {"lat_min": 12.8, "lat_max": 15.0, "lon_min": 73.5, "lon_max": 75.5},
}

# 8-day epoch base (matches MODIS L3m compositing)
EIGHT_DAY_EPOCH = "2000-01-01"

# Feature engineering — must stay in sync with training notebook
NUMERIC_BASE = [
    "chlor_a_mean", "chlor_a_max", "chlor_a_min", "chlor_a_std",
    "sst_mean", "sst_max", "sst_min", "sst_std",
    "rainfall_mm_total_mean", "rainfall_mm_total_max", "rainy_days_mean",
]
LAG_WEEKS = (1, 2)
ROLL_WEEKS = (4, 8)

BLOOM_THRESHOLD_CHL_A = 2.0   # mg/m³, per CMFRI/NASA convention

XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    eval_metric="logloss", random_state=7, n_jobs=-1,
)

MODEL_VERSION = "bloomwatch-xgb-v1"
