"""Feature-engineering sanity tests."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import HISTORICAL_CSV, NUMERIC_BASE
from pipeline.features import engineer_features, feature_columns, add_forecast_target


@pytest.fixture(scope="module")
def hist():
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["date_start", "date_end"])
    return df.sort_values(["region", "date_start"]).reset_index(drop=True)


def test_feature_count_is_69(hist):
    dfE = engineer_features(add_forecast_target(hist))
    assert len(feature_columns(dfE)) == 69


def test_no_target_leakage(hist):
    dfE = engineer_features(add_forecast_target(hist))
    feats = feature_columns(dfE)
    for banned in ("bloom_next_week", "bloom_or_documented", "bloom"):
        assert banned not in feats, f"{banned} leaked into features"


def test_lag_features_have_expected_nans(hist):
    dfE = engineer_features(add_forecast_target(hist))
    # First two rows per region should have NaN for lag1/lag2
    firsts = dfE.groupby("region").head(2)
    for col in NUMERIC_BASE:
        assert firsts[f"{col}_lag2"].isna().sum() >= 1


def test_cyclic_features_bounded(hist):
    dfE = engineer_features(add_forecast_target(hist))
    assert dfE["doy_sin"].between(-1, 1).all()
    assert dfE["doy_cos"].between(-1, 1).all()


def test_region_one_hot(hist):
    dfE = engineer_features(add_forecast_target(hist))
    assert ((dfE["is_kerala"] + dfE["is_karnataka"]) == 1).all()
