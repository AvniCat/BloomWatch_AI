"""End-to-end pipeline test in simulation mode."""
import json
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]


def _run(module_or_script: str):
    if module_or_script.startswith("scripts/"):
        r = subprocess.run([sys.executable, module_or_script], cwd=APP_ROOT, capture_output=True, text=True)
    else:
        r = subprocess.run([sys.executable, "-m", module_or_script], cwd=APP_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{module_or_script} failed:\n{r.stdout}\n{r.stderr}"
    return r


def test_end_to_end_sim():
    _run("scripts/simulate_current_week.py")
    _run("pipeline.build_features")
    _run("pipeline.predict")

    forecast = json.loads((APP_ROOT / "data/current_forecast.json").read_text())
    assert "regions" in forecast
    assert set(forecast["regions"]) == {"Kerala", "Karnataka"}
    for region, r in forecast["regions"].items():
        assert 0.0 <= r["P_bloom_next_week"] <= 1.0
        assert r["risk_band"] in {"Low", "Elevated", "High", "Very High"}
