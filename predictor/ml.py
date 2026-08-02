"""
Loads the trained model ONCE when the server starts, then answers
predictions in milliseconds. Nothing here retrains anything.

Files live in <project>/ml_artifacts/:
    model.pkl      fitted pipeline (returns rent in taka)
    loc_freq.pkl   {cleaned_location: count} for the Location_freq feature
    locations.json known locations (used for the dropdown)
    features.py    the SAME feature-engineering used in training
"""
import json
import sys
from pathlib import Path

import joblib

ARTIFACTS = Path(__file__).resolve().parent.parent / "ml_artifacts"

# Make features.py importable, then reuse the exact training-time logic.
sys.path.insert(0, str(ARTIFACTS))
from features import build_features, clean_location  # noqa: E402

# --- load once at import time ---
_model = joblib.load(ARTIFACTS / "model.pkl")
_loc_freq = joblib.load(ARTIFACTS / "loc_freq.pkl")

with open(ARTIFACTS / "locations.json", encoding="utf-8") as f:
    _locations_raw = json.load(f)

_known = set(_loc_freq.keys())


def _prettify(loc):
    """'gulshan 1,gulshan,dhaka' -> 'Gulshan 1, Gulshan, Dhaka'"""
    return ", ".join(part.strip().title() for part in loc.split(","))


# Pretty labels for the dropdown; clean_location() reverses them on submit.
LOCATION_CHOICES = sorted(_prettify(l) for l in _locations_raw)


def is_known_location(location):
    return clean_location(location) in _known


def predict_rent(location, area, bed, bath):
    """Return estimated monthly rent in taka (float)."""
    X = build_features(location, area, bed, bath, _loc_freq)
    return float(_model.predict(X)[0])
