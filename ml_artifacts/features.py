"""
Feature engineering for the Dhaka house-rent model.

CRITICAL: this exact logic must be used at BOTH training time and
prediction time, otherwise the model sees different features than it
was trained on ("train/serve skew") and predictions become garbage.

Raw inputs from the website form:
    location : str   e.g. "Gulshan 1, Gulshan, Dhaka"
    area     : float sqft, e.g. 1200
    bed      : int
    bath     : int

Everything the model needs is derived from those four values plus the
saved `loc_freq` lookup table.
"""
import re
import numpy as np
import pandas as pd

# The 17 columns the model expects, IN ORDER. Do not reorder.
FEATURE_COLS = [
    "Area_num", "Bed", "Bath",
    "Area_per_bed", "Area_per_bath", "Bath_per_bed", "Total_rooms",
    "loc_parts", "has_block", "has_sector", "has_road", "has_residential",
    "Location_freq",
    "loc_first", "loc_second", "loc_last", "Location_clean",
]
CATEGORICAL_COLS = ["loc_first", "loc_second", "loc_last", "Location_clean"]


def clean_location(x):
    s = str(x).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(", ", ",")
    return s


def parse_area(x):
    """Accepts '1,600 sqft' or a plain number like 1600."""
    s = str(x).replace(",", "").lower().strip()
    m = re.search(r"([\d.]+)", s)
    return float(m.group(1)) if m else np.nan


def _split_loc(location_clean):
    x = location_clean
    parts = [p.strip() for p in str(x).split(",") if p.strip()]
    return {
        "loc_first": parts[0] if len(parts) > 0 else "unknown",
        "loc_second": parts[1] if len(parts) > 1 else "unknown",
        "loc_last": parts[-1] if len(parts) > 0 else "unknown",
        "loc_parts": len(parts),
        "has_block": int("block" in str(x)),
        "has_sector": int("sector" in str(x)),
        "has_road": int("road" in str(x)),
        "has_residential": int("residential" in str(x)),
    }


def build_features(location, area, bed, bath, loc_freq):
    """
    Turn 4 raw inputs into the 1-row DataFrame the model expects.

    loc_freq: dict mapping cleaned-location -> count seen in training.
              Unknown locations fall back to 1.
    Returns a pandas DataFrame with exactly FEATURE_COLS, in order.
    """
    area = float(parse_area(area))
    bed = float(bed)
    bath = float(bath)

    loc_clean = clean_location(location)
    parts = _split_loc(loc_clean)

    bed_safe = bed if bed != 0 else np.nan
    bath_safe = bath if bath != 0 else np.nan

    row = {
        "Area_num": area,
        "Bed": bed,
        "Bath": bath,
        "Area_per_bed": area / bed_safe,
        "Area_per_bath": area / bath_safe,
        "Bath_per_bed": bath / bed_safe,
        "Total_rooms": bed + bath,
        "loc_parts": parts["loc_parts"],
        "has_block": parts["has_block"],
        "has_sector": parts["has_sector"],
        "has_road": parts["has_road"],
        "has_residential": parts["has_residential"],
        "Location_freq": float(loc_freq.get(loc_clean, 1)),
        "loc_first": parts["loc_first"],
        "loc_second": parts["loc_second"],
        "loc_last": parts["loc_last"],
        "Location_clean": loc_clean,
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)
