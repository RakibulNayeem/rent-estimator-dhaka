# =====================================================================
# PASTE THIS AS A NEW CELL AT THE END OF YOUR KAGGLE NOTEBOOK, THEN RUN.
# It trains ONE final model on all data and saves the 3 files the
# website needs. Download them from the Kaggle "Output" panel afterward.
# =====================================================================
import re, json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

# ---- feature engineering (same as your notebook) ----
def clean_location(x):
    s = str(x).lower().strip(); s = re.sub(r"\s+", " ", s); return s.replace(", ", ",")
def parse_area(x):
    m = re.search(r"([\d.]+)", str(x).replace(",", "").lower()); return float(m.group(1)) if m else np.nan
def parse_price(x):
    s = str(x).replace(",", "").lower(); m = re.search(r"([\d.]+)", s)
    if not m: return np.nan
    v = float(m.group(1)); return v*100000 if "lakh" in s else v*1000
def split_loc(x):
    parts = [p.strip() for p in str(x).split(",") if p.strip()]
    return pd.Series({
        "loc_first": parts[0] if parts else "unknown",
        "loc_second": parts[1] if len(parts) > 1 else "unknown",
        "loc_last": parts[-1] if parts else "unknown",
        "loc_parts": len(parts),
        "has_block": int("block" in str(x)), "has_sector": int("sector" in str(x)),
        "has_road": int("road" in str(x)), "has_residential": int("residential" in str(x))})

df = pd.read_csv(file_path).drop(columns=["Unnamed: 0"], errors="ignore").copy()
df["Location_clean"] = df["Location"].apply(clean_location)
df["Area_num"] = df["Area"].apply(parse_area)
df["Rent"] = df["Price"].apply(parse_price)
df = pd.concat([df, df["Location_clean"].apply(split_loc)], axis=1)
df = df.dropna(subset=["Area_num", "Rent", "Bed", "Bath"]).copy()
lo, hi = df["Rent"].quantile([0.01, 0.99]); df["Rent_clipped"] = df["Rent"].clip(lo, hi)
df["Area_per_bed"] = df["Area_num"]/df["Bed"].replace(0, np.nan)
df["Area_per_bath"] = df["Area_num"]/df["Bath"].replace(0, np.nan)
df["Bath_per_bed"] = df["Bath"]/df["Bed"].replace(0, np.nan)
df["Total_rooms"] = df["Bed"] + df["Bath"]
loc_freq = df["Location_clean"].value_counts()
df["Location_freq"] = df["Location_clean"].map(loc_freq)
df = df.drop_duplicates(subset=["Location_clean","Area_num","Bed","Bath","Rent"]).copy()

feature_cols = ["Area_num","Bed","Bath","Area_per_bed","Area_per_bath","Bath_per_bed",
    "Total_rooms","loc_parts","has_block","has_sector","has_road","has_residential",
    "Location_freq","loc_first","loc_second","loc_last","Location_clean"]
cat_cols = ["loc_first","loc_second","loc_last","Location_clean"]
num_cols = [c for c in feature_cols if c not in cat_cols]

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)])
model = TransformedTargetRegressor(
    regressor=Pipeline([("prep", pre),
        ("reg", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                              max_depth=8, random_state=42))]),
    func=np.log1p, inverse_func=np.expm1)

model.fit(df[feature_cols], df["Rent_clipped"])

joblib.dump(model, "model.pkl")
joblib.dump(loc_freq.to_dict(), "loc_freq.pkl")
json.dump(sorted(loc_freq.index.tolist()), open("locations.json", "w"), ensure_ascii=False)
print("Saved model.pkl, loc_freq.pkl, locations.json to /kaggle/working — download them from the Output panel.")
