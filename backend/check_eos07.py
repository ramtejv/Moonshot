import json
import numpy as np
import pandas as pd

with open("show.json", "r", encoding="utf-8") as f:
  rows = json.load(f)

df = pd.DataFrame(rows)
df["epoch_dt"] = pd.to_datetime(df["EPOCH"], utc=True)
df["creation_dt"] = pd.to_datetime(df["CREATION_DATE"], utc=True)
df["sma_km"] = pd.to_numeric(df["SEMIMAJOR_AXIS"])
df["mean_motion"] = pd.to_numeric(df["MEAN_MOTION"])
df["bstar"] = pd.to_numeric(df["BSTAR"])

# 1. Deduplicate identical EPOCHs (keep latest catalog creation)
df = (
    df.sort_values(["epoch_dt", "creation_dt"])
    .drop_duplicates(subset=["EPOCH"], keep="last")
    .sort_values("epoch_dt")
    .reset_index(drop=True)
)

# 2. Compute step-to-step semi-major axis change (metres/day)
dt_days = df["epoch_dt"].diff().dt.total_seconds() / 86400.0
da_m = df["sma_km"].diff() * 1000.0  # negative = orbital decay
df["da_m_per_day"] = da_m / dt_days

# Flag positive jumps (likely station-keeping burns)
df["manoeuvre_flag"] = df["da_m_per_day"] > 50.0

print(f"Total unique epochs: {len(df)}")
print(
    df[[
        "epoch_dt",
        "sma_km",
        "da_m_per_day",
        "manoeuvre_flag",
        "bstar",
    ]].to_string(index=False)
)

# 3. Compare pre-storm drag (May 1-9, excluding burns) vs storm peak (May 10-12)
pre_storm = df[
    (df["epoch_dt"] >= "2024-05-01")
    & (df["epoch_dt"] < "2024-05-10")
    & (~df["manoeuvre_flag"])
]
storm_window = df[
    (df["epoch_dt"] >= "2024-05-10") & (df["epoch_dt"] <= "2024-05-12T23:59:59Z")
]

# Decay rate is positive when altitude drops (hence minus sign)
pre_decay_m_day = -pre_storm["da_m_per_day"].mean()
storm_decay_m_day = -storm_window["da_m_per_day"].mean()
storm_max_m_day = -storm_window["da_m_per_day"].min()

print(f"\nPre-storm mean decay (May 1-9): {pre_decay_m_day:.1f} m/day")
print(f"Storm window mean decay (May 10-12): {storm_decay_m_day:.1f} m/day")
print(f"Storm window peak single-step decay: {storm_max_m_day:.1f} m/day")
if pre_decay_m_day > 0:
  print(f"Observed mean storm ratio: {storm_decay_m_day / pre_decay_m_day:.2f}x")