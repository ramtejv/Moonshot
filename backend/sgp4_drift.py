import json
import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

with open("show.json", "r", encoding="utf-8") as f:
  rows = json.load(f)

df = pd.DataFrame(rows)
df["epoch_dt"] = pd.to_datetime(df["EPOCH"], utc=True)
df["creation_dt"] = pd.to_datetime(df["CREATION_DATE"], utc=True)
df["sma_km"] = pd.to_numeric(df["SEMIMAJOR_AXIS"])

# Deduplicate identical EPOCHs (keep latest catalog record)
df = (
    df.sort_values(["epoch_dt", "creation_dt"])
    .drop_duplicates(subset=["EPOCH"], keep="last")
    .sort_values("epoch_dt")
    .reset_index(drop=True)
)

# 1. Identify thruster burn timestamps (upward semi-major-axis jumps)
dt_days = df["epoch_dt"].diff().dt.total_seconds() / 86400.0
step_rate_m_day = (df["sma_km"].diff() * 1000.0) / dt_days
burn_epochs = df.loc[step_rate_m_day > 50.0, "epoch_dt"].tolist()


def crosses_burn(t0: pd.Timestamp, t1: pd.Timestamp) -> bool:
  start, end = min(t0, t1), max(t0, t1)
  return any(start < b <= end for b in burn_epochs)


def dt_to_jd(ts: pd.Timestamp):
  return jday(
      ts.year,
      ts.month,
      ts.day,
      ts.hour,
      ts.minute,
      ts.second + ts.microsecond / 1e6,
  )


def propagate_teme(line1: str, line2: str, target_ts: pd.Timestamp):
  sat = Satrec.twoline2rv(line1, line2)
  jd, fr = dt_to_jd(target_ts)
  err, r_km, _ = sat.sgp4(jd, fr)
  if err != 0:
    return None
  return np.array(r_km, dtype=float)


records = []
epochs = df["epoch_dt"].tolist()

for i, row in df.iterrows():
  t0 = row["epoch_dt"]
  target = t0 + pd.Timedelta(hours=24)

  diffs = [abs((e - target).total_seconds()) for e in epochs]
  j = int(np.argmin(diffs))
  if diffs[j] > 6 * 3600 or j == i:
    continue

  actual_row = df.iloc[j]
  t1 = actual_row["epoch_dt"]
  lead_hours = (t1 - t0).total_seconds() / 3600.0

  # 2. Filter lead_hours to 20-28h and exclude burn-crossing intervals
  if not (20.0 <= lead_hours <= 28.0):
    continue
  if crosses_burn(t0, t1):
    continue

  r_pred_at_actual = propagate_teme(row["TLE_LINE1"], row["TLE_LINE2"], t1)
  r_actual = propagate_teme(actual_row["TLE_LINE1"], actual_row["TLE_LINE2"], t1)
  if r_pred_at_actual is None or r_actual is None:
    continue

  drift_km = float(np.linalg.norm(r_pred_at_actual - r_actual))
  records.append({
      "from_epoch": t0,
      "to_epoch": t1,
      "lead_hours": round(lead_hours, 1),
      "drift_24h_km": round(drift_km, 3),
  })

drift_df = pd.DataFrame(records)
print(drift_df.to_string(index=False))

calm = drift_df[drift_df["from_epoch"] < "2024-05-08"]
storm = drift_df[
    (drift_df["from_epoch"] >= "2024-05-10")
    & (drift_df["from_epoch"] <= "2024-05-12")
]

print(
    f"\nClean non-burn pre-storm median 24h drift: {calm['drift_24h_km'].median():.2f} km"
)
print(
    f"Clean non-burn May 10-12 max 24h drift:    {storm['drift_24h_km'].max():.2f} km"
)
drift_df.to_csv("eos07_sgp4_drift_clean.csv", index=False)