import json
import sys
import numpy as np
import pandas as pd
from pymsis import msis
from sgp4.api import Satrec, jday

MU_EARTH = 3.986004418e14


def load_target_df(json_path: str, norad_id: str) -> pd.DataFrame:
  with open(json_path, "r", encoding="utf-8") as f:
    rows = json.load(f)

  df = pd.DataFrame(rows)
  df = df[df["NORAD_CAT_ID"].astype(str) == str(norad_id)].copy()
  if df.empty:
    raise ValueError(f"No rows found for NORAD_CAT_ID={norad_id} in {json_path}")

  df["epoch_dt"] = pd.to_datetime(df["EPOCH"], utc=True)
  df["creation_dt"] = pd.to_datetime(df["CREATION_DATE"], utc=True)
  df["sma_km"] = pd.to_numeric(df["SEMIMAJOR_AXIS"])
  df["inc_deg"] = pd.to_numeric(df["INCLINATION"])
  df["raan_deg"] = pd.to_numeric(df["RA_OF_ASC_NODE"])

  df = (
      df.sort_values(["epoch_dt", "creation_dt"])
      .drop_duplicates(subset=["EPOCH"], keep="last")
      .sort_values("epoch_dt")
      .reset_index(drop=True)
  )

  dt_days = df["epoch_dt"].diff().dt.total_seconds() / 86400.0
  step_rate_m_day = (df["sma_km"].diff() * 1000.0) / dt_days
  df["manoeuvre_flag"] = step_rate_m_day > 50.0
  return df


def run_drag(df: pd.DataFrame) -> pd.DataFrame:
  clean = df[~df["manoeuvre_flag"]].set_index("epoch_dt").sort_index()
  grid = clean[["sma_km", "inc_deg", "raan_deg"]].resample("6h").mean().interpolate(method="time")
  grid["obs_decay_24h_m"] = (grid["sma_km"].shift(4) - grid["sma_km"]) * 1000.0
  grid["alt_km"] = grid["sma_km"] - 6378.137
  grid = grid.dropna(subset=["obs_decay_24h_m"]).reset_index()

  t_hr = (grid["epoch_dt"] - grid["epoch_dt"].iloc[0]).dt.total_seconds() / 3600.0
  grid["lat_deg"] = np.degrees(
      np.arcsin(np.sin(np.radians(grid["inc_deg"].to_numpy())) * np.sin(np.radians(0.06 * t_hr)))
  )
  grid["lon_deg"] = (
      (grid["raan_deg"].to_numpy() + np.degrees(0.06 * t_hr) - 15.0 * t_hr + 180.0) % 360.0
  ) - 180.0

  dates = grid["epoch_dt"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
  msis_options = [1.0] * 25
  rho = np.empty(len(grid), dtype=float)
  for i in range(len(grid)):
    out = msis.run(
        dates[i],
        float(grid["lon_deg"].iloc[i]),
        float(grid["lat_deg"].iloc[i]),
        float(grid["alt_km"].iloc[i]),
        version=2.1,
        options=msis_options,
    )
    rho[i] = np.asarray(out, dtype=float).reshape(-1, 11)[0, 0]

  grid["rho_kg_m3"] = rho
  grid["drag_proxy"] = rho * np.sqrt(MU_EARTH * (grid["sma_km"].to_numpy() * 1000.0))

  cal_mask = (grid["epoch_dt"] >= "2024-04-26") & (grid["epoch_dt"] < "2024-05-09")
  denom = np.sum(grid.loc[cal_mask, "drag_proxy"] ** 2)
  k_eff = (
      np.sum(grid.loc[cal_mask, "obs_decay_24h_m"] * grid.loc[cal_mask, "drag_proxy"]) / denom
      if denom > 0
      else 1.0
  )
  grid["model_decay_24h_m"] = k_eff * grid["drag_proxy"]
  quiet_ref = grid.loc[cal_mask, "model_decay_24h_m"].median()
  grid["drag_ratio_vs_quiet"] = grid["obs_decay_24h_m"] / quiet_ref if quiet_ref else np.nan
  return grid


def dt_to_jd(ts: pd.Timestamp):
  return jday(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second + ts.microsecond / 1e6)


def propagate_teme(line1: str, line2: str, target_ts: pd.Timestamp):
  sat = Satrec.twoline2rv(line1, line2)
  jd, fr = dt_to_jd(target_ts)
  err, r_km, _ = sat.sgp4(jd, fr)
  return np.array(r_km, dtype=float) if err == 0 else None


def run_drift(df: pd.DataFrame) -> pd.DataFrame:
  burn_epochs = df.loc[df["manoeuvre_flag"], "epoch_dt"].tolist()
  epochs = df["epoch_dt"].tolist()
  records = []

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
    if not (20.0 <= lead_hours <= 28.0):
      continue
    if any(min(t0, t1) < b <= max(t0, t1) for b in burn_epochs):
      continue

    r_pred = propagate_teme(row["TLE_LINE1"], row["TLE_LINE2"], t1)
    r_actual = propagate_teme(actual_row["TLE_LINE1"], actual_row["TLE_LINE2"], t1)
    if r_pred is None or r_actual is None:
      continue

    records.append({
        "from_epoch": t0,
        "to_epoch": t1,
        "lead_hours": round(lead_hours, 1),
        "drift_24h_km": round(float(np.linalg.norm(r_pred - r_actual)), 3),
    })
  return pd.DataFrame(records)


def main():
  if len(sys.argv) < 3:
    print("Usage: python analyze_target.py <json_file> <norad_cat_id>")
    sys.exit(1)

  json_path, norad_id = sys.argv[1], sys.argv[2]
  df = load_target_df(json_path, norad_id)
  obj_name = df["OBJECT_NAME"].iloc[0]
  drag_df = run_drag(df)
  drift_df = run_drift(df)

  drag_df.to_csv(f"{norad_id}_drag.csv", index=False)
  drift_df.to_csv(f"{norad_id}_drift.csv", index=False)

  may11 = drag_df[drag_df["epoch_dt"].dt.strftime("%Y-%m-%d") == "2024-05-11"]
  calm_drift = drift_df[drift_df["from_epoch"] < "2024-05-08"]["drift_24h_km"].median()
  storm_drift = drift_df[
      (drift_df["from_epoch"] >= "2024-05-10") & (drift_df["from_epoch"] <= "2024-05-12")
  ]["drift_24h_km"].max()

  print(f"Target: {obj_name} (NORAD {norad_id}) | Unique epochs: {len(df)} | Burns flagged: {int(df['manoeuvre_flag'].sum())}")
  if not may11.empty:
    print(f"May 11 peak 24h observed decay: {may11['obs_decay_24h_m'].max():.1f} m/day")
  print(f"Calm non-burn median 24h SGP4 drift: {calm_drift:.2f} km")
  print(f"May 10-12 max 24h SGP4 drift:        {storm_drift:.2f} km")
  print(f"Saved {norad_id}_drag.csv and {norad_id}_drift.csv")


if __name__ == "__main__":
  main()