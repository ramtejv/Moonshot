import json
import numpy as np
import pandas as pd
from pymsis import msis

MU_EARTH = 3.986004418e14  # m^3 / s^2


def load_clean_sma(json_path: str) -> pd.DataFrame:
  with open(json_path, "r", encoding="utf-8") as f:
    rows = json.load(f)

  df = pd.DataFrame(rows)
  df["epoch_dt"] = pd.to_datetime(df["EPOCH"], utc=True)
  df["creation_dt"] = pd.to_datetime(df["CREATION_DATE"], utc=True)
  df["sma_km"] = pd.to_numeric(df["SEMIMAJOR_AXIS"])
  df["inc_deg"] = pd.to_numeric(df["INCLINATION"])
  df["raan_deg"] = pd.to_numeric(df["RA_OF_ASC_NODE"])

  # Deduplicate identical EPOCHs (keep newest catalog record)
  df = (
      df.sort_values(["epoch_dt", "creation_dt"])
      .drop_duplicates(subset=["EPOCH"], keep="last")
      .sort_values("epoch_dt")
      .reset_index(drop=True)
  )

  # Flag upward semi-major-axis jumps (thruster / station-keeping burns)
  dt_days = df["epoch_dt"].diff().dt.total_seconds() / 86400.0
  step_rate_m_day = (df["sma_km"].diff() * 1000.0) / dt_days
  df["manoeuvre_flag"] = step_rate_m_day > 50.0
  return df


def compute_24h_decay(df: pd.DataFrame) -> pd.DataFrame:
  # Keep non-manoeuvre points for smooth drag trajectory
  clean = df[~df["manoeuvre_flag"]].copy()
  clean = clean.set_index("epoch_dt").sort_index()

  # Resample onto 6-hour grid by time interpolation, then take exact 24h (4-step) drop
  grid = clean[["sma_km", "inc_deg", "raan_deg"]].resample("6h").mean()
  grid["sma_km"] = grid["sma_km"].interpolate(method="time")
  grid["inc_deg"] = grid["inc_deg"].interpolate(method="time")
  grid["raan_deg"] = grid["raan_deg"].interpolate(method="time")

  # Positive value = metres lost over the past 24 hours
  grid["obs_decay_24h_m"] = (grid["sma_km"].shift(4) - grid["sma_km"]) * 1000.0
  grid["alt_km"] = grid["sma_km"] - 6378.137

  # Approximate sub-satellite latitude & longitude for MSIS query
  t_hr = (grid.index - grid.index[0]).total_seconds() / 3600.0
  lat_deg = np.degrees(
      np.arcsin(
          np.sin(np.radians(grid["inc_deg"].to_numpy()))
          * np.sin(np.radians(0.06 * t_hr))
      )
  )
  lon_deg = (
      (grid["raan_deg"].to_numpy() + np.degrees(0.06 * t_hr) - 15.0 * t_hr + 180.0)
      % 360.0
  ) - 180.0

  grid["lat_deg"] = lat_deg
  grid["lon_deg"] = lon_deg
  return grid.dropna(subset=["obs_decay_24h_m"]).reset_index()


def add_msis_density(grid: pd.DataFrame) -> pd.DataFrame:
  dates = grid["epoch_dt"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
  lons = grid["lon_deg"].to_numpy(dtype=float)
  lats = grid["lat_deg"].to_numpy(dtype=float)
  alts = grid["alt_km"].to_numpy(dtype=float)

  # pymsis requires a 25-element options list (1.0 = default scaling for each option)
  msis_options = [1.0] * 25

  rho = np.empty(len(grid), dtype=float)
  for i in range(len(grid)):
    out = msis.run(
        dates[i],
        lons[i],
        lats[i],
        alts[i],
        version=2.1,
        options=msis_options,
    )
    arr = np.asarray(out, dtype=float)
    # Index 0 along the output species/variable axis is total mass density (kg/m^3)
    rho[i] = arr.reshape(-1, 11)[0, 0]

  grid["rho_kg_m3"] = rho
  a_m = grid["sma_km"].to_numpy(dtype=float) * 1000.0
  grid["drag_proxy"] = rho * np.sqrt(MU_EARTH * a_m)
  return grid

def main():
  df = load_clean_sma("show.json")
  grid = compute_24h_decay(df)
  grid = add_msis_density(grid)

  # Calibrate k_eff on pre-storm window (Apr 26 - May 8, avoiding early-window edge & May 10-12 storm)
  cal_mask = (grid["epoch_dt"] >= "2024-04-26") & (
      grid["epoch_dt"] < "2024-05-09"
  )
  k_eff = np.sum(
      grid.loc[cal_mask, "obs_decay_24h_m"] * grid.loc[cal_mask, "drag_proxy"]
  ) / np.sum(grid.loc[cal_mask, "drag_proxy"] ** 2)

  grid["model_decay_24h_m"] = k_eff * grid["drag_proxy"]

  # Quiet reference = median model decay during the pre-storm calibration window
  quiet_ref_m_day = grid.loc[cal_mask, "model_decay_24h_m"].median()
  grid["quiet_ref_m_day"] = quiet_ref_m_day
  grid["drag_ratio_vs_quiet"] = grid["obs_decay_24h_m"] / quiet_ref_m_day

  storm_slice = grid[
      (grid["epoch_dt"] >= "2024-05-10") & (grid["epoch_dt"] <= "2024-05-13")
  ]
  print(f"Calibrated k_eff: {k_eff:.6e}")
  print(f"Quiet-window median expected decay: {quiet_ref_m_day:.1f} m/day\n")
  print(
      storm_slice[[
          "epoch_dt",
          "alt_km",
          "obs_decay_24h_m",
          "model_decay_24h_m",
          "drag_ratio_vs_quiet",
      ]].to_string(index=False)
  )

  print(
      "\nMay 11 peak 24h observed decay:"
      f" {storm_slice[storm_slice['epoch_dt'].dt.strftime('%Y-%m-%d') == '2024-05-11']['obs_decay_24h_m'].max():.1f} m/day"
  )
  print(
      "May 11 peak drag ratio vs quiet:"
      f" {storm_slice[storm_slice['epoch_dt'].dt.strftime('%Y-%m-%d') == '2024-05-11']['drag_ratio_vs_quiet'].max():.2f}x"
  )
  grid.to_csv("eos07_drag_analysis.csv", index=False)
  print("Saved eos07_drag_analysis.csv")


if __name__ == "__main__":
  main()