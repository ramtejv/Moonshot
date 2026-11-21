import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pymsis import msis
import requests

BASE_DIR = Path(__file__).resolve().parent
MU_EARTH = 3.986004418e14

TARGET_META = {
    "55562": {
        "name": "EOS-07",
        "role": "Active EO (Thruster-Filtered)",
        "band_km": "416-425 km",
    },
    "51658": {
        "name": "INS-2TD",
        "role": "Passive Payload Tracer",
        "band_km": "435-451 km",
    },
    "27151": {
        "name": "PSLV DEB",
        "role": "Passive High-Drag Tracer",
        "band_km": "447-480 km",
    },
    "51659": {
        "name": "PSLV R/B",
        "role": "Passive Rocket Body Tracer",
        "band_km": "464-502 km",
    },
}

app = FastAPI(title="ATMOS-GUARD API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
  return {"message": "ATMOS-GUARD Backend Running"}


@app.get("/api/health")
def health():
  return {"status": "ok"}


def fetch_noaa_kp() -> list[dict]:
  try:
    resp = requests.get(
        "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
        timeout=6,
    )
    resp.raise_for_status()
    raw = resp.json()
    out = []
    for r in raw[1:]:
      ts = pd.to_datetime(r[0], utc=True, errors="coerce")
      if pd.notna(ts) and "2024-04-25" <= ts.strftime("%Y-%m-%d") <= "2024-05-20":
        out.append({"time_utc": r[0], "kp": float(r[1])})
    return out
  except Exception:
    return []


def compute_forward_forecast(
    alt_km: float,
    inc_deg: float,
    raan_deg: float,
    k_eff: float,
    calm_ref_decay: float,
    calm_drift_km: float,
    expected_kp: float,
) -> dict:
  """Forecast 24h, 48h, and 72h orbital decay and SGP4 position drift risk."""
  now_utc = pd.Timestamp.now("UTC").tz_localize(None)
  msis_options = [1.0] * 25

  # Map expected Kp (0-9) into geomagnetic Ap index for MSIS
  ap_est = float(np.clip(3.5 * np.exp(0.55 * expected_kp), 4.0, 400.0))

  horizons = [24, 48, 72]
  steps = []
  cur_alt = float(alt_km)

  for h in horizons:
    t_query = now_utc + pd.Timedelta(hours=h)
    lat_deg = float(np.degrees(np.arcsin(np.sin(np.radians(inc_deg)) * np.sin(np.radians(0.06 * h)))))
    lon_deg = float(((raan_deg + np.degrees(0.06 * h) - 15.0 * h + 180.0) % 360.0) - 180.0)

    out = msis.run(
        t_query.to_datetime64(),
        lon_deg,
        lat_deg,
        cur_alt,
        ap=[[ap_est] * 7],
        version=2.1,
        options=msis_options,
    )
    rho = float(np.asarray(out, dtype=float).reshape(-1, 11)[0, 0])
    sma_m = (6378.137 + cur_alt) * 1000.0
    drag_proxy = rho * np.sqrt(MU_EARTH * sma_m)

    pred_decay_m_day = max(float(k_eff * drag_proxy), 0.1)
    drag_mult = pred_decay_m_day / max(calm_ref_decay, 1.0)
    # Quadratic-like along-track drift growth with horizon and drag amplification
    pred_drift_km = calm_drift_km * max(drag_mult, 1.0) * ((h / 24.0) ** 1.35)

    cur_alt -= pred_decay_m_day / 1000.0
    steps.append({
        "horizon_h": h,
        "projected_alt_km": round(cur_alt, 3),
        "pred_decay_m_day": round(pred_decay_m_day, 1),
        "drag_ratio_vs_quiet": round(drag_mult, 2),
        "pred_sgp4_drift_km": round(pred_drift_km, 2),
    })

  max_drift_24h = steps[0]["pred_sgp4_drift_km"]
  if max_drift_24h >= 15.0 or expected_kp >= 7.0:
    risk_level = "HIGH STORM DRIFT RISK"
  elif max_drift_24h >= 5.0 or expected_kp >= 5.0:
    risk_level = "ELEVATED DRAG WATCH"
  else:
    risk_level = "NOMINAL"

  return {
      "expected_kp": expected_kp,
      "estimated_ap": round(ap_est, 1),
      "risk_level": risk_level,
      "horizons": steps,
  }


def audit_uploaded_records(df: pd.DataFrame) -> dict:
  """Evaluate whether uploaded TLE/telemetry records are trustworthy for drag modeling."""
  flags = []
  score = 100

  cols = {c.upper(): c for c in df.columns}
  required = ["EPOCH", "SEMIMAJOR_AXIS"]
  missing = [r for r in required if r not in cols]
  if missing:
    return {
        "status": "UNTRUSTED",
        "trust_score": 0,
        "flags": [f"Missing required columns: {', '.join(missing)}"],
        "summary_stats": {},
    }

  work = pd.DataFrame()
  work["epoch_dt"] = pd.to_datetime(df[cols["EPOCH"]], utc=True, errors="coerce")
  work["sma_km"] = pd.to_numeric(df[cols["SEMIMAJOR_AXIS"]], errors="coerce")
  work["inc_deg"] = pd.to_numeric(df[cols["INCLINATION"]], errors="coerce") if "INCLINATION" in cols else 97.4
  work["raan_deg"] = pd.to_numeric(df[cols["RA_OF_ASC_NODE"]], errors="coerce") if "RA_OF_ASC_NODE" in cols else 0.0
  work["object_name"] = df[cols["OBJECT_NAME"]].astype(str) if "OBJECT_NAME" in cols else "UPLOADED-SAT"
  work["norad_id"] = df[cols["NORAD_CAT_ID"]].astype(str) if "NORAD_CAT_ID" in cols else "CUSTOM"

  invalid_rows = int(work["epoch_dt"].isna().sum() + work["sma_km"].isna().sum())
  if invalid_rows > 0:
    score -= min(30, invalid_rows * 5)
    flags.append(f"{invalid_rows} row(s) have invalid EPOCH or SEMIMAJOR_AXIS values.")

  work = work.dropna(subset=["epoch_dt", "sma_km"]).sort_values("epoch_dt").reset_index(drop=True)
  if len(work) < 4:
    return {
        "status": "UNTRUSTED",
        "trust_score": 15,
        "flags": flags + ["Too few valid epochs (< 4) to separate orbital decay from noise."],
        "summary_stats": {"valid_epochs": len(work)},
    }

  dup_count = int(work["epoch_dt"].duplicated().sum())
  if dup_count > 0:
    score -= 10
    flags.append(f"{dup_count} duplicate EPOCH records detected (kept latest per epoch).")
    work = work.drop_duplicates(subset=["epoch_dt"], keep="last").reset_index(drop=True)

  work["alt_km"] = work["sma_km"] - 6378.137
  if (work["alt_km"] < 150).any() or (work["alt_km"] > 2000).any():
    score -= 40
    flags.append("Out-of-bounds LEO altitude (< 150 km or > 2000 km) detected.")

  dt_days = work["epoch_dt"].diff().dt.total_seconds() / 86400.0
  step_rate_m_day = (work["sma_km"].diff() * 1000.0) / dt_days

  burns = int((step_rate_m_day > 50.0).sum())
  extreme_drops = int((step_rate_m_day < -3000.0).sum())
  large_gaps = int((dt_days > 2.5).sum())

  if burns > 0:
    penalty = min(35, burns * 12)
    score -= penalty
    flags.append(f"{burns} upward SMA jump(s) > 50 m/day detected — thruster burns contaminate raw drag.")

  if extreme_drops > 0:
    score -= 35
    flags.append(f"{extreme_drops} extreme single-step drop(s) > 3,000 m/day — likely TLE fit outlier.")

  if large_gaps > 0:
    score -= min(20, large_gaps * 7)
    flags.append(f"{large_gaps} telemetry gap(s) > 60 hours reduce 24h interpolation confidence.")

  score = max(0, min(100, score))
  if score >= 80:
    status = "TRUSTED"
    if not flags:
      flags.append("Clean cadence, physical LEO bounds, and zero thruster-burn contamination.")
  elif score >= 50:
    status = "CAUTION"
  else:
    status = "UNTRUSTED"

  clean = work[~(step_rate_m_day.fillna(0) > 50.0)]
  observed_decay_m_day = 0.0
  if len(clean) >= 2:
    span_days = (clean["epoch_dt"].iloc[-1] - clean["epoch_dt"].iloc[0]).total_seconds() / 86400.0
    if span_days > 0:
      observed_decay_m_day = max(0.0, float((clean["sma_km"].iloc[0] - clean["sma_km"].iloc[-1]) * 1000.0 / span_days))

  return {
      "status": status,
      "trust_score": score,
      "flags": flags,
      "summary_stats": {
          "object_name": work["object_name"].iloc[-1],
          "norad_id": work["norad_id"].iloc[-1],
          "valid_epochs": int(len(work)),
          "latest_epoch_utc": work["epoch_dt"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC"),
          "latest_alt_km": round(float(work["alt_km"].iloc[-1]), 2),
          "burns_flagged": burns,
          "mean_clean_decay_m_day": round(observed_decay_m_day, 1),
          "inc_deg": round(float(work["inc_deg"].iloc[-1]), 2),
          "raan_deg": round(float(work["raan_deg"].iloc[-1]), 2),
      },
  }


@app.get("/api/targets")
def list_targets():
  available = []
  for norad_id, meta in TARGET_META.items():
    drag_file = BASE_DIR / f"{norad_id}_drag.csv"
    drift_file = BASE_DIR / f"{norad_id}_drift.csv"
    available.append({
        "norad_id": norad_id,
        **meta,
        "has_drag": drag_file.exists(),
        "has_drift": drift_file.exists(),
    })
  return {"targets": available}


@app.get("/api/satellite/{norad_id}")
def get_satellite_telemetry(norad_id: str, forecast_kp: float = 6.0):
  if norad_id not in TARGET_META:
    raise HTTPException(status_code=404, detail="Unknown NORAD ID")

  drag_path = BASE_DIR / f"{norad_id}_drag.csv"
  drift_path = BASE_DIR / f"{norad_id}_drift.csv"
  if not drag_path.exists() or not drift_path.exists():
    raise HTTPException(
        status_code=404,
        detail=f"Run analyze_target.py first to generate {norad_id}_drag.csv and {norad_id}_drift.csv",
    )

  drag_df = pd.read_csv(drag_path).replace({np.nan: None})
  drift_df = pd.read_csv(drift_path).replace({np.nan: None})

  may11 = [
      r for r in drag_df.to_dict(orient="records")
      if str(r.get("epoch_dt", "")).startswith("2024-05-11")
  ]
  peak_decay = max(
      (r["obs_decay_24h_m"] for r in may11 if r.get("obs_decay_24h_m") is not None),
      default=None,
  )

  calm_vals = [
      r["drift_24h_km"]
      for r in drift_df.to_dict(orient="records")
      if str(r.get("from_epoch", "")) < "2024-05-08" and r.get("drift_24h_km") is not None
  ]
  storm_vals = [
      r["drift_24h_km"]
      for r in drift_df.to_dict(orient="records")
      if "2024-05-10" <= str(r.get("from_epoch", "")) <= "2024-05-12" and r.get("drift_24h_km") is not None
  ]

  # Estimate k_eff and baseline parameters from historical CSV for forward prediction
  valid_drag = drag_df.dropna(subset=["model_decay_24h_m", "drag_proxy", "alt_km"])
  if not valid_drag.empty:
    last_row = valid_drag.iloc[-1]
    k_eff = float(last_row["model_decay_24h_m"]) / max(float(last_row["drag_proxy"]), 1e-12)
    alt_km = float(last_row["alt_km"])
    inc_deg = float(last_row.get("inc_deg") or 97.4)
    raan_deg = float(last_row.get("raan_deg") or 0.0)
    calm_ref = float(valid_drag["model_decay_24h_m"].median())
  else:
    k_eff, alt_km, inc_deg, raan_deg, calm_ref = 1.5e8, 445.0, 97.4, 0.0, 25.0

  calm_med = float(np.median(calm_vals)) if calm_vals else 1.5
  forecast = compute_forward_forecast(
      alt_km=alt_km,
      inc_deg=inc_deg,
      raan_deg=raan_deg,
      k_eff=k_eff,
      calm_ref_decay=calm_ref,
      calm_drift_km=calm_med,
      expected_kp=float(np.clip(forecast_kp, 0.0, 9.0)),
  )

  return {
      "norad_id": norad_id,
      "meta": TARGET_META[norad_id],
      "summary": {
          "may11_peak_decay_m_day": round(float(peak_decay), 1) if peak_decay is not None else None,
          "calm_median_drift_24h_km": round(calm_med, 2) if calm_vals else None,
          "storm_max_drift_24h_km": round(float(np.max(storm_vals)), 2) if storm_vals else None,
      },
      "forecast": forecast,
      "drag": drag_df.to_dict(orient="records"),
      "drift": drift_df.to_dict(orient="records"),
      "noaa_kp": fetch_noaa_kp(),
  }


@app.post("/api/evaluate")
async def evaluate_uploaded_file(
    file: UploadFile = File(...),
    expected_kp: float = Form(6.0),
):
  """Upload Space-Track JSON or CSV TLE data to audit trustworthiness and forecast 24h-72h drag/drift."""
  raw_bytes = await file.read()
  fname = (file.filename or "").lower()

  try:
    if fname.endswith(".json"):
      payload = json.loads(raw_bytes.decode("utf-8"))
      df = pd.DataFrame(payload)
    elif fname.endswith(".csv"):
      df = pd.read_csv(io.BytesIO(raw_bytes))
    else:
      # Try JSON first, fall back to CSV
      try:
        df = pd.DataFrame(json.loads(raw_bytes.decode("utf-8")))
      except Exception:
        df = pd.read_csv(io.BytesIO(raw_bytes))
  except Exception as exc:
    raise HTTPException(status_code=400, detail=f"Could not parse uploaded file: {exc}")

  audit = audit_uploaded_records(df)
  stats = audit.get("summary_stats", {})
  alt_km = float(stats.get("latest_alt_km", 445.0))
  inc_deg = float(stats.get("inc_deg", 97.4))
  raan_deg = float(stats.get("raan_deg", 0.0))
  observed_decay = float(stats.get("mean_clean_decay_m_day", 25.0))

  # Calibrate effective ballistic drag factor from the uploaded object's clean decay rate
  now_utc = pd.Timestamp.now("UTC").tz_localize(None)
  out = msis.run(now_utc.to_datetime64(), 0.0, 0.0, alt_km, version=2.1, options=[1.0] * 25)
  rho_quiet = float(np.asarray(out, dtype=float).reshape(-1, 11)[0, 0])
  proxy_quiet = rho_quiet * np.sqrt(MU_EARTH * ((6378.137 + alt_km) * 1000.0))
  baseline_decay = max(observed_decay, 12.0)
  k_eff = baseline_decay / max(proxy_quiet, 1e-12)

  forecast = compute_forward_forecast(
      alt_km=alt_km,
      inc_deg=inc_deg,
      raan_deg=raan_deg,
      k_eff=k_eff,
      calm_ref_decay=baseline_decay,
      calm_drift_km=1.8,
      expected_kp=float(np.clip(expected_kp, 0.0, 9.0)),
  )

  return {
      "filename": file.filename,
      "trust_audit": audit,
      "forecast": forecast,
  }