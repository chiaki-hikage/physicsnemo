from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. Expected raw columns
# =========================================================
EXPECTED_COLUMNS = {
    "time": "Time (sec)",
    "lat": " Latitude (deg)",
    "lon": " Longitude (deg)",
    "speed": " Vehicle speed (MPH)",
    "gps_speed": " GPS Speed (MPH)",
    "acceleration": " Acceleration (ft/s²)",
    "accel_x": " Accel X (ft/s²)",
    "accel_y": " Accel Y (ft/s²)",
    "accel_z": " Accel Z (ft/s²)",
    "accel_grav_x": " Accel (Grav) X (ft/s²)",
    "accel_grav_y": " Accel (Grav) Y (ft/s²)",
    "accel_grav_z": " Accel (Grav) Z (ft/s²)",
    "rot_x": " Rotation Rate X (deg/s)",
    "rot_y": " Rotation Rate Y (deg/s)",
    "rot_z": " Rotation Rate Z (deg/s)",   # raw sensor Z rotation, not trusted as yaw truth
    "roll": " Roll (deg)",
    "pitch": " Pitch (deg)",
    "rpm": " Engine RPM (RPM)",
    "throttle": " Absolute throttle position (%)",
    "engine_torque": " Engine Torque (lb•ft)",
    "engine_power": " Engine Power (hp)",
    "maf": " Mass air flow rate (lb/min)",
    "fuel_rate": " Fuel rate (gal/hr)",
    "calc_load": " Calculated load value (%)",
    "map_inhg": " Intake manifold absolute pressure (inHg)",
    "boost": " Boost (psi)",
    "intake_air_temp_f": " Intake air temperature (°F)",
    "coolant_temp_f": " Engine coolant temperature (°F)",
    "altitude": " Altitude (ft)",
    "bearing": " Bearing (deg)",
    "horz_accuracy": " Horz Accuracy (ft)",
}


# =========================================================
# 2. Small helpers
# =========================================================
def mph_to_mps(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) * 0.44704


def ftps2_to_mps2(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) * 0.3048


def deg_to_rad(x: np.ndarray) -> np.ndarray:
    return np.deg2rad(np.asarray(x, dtype=float))


def lbft_to_nm(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) * 1.3558179483314004


def lbmin_to_kgs(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) * 0.45359237 / 60.0


def clamp_np(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), lo, hi)


def wrap_to_pi_np(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * torch.pi
    return torch.remainder(x + torch.pi, two_pi) - torch.pi


def latlon_to_local_xy_m(lat_deg: np.ndarray, lon_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)

    valid = np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() == 0:
        return np.full_like(lat, np.nan), np.full_like(lon, np.nan)

    lat0 = lat[valid][0]
    lon0 = lon[valid][0]

    R = 6371000.0
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    x = R * np.cos(lat0_rad) * (lon_rad - lon0_rad)
    y = R * (lat_rad - lat0_rad)
    return x, y


def inverse_sigmoid_from_bounded_value(v: float, lo: float, hi: float) -> float:
    eps = 1e-6
    v = min(max(v, lo + eps), hi - eps)
    z = (v - lo) / (hi - lo)
    return float(np.log(z / (1.0 - z)))


def bounded_value(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(raw)


def set_requires_grad(params, flag: bool) -> None:
    if isinstance(params, torch.Tensor):
        params.requires_grad = flag
        return
    for p in params:
        p.requires_grad = flag


def get_column(df: pd.DataFrame, key: str) -> pd.Series:
    col = EXPECTED_COLUMNS[key]
    if col in df.columns:
        return df[col]
    stripped = {c.strip(): c for c in df.columns}
    if col.strip() in stripped:
        return df[stripped[col.strip()]]
    raise KeyError(f"Missing column for key='{key}': expected '{col}'")


def robust_corr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m &= mask
    if m.sum() < 10:
        return np.nan
    if np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


# =========================================================
# 3. Config / priors
# =========================================================
@dataclass
class VehicleSpecPriors:
    mass_kg_init: float = 2050.0
    wheelbase_m: float = 2.3
    front_weight_fraction: float = 0.55  # fraction on front axle
    cf_n_per_rad_init: float = 80000.0
    cr_n_per_rad_init: float = 90000.0

    # learnable parameter bounds
    mu_bounds: Tuple[float, float] = (0.35, 1.30)
    cf_scale_bounds: Tuple[float, float] = (0.5, 1.8)
    cr_scale_bounds: Tuple[float, float] = (0.5, 1.8)
    iz_scale_bounds: Tuple[float, float] = (0.5, 1.8)
    mass_scale_bounds: Tuple[float, float] = (0.85, 1.15)

    @property
    def lf_m(self) -> float:
        # CG to front axle
        return (1.0 - self.front_weight_fraction) * self.wheelbase_m

    @property
    def lr_m(self) -> float:
        # CG to rear axle
        return self.front_weight_fraction * self.wheelbase_m

    @property
    def iz_init(self) -> float:
        return self.mass_kg_init * self.lf_m * self.lr_m


@dataclass
class TrainConfig:
    device: str = "cpu"
    seq_len: int = 25
    batch_size: int = 64
    num_epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0

    hidden_dim: int = 32
    residual_scale: float = 0.08
    max_steer_rad: float = 0.35

    w_xy: float = 5.0
    w_v: float = 1.0
    w_r: float = 3.0
    w_psi: float = 1.0
    w_phys: float = 0.02
    w_param: float = 0.005
    w_reg: float = 1e-4

    mu_prior_sigma_rel: float = 0.25
    cf_prior_sigma_rel: float = 0.30
    cr_prior_sigma_rel: float = 0.30
    iz_prior_sigma_rel: float = 0.25
    mass_prior_sigma_rel: float = 0.08

    history_steps: int = 3
    eval_stride_div: int = 4

    horizon_steps: int = 1


# =========================================================
# 4. Preprocess
# =========================================================
def preprocess_can_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["t_s"] = pd.to_numeric(get_column(df, "time"), errors="coerce")

    # Speed / acceleration
    out["vx_mps"] = mph_to_mps(pd.to_numeric(get_column(df, "speed"), errors="coerce").to_numpy())
    out["gps_speed_mps"] = mph_to_mps(pd.to_numeric(get_column(df, "gps_speed"), errors="coerce").to_numpy())
    out["ax_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "acceleration"), errors="coerce").to_numpy())

    # Raw accel axes (diagnostics)
    out["accel_x_sensor_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "accel_x"), errors="coerce").to_numpy())
    out["ay_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "accel_y"), errors="coerce").to_numpy())
    out["az_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "accel_z"), errors="coerce").to_numpy())

    # Gravity accel axes
    out["accel_grav_x_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "accel_grav_x"), errors="coerce").to_numpy())
    out["accel_grav_y_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "accel_grav_y"), errors="coerce").to_numpy())
    out["accel_grav_z_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "accel_grav_z"), errors="coerce").to_numpy())

    # Rotation / attitude
    out["rot_x_radps"] = deg_to_rad(pd.to_numeric(get_column(df, "rot_x"), errors="coerce").to_numpy())
    out["rot_y_radps"] = deg_to_rad(pd.to_numeric(get_column(df, "rot_y"), errors="coerce").to_numpy())
    out["raw_z_radps"] = deg_to_rad(pd.to_numeric(get_column(df, "rot_z"), errors="coerce").to_numpy())
    out["roll_rad"] = deg_to_rad(pd.to_numeric(get_column(df, "roll"), errors="coerce").to_numpy())
    out["pitch_rad"] = deg_to_rad(pd.to_numeric(get_column(df, "pitch"), errors="coerce").to_numpy())

    # Powertrain
    out["rpm"] = pd.to_numeric(get_column(df, "rpm"), errors="coerce")
    out["throttle_pct"] = pd.to_numeric(get_column(df, "throttle"), errors="coerce")
    out["engine_torque_nm"] = lbft_to_nm(pd.to_numeric(get_column(df, "engine_torque"), errors="coerce").to_numpy())
    out["engine_power_hp"] = pd.to_numeric(get_column(df, "engine_power"), errors="coerce")
    out["maf_kgs"] = lbmin_to_kgs(pd.to_numeric(get_column(df, "maf"), errors="coerce").to_numpy())
    out["fuel_rate_gph"] = pd.to_numeric(get_column(df, "fuel_rate"), errors="coerce")
    out["calc_load_pct"] = pd.to_numeric(get_column(df, "calc_load"), errors="coerce")
    out["map_inhg"] = pd.to_numeric(get_column(df, "map_inhg"), errors="coerce")
    out["boost_psi"] = pd.to_numeric(get_column(df, "boost"), errors="coerce")
    out["intake_air_temp_f"] = pd.to_numeric(get_column(df, "intake_air_temp_f"), errors="coerce")
    out["coolant_temp_f"] = pd.to_numeric(get_column(df, "coolant_temp_f"), errors="coerce")

    # Position / heading
    out["altitude_m"] = pd.to_numeric(get_column(df, "altitude"), errors="coerce") * 0.3048
    out["bearing_rad"] = deg_to_rad(pd.to_numeric(get_column(df, "bearing"), errors="coerce").to_numpy())
    out["horz_accuracy_m"] = pd.to_numeric(get_column(df, "horz_accuracy"), errors="coerce") * 0.3048

    lat = pd.to_numeric(get_column(df, "lat"), errors="coerce").to_numpy()
    lon = pd.to_numeric(get_column(df, "lon"), errors="coerce").to_numpy()
    out["lat_deg"] = lat
    out["lon_deg"] = lon

    x_m, y_m = latlon_to_local_xy_m(lat, lon)
    out["x_meas_m"] = x_m
    out["y_meas_m"] = y_m

    out = out.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    out = out.sort_values("t_s").reset_index(drop=True)

    dt = np.gradient(out["t_s"].to_numpy())
    dt = clamp_np(dt, 1e-2, 10.0)
    out["dt_s"] = dt

    # GPS-based course
    out["xdot_meas_mps"] = np.gradient(out["x_meas_m"].to_numpy(), out["t_s"].to_numpy())
    out["ydot_meas_mps"] = np.gradient(out["y_meas_m"].to_numpy(), out["t_s"].to_numpy())

    xdot_s = pd.Series(out["xdot_meas_mps"]).rolling(7, center=True, min_periods=1).mean().to_numpy()
    ydot_s = pd.Series(out["ydot_meas_mps"]).rolling(7, center=True, min_periods=1).mean().to_numpy()

    course = np.unwrap(np.arctan2(ydot_s, xdot_s))
    out["course_rad"] = course
    out["xy_speed_mps"] = np.sqrt(xdot_s**2 + ydot_s**2)

    r_xy = np.gradient(out["course_rad"].to_numpy(), out["t_s"].to_numpy())
    r_xy = pd.Series(r_xy).rolling(7, center=True, min_periods=1).mean().to_numpy()
    r_xy = clamp_np(r_xy, -1.5, 1.5)
    out["r_xy_radps"] = r_xy

    out["psi_meas_rad"] = np.unwrap(out["bearing_rad"].to_numpy())
    beta_proxy = np.unwrap(out["course_rad"].to_numpy() - out["psi_meas_rad"].to_numpy())
    out["beta_proxy_rad"] = clamp_np(beta_proxy, -0.35, 0.35)

    # XY reliability
    out["is_xy_reliable"] = (
        (out["vx_mps"].to_numpy() >= 1.0) &
        (out["horz_accuracy_m"].to_numpy() <= 15.0)
    ).astype(float)

    # Gravity norm + yaw-like rate
    gx = out["accel_grav_x_mps2"].to_numpy()
    gy = out["accel_grav_y_mps2"].to_numpy()
    gz = out["accel_grav_z_mps2"].to_numpy()
    gnorm = np.sqrt(gx**2 + gy**2 + gz**2)
    out["accel_grav_norm_mps2"] = gnorm

    gnorm_safe = np.where(gnorm > 1e-6, gnorm, np.nan)
    ux = gx / gnorm_safe
    uy = gy / gnorm_safe
    uz = gz / gnorm_safe

    wx = out["rot_x_radps"].to_numpy()
    wy = out["rot_y_radps"].to_numpy()
    wz = out["raw_z_radps"].to_numpy()

    yaw_like = -(wx * ux + wy * uy + wz * uz)
    yaw_like = pd.Series(yaw_like).rolling(5, center=True, min_periods=1).mean().to_numpy()
    yaw_like = clamp_np(yaw_like, -1.5, 1.5)
    out["yaw_like_rate_radps"] = yaw_like

    out["is_yaw_reliable"] = (
        (out["vx_mps"].to_numpy() >= 1.0) &
        np.isfinite(out["yaw_like_rate_radps"].to_numpy()) &
        np.isfinite(out["r_xy_radps"].to_numpy())
    ).astype(float)

    # Supervised delta targets
    x_arr = out["x_meas_m"].to_numpy()
    y_arr = out["y_meas_m"].to_numpy()
    vx_arr = out["vx_mps"].to_numpy()
    psi_arr = out["course_rad"].to_numpy()

    out["dx_true_m"] = np.concatenate([[0.0], np.diff(x_arr)])
    out["dy_true_m"] = np.concatenate([[0.0], np.diff(y_arr)])
    out["dpsi_true_rad"] = np.concatenate([[0.0], wrap_to_pi_np(np.diff(psi_arr))])
    out["dvx_true_mps"] = np.concatenate([[0.0], np.diff(vx_arr)])

    return out


def build_training_quality_mask(
    df: pd.DataFrame,
    vx_min_mps: float = 8.0,
    abs_rproxy_max_radps: float = 0.35,
) -> pd.DataFrame:
    out = df.copy()

    out["ok_vx"] = (out["vx_mps"] >= vx_min_mps).astype(int)
    out["ok_rproxy"] = (np.abs(out["yaw_like_rate_radps"]) <= abs_rproxy_max_radps).astype(int)

    out["quality_score"] = (
        1.0 * out["ok_vx"] +
        1.0 * out["ok_rproxy"]
    )

    out["quality_mask"] = (
        (out["ok_vx"] == 1) &
        (out["ok_rproxy"] == 1)
    ).astype(int)

    return out


def find_good_time_segments(
    dfq: pd.DataFrame,
    min_duration_s: float = 120.0,
) -> pd.DataFrame:
    """
    dfq: build_training_quality_mask() 後の DataFrame
    return: 良質な連続区間一覧
    """
    x = dfq.sort_values("t_s").reset_index(drop=True).copy()
    mask = x["quality_mask"].to_numpy().astype(int)
    t = x["t_s"].to_numpy()

    segments = []
    start_idx = None

    for i, m in enumerate(mask):
        if m == 1 and start_idx is None:
            start_idx = i
        elif m == 0 and start_idx is not None:
            end_idx = i - 1
            t0 = float(t[start_idx])
            t1 = float(t[end_idx])
            dur = t1 - t0
            if dur >= min_duration_s:
                seg = x.iloc[start_idx:end_idx + 1]
                segments.append({
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "t_start": t0,
                    "t_end": t1,
                    "duration_s": dur,
                    "n_rows": int(len(seg)),
                    "mean_quality_score": float(seg["quality_score"].mean()),
                    "mean_vx_mps": float(seg["vx_mps"].mean()),
                    "mean_abs_rxy": float(np.abs(seg["r_xy_radps"]).mean()),
                })
            start_idx = None

    if start_idx is not None:
        end_idx = len(x) - 1
        t0 = float(t[start_idx])
        t1 = float(t[end_idx])
        dur = t1 - t0
        if dur >= min_duration_s:
            seg = x.iloc[start_idx:end_idx + 1]
            segments.append({
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "t_start": t0,
                "t_end": t1,
                "duration_s": dur,
                "n_rows": int(len(seg)),
                "mean_quality_score": float(seg["quality_score"].mean()),
                "mean_vx_mps": float(seg["vx_mps"].mean()),
                "mean_abs_rxy": float(np.abs(seg["r_xy_radps"]).mean()),
            })

    seg_df = pd.DataFrame(segments)
    if len(seg_df) == 0:
        return seg_df

    seg_df = seg_df.sort_values(
        ["mean_quality_score", "duration_s", "mean_vx_mps"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    return seg_df


def propose_train_val_test_windows(
    seg_df: pd.DataFrame,
    min_gap_s: float = 30.0,
) -> Dict[str, Tuple[float, float]]:
    """
    良質区間一覧から、train / val / test 用の時間窓を提案する
    なるべく別区間を選ぶ
    """
    if len(seg_df) < 3:
        raise ValueError(
            f"良質区間が不足しています。3区間以上必要ですが {len(seg_df)} 区間しかありません。"
        )

    selected = []
    for _, row in seg_df.iterrows():
        t0 = float(row["t_start"])
        t1 = float(row["t_end"])

        ok = True
        for s0, s1 in selected:
            if not (t1 + min_gap_s < s0 or t0 - min_gap_s > s1):
                ok = False
                break

        if ok:
            selected.append((t0, t1))

        if len(selected) == 3:
            break

    if len(selected) < 3:
        raise ValueError("十分に離れた train / val / test 用区間を3つ選べませんでした。")

    return {
        "train_window": selected[0],
        "val_window": selected[1],
        "test_window": selected[2],
    }

def slice_df_by_time_window(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
) -> pd.DataFrame:
    out = df[(df["t_s"] >= t_start) & (df["t_s"] <= t_end)].copy()
    out = out.sort_values("t_s").reset_index(drop=True)
    if len(out) < 50:
        raise ValueError(f"区間 [{t_start}, {t_end}] の行数が少なすぎます: len={len(out)}")
    return out



# =========================================================
# 5. Diagnostics
# =========================================================
def validate_can_training_signals(df: pd.DataFrame) -> Dict[str, float]:
    stats = {}

    vx = df["vx_mps"].to_numpy()
    gps = df["gps_speed_mps"].to_numpy()
    ax = df["ax_mps2"].to_numpy()
    t = df["t_s"].to_numpy()
    dvx_dt = np.gradient(vx, t)
    dvx_dt = pd.Series(dvx_dt).rolling(5, center=True, min_periods=1).mean().to_numpy()

    yaw_like = df["yaw_like_rate_radps"].to_numpy()
    r_xy = df["r_xy_radps"].to_numpy()
    raw_z = df["raw_z_radps"].to_numpy()

    low_speed_mask = vx >= 1.0

    stats["corr_speed_gps"] = robust_corr(vx, gps)
    stats["corr_ax_dvxdt"] = robust_corr(ax, dvx_dt)
    stats["corr_rawz_rxy_all"] = robust_corr(raw_z, r_xy)
    stats["corr_yawlike_rxy_all"] = robust_corr(yaw_like, r_xy)
    stats["corr_yawlike_rxy_vxge5"] = robust_corr(yaw_like, r_xy, low_speed_mask)
    stats["grav_norm_mean_mps2"] = float(np.nanmean(df["accel_grav_norm_mps2"].to_numpy()))
    stats["yaw_reliable_ratio"] = float(np.mean(df["is_yaw_reliable"].to_numpy()))

    print("[CAN validation]", stats)
    return stats


# =========================================================
# 6. Arrays / dataset
# =========================================================
def make_history_features(df: pd.DataFrame, history_steps: int = 3) -> np.ndarray:
    """
    Base channels:
      0 ax_mps2
      1 throttle_pct
      2 rpm
      3 vx_mps
      4 yaw_like_rate_radps
      5 is_yaw_reliable
    """
    base = np.stack([
        df["ax_mps2"].to_numpy(dtype=np.float32),
        df["throttle_pct"].to_numpy(dtype=np.float32),
        df["rpm"].to_numpy(dtype=np.float32),
        df["vx_mps"].to_numpy(dtype=np.float32),
        df["yaw_like_rate_radps"].to_numpy(dtype=np.float32),
        df["is_yaw_reliable"].to_numpy(dtype=np.float32),
    ], axis=-1)

    feats = []
    for h in range(history_steps):
        shifted = np.roll(base, shift=h, axis=0)
        if h > 0:
            shifted[:h, :] = shifted[h:h+1, :]
        feats.append(shifted)

    return np.concatenate(feats, axis=-1).astype(np.float32)


def build_can_target_arrays(
    df: pd.DataFrame,
    history_steps: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    can_feat = make_history_features(df, history_steps=history_steps)

    target_feat = np.stack([
        df["x_meas_m"].to_numpy(dtype=np.float32),            # 0
        df["y_meas_m"].to_numpy(dtype=np.float32),            # 1
        df["course_rad"].to_numpy(dtype=np.float32),          # 2
        df["vx_mps"].to_numpy(dtype=np.float32),              # 3
        df["yaw_like_rate_radps"].to_numpy(dtype=np.float32), # 4 yaw proxy
        df["dx_true_m"].to_numpy(dtype=np.float32),           # 5
        df["dy_true_m"].to_numpy(dtype=np.float32),           # 6
        df["dpsi_true_rad"].to_numpy(dtype=np.float32),       # 7
        df["dvx_true_mps"].to_numpy(dtype=np.float32),        # 8
    ], axis=-1)

    dt = df["dt_s"].to_numpy(dtype=np.float32)
    return can_feat, target_feat, dt


# =========================================================
# NEW: Center-based dataset (valid_sample対応)
# =========================================================
def build_valid_center_indices(df, history_steps, horizon_steps):
    valid = df["valid_sample"].to_numpy().astype(int)
    n = len(valid)

    centers = []
    for t in range(history_steps, n - horizon_steps):
        window = valid[t - history_steps : t + horizon_steps + 1]
        if np.all(window == 1):
            centers.append(t)

    return np.array(centers)


class CenteredDataset(Dataset):
    def __init__(self, df: pd.DataFrame, history_steps: int, horizon_steps: int):
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps

        self.can_feat, self.target_feat, self.dt = build_can_target_arrays(
            df, history_steps=history_steps
        )

        self.centers = build_valid_center_indices(
            df, history_steps, horizon_steps
        )

        if len(self.centers) == 0:
            raise ValueError("No valid centers found. Check valid_sample or window sizes.")

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx: int):
        t = self.centers[idx]

        s = t - self.history_steps
        e = t + self.horizon_steps + 1

        can_seq = torch.from_numpy(self.can_feat[s:e])
        target_seq = torch.from_numpy(self.target_feat[s:e])
        dt_seq = torch.from_numpy(self.dt[s:e])

        return can_seq, target_seq, dt_seq

# =========================================================
# 7. Model
# =========================================================
class ResidualMLP_org(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

        # 最終層を 0 初期化
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PhysicsInformedVehicleModel(nn.Module):
    """
    State: [x, y, psi, vx, beta, r]
    Input: history-stacked CAN features

    Physics core:
      - dynamic bicycle model with explicit mu / Cf / Cr / Iz / mass
      - latent steering inferred from CAN history
      - small residual corrections on beta_dot / r_dot / dvx
    """

    def __init__(
        self,
        priors: VehicleSpecPriors,
        cfg: TrainConfig,
        mass_kg: float,
        drag_terms: Dict[str, float],
        input_dim: int = 18,  # 6 base channels * history_steps
    ):
        super().__init__()
        self.priors = priors
        self.cfg = cfg
        self.mass_fixed_init = float(mass_kg)
        self.drag_terms = drag_terms

        # Learnable physical parameters (bounded around prior/init)
        self.raw_mu = nn.Parameter(torch.tensor(
            inverse_sigmoid_from_bounded_value(0.95, *self.priors.mu_bounds),
            dtype=torch.float32
        ))
        self.raw_cf_scale = nn.Parameter(torch.tensor(
            inverse_sigmoid_from_bounded_value(1.0, *self.priors.cf_scale_bounds),
            dtype=torch.float32
        ))
        self.raw_cr_scale = nn.Parameter(torch.tensor(
            inverse_sigmoid_from_bounded_value(1.0, *self.priors.cr_scale_bounds),
            dtype=torch.float32
        ))
        self.raw_iz_scale = nn.Parameter(torch.tensor(
            inverse_sigmoid_from_bounded_value(1.0, *self.priors.iz_scale_bounds),
            dtype=torch.float32
        ))
        self.raw_mass_scale = nn.Parameter(torch.tensor(
            inverse_sigmoid_from_bounded_value(1.0, *self.priors.mass_scale_bounds),
            dtype=torch.float32
        ))

        # Small residual corrections
        self.residual = ResidualMLP(in_dim=input_dim + 6, hidden=cfg.hidden_dim, out_dim=3)

        self.raw_delta_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))


    def mu(self) -> torch.Tensor:
        return bounded_value(self.raw_mu, *self.priors.mu_bounds)

    def cf(self) -> torch.Tensor:
        scale = bounded_value(self.raw_cf_scale, *self.priors.cf_scale_bounds)
        return self.priors.cf_n_per_rad_init * scale

    def cr(self) -> torch.Tensor:
        scale = bounded_value(self.raw_cr_scale, *self.priors.cr_scale_bounds)
        return self.priors.cr_n_per_rad_init * scale

    def iz(self) -> torch.Tensor:
        scale = bounded_value(self.raw_iz_scale, *self.priors.iz_scale_bounds)
        return self.priors.iz_init * scale

    def mass_kg(self) -> torch.Tensor:
        scale = bounded_value(self.raw_mass_scale, *self.priors.mass_scale_bounds)
        return self.mass_fixed_init * scale
    
    def delta_gain(self) -> torch.Tensor:
        # 0.5 ~ 2.0 の範囲
        return 0.5 + 1.5 * torch.sigmoid(self.raw_delta_gain)

    def step(
        self,
        state: torch.Tensor,
        can_step: torch.Tensor,
        dt_step: torch.Tensor,
        delta_input: Optional[torch.Tensor] = None,
        mu_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x, y, psi, vx, beta, r = [state[..., i] for i in range(6)]

        ax_meas = can_step[..., 0]
        vx_meas = can_step[..., 3]
        r_proxy = can_step[..., 4]
        yaw_rel = can_step[..., 5]

        dt = torch.clamp(dt_step.squeeze(-1), min=1e-2)
        vx_safe = torch.clamp(vx, min=1.0)

        wheelbase = self.priors.wheelbase_m
        vx_for_delta = torch.clamp(vx_meas, min=5.0)

        if delta_input is not None:
            delta_eff = delta_input
        else:
            vx_safe = torch.clamp(vx, min=1.0)
            delta_proxy = wheelbase * r_proxy / vx_for_delta
            delta_eff = self.delta_gain() * delta_proxy

        model_in = torch.cat([state, can_step], dim=-1)
        res = self.cfg.residual_scale * self.residual(model_in)

        beta_dot_res = res[..., 0]
        r_dot_res = res[..., 1]
        dvx_res = res[..., 2]

        m = self.mass_kg()
        if mu_override is not None:
            mu = mu_override.to(state.device)
            if mu.ndim == 0:
                mu = mu.expand_as(vx)
        else:
            mu = self.mu()
        cf = self.cf()
        cr = self.cr()
        iz = self.iz()
        g = 9.81

        lf = self.priors.lf_m
        lr = self.priors.lr_m

        alpha_f = beta + lf * r / vx_safe - delta_eff
        alpha_r = beta - lr * r / vx_safe

        fzf = m * g * lr / (lf + lr)
        fzr = m * g * lf / (lf + lr)

        fyf_lin = -cf * alpha_f
        fyr_lin = -cr * alpha_r

        fyf = torch.clamp(fyf_lin, -mu * fzf, mu * fzf)
        fyr = torch.clamp(fyr_lin, -mu * fzr, mu * fzr)

        beta_dot_phys = (fyf + fyr) / (m * vx_safe) - r
        r_dot_phys = (lf * fyf - lr * fyr) / iz

        beta_dot_eff = beta_dot_phys + beta_dot_res
        r_dot_eff = r_dot_phys + r_dot_res
        dvx_eff = ax_meas + dvx_res

        dbeta = beta_dot_eff * dt
        dr = r_dot_eff * dt
        dvx = dvx_eff * dt

        beta_next = torch.clamp(beta + dbeta, -0.12, 0.12)
        r_next = torch.clamp(r + dr, -0.4, 0.4)

        dpsi = r * dt
        dx = vx * torch.cos(psi + beta) * dt
        dy = vx * torch.sin(psi + beta) * dt
        
        """
        vx_mid = vx + 0.5 * dvx
        r_mid = r + 0.5 * dr
        beta_mid = beta + 0.5 * dbeta

        dpsi = r_mid * dt
        psi_mid = psi + 0.5 * dpsi

        dx = vx_mid * torch.cos(psi_mid + beta_mid) * dt
        dy = vx_mid * torch.sin(psi_mid + beta_mid) * dt
        """

        next_state = torch.stack([
            x + dx,
            y + dy,
            psi + dpsi,
            vx + dvx,
            beta_next,
            r_next,
        ], dim=-1)

        aux = {
            "dx_pred": dx,
            "dy_pred": dy,
            "dpsi_pred": dpsi,
            "dvx_pred": dvx,
            "dbeta_pred": dbeta,
            "dr_pred": dr,
            "r_pred": r_next,
            "beta_dot_phys": beta_dot_phys,
            "r_dot_phys": r_dot_phys,
            "beta_dot_eff": beta_dot_eff,
            "r_dot_eff": r_dot_eff,
            "dvx_eff": dvx_eff,
            "delta_eff": delta_eff,
            "r_proxy": r_proxy,
            "yaw_rel": yaw_rel,
            "mu": mu,
            "cf": cf,
            "cr": cr,
            "iz": iz,
            "mass_kg": m,
            "r_dot_res": r_dot_res,
            "delta_gain": self.delta_gain(),
            "beta_dot_res": beta_dot_res,
            "r_dot_res": r_dot_res,
            "dvx_res": dvx_res,
        }
        return next_state, aux
    
    def short_rollout(
        self,
        init_state: torch.Tensor,   # [B, 6]
        can_seq: torch.Tensor,      # [B, H, F]
        dt_seq: torch.Tensor,       # [B, H]
    ):
        """
        init_state から horizon_steps だけ open-loop rollout する。
        """
        batch, steps, _ = can_seq.shape
        state = init_state

        state_hist = []
        keys_bt = [
            "dx_pred", "dy_pred", "dpsi_pred", "dvx_pred",
            "dbeta_pred", "dr_pred", "r_pred",
            "beta_dot_phys", "r_dot_phys",
            "beta_dot_eff", "r_dot_eff",
            "beta_dot_res", "r_dot_res", "dvx_res",
            "dvx_eff", "delta_eff", "r_proxy", "yaw_rel",
        ]
        hist = {k: [] for k in keys_bt}

        for k in range(steps):
            dt = dt_seq[:, k].unsqueeze(-1)
            state, aux = self.step(state, can_seq[:, k, :], dt)
            state_hist.append(state)

            for kk in keys_bt:
                hist[kk].append(aux[kk])

        states = torch.stack(state_hist, dim=1)      # [B, H, 6]
        aux_seq = {k: torch.stack(v, dim=1) for k, v in hist.items()}
        return states, aux_seq

    def rollout(
        self,
        can_seq: torch.Tensor,
        target_seq: torch.Tensor,
        dt_seq: torch.Tensor,
    ):
        batch, steps, _ = can_seq.shape
        device = can_seq.device

        x0 = target_seq[:, 0, 0]
        y0 = target_seq[:, 0, 1]
        psi0 = target_seq[:, 0, 2]
        vx0 = target_seq[:, 0, 3]
        r0 = target_seq[:, 0, 4]
        beta0 = torch.zeros_like(vx0)  # can be replaced later with better beta init

        state = torch.stack([x0, y0, psi0, vx0, beta0, r0], dim=-1).to(device)

        state_hist = []
        keys_bt = [
            "dx_pred", "dy_pred", "dpsi_pred", "dvx_pred", "dbeta_pred", "dr_pred",
            "r_pred", "beta_dot_phys", "r_dot_phys", "beta_dot_eff", "r_dot_eff",
            "dvx_eff", "delta_eff", "r_proxy", "yaw_rel"
        ]
        hist = {k: [] for k in keys_bt}
        scalars_last = {}

        for k in range(steps):
            dt = dt_seq[:, k].unsqueeze(-1)
            state, aux = self.step(state, can_seq[:, k, :], dt)
            state_hist.append(state)

            for kk in keys_bt:
                hist[kk].append(aux[kk])

            scalars_last["mu"] = aux["mu"]
            scalars_last["cf"] = aux["cf"]
            scalars_last["cr"] = aux["cr"]
            scalars_last["iz"] = aux["iz"]
            scalars_last["mass_kg"] = aux["mass_kg"]

        states = torch.stack(state_hist, dim=1)
        aux_seq = {k: torch.stack(v, dim=1) for k, v in hist.items()}
        aux_seq.update(scalars_last)
        return states, aux_seq


# =========================================================
# 8. Loss
# =========================================================

def compute_losses(
    model: PhysicsInformedVehicleModel,
    can_seq: torch.Tensor,      # [B, T, F]
    target_seq: torch.Tensor,   # [B, T, 9]
    dt_seq: torch.Tensor,       # [B, T]
    cfg,
):
    """
    Center-based short-horizon rollout loss
    中心点（history_steps）から horizon_steps だけ rollout して loss 計算
    """

    B, T, _ = can_seq.shape
    H = cfg.horizon_steps

    if T < H:
        raise ValueError(f"T={T} < horizon_steps={H}")

    # =========================
    # ★ center index を使用
    # =========================
    center = cfg.history_steps
    s = center

    # =========================
    # 初期状態（中心時刻）
    # =========================
    x0 = target_seq[:, s, 0]
    y0 = target_seq[:, s, 1]
    psi0 = target_seq[:, s, 2]
    vx0 = target_seq[:, s, 3]
    r0 = target_seq[:, s, 4]
    beta0 = torch.zeros_like(vx0)

    init_state = torch.stack([x0, y0, psi0, vx0, beta0, r0], dim=-1)

    # =========================
    # horizon部分
    # =========================
    can_h = can_seq[:, s:s+H, :]
    target_h = target_seq[:, s:s+H, :]
    dt_h = dt_seq[:, s:s+H]

    # =========================
    # rollout
    # =========================
    states_pred, aux_h = model.short_rollout(init_state, can_h, dt_h)

    dx_pred = aux_h["dx_pred"]
    dy_pred = aux_h["dy_pred"]
    dpsi_pred = aux_h["dpsi_pred"]
    dvx_pred = aux_h["dvx_pred"]
    r_pred = aux_h["r_pred"]

    dx_true = target_h[..., 5]
    dy_true = target_h[..., 6]
    dpsi_true = target_h[..., 7]
    dvx_true = target_h[..., 8]
    r_true = target_h[..., 4]

    yaw_rel = can_h[..., 5]

    # =========================
    # 正規化スケール
    # =========================
    sigma_dx = torch.clamp(dx_true.std(), min=1e-1)
    sigma_dy = torch.clamp(dy_true.std(), min=1e-1)
    sigma_dpsi = torch.clamp(dpsi_true.std(), min=1e-2)
    sigma_dvx = torch.clamp(dvx_true.std(), min=1e-2)
    sigma_r = torch.clamp(r_true.std(), min=5e-2)

    err_dx = (dx_pred - dx_true) / sigma_dx
    err_dy = (dy_pred - dy_true) / sigma_dy
    err_dvx = (dvx_pred - dvx_true) / sigma_dvx
    err_r = ((r_pred - r_true) * yaw_rel) / sigma_r
    err_psi = wrap_to_pi_torch((dpsi_pred - dpsi_true) * yaw_rel) / sigma_dpsi

    # =========================
    # loss各種
    # =========================
    loss_xy = (
        torch.nn.functional.huber_loss(err_dx, torch.zeros_like(err_dx), delta=2.0)
        + torch.nn.functional.huber_loss(err_dy, torch.zeros_like(err_dy), delta=2.0)
    )
    loss_v = torch.nn.functional.huber_loss(err_dvx, torch.zeros_like(err_dvx), delta=2.0)
    loss_r = torch.nn.functional.huber_loss(err_r, torch.zeros_like(err_r), delta=2.0)
    loss_psi = torch.nn.functional.huber_loss(err_psi, torch.zeros_like(err_psi), delta=2.0)

    # =========================
    # 物理整合性
    # =========================
    beta_dot_phys = aux_h["beta_dot_phys"]
    r_dot_phys = aux_h["r_dot_phys"]
    beta_dot_eff = aux_h["beta_dot_eff"]
    r_dot_eff = aux_h["r_dot_eff"]

    sigma_beta_dot = torch.clamp(beta_dot_phys.std().detach(), min=5e-2)
    sigma_r_dot = torch.clamp(r_dot_phys.std().detach(), min=5e-2)

    err_bicycle_beta = (beta_dot_eff - beta_dot_phys) / sigma_beta_dot
    err_bicycle_r = (r_dot_eff - r_dot_phys) / sigma_r_dot

    loss_bicycle = (
        torch.nn.functional.huber_loss(err_bicycle_beta, torch.zeros_like(err_bicycle_beta), delta=2.0)
        + torch.nn.functional.huber_loss(err_bicycle_r, torch.zeros_like(err_bicycle_r), delta=2.0)
    )

    # =========================
    # residual regularization
    # =========================
    beta_dot_res = aux_h["beta_dot_res"]
    r_dot_res = aux_h["r_dot_res"]
    dvx_res = aux_h["dvx_res"]
    delta_eff = aux_h["delta_eff"]
    dvx_eff = aux_h["dvx_eff"]

    loss_reg = (
        1e-2 * (delta_eff ** 2).mean()
        + 1e-3 * (beta_dot_res ** 2).mean()
        + 1e-3 * (r_dot_res ** 2).mean()
        + 1e-3 * (dvx_res ** 2).mean()
        + 1e-2 * ((dvx_eff - can_h[..., 0]) ** 2).mean()
    )

    # =========================
    # parameter prior
    # =========================
    mu_rel = model.mu() / 0.95 - 1.0
    cf_rel = model.cf() / model.priors.cf_n_per_rad_init - 1.0
    cr_rel = model.cr() / model.priors.cr_n_per_rad_init - 1.0
    iz_rel = model.iz() / model.priors.iz_init - 1.0
    mass_rel = model.mass_kg() / model.mass_fixed_init - 1.0

    loss_param = (
        (mu_rel / cfg.mu_prior_sigma_rel) ** 2
        + (cf_rel / cfg.cf_prior_sigma_rel) ** 2
        + (cr_rel / cfg.cr_prior_sigma_rel) ** 2
        + (iz_rel / cfg.iz_prior_sigma_rel) ** 2
        + (mass_rel / cfg.mass_prior_sigma_rel) ** 2
    )

    # =========================
    # total loss
    # =========================
    total = (
        cfg.w_xy * loss_xy
        + cfg.w_v * loss_v
        + cfg.w_r * loss_r
        + cfg.w_psi * loss_psi
        + cfg.w_phys * loss_bicycle
        + cfg.w_param * loss_param
        + cfg.w_reg * loss_reg
    )

    # =========================
    # logging
    # =========================
    beta_dot_res_cat = beta_dot_res.reshape(-1)
    r_dot_res_cat = r_dot_res.reshape(-1)
    dvx_res_cat = dvx_res.reshape(-1)

    logs = {
        "loss": float(total.detach().cpu().item()),
        "loss_xy": float(loss_xy.detach().cpu().item()),
        "loss_v": float(loss_v.detach().cpu().item()),
        "loss_r": float(loss_r.detach().cpu().item()),
        "loss_psi": float(loss_psi.detach().cpu().item()),
        "loss_bicycle": float(loss_bicycle.detach().cpu().item()),
        "loss_param": float(loss_param.detach().cpu().item()),
        "loss_reg": float(loss_reg.detach().cpu().item()),
        "mu": float(model.mu().detach().cpu().item()),
        "cf": float(model.cf().detach().cpu().item()),
        "cr": float(model.cr().detach().cpu().item()),
        "iz": float(model.iz().detach().cpu().item()),
        "mass_kg": float(model.mass_kg().detach().cpu().item()),
        "mean_beta_dot_res": float(beta_dot_res_cat.mean().detach().cpu().item()),
        "mean_abs_beta_dot_res": float(beta_dot_res_cat.abs().mean().detach().cpu().item()),
        "mean_r_dot_res": float(r_dot_res_cat.mean().detach().cpu().item()),
        "mean_abs_r_dot_res": float(r_dot_res_cat.abs().mean().detach().cpu().item()),
        "mean_dvx_res": float(dvx_res_cat.mean().detach().cpu().item()),
        "mean_abs_dvx_res": float(dvx_res_cat.abs().mean().detach().cpu().item()),
        "delta_gain": float(model.delta_gain().detach().cpu().item()),
    }

    return total, logs


# =========================================================
# 9. Split / crop helpers
# =========================================================
def crop_df_by_time_range(
    df: pd.DataFrame,
    min_time: float | None = None,
    max_time: float | None = None,
) -> pd.DataFrame:
    out = df.sort_values("t_s").reset_index(drop=True).copy()
    if min_time is not None:
        out = out[out["t_s"] >= float(min_time)]
    if max_time is not None:
        out = out[out["t_s"] <= float(max_time)]
    out = out.reset_index(drop=True)

    if len(out) < 50:
        raise ValueError(
            f"Too few rows after time filtering: len={len(out)}, min_time={min_time}, max_time={max_time}"
        )
    return out


def make_time_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, pd.DataFrame]:
    df = df.sort_values("t_s").reset_index(drop=True)
    n = len(df)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n_train < 100 or n_val < 50 or n_test < 50:
        raise ValueError(
            f"Not enough samples after split: n={n}, n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )

    return {
        "train": df.iloc[:n_train].reset_index(drop=True),
        "val": df.iloc[n_train:n_train + n_val].reset_index(drop=True),
        "test": df.iloc[n_train + n_val:].reset_index(drop=True),
    }


# =========================================================
# 10. Evaluation / plots
# =========================================================
@torch.no_grad()
def evaluate_model_on_windows(
    model: torch.nn.Module,
    df: pd.DataFrame,
    cfg,
    seq_len: int | None = None,
    stride: int | None = None,
    history_steps: int = 3,
) -> Dict[str, float]:
    model.eval()

    can_feat, target_feat, dt = build_can_target_arrays(df, history_steps=history_steps)

    H = cfg.horizon_steps
    n = len(df)

    if stride is None:
        stride = 1

    centers = build_valid_center_indices(df, history_steps, cfg.horizon_steps)

    dx_true_all, dx_pred_all = [], []
    dy_true_all, dy_pred_all = [], []
    dvx_true_all, dvx_pred_all = [], []
    r_true_all, r_pred_all = [], []
    psi_true_all, psi_pred_all = [], []

    dx_errs, dy_errs, dvx_errs, r_errs = [], [], [], []

    for t in centers:
        s = t
        e = s + H
        if e > n:
            break

        can_h = torch.from_numpy(can_feat[s:e]).unsqueeze(0).to(cfg.device)
        target_h = torch.from_numpy(target_feat[s:e]).unsqueeze(0).to(cfg.device)
        dt_h = torch.from_numpy(dt[s:e]).unsqueeze(0).to(cfg.device)

        x0 = target_h[:, 0, 0]
        y0 = target_h[:, 0, 1]
        psi0 = target_h[:, 0, 2]
        vx0 = target_h[:, 0, 3]
        r0 = target_h[:, 0, 4]
        beta0 = torch.zeros_like(vx0)
        init_state = torch.stack([x0, y0, psi0, vx0, beta0, r0], dim=-1)

        states, aux_h = model.short_rollout(init_state, can_h, dt_h)

        dx_pred = aux_h["dx_pred"][0].detach().cpu().numpy()
        dy_pred = aux_h["dy_pred"][0].detach().cpu().numpy()
        dvx_pred = aux_h["dvx_pred"][0].detach().cpu().numpy()
        r_pred = aux_h["r_pred"][0].detach().cpu().numpy()
        psi_pred = states[0, :, 2].detach().cpu().numpy()

        dx_true = target_h[0, :, 5].detach().cpu().numpy()
        dy_true = target_h[0, :, 6].detach().cpu().numpy()
        dvx_true = target_h[0, :, 8].detach().cpu().numpy()
        r_true = target_h[0, :, 4].detach().cpu().numpy()
        psi_true = target_h[0, :, 2].detach().cpu().numpy()

        dx_errs.append(float(np.sqrt(np.mean((dx_pred - dx_true) ** 2))))
        dy_errs.append(float(np.sqrt(np.mean((dy_pred - dy_true) ** 2))))
        dvx_errs.append(float(np.sqrt(np.mean((dvx_pred - dvx_true) ** 2))))
        r_errs.append(float(np.sqrt(np.mean((r_pred - r_true) ** 2))))

        dx_true_all.append(dx_true); dx_pred_all.append(dx_pred)
        dy_true_all.append(dy_true); dy_pred_all.append(dy_pred)
        dvx_true_all.append(dvx_true); dvx_pred_all.append(dvx_pred)
        r_true_all.append(r_true); r_pred_all.append(r_pred)
        psi_true_all.append(psi_true); psi_pred_all.append(psi_pred)

    def _cat(xs):
        return np.concatenate(xs) if len(xs) > 0 else np.array([])

    return {
        "rmse_dx": float(np.mean(dx_errs)) if dx_errs else np.nan,
        "rmse_dy": float(np.mean(dy_errs)) if dy_errs else np.nan,
        "rmse_dvx": float(np.mean(dvx_errs)) if dvx_errs else np.nan,
        "rmse_r_proxy": float(np.mean(r_errs)) if r_errs else np.nan,

        "dx_true_all": _cat(dx_true_all),
        "dx_pred_all": _cat(dx_pred_all),
        "dy_true_all": _cat(dy_true_all),
        "dy_pred_all": _cat(dy_pred_all),
        "dvx_true_all": _cat(dvx_true_all),
        "dvx_pred_all": _cat(dvx_pred_all),
        "r_true_all": _cat(r_true_all),
        "r_pred_all": _cat(r_pred_all),
        "psi_true_all": _cat(psi_true_all),
        "psi_pred_all": _cat(psi_pred_all),
    }


@torch.no_grad()
def evaluate_state_timeseries(
    model: nn.Module,
    df: pd.DataFrame,
    cfg,
    history_steps: int = 3,
) -> pd.DataFrame:
    model.eval()

    can_feat, target_feat, dt = build_can_target_arrays(df, history_steps=history_steps)

    n = len(df)
    rows = []

    centers = build_valid_center_indices(df, history_steps, 1)

    for i in centers:
        can_step = torch.from_numpy(can_feat[i:i+1]).to(cfg.device)          # [1, F]
        target_step = torch.from_numpy(target_feat[i:i+1]).to(cfg.device)    # [1, 9]
        dt_step = torch.tensor([[dt[i]]], dtype=torch.float32, device=cfg.device)  # [1,1]

        # state initialization from current target
        x0 = target_step[:, 0]
        y0 = target_step[:, 1]
        psi0 = target_step[:, 2]
        vx0 = target_step[:, 3]
        r0 = target_step[:, 4]
        beta0 = torch.zeros_like(vx0)

        state = torch.stack([x0, y0, psi0, vx0, beta0, r0], dim=-1)

        next_state, aux = model.step(state, can_step, dt_step)

        row = {
            "t_s": float(df["t_s"].iloc[i]),

            "x_true": float(target_step[0, 0].detach().cpu().item()),
            "y_true": float(target_step[0, 1].detach().cpu().item()),
            "course_true": float(target_step[0, 2].detach().cpu().item()),
            "vx_true": float(target_step[0, 3].detach().cpu().item()),
            "r_proxy_true": float(target_step[0, 4].detach().cpu().item()),

            "psi_pred": float(next_state[0, 2].detach().cpu().item()),
            "vx_pred": float(next_state[0, 3].detach().cpu().item()),
            "beta_pred": float(next_state[0, 4].detach().cpu().item()),
            "r_pred": float(next_state[0, 5].detach().cpu().item()),

            "dpsi_pred": float(aux["dpsi_pred"][0].detach().cpu().item()),
            "dx_pred": float(aux["dx_pred"][0].detach().cpu().item()),
            "dy_pred": float(aux["dy_pred"][0].detach().cpu().item()),
            "dvx_pred": float(aux["dvx_pred"][0].detach().cpu().item()),
            "delta_eff": float(aux["delta_eff"][0].detach().cpu().item()),

            "yaw_like_rate": float(df["yaw_like_rate_radps"].iloc[i]),
            "r_xy_true": float(df["r_xy_radps"].iloc[i]),
        }
        rows.append(row)

    out = pd.DataFrame(rows)

    out["psi_plus_beta_pred"] = wrap_to_pi_np(
        out["psi_pred"].to_numpy() + out["beta_pred"].to_numpy()
    )

    out["course_minus_psibeta"] = wrap_to_pi_np(
        out["course_true"].to_numpy() - out["psi_plus_beta_pred"].to_numpy()
    )

    return out

def _scatter_panel(ax, x, y, title, xlabel, ylabel):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) == 0:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return

    ax.scatter(x, y, s=8, alpha=0.35)
    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.0)

    corr = np.corrcoef(x, y)[0, 1] if (len(x) >= 2 and np.std(x) > 1e-12 and np.std(y) > 1e-12) else np.nan
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    mae = float(np.mean(np.abs(y - x)))

    ax.text(
        0.03, 0.97,
        f"n={len(x)}\nr={corr:.3f}\nRMSE={rmse:.3f}\nMAE={mae:.3f}",
        ha="left", va="top", transform=ax.transAxes
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_final_scatter_plots_windowed(preds: Dict[str, np.ndarray], split_name: str, outdir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    axes = axes.ravel()

    _scatter_panel(axes[0], preds["dx_true_all"], preds["dx_pred_all"], f"{split_name}: dX_pred vs dX_true", "dX_true [m]", "dX_pred [m]")
    _scatter_panel(axes[1], preds["dy_true_all"], preds["dy_pred_all"], f"{split_name}: dY_pred vs dY_true", "dY_true [m]", "dY_pred [m]")
    _scatter_panel(axes[2], preds["dvx_true_all"], preds["dvx_pred_all"], f"{split_name}: dVx_pred vs dVx_true", "dVx_true [m/s]", "dVx_pred [m/s]")
    _scatter_panel(axes[3], preds["r_true_all"], preds["r_pred_all"], f"{split_name}: r_pred vs yaw_proxy", "r_proxy [rad/s]", "r_pred [rad/s]")
    _scatter_panel(axes[4], preds["psi_true_all"], preds["psi_pred_all"], f"{split_name}: psi_pred vs psi_true", "psi_true [rad]", "psi_pred [rad]")
    axes[5].axis("off")

    fig.tight_layout()
    fig.savefig(outdir / f"{split_name.lower()}_scatter_final.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_metric_history(history_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()

    metric_names = [
        "loss",
        "rmse_dx",
        "rmse_dy",
        "rmse_dvx",
        "rmse_r_proxy",
    ]

    for ax, m in zip(axes, metric_names):
        if m == "loss":
            ax.plot(history_df["epoch"], history_df["train_loss"], label="train")
            ax.plot(history_df["epoch"], history_df["val_loss"], label="val")
            ax.plot(history_df["epoch"], history_df["test_loss"], label="test")
            ax.set_ylabel("loss")
        else:
            ax.plot(history_df["epoch"], history_df[f"train_{m}"], label="train")
            ax.plot(history_df["epoch"], history_df[f"val_{m}"], label="val")
            ax.plot(history_df["epoch"], history_df[f"test_{m}"], label="test")
            ax.set_ylabel(m)

        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].axis("off")

    for ax in axes[:-1]:
        ax.set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(outdir / "epoch_split_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss_components(history_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "train_loss",
        "loss_xy",
        "loss_v",
        "loss_r",
        "loss_psi",
        "loss_bicycle",
        "loss_param",
        "loss_reg",
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True)
    axes = axes.ravel()

    for ax, m in zip(axes, metrics):
        if m in history_df.columns:
            ax.plot(history_df["epoch"], history_df[m], linewidth=1.5)
            ax.set_title(m)
            ax.set_ylabel(m)
            ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(outdir / "training_loss_components.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_history(history_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = ["mu", "cf", "cr", "iz", "mass_kg", "delta_gain"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        if m in history_df.columns:
            ax.plot(history_df["epoch"], history_df[m], linewidth=1.5)
            ax.set_ylabel(m)
            ax.set_title(m)
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(outdir / "parameter_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_preprocessed_can_dataframe(df: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    t = df["t_s"].to_numpy()

    plot_groups = [
        {
            "filename": "can_overview_motion.png",
            "title": "CAN Overview: Motion States",
            "columns": [
                ("vx_mps", "vx [m/s]"),
                ("gps_speed_mps", "gps speed [m/s]"),
                ("ax_mps2", "ax [m/s^2]"),
                ("yaw_like_rate_radps", "yaw-like [rad/s]"),
                ("r_xy_radps", "r_xy GPS [rad/s]"),
                ("raw_z_radps", "raw rot Z [rad/s]"),
                ("is_yaw_reliable", "yaw reliable"),
            ],
        },
        {
            "filename": "can_overview_powertrain.png",
            "title": "CAN Overview: Powertrain",
            "columns": [
                ("rpm", "rpm"),
                ("throttle_pct", "throttle [%]"),
            ],
        },
        {
            "filename": "can_overview_position.png",
            "title": "CAN Overview: Position / Heading",
            "columns": [
                ("x_meas_m", "x [m]"),
                ("y_meas_m", "y [m]"),
                ("course_rad", "course [rad]"),
                ("dx_true_m", "dx true [m]"),
                ("dy_true_m", "dy true [m]"),
                ("dvx_true_mps", "dvx true [m/s]"),
            ],
        },
        {
            "filename": "can_overview_gravity.png",
            "title": "CAN Overview: Gravity / Rotation",
            "columns": [
                ("accel_grav_x_mps2", "grav x [m/s^2]"),
                ("accel_grav_y_mps2", "grav y [m/s^2]"),
                ("accel_grav_z_mps2", "grav z [m/s^2]"),
                ("accel_grav_norm_mps2", "grav norm [m/s^2]"),
                ("rot_x_radps", "rot x [rad/s]"),
                ("rot_y_radps", "rot y [rad/s]"),
                ("raw_z_radps", "rot z [rad/s]"),
            ],
        },
    ]

    for group in plot_groups:
        cols = [(c, label) for c, label in group["columns"] if c in df.columns]
        if not cols:
            continue

        fig, axes = plt.subplots(len(cols), 1, figsize=(14, 2.6 * len(cols)), sharex=True)
        if len(cols) == 1:
            axes = [axes]

        for ax, (col, label) in zip(axes, cols):
            ax.plot(t, df[col].to_numpy(), linewidth=1.0)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("time [s]")
        fig.suptitle(group["title"])
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, group["filename"]), dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_state_timeseries_diagnostics(
    state_df: pd.DataFrame,
    outdir: Path,
    suffix: str = "",
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    t = state_df["t_s"].to_numpy()

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    # 1) course と psi+beta
    axes[0].plot(t, state_df["course_true"].to_numpy(), label="course_true", linewidth=1.5)
    axes[0].plot(t, state_df["psi_pred"].to_numpy(), label="psi_pred", linewidth=1.2)
    axes[0].plot(t, state_df["psi_plus_beta_pred"].to_numpy(), label="psi_pred + beta_pred", linewidth=1.2)
    axes[0].set_ylabel("angle [rad]")
    axes[0].set_title("Course vs psi / (psi+beta)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2) beta
    axes[1].plot(t, state_df["beta_pred"].to_numpy(), label="beta_pred", linewidth=1.2)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("beta [rad]")
    axes[1].set_title("Predicted slip angle beta")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3) yaw rate / course derivative
    axes[2].plot(t, state_df["r_xy_true"].to_numpy(), label="r_xy_true = d(course)/dt", linewidth=1.5)
    axes[2].plot(t, state_df["yaw_like_rate"].to_numpy(), label="yaw_like_rate", linewidth=1.2)
    axes[2].plot(t, state_df["r_pred"].to_numpy(), label="r_pred", linewidth=1.2)
    axes[2].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("yaw rate [rad/s]")
    axes[2].set_title("Yaw-related signals")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # 4) delta
    axes[3].plot(t, state_df["delta_eff"].to_numpy(), label="delta_eff", linewidth=1.2)
    axes[3].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("delta [rad]")
    axes[3].set_title("Effective steering proxy")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    # 5) course - (psi+beta)
    axes[4].plot(t, state_df["course_minus_psibeta"].to_numpy(), label="course_true - (psi_pred+beta_pred)", linewidth=1.2)
    axes[4].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[4].set_ylabel("angle diff [rad]")
    axes[4].set_xlabel("time [s]")
    axes[4].set_title("Mismatch between course and predicted motion direction")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    fig.tight_layout()
    out_png = outdir / f"state_timeseries_diagnostics{suffix}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_state_timeseries_zoom(
    state_df: pd.DataFrame,
    outdir: Path,
    t_min: float,
    t_max: float,
    suffix: str = "",
) -> None:
    sub = state_df[(state_df["t_s"] >= t_min) & (state_df["t_s"] <= t_max)].copy()
    if len(sub) < 5:
        return

    plot_state_timeseries_diagnostics(
        sub,
        outdir=outdir,
        suffix=suffix if suffix else f"_zoom_{int(t_min)}_{int(t_max)}",
    )


def plot_quality_and_selected_windows(
    dfq: pd.DataFrame,
    windows: Dict[str, Tuple[float, float]],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    t = dfq["t_s"].to_numpy()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(t, dfq["quality_score"].to_numpy(), label="quality_score")
    axes[0].set_ylabel("quality")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, dfq["vx_mps"].to_numpy(), label="vx_mps")
    axes[1].set_ylabel("vx [m/s]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, dfq["r_xy_radps"].to_numpy(), label="r_xy_radps")
    axes[2].plot(t, dfq["yaw_like_rate_radps"].to_numpy(), label="yaw_like_rate_radps", alpha=0.8)
    axes[2].set_ylabel("yaw [rad/s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(t, dfq["horz_accuracy_m"].to_numpy(), label="horz_accuracy_m")
    axes[3].set_ylabel("horz acc [m]")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(True, alpha=0.3)

    colors = {
        "train_window": "#1f77b4",
        "val_window": "#ff7f0e",
        "test_window": "#2ca02c",
    }

    for ax in axes:
        for k, (t0, t1) in windows.items():
            ax.axvspan(t0, t1, color=colors[k], alpha=0.12)

    fig.tight_layout()
    fig.savefig(outdir / "quality_and_selected_windows.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def split_df_by_time_ratio(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, pd.DataFrame]:
    df = df.sort_values("t_s").reset_index(drop=True)

    n = len(df)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:i_train].copy().reset_index(drop=True)
    val_df = df.iloc[i_train:i_val].copy().reset_index(drop=True)
    test_df = df.iloc[i_val:].copy().reset_index(drop=True)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

def build_training_quality_mask(
    df: pd.DataFrame,
    vx_min_mps: float = 1.0,
    abs_rproxy_max_radps: float = 0.35,
) -> pd.DataFrame:
    out = df.copy()

    out["ok_vx"] = (out["vx_mps"] >= vx_min_mps).astype(int)
    out["ok_rproxy"] = (np.abs(out["yaw_like_rate_radps"]) <= abs_rproxy_max_radps).astype(int)

    out["quality_score"] = (
        1.0 * out["ok_vx"] +
        1.0 * out["ok_rproxy"]
    )

    out["quality_ok"] = (
        (out["ok_vx"] == 1) &
        (out["ok_rproxy"] == 1)
    ).astype(int)

    return out


def add_valid_sample_mask(
    df: pd.DataFrame,
    history_steps: int,
    horizon_steps: int,
    quality_col: str = "quality_ok",
) -> pd.DataFrame:
    """
    各時刻 t について、
    - t-history_steps+1 ... t がすべて quality 条件を満たす
    - t ... t+horizon_steps-1 がデータ範囲内にある
    とき valid_sample=1 とする
    """
    out = df.copy().reset_index(drop=True)
    n = len(out)

    q = out[quality_col].to_numpy().astype(int)
    valid = np.zeros(n, dtype=int)

    for t in range(n):
        hist_start = t - history_steps + 1
        fut_end = t + horizon_steps - 1

        if hist_start < 0:
            continue
        if fut_end >= n:
            continue

        if np.all(q[hist_start:t+1] == 1):
            valid[t] = 1

    out["valid_sample"] = valid
    return out


def prepare_train_val_test_splits(
    df: pd.DataFrame,
    outdir: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    vx_min_mps: float = 1.0,
    abs_rproxy_max_radps: float = 0.35,
    history_steps: int = 3,
    horizon_steps: int = 1,
) -> Dict[str, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)

    splits = split_df_by_time_ratio(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    result = {}
    summary_rows = []

    for split_name, sdf in splits.items():
        sdf = build_training_quality_mask(
            sdf,
            vx_min_mps=vx_min_mps,
            abs_rproxy_max_radps=abs_rproxy_max_radps,
        )
        sdf = add_valid_sample_mask(
            sdf,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            quality_col="quality_ok",
        )

        sdf.to_csv(outdir / f"{split_name}_split_with_quality.csv", index=False)

        result[split_name] = sdf
        summary_rows.append({
            "split": split_name,
            "n_rows": len(sdf),
            "n_quality_ok": int(sdf["quality_ok"].sum()),
            "n_valid_sample": int(sdf["valid_sample"].sum()),
            "t_start": float(sdf["t_s"].iloc[0]),
            "t_end": float(sdf["t_s"].iloc[-1]),
        })

    pd.DataFrame(summary_rows).to_csv(outdir / "split_summary.csv", index=False)

    return result


def plot_residual_output_timeseries(
    pred_df: pd.DataFrame,
    outdir: str | Path,
    residual_threshold_quantile: float = 0.99,
) -> None:
    """
    residual 出力の時系列を描く。
    大きな residual が出ている時刻を背景色でハイライトする。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t = pred_df["t_s"].to_numpy()

    # どれかの residual が大きい時刻を検出
    max_abs_res = np.maximum.reduce([
        np.abs(pred_df["beta_dot_res"].to_numpy()),
        np.abs(pred_df["r_dot_res"].to_numpy()),
        np.abs(pred_df["dvx_res"].to_numpy()),
    ])
    thr = float(np.quantile(max_abs_res, residual_threshold_quantile))
    bad_mask = max_abs_res >= thr

    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)

    # 1) beta_dot_res
    axes[0].plot(t, pred_df["beta_dot_res"], label="beta_dot_res", linewidth=1.2)
    axes[0].plot(t, pred_df["beta_dot_phys"], label="beta_dot_phys", linewidth=1.0)
    axes[0].plot(t, pred_df["beta_dot_eff"], label="beta_dot_eff", linewidth=1.0)
    axes[0].set_ylabel("beta_dot")
    axes[0].set_title("beta residual / phys / eff")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2) r_dot_res
    axes[1].plot(t, pred_df["r_dot_res"], label="r_dot_res", linewidth=1.2)
    axes[1].plot(t, pred_df["r_dot_phys"], label="r_dot_phys", linewidth=1.0)
    axes[1].plot(t, pred_df["r_dot_eff"], label="r_dot_eff", linewidth=1.0)
    axes[1].set_ylabel("r_dot")
    axes[1].set_title("r residual / phys / eff")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3) dvx_res
    axes[2].plot(t, pred_df["dvx_res"], label="dvx_res", linewidth=1.2)
    axes[2].plot(t, pred_df["dvx_eff"], label="dvx_eff", linewidth=1.0)
    if "ax_mps2" in pred_df.columns:
        axes[2].plot(t, pred_df["ax_mps2"], label="ax_mps2", linewidth=1.0)
    axes[2].set_ylabel("dvx")
    axes[2].set_title("dvx residual / eff / meas")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # 4) r_pred / r_true / yaw proxy
    axes[3].plot(t, pred_df["r_true"], label="r_true", linewidth=1.2)
    axes[3].plot(t, pred_df["r_pred"], label="r_pred", linewidth=1.2)
    if "yaw_like_rate_radps" in pred_df.columns:
        axes[3].plot(t, pred_df["yaw_like_rate_radps"], label="yaw_like_rate", linewidth=1.0)
    axes[3].set_ylabel("r [rad/s]")
    axes[3].set_title("r prediction")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    # 5) step errors
    axes[4].plot(t, pred_df["err_r"], label="err_r", linewidth=1.1)
    axes[4].plot(t, pred_df["err_dx"], label="err_dx", linewidth=1.1)
    axes[4].plot(t, pred_df["err_dy"], label="err_dy", linewidth=1.1)
    axes[4].plot(t, pred_df["err_dvx"], label="err_dvx", linewidth=1.1)
    axes[4].set_ylabel("prediction errors")
    axes[4].set_title("prediction errors")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(ncol=4)

    # 6) related vars
    plotted = False
    for c in ["vx_mps", "throttle_pct", "rpm", "raw_z_radps", "course_rate_radps"]:
        if c in pred_df.columns:
            axes[5].plot(t, pred_df[c], label=c, linewidth=1.0)
            plotted = True
    axes[5].set_ylabel("related vars")
    axes[5].set_xlabel("time [s]")
    axes[5].set_title("related signals")
    axes[5].grid(True, alpha=0.3)
    if plotted:
        axes[5].legend(ncol=4)

    # bad residual 時刻をハイライト
    bad_idx = np.where(bad_mask)[0]
    for ax in axes:
        for idx in bad_idx:
            ax.axvspan(t[idx], t[idx], alpha=0.15, color="red")

    fig.tight_layout()
    fig.savefig(outdir / "residual_output_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_residual_segments(
    pred_df: pd.DataFrame,
    outdir: str | Path,
    top_k: int = 8,
    pad_steps: int = 40,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    max_abs_res = np.maximum.reduce([
        np.abs(pred_df["beta_dot_res"].to_numpy()),
        np.abs(pred_df["r_dot_res"].to_numpy()),
        np.abs(pred_df["dvx_res"].to_numpy()),
    ])
    pred_df = pred_df.copy()
    pred_df["max_abs_res"] = max_abs_res

    top_idx = np.argsort(-pred_df["max_abs_res"].to_numpy())[:top_k]

    for rank, center in enumerate(top_idx, start=1):
        s = max(int(center) - pad_steps, 0)
        e = min(int(center) + pad_steps, len(pred_df) - 1)
        sub = pred_df.iloc[s:e+1].copy()
        t = sub["t_s"].to_numpy()

        fig, axes = plt.subplots(5, 1, figsize=(16, 15), sharex=True)

        axes[0].plot(t, sub["beta_dot_res"], label="beta_dot_res")
        axes[0].plot(t, sub["r_dot_res"], label="r_dot_res")
        axes[0].plot(t, sub["dvx_res"], label="dvx_res")
        axes[0].set_ylabel("residual outputs")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(t, sub["beta_dot_phys"], label="beta_dot_phys")
        axes[1].plot(t, sub["r_dot_phys"], label="r_dot_phys")
        axes[1].plot(t, sub["dvx_eff"], label="dvx_eff")
        axes[1].set_ylabel("phys / eff")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(t, sub["r_true"], label="r_true")
        axes[2].plot(t, sub["r_pred"], label="r_pred")
        if "yaw_like_rate_radps" in sub.columns:
            axes[2].plot(t, sub["yaw_like_rate_radps"], label="yaw_like_rate")
        axes[2].set_ylabel("r")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        axes[3].plot(t, sub["err_r"], label="err_r")
        axes[3].plot(t, sub["err_dx"], label="err_dx")
        axes[3].plot(t, sub["err_dy"], label="err_dy")
        axes[3].plot(t, sub["err_dvx"], label="err_dvx")
        axes[3].set_ylabel("errors")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(ncol=4)

        plotted = False
        for c in ["vx_mps", "throttle_pct", "rpm", "ax_mps2", "raw_z_radps"]:
            if c in sub.columns:
                axes[4].plot(t, sub[c], label=c)
                plotted = True
        axes[4].set_ylabel("related vars")
        axes[4].set_xlabel("time [s]")
        axes[4].grid(True, alpha=0.3)
        if plotted:
            axes[4].legend(ncol=4)

        t_center = pred_df["t_s"].iloc[center]
        for ax in axes:
            ax.axvline(t_center, linestyle="--", alpha=0.5, color="red")

        fig.tight_layout()
        fig.savefig(outdir / f"top_residual_segment_{rank:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_residual_history(history_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    metrics = [
        "mean_beta_dot_res",
        "mean_abs_beta_dot_res",
        "mean_r_dot_res",
        "mean_abs_r_dot_res",
        "mean_dvx_res",
        "mean_abs_dvx_res",
    ]

    for ax, m in zip(axes, metrics):
        if m in history_df.columns:
            ax.plot(history_df["epoch"], history_df[m], label=m)
            ax.set_title(m)
            ax.grid(True, alpha=0.3)
            ax.legend()

    for ax in axes:
        ax.set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(outdir / "residual_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# =========================================================
# 11. Parameter summary
# =========================================================
def summarize_parameters(model: PhysicsInformedVehicleModel, mass_fit: Dict[str, float]) -> Dict[str, float]:
    return {
        "mass_fit_input_kg": float(mass_fit.get("mass_kg", np.nan)),
        "mass_kg": float(model.mass_kg().detach().cpu().item()),
        "mu": float(model.mu().detach().cpu().item()),
        "cf_n_per_rad": float(model.cf().detach().cpu().item()),
        "cr_n_per_rad": float(model.cr().detach().cpu().item()),
        "iz": float(model.iz().detach().cpu().item()),
    }


# =========================================================
# 12. Training
# =========================================================

def train_vehicle_model(
    df: pd.DataFrame,
    priors: VehicleSpecPriors,
    cfg: TrainConfig,
    mass_fit: Dict[str, float],
    outdir: Path,
    plot_every: int = 10,
    history_steps: int = 3,
    auto_find_windows: bool = True,
    train_window: Tuple[float, float] | None = None,
    val_window: Tuple[float, float] | None = None,
    test_window: Tuple[float, float] | None = None,
) -> PhysicsInformedVehicleModel:

    outdir.mkdir(parents=True, exist_ok=True)


    if auto_find_windows:
        splits = prepare_train_val_test_splits(
            df=df,
            outdir=outdir / "window_selection",
            train_ratio=0.6,
            val_ratio=0.2,
            vx_min_mps=1.0,
            abs_rproxy_max_radps=0.35,
            history_steps=history_steps,
            horizon_steps=cfg.horizon_steps,
        )

        train_df = splits["train"]
        val_df = splits["val"]
        test_df = splits["test"]
    else:
        if train_window is None or val_window is None or test_window is None:
            raise ValueError(
                "auto_find_windows=False の場合は train_window, val_window, test_window を全て指定してください。"
            )
        train_df = slice_df_by_time_window(df, *train_window)
        val_df = slice_df_by_time_window(df, *val_window)
        test_df = slice_df_by_time_window(df, *test_window)

        # manual 指定時は quality 制限なしで全部使う
        for sdf in [train_df, val_df, test_df]:
            sdf["quality_ok"] = 1
            sdf["valid_sample"] = 1
            sdf["quality_score"] = 1.0

    validate_can_training_signals(train_df)
    plot_preprocessed_can_dataframe(train_df, str(outdir / "preprocessed_overview_train"))
    plot_preprocessed_can_dataframe(val_df, str(outdir / "preprocessed_overview_val"))
    plot_preprocessed_can_dataframe(test_df, str(outdir / "preprocessed_overview_test"))

    train_dataset = CenteredDataset(
        train_df,
        history_steps=cfg.history_steps,
        horizon_steps=cfg.horizon_steps,
    )
    val_dataset = CenteredDataset(
        val_df,
        history_steps=cfg.history_steps,
        horizon_steps=cfg.horizon_steps,
    )
    test_dataset = CenteredDataset(
        test_df,
        history_steps=cfg.history_steps,
        horizon_steps=cfg.horizon_steps,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    input_dim = 6 * history_steps

    model = PhysicsInformedVehicleModel(
        priors=priors,
        cfg=cfg,
        mass_kg=mass_fit["mass_kg"],
        drag_terms=mass_fit,
        input_dim=input_dim,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = np.inf
    best_state = None
    history: List[Dict[str, float]] = []

    state_df_train = evaluate_state_timeseries(model, train_df, cfg, history_steps=history_steps)
    state_df_val = evaluate_state_timeseries(model, val_df, cfg, history_steps=history_steps)
    state_df_test = evaluate_state_timeseries(model, test_df, cfg, history_steps=history_steps)

    plot_state_timeseries_diagnostics(state_df_train, outdir, suffix="_train")
    plot_state_timeseries_diagnostics(state_df_val, outdir, suffix="_val")
    plot_state_timeseries_diagnostics(state_df_test, outdir, suffix="_test")

    for epoch in range(cfg.num_epochs):
        model.train()

        # -------------------------------------------------
        # Stage 1: fit Iz/Cf/Cr/mass + latent steering
        #          mu frozen, residual frozen
        # Stage 2: unfreeze mu
        # Stage 3: unfreeze residual as small correction
        # -------------------------------------------------
        if epoch < 10:
            set_requires_grad([model.raw_mu], False)
            set_requires_grad(model.residual.parameters(), False)
            cfg.w_phys = 0.03
            cfg.w_r = 3.0
            cfg.w_psi = 1.0
            cfg.w_xy = 5.0
        elif epoch < 20:
            set_requires_grad([model.raw_mu], True)
            set_requires_grad(model.residual.parameters(), False)
            cfg.w_phys = 0.03
            cfg.w_r = 3.0
            cfg.w_psi = 1.0
            cfg.w_xy = 5.0
        else:
            set_requires_grad([model.raw_mu], True)
            set_requires_grad(model.residual.parameters(), True)
            cfg.w_phys = 0.04
            cfg.w_r = 3.0
            cfg.w_psi = 1.0
            cfg.w_xy = 5.0

        epoch_logs = []
        for can_seq, target_seq, dt_seq in train_loader:
            can_seq = can_seq.to(cfg.device)
            target_seq = target_seq.to(cfg.device)
            dt_seq = dt_seq.to(cfg.device)

            optimizer.zero_grad()
            loss, logs = compute_losses(model, can_seq, target_seq, dt_seq, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_logs.append(logs)

        mean_train_logs = {
            k: float(np.mean([x[k] for x in epoch_logs]))
            for k in epoch_logs[0].keys()
        }

        train_preds = evaluate_model_on_windows(model, train_df, cfg, history_steps=history_steps)
        val_preds = evaluate_model_on_windows(model, val_df, cfg, history_steps=history_steps)
        test_preds = evaluate_model_on_windows(model, test_df, cfg, history_steps=history_steps)

        val_loss = val_preds["rmse_dx"] + val_preds["rmse_dy"] + val_preds["rmse_dvx"]
        test_loss = test_preds["rmse_dx"] + test_preds["rmse_dy"] + test_preds["rmse_dvx"]

        row = {
            "epoch": epoch,
            "train_loss": mean_train_logs["loss"],
            "val_loss": val_loss,
            "test_loss": test_loss,

            "train_rmse_dx": train_preds["rmse_dx"],
            "train_rmse_dy": train_preds["rmse_dy"],
            "train_rmse_dvx": train_preds["rmse_dvx"],
            "train_rmse_r_proxy": train_preds["rmse_r_proxy"],

            "val_rmse_dx": val_preds["rmse_dx"],
            "val_rmse_dy": val_preds["rmse_dy"],
            "val_rmse_dvx": val_preds["rmse_dvx"],
            "val_rmse_r_proxy": val_preds["rmse_r_proxy"],

            "test_rmse_dx": test_preds["rmse_dx"],
            "test_rmse_dy": test_preds["rmse_dy"],
            "test_rmse_dvx": test_preds["rmse_dvx"],
            "test_rmse_r_proxy": test_preds["rmse_r_proxy"],

            # train loss components
            "loss_xy": mean_train_logs["loss_xy"],
            "loss_v": mean_train_logs["loss_v"],
            "loss_r": mean_train_logs["loss_r"],
            "loss_psi": mean_train_logs["loss_psi"],
            "loss_bicycle": mean_train_logs["loss_bicycle"],
            "loss_param": mean_train_logs["loss_param"],
            "loss_reg": mean_train_logs["loss_reg"],

            # learned parameters
            "mu": mean_train_logs["mu"],
            "cf": mean_train_logs["cf"],
            "cr": mean_train_logs["cr"],
            "iz": mean_train_logs["iz"],
            "mass_kg": mean_train_logs["mass_kg"],

            # residual history を追加
            "mean_beta_dot_res": mean_train_logs["mean_beta_dot_res"],
            "mean_abs_beta_dot_res": mean_train_logs["mean_abs_beta_dot_res"],
            "mean_r_dot_res": mean_train_logs["mean_r_dot_res"],
            "mean_abs_r_dot_res": mean_train_logs["mean_abs_r_dot_res"],
            "mean_dvx_res": mean_train_logs["mean_dvx_res"],
            "mean_abs_dvx_res": mean_train_logs["mean_abs_dvx_res"],

            "delta_gain": mean_train_logs["delta_gain"],
        }
        history.append(row)

        print(
            f"epoch={epoch:04d} "
            f"train_rmse=({row['train_rmse_dx']:.3f}, {row['train_rmse_dy']:.3f}, {row['train_rmse_dvx']:.3f}, r={row['train_rmse_r_proxy']:.3f}) "
            f"val_rmse=({row['val_rmse_dx']:.3f}, {row['val_rmse_dy']:.3f}, {row['val_rmse_dvx']:.3f}, r={row['val_rmse_r_proxy']:.3f}) "
            f"test_rmse=({row['test_rmse_dx']:.3f}, {row['test_rmse_dy']:.3f}, {row['test_rmse_dvx']:.3f}, r={row['test_rmse_r_proxy']:.3f}) "
            f"mu={row['mu']:.3f} cf={row['cf']:.1f} cr={row['cr']:.1f} iz={row['iz']:.1f} mass={row['mass_kg']:.1f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch % plot_every == 0) or (epoch == cfg.num_epochs - 1):
            hist_df = pd.DataFrame(history)
            hist_df.to_csv(outdir / "train_history.csv", index=False)
            plot_split_metric_history(hist_df, outdir)
            plot_training_loss_components(hist_df, outdir)
            plot_parameter_history(hist_df, outdir)
            plot_residual_history(hist_df, outdir)

    if best_state is not None:
        model.load_state_dict(best_state)

    train_preds = evaluate_model_on_windows(model, train_df, cfg, history_steps=history_steps)
    val_preds = evaluate_model_on_windows(model, val_df, cfg, history_steps=history_steps)
    test_preds = evaluate_model_on_windows(model, test_df, cfg, history_steps=history_steps)

    plot_final_scatter_plots_windowed(train_preds, "Train", outdir)
    plot_final_scatter_plots_windowed(val_preds, "Val", outdir)
    plot_final_scatter_plots_windowed(test_preds, "Test", outdir)

    # ---------------------------------------------------------
    # r error diagnostics after final training
    # ---------------------------------------------------------
    if auto_find_windows == False:
        run_r_error_diagnostics(
            model=model,
            df=train_df,
            cfg=cfg,
            outdir=outdir / "r_error_diagnostics_train",
            history_steps=history_steps,
            top_k=8,
            pad_steps=40,
            threshold_quantile=0.99,
        )

    run_residual_output_diagnostics(
        model=model,
        df=train_df,
        cfg=cfg,
        outdir=outdir / "residual_output_diagnostics_train",
        history_steps=history_steps,
        top_k=8,
        pad_steps=40,
    )

    final_metrics = pd.DataFrame([
        {
            "split": "train",
            "rmse_dx": train_preds["rmse_dx"],
            "rmse_dy": train_preds["rmse_dy"],
            "rmse_dvx": train_preds["rmse_dvx"],
            "rmse_r_proxy": train_preds["rmse_r_proxy"],
        },
        {
            "split": "val",
            "rmse_dx": val_preds["rmse_dx"],
            "rmse_dy": val_preds["rmse_dy"],
            "rmse_dvx": val_preds["rmse_dvx"],
            "rmse_r_proxy": val_preds["rmse_r_proxy"],
        },
        {
            "split": "test",
            "rmse_dx": test_preds["rmse_dx"],
            "rmse_dy": test_preds["rmse_dy"],
            "rmse_dvx": test_preds["rmse_dvx"],
            "rmse_r_proxy": test_preds["rmse_r_proxy"],
        },
    ])
    final_metrics.to_csv(outdir / "final_split_metrics.csv", index=False)

    summary = summarize_parameters(model, mass_fit)
    pd.DataFrame([summary]).to_csv(outdir / "fit_summary.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "summary": summary}, outdir / "vehicle_model.pt")

    torch.save(model.state_dict(), "model.pth")

    return model


@torch.no_grad()
def collect_residual_timeseries(
    model: torch.nn.Module,
    df: pd.DataFrame,
    cfg,
    history_steps: int = 3,
) -> pd.DataFrame:
    """
    各時刻について 1-step prediction を行い、
    residual 出力と関連量を DataFrame にまとめる。
    """
    model.eval()

    can_feat, target_feat, dt = build_can_target_arrays(df, history_steps=history_steps)

    rows = []
    n = len(df)

    for i in range(n):
        can_step = torch.from_numpy(can_feat[i:i+1]).to(cfg.device)        # [1, F]
        target_step = torch.from_numpy(target_feat[i:i+1]).to(cfg.device)  # [1, 9]
        dt_step = torch.tensor([[dt[i]]], dtype=torch.float32, device=cfg.device)

        x0 = target_step[:, 0]
        y0 = target_step[:, 1]
        psi0 = target_step[:, 2]
        vx0 = target_step[:, 3]
        r0 = target_step[:, 4]
        beta0 = torch.zeros_like(vx0)

        state = torch.stack([x0, y0, psi0, vx0, beta0, r0], dim=-1)

        next_state, aux = model.step(state, can_step, dt_step)

        row = {
            "idx": i,
            "t_s": float(df["t_s"].iloc[i]) if "t_s" in df.columns else float(i),

            # residual outputs
            "beta_dot_res": float(aux["beta_dot_res"][0].detach().cpu().item()),
            "r_dot_res": float(aux["r_dot_res"][0].detach().cpu().item()),
            "dvx_res": float(aux["dvx_res"][0].detach().cpu().item()),

            # physics / effective values
            "beta_dot_phys": float(aux["beta_dot_phys"][0].detach().cpu().item()),
            "r_dot_phys": float(aux["r_dot_phys"][0].detach().cpu().item()),
            "beta_dot_eff": float(aux["beta_dot_eff"][0].detach().cpu().item()),
            "r_dot_eff": float(aux["r_dot_eff"][0].detach().cpu().item()),
            "dvx_eff": float(aux["dvx_eff"][0].detach().cpu().item()),

            # predictions / truth
            "r_pred": float(aux["r_pred"][0].detach().cpu().item()),
            "dx_pred": float(aux["dx_pred"][0].detach().cpu().item()),
            "dy_pred": float(aux["dy_pred"][0].detach().cpu().item()),
            "dpsi_pred": float(aux["dpsi_pred"][0].detach().cpu().item()),
            "dvx_pred": float(aux["dvx_pred"][0].detach().cpu().item()),

            "r_true": float(target_step[0, 4].detach().cpu().item()),
            "dx_true": float(target_step[0, 5].detach().cpu().item()),
            "dy_true": float(target_step[0, 6].detach().cpu().item()),
            "dpsi_true": float(target_step[0, 7].detach().cpu().item()),
            "dvx_true": float(target_step[0, 8].detach().cpu().item()),

            # errors
            "err_r": float(aux["r_pred"][0].detach().cpu().item() - target_step[0, 4].detach().cpu().item()),
            "err_dx": float(aux["dx_pred"][0].detach().cpu().item() - target_step[0, 5].detach().cpu().item()),
            "err_dy": float(aux["dy_pred"][0].detach().cpu().item() - target_step[0, 6].detach().cpu().item()),
            "err_dvx": float(aux["dvx_pred"][0].detach().cpu().item() - target_step[0, 8].detach().cpu().item()),
        }

        # 元データ側の関連列があれば保存
        optional_cols = [
            "vx_mps",
            "yaw_like_rate_radps",
            "raw_z_radps",
            "course_rate_radps",
            "throttle_pct",
            "rpm",
            "ax_mps2",
            "ay_mps2",
            "delta_proxy_rad",
        ]
        for c in optional_cols:
            if c in df.columns:
                row[c] = float(df[c].iloc[i])

        rows.append(row)

    return pd.DataFrame(rows)

def collect_one_step_predictions_for_debug(
    model: torch.nn.Module,
    df: pd.DataFrame,
    cfg,
    history_steps: int = 3,
) -> pd.DataFrame:
    """
    各時刻 t について 1-step 予測を行い、
    r / dX / dY / dVx / dPsi の予測値・実測値・誤差をまとめて返す。
    """
    model.eval()

    can_feat, target_feat, dt = build_can_target_arrays(df, history_steps=history_steps)

    rows = []
    n = len(df)

    with torch.no_grad():
        for i in range(n):
            can_seq = torch.from_numpy(can_feat[i:i+1]).unsqueeze(0).to(cfg.device)      # [1,1,F]
            target_seq = torch.from_numpy(target_feat[i:i+1]).unsqueeze(0).to(cfg.device) # [1,1,T]
            dt_seq = torch.from_numpy(dt[i:i+1]).unsqueeze(0).to(cfg.device)              # [1,1]

            _, aux = model.rollout(can_seq, target_seq, dt_seq)

            dx_pred = float(aux["dx_pred"][0, 0].detach().cpu().item())
            dy_pred = float(aux["dy_pred"][0, 0].detach().cpu().item())
            dpsi_pred = float(aux["dpsi_pred"][0, 0].detach().cpu().item())
            dvx_pred = float(aux["dvx_pred"][0, 0].detach().cpu().item())
            r_pred = float(aux["r_pred"][0, 0].detach().cpu().item())

            # target_feat の並び:
            # [x, y, psi, vx, r, dx, dy, dpsi, dvx]
            x_true = float(target_feat[i, 0])
            y_true = float(target_feat[i, 1])
            psi_true = float(target_feat[i, 2])
            vx_true = float(target_feat[i, 3])
            r_true = float(target_feat[i, 4])
            dx_true = float(target_feat[i, 5])
            dy_true = float(target_feat[i, 6])
            dpsi_true = float(target_feat[i, 7])
            dvx_true = float(target_feat[i, 8])

            row = {
                "idx": i,
                "t_s": float(df["t_s"].iloc[i]) if "t_s" in df.columns else float(i),

                "x_true": x_true,
                "y_true": y_true,
                "psi_true": psi_true,
                "vx_true": vx_true,
                "r_true": r_true,
                "dx_true": dx_true,
                "dy_true": dy_true,
                "dpsi_true": dpsi_true,
                "dvx_true": dvx_true,

                "dx_pred": dx_pred,
                "dy_pred": dy_pred,
                "dpsi_pred": dpsi_pred,
                "dvx_pred": dvx_pred,
                "r_pred": r_pred,

                "err_r": r_pred - r_true,
                "err_dx": dx_pred - dx_true,
                "err_dy": dy_pred - dy_true,
                "err_dpsi": dpsi_pred - dpsi_true,
                "err_dvx": dvx_pred - dvx_true,
                "abs_err_r": abs(r_pred - r_true),
                "abs_err_dx": abs(dx_pred - dx_true),
                "abs_err_dy": abs(dy_pred - dy_true),
                "abs_err_dpsi": abs(dpsi_pred - dpsi_true),
                "abs_err_dvx": abs(dvx_pred - dvx_true),
            }

            # 関連しそうな raw / proxy 量もあれば追加
            optional_cols = [
                "vx_mps",
                "gps_speed_mps",
                "yaw_like_rate_radps",
                "raw_z_radps",
                "psi_xy_rad",
                "course_xy_rad",
                "course_rate_radps",
                "ax_mps2",
                "ay_mps2",
                "throttle_pct",
                "rpm",
                "engine_torque_nm",
                "engine_power_hp",
                "maf_kgs",
                "beta_proxy_rad",
                "xy_valid_rel",
                "yaw_rel",
                "delta_proxy_rad",
            ]
            for c in optional_cols:
                if c in df.columns:
                    row[c] = float(df[c].iloc[i])

            rows.append(row)

    pred_df = pd.DataFrame(rows)

    # wrap 誤差も見たい場合
    if "dpsi_true" in pred_df.columns and "dpsi_pred" in pred_df.columns:
        d = pred_df["dpsi_pred"].to_numpy() - pred_df["dpsi_true"].to_numpy()
        d = (d + np.pi) % (2 * np.pi) - np.pi
        pred_df["err_dpsi_wrap"] = d
        pred_df["abs_err_dpsi_wrap"] = np.abs(d)

    return pred_df


def _find_contiguous_error_segments(
    pred_df: pd.DataFrame,
    err_col: str = "abs_err_r",
    threshold_quantile: float = 0.99,
    min_len: int = 1,
) -> pd.DataFrame:
    """
    abs error が大きい点を抽出し、連続区間としてまとめる。
    """
    thr = float(pred_df[err_col].quantile(threshold_quantile))
    mask = pred_df[err_col].to_numpy() >= thr

    segments = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif (not flag) and (start is not None):
            end = i - 1
            if end - start + 1 >= min_len:
                seg = pred_df.iloc[start:end+1]
                peak_local = int(seg[err_col].to_numpy().argmax())
                peak_idx = int(seg.iloc[peak_local]["idx"])
                segments.append({
                    "start_idx": int(pred_df.iloc[start]["idx"]),
                    "end_idx": int(pred_df.iloc[end]["idx"]),
                    "peak_idx": peak_idx,
                    "n_points": int(end - start + 1),
                    "max_abs_err": float(seg[err_col].max()),
                    "mean_abs_err": float(seg[err_col].mean()),
                    "t_start": float(seg["t_s"].iloc[0]),
                    "t_end": float(seg["t_s"].iloc[-1]),
                    "t_peak": float(pred_df.loc[pred_df["idx"] == peak_idx, "t_s"].iloc[0]),
                })
            start = None

    if start is not None:
        end = len(mask) - 1
        if end - start + 1 >= min_len:
            seg = pred_df.iloc[start:end+1]
            peak_local = int(seg[err_col].to_numpy().argmax())
            peak_idx = int(seg.iloc[peak_local]["idx"])
            segments.append({
                "start_idx": int(pred_df.iloc[start]["idx"]),
                "end_idx": int(pred_df.iloc[end]["idx"]),
                "peak_idx": peak_idx,
                "n_points": int(end - start + 1),
                "max_abs_err": float(seg[err_col].max()),
                "mean_abs_err": float(seg[err_col].mean()),
                "t_start": float(seg["t_s"].iloc[0]),
                "t_end": float(seg["t_s"].iloc[-1]),
                "t_peak": float(pred_df.loc[pred_df["idx"] == peak_idx, "t_s"].iloc[0]),
            })

    seg_df = pd.DataFrame(segments)
    if len(seg_df) > 0:
        seg_df = seg_df.sort_values(["max_abs_err", "n_points"], ascending=[False, False]).reset_index(drop=True)
    return seg_df


def plot_global_error_overview(
    pred_df: pd.DataFrame,
    outdir: str | Path,
    err_col: str = "abs_err_r",
    threshold_quantile: float = 0.99,
) -> None:
    """
    全体時系列の overview。大きな r 誤差区間を背景色でハイライト。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seg_df = _find_contiguous_error_segments(
        pred_df,
        err_col=err_col,
        threshold_quantile=threshold_quantile,
        min_len=1,
    )

    t = pred_df["t_s"].to_numpy()

    fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)

    # 1) r_true / r_pred
    axes[0].plot(t, pred_df["r_true"], label="r_true", linewidth=1.2)
    axes[0].plot(t, pred_df["r_pred"], label="r_pred", linewidth=1.2)
    axes[0].set_ylabel("r [rad/s]")
    axes[0].set_title("Yaw-rate prediction overview")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2) r error
    axes[1].plot(t, pred_df["err_r"], label="r_pred - r_true", linewidth=1.1)
    axes[1].plot(t, pred_df["abs_err_r"], label="|r_pred - r_true|", linewidth=1.1)
    axes[1].set_ylabel("r error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3) yaw proxy 系
    if "yaw_like_rate_radps" in pred_df.columns:
        axes[2].plot(t, pred_df["yaw_like_rate_radps"], label="yaw_like_rate", linewidth=1.2)
    if "raw_z_radps" in pred_df.columns:
        axes[2].plot(t, pred_df["raw_z_radps"], label="raw_z_rate", linewidth=1.0)
    if "course_rate_radps" in pred_df.columns:
        axes[2].plot(t, pred_df["course_rate_radps"], label="d(course)/dt", linewidth=1.0)
    axes[2].set_ylabel("yaw proxies")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # 4) dX, dY error
    axes[3].plot(t, pred_df["err_dx"], label="dx_pred - dx_true", linewidth=1.0)
    axes[3].plot(t, pred_df["err_dy"], label="dy_pred - dy_true", linewidth=1.0)
    axes[3].set_ylabel("dX / dY error")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    # 5) speed / throttle
    if "vx_mps" in pred_df.columns:
        axes[4].plot(t, pred_df["vx_mps"], label="vx_mps", linewidth=1.2)
    elif "vx_true" in pred_df.columns:
        axes[4].plot(t, pred_df["vx_true"], label="vx_true", linewidth=1.2)
    if "throttle_pct" in pred_df.columns:
        axes[4].plot(t, pred_df["throttle_pct"], label="throttle_pct", linewidth=1.0)
    axes[4].set_ylabel("speed / throttle")
    axes[4].set_xlabel("time [s]")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    # 大きな誤差区間をハイライト
    for _, seg in seg_df.iterrows():
        for ax in axes:
            ax.axvspan(seg["t_start"], seg["t_end"], alpha=0.15, color="red")
            ax.axvline(seg["t_peak"], linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(outdir / "r_error_overview_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if len(seg_df) > 0:
        seg_df.to_csv(outdir / "r_error_segments.csv", index=False)


def plot_top_r_error_segments(
    pred_df: pd.DataFrame,
    outdir: str | Path,
    err_col: str = "abs_err_r",
    threshold_quantile: float = 0.99,
    top_k: int = 8,
    pad_steps: int = 30,
) -> None:
    """
    r 誤差が大きい区間ごとに詳細時系列を保存する。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seg_df = _find_contiguous_error_segments(
        pred_df,
        err_col=err_col,
        threshold_quantile=threshold_quantile,
        min_len=1,
    )
    if len(seg_df) == 0:
        return

    seg_df = seg_df.head(top_k).copy()
    seg_df.to_csv(outdir / "top_r_error_segments.csv", index=False)

    n = len(pred_df)

    for rank, seg in enumerate(seg_df.itertuples(index=False), start=1):
        s = max(int(seg.start_idx) - pad_steps, 0)
        e = min(int(seg.end_idx) + pad_steps, n - 1)

        sub = pred_df.iloc[s:e+1].copy()
        t = sub["t_s"].to_numpy()

        fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)

        # 1) r_true / r_pred
        axes[0].plot(t, sub["r_true"], label="r_true", linewidth=1.4)
        axes[0].plot(t, sub["r_pred"], label="r_pred", linewidth=1.4)
        if "yaw_like_rate_radps" in sub.columns:
            axes[0].plot(t, sub["yaw_like_rate_radps"], label="yaw_like_rate", linewidth=1.1)
        axes[0].set_ylabel("r [rad/s]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # 2) r error
        axes[1].plot(t, sub["err_r"], label="err_r", linewidth=1.3)
        axes[1].plot(t, sub["abs_err_r"], label="abs_err_r", linewidth=1.2)
        axes[1].set_ylabel("r error")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # 3) dX / dY compare
        axes[2].plot(t, sub["dx_true"], label="dx_true", linewidth=1.2)
        axes[2].plot(t, sub["dx_pred"], label="dx_pred", linewidth=1.2)
        axes[2].plot(t, sub["dy_true"], label="dy_true", linewidth=1.2)
        axes[2].plot(t, sub["dy_pred"], label="dy_pred", linewidth=1.2)
        axes[2].set_ylabel("dX / dY [m]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(ncol=2)

        # 4) dX / dY error
        axes[3].plot(t, sub["err_dx"], label="err_dx", linewidth=1.1)
        axes[3].plot(t, sub["err_dy"], label="err_dy", linewidth=1.1)
        if "err_dpsi_wrap" in sub.columns:
            axes[3].plot(t, sub["err_dpsi_wrap"], label="err_dpsi_wrap", linewidth=1.0)
        axes[3].set_ylabel("step errors")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

        # 5) 速度・加速度・スロットル
        if "vx_mps" in sub.columns:
            axes[4].plot(t, sub["vx_mps"], label="vx_mps", linewidth=1.2)
        else:
            axes[4].plot(t, sub["vx_true"], label="vx_true", linewidth=1.2)
        if "ax_mps2" in sub.columns:
            axes[4].plot(t, sub["ax_mps2"], label="ax_mps2", linewidth=1.1)
        if "throttle_pct" in sub.columns:
            axes[4].plot(t, sub["throttle_pct"], label="throttle_pct", linewidth=1.1)
        axes[4].set_ylabel("speed / accel / throttle")
        axes[4].grid(True, alpha=0.3)
        axes[4].legend()

        # 6) そのほか関連しそうな量
        plotted = False
        for c in ["rpm", "engine_torque_nm", "beta_proxy_rad", "delta_proxy_rad", "xy_valid_rel", "yaw_rel"]:
            if c in sub.columns:
                axes[5].plot(t, sub[c], label=c, linewidth=1.0)
                plotted = True
        if plotted:
            axes[5].legend(ncol=3)
        axes[5].set_ylabel("related vars")
        axes[5].set_xlabel("time [s]")
        axes[5].grid(True, alpha=0.3)

        # 誤差が大きい中心区間を強調
        t_start = float(pred_df.loc[pred_df["idx"] == seg.start_idx, "t_s"].iloc[0])
        t_end = float(pred_df.loc[pred_df["idx"] == seg.end_idx, "t_s"].iloc[0])
        t_peak = float(pred_df.loc[pred_df["idx"] == seg.peak_idx, "t_s"].iloc[0])

        for ax in axes:
            ax.axvspan(t_start, t_end, alpha=0.18, color="red")
            ax.axvline(t_peak, linestyle="--", alpha=0.5, color="red")

        fig.suptitle(
            f"Top-{rank} r-error segment | idx=[{seg.start_idx}, {seg.end_idx}] | "
            f"peak_idx={seg.peak_idx} | max_abs_err={seg.max_abs_err:.4f}",
            y=0.995
        )
        fig.tight_layout()
        fig.savefig(outdir / f"r_error_segment_{rank:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_r_error_scatter_relationships(
    pred_df: pd.DataFrame,
    outdir: str | Path,
) -> None:
    """
    r 誤差と他の誤差・関連量との関係を散布図で見る。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for c in [
        "err_dx",
        "err_dy",
        "abs_err_dx",
        "abs_err_dy",
        "vx_true",
        "vx_mps",
        "yaw_like_rate_radps",
        "raw_z_radps",
        "throttle_pct",
        "rpm",
        "engine_torque_nm",
    ]:
        if c in pred_df.columns:
            candidates.append(c)

    if len(candidates) == 0:
        return

    ncols = 2
    nrows = int(np.ceil(len(candidates) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    x = pred_df["err_r"].to_numpy()
    high_mask = pred_df["abs_err_r"].to_numpy() >= pred_df["abs_err_r"].quantile(0.99)

    for ax, c in zip(axes, candidates):
        y = pred_df[c].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() == 0:
            continue

        ax.scatter(x[ok & (~high_mask)], y[ok & (~high_mask)], s=8, alpha=0.35, label="normal")
        ax.scatter(x[ok & high_mask], y[ok & high_mask], s=12, alpha=0.8, label="high |r error|")

        corr = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() >= 3 else np.nan
        ax.set_xlabel("err_r = r_pred - r_true")
        ax.set_ylabel(c)
        ax.set_title(f"{c} vs err_r (corr={corr:.3f})")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[len(candidates):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(outdir / "r_error_relationship_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_r_error_diagnostics(
    model: torch.nn.Module,
    df: pd.DataFrame,
    cfg,
    outdir: str | Path,
    history_steps: int = 3,
    top_k: int = 8,
    pad_steps: int = 30,
    threshold_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    一括実行用。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred_df = collect_one_step_predictions_for_debug(
        model=model,
        df=df,
        cfg=cfg,
        history_steps=history_steps,
    )
    pred_df.to_csv(outdir / "one_step_predictions_debug.csv", index=False)

    plot_global_error_overview(
        pred_df=pred_df,
        outdir=outdir,
        err_col="abs_err_r",
        threshold_quantile=threshold_quantile,
    )

    plot_top_r_error_segments(
        pred_df=pred_df,
        outdir=outdir,
        err_col="abs_err_r",
        threshold_quantile=threshold_quantile,
        top_k=top_k,
        pad_steps=pad_steps,
    )

    plot_r_error_scatter_relationships(
        pred_df=pred_df,
        outdir=outdir,
    )

    return pred_df


def run_residual_output_diagnostics(
    model: torch.nn.Module,
    df: pd.DataFrame,
    cfg,
    outdir: str | Path,
    history_steps: int = 3,
    top_k: int = 8,
    pad_steps: int = 40,
) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred_df = collect_residual_timeseries(
        model=model,
        df=df,
        cfg=cfg,
        history_steps=history_steps,
    )
    pred_df.to_csv(outdir / "residual_timeseries.csv", index=False)

    plot_residual_output_timeseries(
        pred_df=pred_df,
        outdir=outdir,
        residual_threshold_quantile=0.99,
    )

    plot_top_residual_segments(
        pred_df=pred_df,
        outdir=outdir,
        top_k=top_k,
        pad_steps=pad_steps,
    )

    return pred_df

# =========================================================
# 13. Example main
# =========================================================
if __name__ == "__main__":
    # Example:
    raw_df = pd.read_csv("./input_can/auto_iowa.csv")
    df = preprocess_can_dataframe(raw_df)
    priors = VehicleSpecPriors(mass_kg_init=2050.0, wheelbase_m=2.3, front_weight_fraction=0.55)
    cfg = TrainConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    mass_fit = {"mass_kg": 2050.0}
    outdir = Path("./outputs_training")

    model = train_vehicle_model(
        df=df,
        priors=priors,
        cfg=cfg,
        mass_fit=mass_fit,
        outdir=outdir,
        plot_every=10,
        history_steps=3,
        auto_find_windows=True,
#        train_window=(0.0, 2400.0),
#        val_window=(2400.0, 3200.0),
#        test_window=(3200.0, 4000.0),
    )
    pass