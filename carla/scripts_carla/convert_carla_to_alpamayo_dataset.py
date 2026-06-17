# -*- coding: utf-8 -*-
"""
Convert egomotion CSV + PNG frames into Alpamayo scene .pt + parquets.

Assumptions
- egomotion csv columns:
  (t_s, timestamp_us, x_m, y_m, z_m, yaw_rad, qx, qy, qz, qw, vx_mps, beta_rad, yaw_rate_radps)
- images: $BASE/images/{camera_name}/%06d.png  (frame index 0,1,2,...)
- egomotion is 10Hz (0.1s); image frame indices are aligned.

Output layout:
  out_dir/
    clip_id_dataops.parquet   (clip_id column; 1 row per clip with ≥1 valid sample)
    cot.parquet               (clip_id, t0_us, sample_idx, cot_text; 1 row per sample)
    skipped_samples.csv       (records of skipped (clip_id, offset) pairs)
    scenes/
      {clip_id}_{t0_us}.pt

Usage – manifest (multiple clips):
  python convert_carla_to_alpamayo_dataset.py \
    --manifest clips_manifest.csv \
    --out-dir ./alpamayo_dataset_out \
    --t0-offsets-in-window-s 2.0 4.0 6.0 8.0 10.0 12.0

Usage – single clip, multiple t0:
  python convert_carla_to_alpamayo_dataset.py \
    --clip-id clip_A \
    --base-dir ./output/clip_A \
    --egomotion-csv ./output/clip_A/egomotion.csv \
    --out-dir ./alpamayo_dataset_out \
    --cut-start-s 0.0 --window-s 20.0 \
    --t0-offsets-in-window-s 2.0 4.0 6.0 8.0 \
    --cot-text "Slow down because the road is wet."

Usage – legacy single t0:
  python convert_carla_to_alpamayo_dataset.py \
    --clip-id test_20260520_01 \
    --base-dir ./output/town10_sp147_rain \
    --egomotion-csv ./outputs_physicsnemo/carla_route/Town10HD_Opt_sp147/egomotion.csv \
    --out-dir ./alpamayo_dataset_out \
    --cut-start-s 0.0 --window-s 20.0 \
    --t0-us 5100000 --t0-offset-in-window-s 2.0 \
    --cot-text "Slow down because the road is wet and visibility is reduced."
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMERAS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]
FRAME_OFFSETS = [-0.3, -0.2, -0.1, 0.0]  # seconds relative to t0
_HIST_STEPS = 16
_FUT_STEPS = 64
_DT = 0.1  # 10 Hz

_FILENAME_SAFE_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_EGOMOTION_REQUIRED_COLS = [
    "t_s", "timestamp_us", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClipSpec:
    clip_id: str
    base_dir: Path
    egomotion_csv: Path
    cut_start_s: float
    window_s: float
    cot_text: Optional[str]
    offsets: list[float]


@dataclass
class SampleResult:
    clip_id: str
    t0_us: int
    pt_path: Path
    cot_text: Optional[str]


@dataclass
class SkipRecord:
    clip_id: str
    offset_s: float
    requested_t0_s: float
    actual_t0_s: float
    actual_t0_us: int
    reason: str
    data_start_s: float
    data_end_s: float
    required_start_s: float
    required_end_s: float
    detail: str


@dataclass
class BuildOptions:
    """Legacy single-clip options (kept for backward compatibility)."""
    clip_id: str
    base_dir: Path
    egomotion_csv: Path
    out_dir: Path
    cut_start_s: float = 0.0
    window_s: float = 20.0
    t0_us: int = 5100000
    t0_offset_in_window_s: float = 2.0
    t0_offsets_in_window_s: list[float] = field(default_factory=list)
    hist_steps: int = _HIST_STEPS
    fut_steps: int = _FUT_STEPS
    dt: float = _DT
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Utility / pure functions
# ---------------------------------------------------------------------------


def validate_clip_id(clip_id: str) -> None:
    if not _FILENAME_SAFE_RE.match(clip_id):
        raise ValueError(
            f"clip_id {clip_id!r} contains characters not safe for filenames. "
            "Only A-Z, a-z, 0-9, _, - are allowed."
        )


def nearest_index(times: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times - t)))


def rot_mats_from_quat(df: pd.DataFrame) -> np.ndarray:
    rot = R.from_quat(df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64))
    return rot.as_matrix().astype(np.float32)


def localize_trajectory(
    xyz_global: np.ndarray,
    rot_global: np.ndarray,
    idx0: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz_local, rot_local) relative to pose at idx0."""
    xyz0 = xyz_global[idx0]
    R0_inv = R.from_matrix(rot_global[idx0]).inv()
    xyz_local = R0_inv.apply(xyz_global - xyz0).astype(np.float32)
    rot_local = (R0_inv * R.from_matrix(rot_global)).as_matrix().astype(np.float32)
    return xyz_local, rot_local


def image_path(base_dir: Path, camera: str, frame_idx: int) -> Path:
    return base_dir / "images" / camera / f"{frame_idx:06d}.png"


def load_png_uint8_chw(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(np.transpose(arr, (2, 0, 1))))


def _compute_required_times(
    actual_t0_s: float,
    hist_steps: int = _HIST_STEPS,
    fut_steps: int = _FUT_STEPS,
    dt: float = _DT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (hist_times, fut_times, frame_times) as float64 arrays."""
    hist_times = np.array(
        [actual_t0_s - 1.5 + dt * i for i in range(hist_steps)], dtype=np.float64
    )
    fut_times = np.array(
        [actual_t0_s + dt * (i + 1) for i in range(fut_steps)], dtype=np.float64
    )
    frame_times = np.array(
        [actual_t0_s + off for off in FRAME_OFFSETS], dtype=np.float64
    )
    return hist_times, fut_times, frame_times


def validate_offsets(offsets: list[float]) -> None:
    """Hard validation. Raises ValueError on duplicates, non-ascending, non-finite."""
    if not offsets:
        raise ValueError("offset list is empty")
    for o in offsets:
        if not np.isfinite(o):
            raise ValueError(f"offset {o} is not finite")
    if len(offsets) != len(set(offsets)):
        seen: set[float] = set()
        dups = [o for o in offsets if o in seen or seen.add(o)]  # type: ignore[func-returns-value]
        raise ValueError(f"duplicate offsets: {dups}")
    for i in range(len(offsets) - 1):
        if offsets[i] >= offsets[i + 1]:
            raise ValueError(
                f"offsets must be strictly ascending, got {offsets[i]} >= {offsets[i + 1]}"
            )


def check_sample_feasibility(
    times: np.ndarray,
    actual_t0_s: float,
    base_dir: Path,
    dt: float = _DT,
    hist_steps: int = _HIST_STEPS,
    fut_steps: int = _FUT_STEPS,
) -> tuple[bool, str, str, float, float]:
    """
    Returns (ok, reason, detail, required_start_s, required_end_s).

    Checks:
    - CSV data covers the required time range (within tolerance)
    - No per-timestamp gap in hist or fut times exceeds tolerance
    - All image files exist and are readable
    """
    tolerance = dt / 2 + 1e-6

    hist_times, fut_times, frame_times = _compute_required_times(
        actual_t0_s, hist_steps, fut_steps, dt
    )
    required_start_s = float(min(hist_times.min(), frame_times.min()))
    required_end_s = float(fut_times.max())
    data_start = float(times.min())
    data_end = float(times.max())

    if required_start_s < data_start - tolerance:
        return (
            False,
            "insufficient_history",
            f"required_start={required_start_s:.4f}s but data starts at {data_start:.4f}s",
            required_start_s,
            required_end_s,
        )
    if required_end_s > data_end + tolerance:
        return (
            False,
            "insufficient_future",
            f"required_end={required_end_s:.4f}s but data ends at {data_end:.4f}s",
            required_start_s,
            required_end_s,
        )

    for t in hist_times:
        idx = nearest_index(times, t)
        gap = abs(times[idx] - t)
        if gap > tolerance:
            return (
                False,
                "missing_history_timestamp",
                f"required t={t:.4f}s, nearest={times[idx]:.4f}s, gap={gap:.6f}s > tol={tolerance:.6f}s",
                required_start_s,
                required_end_s,
            )

    for t in fut_times:
        idx = nearest_index(times, t)
        gap = abs(times[idx] - t)
        if gap > tolerance:
            return (
                False,
                "missing_future_timestamp",
                f"required t={t:.4f}s, nearest={times[idx]:.4f}s, gap={gap:.6f}s > tol={tolerance:.6f}s",
                required_start_s,
                required_end_s,
            )

    def to_frame_idx(t_s: float) -> int:
        return int(round(t_s / dt))

    for cam in CAMERAS:
        for tt in frame_times:
            fi = to_frame_idx(tt)
            p = image_path(base_dir, cam, fi)
            if not p.exists():
                return (
                    False,
                    "missing_image",
                    f"file not found: {p} (t={tt:.3f}s, frame={fi})",
                    required_start_s,
                    required_end_s,
                )
            try:
                with Image.open(p) as img:
                    img.convert("RGB")
            except Exception as exc:
                return (
                    False,
                    "invalid_image",
                    f"cannot read {p}: {exc}",
                    required_start_s,
                    required_end_s,
                )

    return True, "", "", required_start_s, required_end_s


# ---------------------------------------------------------------------------
# Core build functions
# ---------------------------------------------------------------------------


def build_sample_at_t0(
    clip_id: str,
    base_dir: Path,
    out_dir: Path,
    times: np.ndarray,
    xyz_global: np.ndarray,
    rot_global: np.ndarray,
    actual_t0_s: float,
    actual_t0_us: int,
    hist_steps: int = _HIST_STEPS,
    fut_steps: int = _FUT_STEPS,
    dt: float = _DT,
    overwrite: bool = False,
) -> tuple[Path, dict]:
    """Build and save one .pt scene file. Assumes feasibility already checked."""
    scene_dir = out_dir / "scenes"
    out_pt = scene_dir / f"{clip_id}_{actual_t0_us}.pt"

    if not overwrite and out_pt.exists():
        raise FileExistsError(f"Output file already exists: {out_pt}")

    idx_t0 = nearest_index(times, actual_t0_s)
    xyz_local, rot_local = localize_trajectory(xyz_global, rot_global, idx_t0)

    hist_times, fut_times, frame_times = _compute_required_times(
        actual_t0_s, hist_steps, fut_steps, dt
    )

    hist_idx = [nearest_index(times, tt) for tt in hist_times]
    fut_idx = [nearest_index(times, tt) for tt in fut_times]

    ego_history_xyz = torch.from_numpy(xyz_local[hist_idx]).view(1, 1, hist_steps, 3)
    ego_future_xyz = torch.from_numpy(xyz_local[fut_idx]).view(1, 1, fut_steps, 3)
    ego_history_rot = torch.from_numpy(rot_local[hist_idx]).view(1, 1, hist_steps, 3, 3)
    ego_future_rot = torch.from_numpy(rot_local[fut_idx]).view(1, 1, fut_steps, 3, 3)

    def to_frame_idx(t_s: float) -> int:
        return int(round(t_s / dt))

    image_tensor_list = []
    for cam in CAMERAS:
        frames = []
        for tt in frame_times:
            fi = to_frame_idx(tt)
            frames.append(load_png_uint8_chw(image_path(base_dir, cam, fi)))
        image_tensor_list.append(torch.stack(frames, dim=0))
    image_frames = torch.stack(image_tensor_list, dim=0).contiguous()  # (4,4,3,H,W)

    scene: dict = {
        "clip_id": clip_id,
        "t0_us": int(actual_t0_us),
        "image_frames": image_frames,
        "ego_history_xyz": ego_history_xyz.to(torch.float32).contiguous(),
        "ego_history_rot": ego_history_rot.to(torch.float32).contiguous(),
        "ego_future_xyz": ego_future_xyz.to(torch.float32).contiguous(),
        "ego_future_rot": ego_future_rot.to(torch.float32).contiguous(),
    }
    torch.save(scene, out_pt)
    return out_pt, scene


def build_clip(
    spec: ClipSpec,
    out_dir: Path,
    overwrite: bool,
) -> tuple[list[SampleResult], list[SkipRecord]]:
    """
    Process one clip: validate, skip infeasible t0s, build feasible ones.
    Hard errors (bad config, duplicate offsets, etc.) are raised.
    Data-quality issues are recorded as SkipRecords.
    """
    validate_clip_id(spec.clip_id)

    df = pd.read_csv(spec.egomotion_csv)
    missing_cols = [c for c in _EGOMOTION_REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Clip {spec.clip_id!r}: missing CSV columns: {missing_cols}"
        )

    cut_start_s = float(spec.cut_start_s)
    window_end = cut_start_s + float(spec.window_s)
    df_win = df[
        (df["t_s"] >= cut_start_s) & (df["t_s"] <= window_end + 1e-9)
    ].copy().reset_index(drop=True)

    if len(df_win) == 0:
        raise ValueError(
            f"Clip {spec.clip_id!r}: no samples in window "
            f"[{cut_start_s}, {window_end}]"
        )

    times = df_win["t_s"].to_numpy(dtype=np.float64)
    xyz_global = df_win[["x_m", "y_m", "z_m"]].to_numpy(dtype=np.float32)
    rot_global = rot_mats_from_quat(df_win)

    # Hard validation: offset list
    validate_offsets(spec.offsets)

    # Resolve offsets → actual (t0_s, t0_us)
    Resolved = tuple[float, float, float, int]  # (offset, requested, actual_s, actual_us)
    resolved: list[Resolved] = []
    for off in spec.offsets:
        requested = cut_start_s + off
        idx = nearest_index(times, requested)
        actual_t0_s = float(times[idx])
        actual_t0_us = int(df_win.iloc[idx]["timestamp_us"])
        resolved.append((off, requested, actual_t0_s, actual_t0_us))

    # Hard check: different offsets must not map to the same actual_t0_us
    seen_us: dict[int, float] = {}
    for off, _, _, actual_t0_us in resolved:
        if actual_t0_us in seen_us:
            raise ValueError(
                f"Clip {spec.clip_id!r}: offsets {seen_us[actual_t0_us]} and {off} "
                f"both map to actual_t0_us={actual_t0_us}"
            )
        seen_us[actual_t0_us] = off

    # Check feasibility; separate valid from skip
    data_start_s = float(times.min())
    data_end_s = float(times.max())
    valid: list[Resolved] = []
    skips: list[SkipRecord] = []

    for off, requested, actual_t0_s, actual_t0_us in resolved:
        ok, reason, detail, req_start, req_end = check_sample_feasibility(
            times, actual_t0_s, spec.base_dir
        )
        if ok:
            valid.append((off, requested, actual_t0_s, actual_t0_us))
        else:
            warnings.warn(
                f"[SKIP] clip={spec.clip_id!r} offset={off}: {reason} — {detail}"
            )
            skips.append(
                SkipRecord(
                    clip_id=spec.clip_id,
                    offset_s=off,
                    requested_t0_s=requested,
                    actual_t0_s=actual_t0_s,
                    actual_t0_us=actual_t0_us,
                    reason=reason,
                    data_start_s=data_start_s,
                    data_end_s=data_end_s,
                    required_start_s=req_start,
                    required_end_s=req_end,
                    detail=detail,
                )
            )

    # Check .pt existence upfront for all valid samples
    scene_dir = out_dir / "scenes"
    if not overwrite:
        for _, _, _, actual_t0_us in valid:
            pt = scene_dir / f"{spec.clip_id}_{actual_t0_us}.pt"
            if pt.exists():
                raise FileExistsError(f"Output file already exists: {pt}")

    # Build valid samples
    results: list[SampleResult] = []
    for _, _, actual_t0_s, actual_t0_us in valid:
        pt_path, _ = build_sample_at_t0(
            clip_id=spec.clip_id,
            base_dir=spec.base_dir,
            out_dir=out_dir,
            times=times,
            xyz_global=xyz_global,
            rot_global=rot_global,
            actual_t0_s=actual_t0_s,
            actual_t0_us=actual_t0_us,
            overwrite=overwrite,
        )
        results.append(
            SampleResult(
                clip_id=spec.clip_id,
                t0_us=actual_t0_us,
                pt_path=pt_path,
                cot_text=spec.cot_text,
            )
        )

    if not results:
        warnings.warn(
            f"Clip {spec.clip_id!r}: all {len(spec.offsets)} offsets skipped; "
            "excluding from dataset"
        )

    return results, skips


def load_manifest(manifest_path: Path, cli_offsets: list[float]) -> list[ClipSpec]:
    df = pd.read_csv(manifest_path)
    required = ["clip_id", "base_dir", "egomotion_csv", "cut_start_s", "window_s", "cot_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    dup = df["clip_id"][df["clip_id"].duplicated()].tolist()
    if dup:
        raise ValueError(f"Duplicate clip_ids in manifest: {dup}")

    specs: list[ClipSpec] = []
    for _, row in df.iterrows():
        clip_id = str(row["clip_id"])
        validate_clip_id(clip_id)

        # Resolve offsets: per-row column > CLI
        offsets: list[float]
        if "t0_offsets_s" in df.columns and pd.notna(row.get("t0_offsets_s")):
            raw = str(row["t0_offsets_s"]).strip()
            offsets = [float(x.strip()) for x in raw.split(";") if x.strip()]
        else:
            offsets = list(cli_offsets)

        if not offsets:
            raise ValueError(
                f"No offsets for clip {clip_id!r}: "
                "set --t0-offsets-in-window-s or add t0_offsets_s to manifest"
            )

        cot: Optional[str] = None
        if not pd.isna(row["cot_text"]):
            cot = str(row["cot_text"])

        specs.append(
            ClipSpec(
                clip_id=clip_id,
                base_dir=Path(str(row["base_dir"])),
                egomotion_csv=Path(str(row["egomotion_csv"])),
                cut_start_s=float(row["cut_start_s"]),
                window_s=float(row["window_s"]),
                cot_text=cot,
                offsets=offsets,
            )
        )
    return specs


def build_dataset_from_manifest(
    manifest_path: Path,
    out_dir: Path,
    cli_offsets: list[float],
    overwrite: bool,
) -> tuple[list[SampleResult], list[SkipRecord]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenes").mkdir(parents=True, exist_ok=True)

    specs = load_manifest(manifest_path, cli_offsets)

    all_results: list[SampleResult] = []
    all_skips: list[SkipRecord] = []

    for spec in specs:
        results, skips = build_clip(spec, out_dir, overwrite)
        all_results.extend(results)
        all_skips.extend(skips)

    # Cross-dataset (clip_id, t0_us) uniqueness (should be guaranteed by per-clip checks
    # since clip_ids are unique, but verify as a safety net)
    seen: dict[tuple[str, int], str] = {}
    for r in all_results:
        key = (r.clip_id, r.t0_us)
        if key in seen:
            raise ValueError(
                f"Duplicate (clip_id, t0_us) in dataset: "
                f"clip_id={r.clip_id!r}, t0_us={r.t0_us}"
            )
        seen[key] = r.pt_path.name

    if not all_results:
        raise RuntimeError(
            "All samples were skipped; no valid samples generated. "
            "No Parquet files will be written."
        )

    return all_results, all_skips


def _write_atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(str(tmp), str(path))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def write_dataset_metadata(
    results: list[SampleResult],
    skips: list[SkipRecord],
    out_dir: Path,
    overwrite: bool,
) -> None:
    """Write clip_id_dataops.parquet, cot.parquet, skipped_samples.csv atomically."""
    clip_parquet = out_dir / "clip_id_dataops.parquet"
    cot_parquet = out_dir / "cot.parquet"
    skipped_csv = out_dir / "skipped_samples.csv"

    if not overwrite:
        if clip_parquet.exists():
            raise FileExistsError(f"Output parquet already exists: {clip_parquet}")
        if cot_parquet.exists():
            raise FileExistsError(f"Output parquet already exists: {cot_parquet}")

    # clip_id_dataops: 1 row per clip (preserve order of first appearance)
    clip_ids_ordered = list(dict.fromkeys(r.clip_id for r in results))
    clip_df = pd.DataFrame({"clip_id": clip_ids_ordered})
    _write_atomic_parquet(clip_df, clip_parquet)

    # cot.parquet: 1 row per (clip_id, t0_us); all sample_idx=0
    cot_df = pd.DataFrame(
        [
            {
                "clip_id": r.clip_id,
                "t0_us": r.t0_us,
                "sample_idx": 0,
                "cot_text": r.cot_text,
            }
            for r in results
        ]
    )
    _write_atomic_parquet(cot_df, cot_parquet)

    # skipped_samples.csv (always written if there are any skips)
    if skips:
        skip_df = pd.DataFrame(
            [
                {
                    "clip_id": s.clip_id,
                    "offset_s": s.offset_s,
                    "requested_t0_s": s.requested_t0_s,
                    "actual_t0_s": s.actual_t0_s,
                    "actual_t0_us": s.actual_t0_us,
                    "reason": s.reason,
                    "data_start_s": s.data_start_s,
                    "data_end_s": s.data_end_s,
                    "required_start_s": s.required_start_s,
                    "required_end_s": s.required_end_s,
                    "detail": s.detail,
                }
                for s in skips
            ]
        )
        skip_df.to_csv(skipped_csv, index=False)


# ---------------------------------------------------------------------------
# Legacy single-clip API (backward compatibility)
# ---------------------------------------------------------------------------


def build_one_scene(opt: BuildOptions, cot_text: str | None = None) -> list[Path]:
    """
    Legacy entry point for single-clip builds.

    - t0_offsets_in_window_s set → multi-t0 mode via build_clip (data issues → skip)
    - otherwise              → single-t0 legacy (data issues → raise ValueError)
    """
    validate_clip_id(opt.clip_id)
    opt.out_dir.mkdir(parents=True, exist_ok=True)
    (opt.out_dir / "scenes").mkdir(parents=True, exist_ok=True)

    if opt.t0_offsets_in_window_s:
        # Multi-t0 path
        spec = ClipSpec(
            clip_id=opt.clip_id,
            base_dir=opt.base_dir,
            egomotion_csv=opt.egomotion_csv,
            cut_start_s=opt.cut_start_s,
            window_s=opt.window_s,
            cot_text=cot_text,
            offsets=list(opt.t0_offsets_in_window_s),
        )
        results, skips = build_clip(spec, opt.out_dir, opt.overwrite)
        write_dataset_metadata(results, skips, opt.out_dir, opt.overwrite)
        return [r.pt_path for r in results]

    # ---- legacy single-t0 mode ----
    df = pd.read_csv(opt.egomotion_csv)
    missing_cols = [c for c in _EGOMOTION_REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns in egomotion csv: {missing_cols}")

    cut_start_s = float(opt.cut_start_s)
    window_end = cut_start_s + float(opt.window_s)

    df_win = df[
        (df["t_s"] >= cut_start_s) & (df["t_s"] <= window_end + 1e-9)
    ].copy().reset_index(drop=True)

    if len(df_win) == 0:
        raise ValueError("No samples found in the requested cut window.")

    times = df_win["t_s"].to_numpy(dtype=np.float64)
    xyz_global = df_win[["x_m", "y_m", "z_m"]].to_numpy(dtype=np.float32)
    rot_global = rot_mats_from_quat(df_win)

    t0_s = cut_start_s + opt.t0_offset_in_window_s
    idx_t0 = nearest_index(times, t0_s)
    actual_t0_s = float(times[idx_t0])

    # Validate range (raise for legacy mode)
    ok, reason, detail, req_start, req_end = check_sample_feasibility(
        times, actual_t0_s, opt.base_dir, opt.dt, opt.hist_steps, opt.fut_steps
    )
    if not ok:
        raise ValueError(
            f"t0 at offset {opt.t0_offset_in_window_s}s is infeasible: "
            f"{reason} — {detail}"
        )

    clip_parquet = opt.out_dir / "clip_id_dataops.parquet"
    cot_parquet = opt.out_dir / "cot.parquet"
    out_pt = opt.out_dir / "scenes" / f"{opt.clip_id}_{opt.t0_us}.pt"

    if not opt.overwrite:
        if out_pt.exists():
            raise FileExistsError(f"Output file already exists: {out_pt}")
        if clip_parquet.exists():
            raise FileExistsError(f"Output parquet already exists: {clip_parquet}")
        if cot_text is not None and cot_parquet.exists():
            raise FileExistsError(f"Output parquet already exists: {cot_parquet}")

    # Build using opt.t0_us as the t0_us (legacy behaviour: caller supplies t0_us)
    xyz_local, rot_local = localize_trajectory(xyz_global, rot_global, idx_t0)
    hist_times, fut_times, frame_times = _compute_required_times(
        actual_t0_s, opt.hist_steps, opt.fut_steps, opt.dt
    )

    hist_idx = [nearest_index(times, tt) for tt in hist_times]
    fut_idx = [nearest_index(times, tt) for tt in fut_times]

    ego_history_xyz = torch.from_numpy(xyz_local[hist_idx]).view(1, 1, opt.hist_steps, 3)
    ego_future_xyz = torch.from_numpy(xyz_local[fut_idx]).view(1, 1, opt.fut_steps, 3)
    ego_history_rot = torch.from_numpy(rot_local[hist_idx]).view(1, 1, opt.hist_steps, 3, 3)
    ego_future_rot = torch.from_numpy(rot_local[fut_idx]).view(1, 1, opt.fut_steps, 3, 3)

    def to_frame_idx(t_s: float) -> int:
        return int(round(t_s / opt.dt))

    image_tensor_list = []
    for cam in CAMERAS:
        frames = []
        for tt in frame_times:
            fi = to_frame_idx(tt)
            p = image_path(opt.base_dir, cam, fi)
            frames.append(load_png_uint8_chw(p))
        image_tensor_list.append(torch.stack(frames, dim=0))
    image_frames = torch.stack(image_tensor_list, dim=0).contiguous()

    scene: dict = {
        "clip_id": opt.clip_id,
        "t0_us": int(opt.t0_us),
        "image_frames": image_frames,
        "ego_history_xyz": ego_history_xyz.to(torch.float32).contiguous(),
        "ego_history_rot": ego_history_rot.to(torch.float32).contiguous(),
        "ego_future_xyz": ego_future_xyz.to(torch.float32).contiguous(),
        "ego_future_rot": ego_future_rot.to(torch.float32).contiguous(),
    }
    torch.save(scene, out_pt)

    # clip_id_dataops: 1-row, clip_id only
    _write_atomic_parquet(pd.DataFrame({"clip_id": [opt.clip_id]}), clip_parquet)

    if cot_text is not None:
        _write_atomic_parquet(
            pd.DataFrame(
                {
                    "clip_id": [opt.clip_id],
                    "t0_us": [opt.t0_us],
                    "sample_idx": [0],
                    "cot_text": [cot_text],
                }
            ),
            cot_parquet,
        )

    return [out_pt]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Path to clips manifest CSV (columns: clip_id, base_dir, egomotion_csv, "
            "cut_start_s, window_s, cot_text). When set, enables multi-clip mode "
            "and --clip-id/--base-dir/--egomotion-csv are ignored."
        ),
    )
    p.add_argument("--clip-id", default=None, help="clip id string (single-clip mode)")
    p.add_argument(
        "--base-dir", type=Path, default=None,
        help="Base directory containing images/ (single-clip mode)"
    )
    p.add_argument(
        "--egomotion-csv", type=Path, default=None,
        help="Path to egomotion CSV (single-clip mode)"
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--cut-start-s", type=float, default=0.0)
    p.add_argument("--window-s", type=float, default=20.0)
    p.add_argument("--t0-us", type=int, default=5100000,
                   help="t0_us for legacy single-t0 mode")
    p.add_argument("--t0-offset-in-window-s", type=float, default=2.0,
                   help="t0 = cut_start_s + offset (legacy single-t0 mode)")
    p.add_argument(
        "--t0-offsets-in-window-s",
        type=float,
        nargs="+",
        default=None,
        help="Multiple offsets; enables multi-t0 mode for single clip or manifest",
    )
    p.add_argument("--cot-text", type=str, default=None)
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quick-check helper
# ---------------------------------------------------------------------------


def _quick_check(
    pt_paths: list[Path],
    clip_parquet: Path,
    cot_parquet: Path,
) -> None:
    clip_df = pd.read_parquet(clip_parquet)
    cot_df = pd.read_parquet(cot_parquet) if cot_parquet.exists() else None

    print(f"\nGenerated {len(pt_paths)} scene file(s):")
    for pt_path in pt_paths:
        scene = torch.load(pt_path, weights_only=False)
        cid = scene["clip_id"]
        t0u = scene["t0_us"]
        expected_name = f"{cid}_{t0u}.pt"

        print(f"\n  Path:             {pt_path}")
        print(f"  clip_id:          {cid}")
        print(f"  t0_us:            {t0u}")
        print(f"  image_frames:     {tuple(scene['image_frames'].shape)}, {scene['image_frames'].dtype}")
        print(f"  ego_history_xyz:  {tuple(scene['ego_history_xyz'].shape)}, {scene['ego_history_xyz'].dtype}")
        print(f"  ego_future_xyz:   {tuple(scene['ego_future_xyz'].shape)}, {scene['ego_future_xyz'].dtype}")
        print(f"  ego_history_rot:  {tuple(scene['ego_history_rot'].shape)}, {scene['ego_history_rot'].dtype}")
        print(f"  ego_future_rot:   {tuple(scene['ego_future_rot'].shape)}, {scene['ego_future_rot'].dtype}")

        assert pt_path.name == expected_name, (
            f"filename mismatch: {pt_path.name} != {expected_name}"
        )
        assert cid in clip_df["clip_id"].values, (
            f"{cid!r} not in clip_id_dataops.parquet"
        )
        if cot_df is not None:
            row = cot_df[(cot_df["clip_id"] == cid) & (cot_df["t0_us"] == t0u)]
            assert len(row) == 1, f"cot.parquet has no row for ({cid}, {t0u})"
        print(f"  [OK] consistency checks passed")

    print(f"\nclip_id_dataops.parquet:\n{clip_df.to_string(index=False)}")
    if cot_df is not None:
        print(f"\ncot.parquet:\n{cot_df.to_string(index=False)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()

    if args.manifest:
        # ---- manifest (multi-clip) mode ----
        cli_offsets = list(args.t0_offsets_in_window_s or [])
        results, skips = build_dataset_from_manifest(
            manifest_path=args.manifest,
            out_dir=args.out_dir,
            cli_offsets=cli_offsets,
            overwrite=args.overwrite,
        )
        write_dataset_metadata(results, skips, args.out_dir, args.overwrite)
        pt_paths = [r.pt_path for r in results]

    elif args.t0_offsets_in_window_s:
        # ---- single clip, multi-t0 mode ----
        if not args.clip_id or not args.base_dir or not args.egomotion_csv:
            raise SystemExit(
                "Error: --clip-id, --base-dir, and --egomotion-csv are required "
                "in single-clip mode"
            )
        spec = ClipSpec(
            clip_id=args.clip_id,
            base_dir=args.base_dir,
            egomotion_csv=args.egomotion_csv,
            cut_start_s=args.cut_start_s,
            window_s=args.window_s,
            cot_text=args.cot_text,
            offsets=list(args.t0_offsets_in_window_s),
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "scenes").mkdir(parents=True, exist_ok=True)
        results, skips = build_clip(spec, args.out_dir, args.overwrite)
        if not results:
            raise RuntimeError(
                "All offsets were skipped; no valid samples generated."
            )
        write_dataset_metadata(results, skips, args.out_dir, args.overwrite)
        pt_paths = [r.pt_path for r in results]

    else:
        # ---- legacy single-t0 mode ----
        if not args.clip_id or not args.base_dir or not args.egomotion_csv:
            raise SystemExit(
                "Error: --clip-id, --base-dir, and --egomotion-csv are required "
                "in single-clip mode"
            )
        opt = BuildOptions(
            clip_id=args.clip_id,
            base_dir=args.base_dir,
            egomotion_csv=args.egomotion_csv,
            out_dir=args.out_dir,
            cut_start_s=args.cut_start_s,
            window_s=args.window_s,
            t0_us=args.t0_us,
            t0_offset_in_window_s=args.t0_offset_in_window_s,
            overwrite=args.overwrite,
        )
        pt_paths = build_one_scene(opt, cot_text=args.cot_text)

    _quick_check(
        pt_paths,
        args.out_dir / "clip_id_dataops.parquet",
        args.out_dir / "cot.parquet",
    )
