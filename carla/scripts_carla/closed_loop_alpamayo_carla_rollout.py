#!/usr/bin/env python3
"""
Closed-loop CARLA rollout using Alpamayo trajectory predictions.

Initialization:
  1. Load --init-egomotion-csv and extract 16-step ego history at --init-t0-sec.
  2. Teleport ego vehicle to t0 world pose via set_transform.
  3. Initialize 4-frame image buffer (replay or duplicate mode).
  4. Run closed-loop: capture images → Alpamayo predict → apply predicted pose → repeat.

Usage:
    python scripts/closed_loop_alpamayo_carla_rollout.py \
        --map-name Town01 \
        --init-egomotion-csv path/to/egomotion.csv \
        --init-t0-sec 5.1 \
        --output-dir outputs/closed_loop_alpamayo \
        --model-variant base \
        --num-steps 100 \
        --dt 0.1 \
        --control-stride 1 \
        --image-width 800 \
        --image-height 450 \
        --weather-preset rainfog \
        --device cuda
"""

import argparse
import collections
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from queue import Empty, Queue

import carla
import numpy as np
import pandas as pd
import torch

# NOTE: adjust this import path to the actual alpamayo package location before use.
from alpamayo.wrapper import project_trajectory  # type: ignore[import]

log = logging.getLogger(__name__)

FIXED_DELTA_SECONDS = 0.1
WARMUP_TICKS = 3
HISTORY_STEPS = 16      # 10 Hz × 1.6 s (t0-1.5s … t0)
FRAME_BUFFER_SIZE = 4   # t0-0.3s, t0-0.2s, t0-0.1s, t0

# Reused from replay_egomotion_capture_images.py
CAMERA_SPECS = [
    (
        "camera_cross_left_120fov",
        carla.Transform(
            carla.Location(x=1.55, y=-0.35, z=1.60),
            carla.Rotation(yaw=-90.0),
        ),
        120.0,
    ),
    (
        "camera_front_wide_120fov",
        carla.Transform(
            carla.Location(x=1.55, y=0.00, z=1.60),
            carla.Rotation(yaw=0.0),
        ),
        120.0,
    ),
    (
        "camera_cross_right_120fov",
        carla.Transform(
            carla.Location(x=1.55, y=0.35, z=1.60),
            carla.Rotation(yaw=90.0),
        ),
        120.0,
    ),
    (
        "camera_front_tele_30fov",
        carla.Transform(
            carla.Location(x=1.70, y=0.00, z=1.60),
            carla.Rotation(yaw=0.0),
        ),
        30.0,
    ),
]

# Output directory names aligned with Alpamayo camera order
CAM_OUT_NAMES = ["left_cross_120", "front_wide_120", "right_cross_120", "front_tele_30"]


# ─────────────────────────── CARLA helpers ───────────────────────────

def build_weather(preset: str) -> carla.WeatherParameters:
    p = preset.lower().replace("-", "_")
    if p == "clear":
        return carla.WeatherParameters.ClearNoon
    if p in ("hard_rain", "hardrain"):
        return carla.WeatherParameters.HardRainNoon
    if p in ("wet_cloudy", "wetcloudy"):
        return carla.WeatherParameters.WetCloudyNoon
    # rainfog / hard_rain_fog
    return carla.WeatherParameters(
        cloudiness=100, precipitation=100, precipitation_deposits=100,
        wetness=100, fog_density=35, fog_distance=15, fog_falloff=1,
        wind_intensity=80, sun_altitude_angle=20,
    )


def get_image_for_frame(q: Queue, target_frame: int, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Empty(f"Timed out waiting for frame {target_frame}")
        image = q.get(True, max(0.01, remaining))
        if image.frame == target_frame:
            return image
        if image.frame < target_frame:
            continue
        raise Empty(f"Skipped target frame {target_frame}, got {image.frame}")


def drain_queue(q: Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except Empty:
            break


def collect_images(cameras: list, target_frame: int):
    """Returns dict {carla_name: carla.Image} or None on failure (with queues drained)."""
    images = {}
    for carla_name, _, q in cameras:
        try:
            images[carla_name] = get_image_for_frame(q, target_frame)
        except Empty as e:
            log.warning("Camera '%s': %s", carla_name, e)
            for _, _, qq in cameras:
                drain_queue(qq)
            return None
    return images


def get_vehicle_blueprint(world):
    lib = world.get_blueprint_library()
    candidates = sorted(lib.filter("vehicle.lincoln.*"), key=lambda bp: bp.id)
    if not candidates:
        candidates = sorted(lib.filter("vehicle.*"), key=lambda bp: bp.id)
    if not candidates:
        sys.exit("[ERROR] No vehicle blueprints found")
    return candidates[0]


def try_spawn_vehicle(world, vehicle_bp):
    for i, sp in enumerate(world.get_map().get_spawn_points()):
        actor = world.try_spawn_actor(vehicle_bp, sp)
        if actor is not None:
            log.info("Spawned %s at spawn_point[%d]: %s", vehicle_bp.id, i, sp.location)
            return actor
    sys.exit("[ERROR] Failed to spawn vehicle at all spawn points")


def build_camera_blueprint(world, width: int, height: int, fov: float):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))
    return bp


def make_transform(x_m: float, y_m: float, z_m: float, yaw_rad: float,
                   z_offset: float = 0.0) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=float(x_m), y=float(y_m), z=float(z_m) + z_offset),
        carla.Rotation(yaw=math.degrees(float(yaw_rad))),
    )


# ─────────────────────────── coordinate conversion ───────────────────────────

def yaw_to_rot3x3(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def world_to_local_xyz(xk, yk, zk, x0, y0, z0, yaw0_rad):
    """CARLA world coords → t0-local coords."""
    c, s = math.cos(yaw0_rad), math.sin(yaw0_rad)
    dx, dy = xk - x0, yk - y0
    return float(c * dx + s * dy), float(-s * dx + c * dy), float(zk - z0)


def local_to_world_xyz(lx, ly, lz, x0, y0, z0, yaw0_rad):
    """t0-local coords → CARLA world coords."""
    c, s = math.cos(yaw0_rad), math.sin(yaw0_rad)
    return (float(x0 + c * lx - s * ly),
            float(y0 + s * lx + c * ly),
            float(z0 + lz))


def rot3x3_to_local_yaw(R: np.ndarray) -> float:
    """Extract yaw_rad from a t0-local 3×3 rotation matrix (Z-up convention)."""
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


# ─────────────────────────── init CSV helpers ───────────────────────────

def load_init_history(csv_path: Path, t0_sec: float, dt: float = 0.1):
    """
    Returns (history_rows, t0_row):
      history_rows - list of 16 row dicts from t0-1.5s to t0 (oldest first)
      t0_row       - the row closest to t0_sec
    """
    df = pd.read_csv(csv_path)
    if "t_s" in df.columns:
        t_col = "t_s"
    elif "t_sec" in df.columns:
        t_col = "t_sec"
    else:
        sys.exit(f"[ERROR] Init CSV must have 't_s' or 't_sec' column: {csv_path}")

    for col in ("x_m", "y_m", "z_m", "yaw_rad"):
        if col not in df.columns:
            sys.exit(f"[ERROR] Init CSV missing column '{col}': {csv_path}")

    target_times = [t0_sec - (HISTORY_STEPS - 1 - i) * dt
                    for i in range(HISTORY_STEPS)]

    history_rows = []
    for tt in target_times:
        idx = (df[t_col] - tt).abs().idxmin()
        history_rows.append(df.iloc[idx].to_dict())

    t0_row = history_rows[-1]
    return history_rows, t0_row


def build_ego_history_tensors(history_rows: list, t0_row: dict):
    """
    Build Alpamayo inputs from 16-step history dicts in world coords.
    Returns:
      ego_history_xyz : torch.Tensor (1, 1, 16, 3) float32, t0-local
      ego_history_rot : torch.Tensor (1, 1, 16, 3, 3) float32, t0-local
    """
    x0, y0, z0 = float(t0_row["x_m"]), float(t0_row["y_m"]), float(t0_row["z_m"])
    yaw0 = float(t0_row["yaw_rad"])

    xyz = np.zeros((HISTORY_STEPS, 3), dtype=np.float32)
    rot = np.zeros((HISTORY_STEPS, 3, 3), dtype=np.float32)

    for i, row in enumerate(history_rows):
        lx, ly, lz = world_to_local_xyz(
            float(row["x_m"]), float(row["y_m"]), float(row["z_m"]),
            x0, y0, z0, yaw0,
        )
        xyz[i] = [lx, ly, lz]
        rot[i] = yaw_to_rot3x3(float(row["yaw_rad"]) - yaw0)

    ego_history_xyz = torch.from_numpy(xyz).unsqueeze(0).unsqueeze(0)   # (1,1,16,3)
    ego_history_rot = torch.from_numpy(rot).unsqueeze(0).unsqueeze(0)   # (1,1,16,3,3)
    return ego_history_xyz, ego_history_rot


# ─────────────────────────── image helpers ───────────────────────────

def carla_image_to_numpy(img) -> np.ndarray:
    """CARLA BGRA image → numpy uint8 (H, W, 3) RGB."""
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    return arr[:, :, :3][:, :, ::-1].copy()


def images_to_tensor(cam_frame_buffer: list, H: int, W: int) -> torch.Tensor:
    """
    cam_frame_buffer : list of FRAME_BUFFER_SIZE dicts {carla_name: np.ndarray (H,W,3)}
                       ordered [t0-0.3s, t0-0.2s, t0-0.1s, t0]
    Returns torch.Tensor (4, 4, 3, H, W) uint8
      dim-0 (camera): left_cross_120, front_wide_120, right_cross_120, front_tele_30
      dim-1 (frame) : t0-0.3s, t0-0.2s, t0-0.1s, t0
    """
    tensor = torch.zeros(4, FRAME_BUFFER_SIZE, 3, H, W, dtype=torch.uint8)
    for cam_idx, (cam_name, _, _) in enumerate(CAMERA_SPECS):
        for frame_idx, frame_dict in enumerate(cam_frame_buffer):
            arr = frame_dict[cam_name]  # (H, W, 3)
            tensor[cam_idx, frame_idx] = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def blank_frame_dict(H: int, W: int) -> dict:
    return {cn: np.zeros((H, W, 3), dtype=np.uint8) for cn, _, _ in CAMERA_SPECS}


# ─────────────────────────── output helpers ───────────────────────────

def setup_output_dirs(output_dir: Path):
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "cot").mkdir(parents=True, exist_ok=True)
    for name in CAM_OUT_NAMES:
        (output_dir / "camera" / name).mkdir(parents=True, exist_ok=True)


def save_step_images(output_dir: Path, step: int, images: dict):
    for carla_name, out_name in zip(
        [cn for cn, _, _ in CAMERA_SPECS], CAM_OUT_NAMES
    ):
        images[carla_name].save_to_disk(
            str(output_dir / "camera" / out_name / f"step_{step:06d}.png")
        )


def save_prediction(output_dir: Path, step: int,
                    pred_xyz: np.ndarray, pred_rot: np.ndarray):
    data = {
        "step": step,
        "pred_future_xyz": pred_xyz.tolist(),
        "pred_future_rot": pred_rot.tolist(),
    }
    with open(output_dir / "predictions" / f"step_{step:06d}.json", "w") as f:
        json.dump(data, f, indent=2)


def save_cot(output_dir: Path, step: int, cot_text: str):
    with open(output_dir / "cot" / f"step_{step:06d}.txt", "w") as f:
        f.write(cot_text or "")


# ─────────────────────────── CLI ───────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map-name", default=None,
                   help="Expected map name substring (check only; does not load a map)")
    p.add_argument("--init-egomotion-csv", required=True,
                   help="Initial egomotion CSV in CARLA absolute world coords")
    p.add_argument("--init-t0-sec", type=float, required=True,
                   help="t0 timestamp [s] in the init CSV (center of history window)")
    p.add_argument("--output-dir", default="outputs/closed_loop_alpamayo")
    p.add_argument("--model-variant", default="base",
                   help="Alpamayo model variant (e.g. base, ft_001)")
    p.add_argument("--num-steps", type=int, default=100,
                   help="Number of closed-loop steps")
    p.add_argument("--dt", type=float, default=0.1,
                   help="Step duration [s] (should match FIXED_DELTA_SECONDS=0.1)")
    p.add_argument("--control-stride", type=int, default=1,
                   help="Which prediction step index to apply (1 = t0+dt, 2 = t0+2*dt, ...)")
    p.add_argument("--image-width", type=int, default=800)
    p.add_argument("--image-height", type=int, default=450)
    p.add_argument("--weather-preset", default="clear",
                   choices=["clear", "hard_rain", "wet_cloudy", "hard_rain_fog", "rainfog"])
    p.add_argument("--device", default="cpu", help="Torch device (cpu / cuda)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--z-offset", type=float, default=0.3,
                   help="Z offset [m] added to every world pose")
    p.add_argument("--init-buffer-mode", default="replay",
                   choices=["replay", "duplicate"],
                   help="replay: move vehicle to t0-0.3s…t0 poses to capture init frames; "
                        "duplicate: capture at t0 and repeat 4×")
    p.add_argument("--save-debug", action="store_true",
                   help="Save predictions and CoT even on fallback steps")
    return p.parse_args()


# ─────────────────────────── main ───────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                        stream=sys.stdout)
    args = parse_args()

    output_dir = Path(args.output_dir)
    setup_output_dirs(output_dir)

    # ── load init CSV ──
    init_csv = Path(args.init_egomotion_csv)
    if not init_csv.exists():
        sys.exit(f"[ERROR] init CSV not found: {init_csv}")
    history_rows, t0_row = load_init_history(init_csv, args.init_t0_sec, args.dt)
    x0 = float(t0_row["x_m"])
    y0 = float(t0_row["y_m"])
    z0 = float(t0_row["z_m"])
    yaw0 = float(t0_row["yaw_rad"])
    log.info("Init CSV: %s  t0=%.3fs  world=(%.2f, %.2f, %.2f)  yaw=%.4frad",
             init_csv, args.init_t0_sec, x0, y0, z0, yaw0)

    # ── torch device ──
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info("Torch device: %s", device)

    # ── CARLA client ──
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    map_name = world.get_map().name
    log.info("Connected to CARLA, map: %s", map_name)
    if args.map_name and args.map_name not in map_name:
        log.warning("Expected map '%s' but server has '%s'", args.map_name, map_name)

    original_settings = world.get_settings()
    original_weather = world.get_weather()
    ego_vehicle = None
    cameras = []        # list of (carla_name, actor, queue)
    ego_csv_file = None

    # mutable loop state
    ego_history_deque = collections.deque(
        [(float(r["x_m"]), float(r["y_m"]), float(r["z_m"]), float(r["yaw_rad"]))
         for r in history_rows],
        maxlen=HISTORY_STEPS,
    )
    cam_frame_buffer: collections.deque = collections.deque(maxlen=FRAME_BUFFER_SIZE)

    try:
        # ── synchronous mode ──
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        world.set_weather(build_weather(args.weather_preset))
        log.info("Weather: %s", args.weather_preset)

        # ── spawn vehicle ──
        vehicle_bp = get_vehicle_blueprint(world)
        ego_vehicle = try_spawn_vehicle(world, vehicle_bp)
        ego_vehicle.set_autopilot(False)
        ego_vehicle.set_simulate_physics(False)

        # ── move to t0 world pose ──
        t0_transform = make_transform(x0, y0, z0, yaw0, args.z_offset)
        ego_vehicle.set_transform(t0_transform)
        log.info("Ego at t0: %s", t0_transform.location)

        # ── spawn 4 cameras ──
        for carla_name, cam_offset, fov in CAMERA_SPECS:
            cam_bp = build_camera_blueprint(world, args.image_width, args.image_height, fov)
            actor = world.spawn_actor(cam_bp, cam_offset, attach_to=ego_vehicle)
            q: Queue = Queue()
            actor.listen(lambda data, _q=q: _q.put(data))
            cameras.append((carla_name, actor, q))
            log.info("Camera '%s': %dx%d fov=%.0f",
                     carla_name, args.image_width, args.image_height, fov)

        # ── warm-up ──
        for _ in range(WARMUP_TICKS):
            world.tick()
            for _, _, q in cameras:
                drain_queue(q)
        log.info("Warm-up done (%d ticks)", WARMUP_TICKS)

        # ── initialize 4-frame image buffer ──
        if args.init_buffer_mode == "replay":
            # Move vehicle through last 4 history poses (t0-0.3s … t0) and capture
            init_poses = history_rows[-FRAME_BUFFER_SIZE:]
            for pose_row in init_poses:
                tr = make_transform(
                    float(pose_row["x_m"]), float(pose_row["y_m"]), float(pose_row["z_m"]),
                    float(pose_row["yaw_rad"]), args.z_offset,
                )
                ego_vehicle.set_transform(tr)
                tf = world.tick()
                imgs = collect_images(cameras, tf)
                if imgs is None:
                    log.warning("Buffer init (replay): image capture failed, using blank frame")
                    cam_frame_buffer.append(blank_frame_dict(args.image_height, args.image_width))
                else:
                    cam_frame_buffer.append(
                        {cn: carla_image_to_numpy(imgs[cn]) for cn, _, _ in CAMERA_SPECS}
                    )
            # restore to t0 pose for the main loop
            ego_vehicle.set_transform(t0_transform)
            log.info("Image buffer initialized via replay mode (%d frames)", len(cam_frame_buffer))
        else:
            # Capture at t0 and duplicate 4×
            tf = world.tick()
            imgs = collect_images(cameras, tf)
            if imgs is None:
                log.warning("Buffer init (duplicate): image capture failed, using blank frames")
                fd = blank_frame_dict(args.image_height, args.image_width)
            else:
                fd = {cn: carla_image_to_numpy(imgs[cn]) for cn, _, _ in CAMERA_SPECS}
            for _ in range(FRAME_BUFFER_SIZE):
                cam_frame_buffer.append(fd)
            log.info("Image buffer initialized via duplicate mode")

        # ── open CSV ──
        ego_csv_path = output_dir / "closed_loop_egomotion.csv"
        ego_csv_file = open(ego_csv_path, "w", newline="")
        ego_csv_writer = csv.DictWriter(
            ego_csv_file,
            fieldnames=["step", "t_sec", "x_m", "y_m", "z_m", "yaw_rad", "vx_mps", "source"],
        )
        ego_csv_writer.writeheader()
        ego_csv_writer.writerow({
            "step": 0,
            "t_sec": round(args.init_t0_sec, 4),
            "x_m": round(x0, 6), "y_m": round(y0, 6), "z_m": round(z0, 6),
            "yaw_rad": round(yaw0, 6),
            "vx_mps": 0.0,
            "source": "init_csv",
        })

        # ── fallback state ──
        current_x, current_y, current_z, current_yaw = x0, y0, z0, yaw0
        fallback_vx_mps = 0.0    # scalar forward speed in world heading direction
        fallback_dyaw_radps = 0.0

        # ── closed-loop main loop ──
        for step in range(1, args.num_steps + 1):
            t_sec = args.init_t0_sec + step * args.dt

            # current t0 = most recent entry in history deque
            t0_wx, t0_wy, t0_wz, t0_wyaw = ego_history_deque[-1]

            # build Alpamayo inputs
            ego_hist_xyz, ego_hist_rot = build_ego_history_tensors(
                [{"x_m": x, "y_m": y, "z_m": z, "yaw_rad": yaw}
                 for x, y, z, yaw in ego_history_deque],
                {"x_m": t0_wx, "y_m": t0_wy, "z_m": t0_wz, "yaw_rad": t0_wyaw},
            )
            image_frames = images_to_tensor(
                list(cam_frame_buffer), args.image_height, args.image_width,
            )
            ego_hist_xyz = ego_hist_xyz.to(device)
            ego_hist_rot = ego_hist_rot.to(device)
            image_frames = image_frames.to(device)

            # ─ Alpamayo inference ─
            pred_source = "alpamayo"
            try:
                with torch.no_grad():
                    pred_future_xyz, pred_future_rot, cot_text = project_trajectory(
                        image_frames=image_frames,
                        ego_history_xyz=ego_hist_xyz,
                        ego_history_rot=ego_hist_rot,
                        model_variant=args.model_variant,
                    )

                # pred_future_xyz: (1, 1, 16, 3); pick step at control_stride
                stride_idx = min(args.control_stride - 1,
                                 pred_future_xyz.shape[2] - 1)
                pred_lxyz = pred_future_xyz[0, 0, stride_idx].cpu().float().numpy()
                pred_lrot = pred_future_rot[0, 0, stride_idx].cpu().float().numpy()

                new_x, new_y, new_z = local_to_world_xyz(
                    float(pred_lxyz[0]), float(pred_lxyz[1]), float(pred_lxyz[2]),
                    t0_wx, t0_wy, t0_wz, t0_wyaw,
                )
                new_yaw = t0_wyaw + rot3x3_to_local_yaw(pred_lrot)

                # update fallback velocity from this successful prediction
                dx, dy = new_x - current_x, new_y - current_y
                c, s = math.cos(current_yaw), math.sin(current_yaw)
                fallback_vx_mps = (c * dx + s * dy) / args.dt
                fallback_dyaw_radps = (new_yaw - current_yaw) / args.dt

                save_prediction(output_dir, step,
                                pred_future_xyz[0, 0].cpu().numpy(),
                                pred_future_rot[0, 0].cpu().numpy())
                save_cot(output_dir, step, cot_text)

            except Exception as exc:
                log.error("Step %d: inference failed (%s) — using fallback", step, exc)
                pred_source = "fallback"
                c, s = math.cos(current_yaw), math.sin(current_yaw)
                new_x = current_x + c * fallback_vx_mps * args.dt
                new_y = current_y + s * fallback_vx_mps * args.dt
                new_z = current_z
                new_yaw = current_yaw + fallback_dyaw_radps * args.dt
                if args.save_debug:
                    save_cot(output_dir, step, f"FALLBACK: {exc}")

            # ─ apply predicted pose ─
            new_transform = make_transform(new_x, new_y, new_z, new_yaw, args.z_offset)
            ego_vehicle.set_transform(new_transform)
            target_frame = world.tick()

            # ─ collect new images ─
            images = collect_images(cameras, target_frame)
            if images is None:
                log.warning("Step %d: image collection failed, using blank frame", step)
                new_frame_dict = blank_frame_dict(args.image_height, args.image_width)
            else:
                new_frame_dict = {cn: carla_image_to_numpy(images[cn])
                                  for cn, _, _ in CAMERA_SPECS}
                save_step_images(output_dir, step, images)

            # ─ update buffer and history ─
            cam_frame_buffer.append(new_frame_dict)
            ego_history_deque.append((new_x, new_y, new_z, new_yaw))

            # ─ compute scalar vx for CSV ─
            dx, dy = new_x - current_x, new_y - current_y
            c, s = math.cos(new_yaw), math.sin(new_yaw)
            vx_mps = (c * dx + s * dy) / args.dt

            ego_csv_writer.writerow({
                "step": step,
                "t_sec": round(t_sec, 4),
                "x_m": round(new_x, 6), "y_m": round(new_y, 6), "z_m": round(new_z, 6),
                "yaw_rad": round(new_yaw, 6),
                "vx_mps": round(vx_mps, 4),
                "source": pred_source,
            })

            current_x, current_y, current_z, current_yaw = new_x, new_y, new_z, new_yaw

            if step % 10 == 0 or step == args.num_steps:
                log.info("[%d/%d] t=%.2fs  world=(%.2f, %.2f, %.2f)  "
                         "yaw=%.3frad  vx=%.2f m/s  src=%s",
                         step, args.num_steps, t_sec,
                         new_x, new_y, new_z, new_yaw, vx_mps, pred_source)

        ego_csv_file.close()
        ego_csv_file = None
        log.info("closed_loop_egomotion.csv saved: %s", ego_csv_path)

        # ── metadata ──
        metadata = {
            "map": map_name,
            "init_egomotion_csv": str(init_csv),
            "init_t0_sec": args.init_t0_sec,
            "model_variant": args.model_variant,
            "num_steps": args.num_steps,
            "dt": args.dt,
            "control_stride": args.control_stride,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "weather_preset": args.weather_preset,
            "device": str(device),
            "z_offset": args.z_offset,
            "init_buffer_mode": args.init_buffer_mode,
            "fixed_delta_seconds": FIXED_DELTA_SECONDS,
            "warmup_ticks": WARMUP_TICKS,
            "cameras": [
                {
                    "carla_name": cname,
                    "output_name": oname,
                    "fov": fov,
                    "offset": {
                        "x": off.location.x, "y": off.location.y, "z": off.location.z,
                        "yaw_deg": off.rotation.yaw,
                    },
                }
                for (cname, off, fov), oname in zip(CAMERA_SPECS, CAM_OUT_NAMES)
            ],
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log.info("metadata.json saved: %s", meta_path)
        log.info("Done → %s", output_dir)

    finally:
        if ego_csv_file is not None and not ego_csv_file.closed:
            ego_csv_file.close()
        for _, actor, _ in cameras:
            actor.stop()
            actor.destroy()
        if ego_vehicle is not None:
            ego_vehicle.destroy()
        world.apply_settings(original_settings)
        try:
            world.set_weather(original_weather)
        except Exception:
            pass
        log.info("Cleaned up actors and restored world settings")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
