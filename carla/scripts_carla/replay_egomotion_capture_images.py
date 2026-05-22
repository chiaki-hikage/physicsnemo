#!/usr/bin/env python3
"""
Replay PhysicsNeMo egomotion CSV in CARLA and capture 4-camera images.

The egomotion trajectory is replayed via ego_vehicle.set_transform() — physics
simulation is disabled, so tire_friction / weather affect only visual appearance
and metadata, not the vehicle trajectory.

Usage (absolute-coords, CARLA-route egomotion, short test):
    python scripts/replay_egomotion_capture_images.py \
        --csv outputs_physicsnemo/carla_route/Town10HD_Opt_sp147_normal/egomotion.csv \
        --sample-id town10_sp147_4cam_10sec_rain \
        --output-dir output \
        --map Town10HD_Opt \
        --strict-map-check \
        --absolute-coords \
        --duration-sec 10 \
        --image-width 800 \
        --image-height 450 \
        --weather hard_rain_fog \
        --tire-friction 0.6

Usage (relative mode, legacy PhysicsNeMo synthetic egomotion):
    python scripts/replay_egomotion_capture_images.py \
        --csv path/to/egomotion.csv \
        --sample-id sample_001 \
        [--map Town01] [--flip-y]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from queue import Empty, Queue

import carla
import pandas as pd

FIXED_DELTA_SECONDS = 0.1
WARMUP_TICKS = 3

REQUIRED_COLS = {"t_s", "timestamp_us", "x_m", "y_m", "z_m", "yaw_rad",
                 "qx", "qy", "qz", "qw"}

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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Path to PhysicsNeMo egomotion CSV")
    p.add_argument("--sample-id", required=True,
                   help="Sample ID (used as output subdirectory name)")
    p.add_argument("--output-dir", default="output",
                   help="Base output directory (default: output)")
    p.add_argument("--host", default="localhost", help="CARLA server host")
    p.add_argument("--port", type=int, default=2000, help="CARLA server port")
    p.add_argument("--map", default=None,
                   help="Expected map name substring (check only; does not load a map)")
    p.add_argument("--strict-map-check", action="store_true",
                   help="Exit with error if --map does not match the running world map")
    p.add_argument("--flip-y", action="store_true",
                   help="Negate Y and yaw (relative mode only, for coordinate convention check)")
    p.add_argument("--absolute-coords", action="store_true",
                   help="Treat x_m/y_m/z_m/yaw_rad as CARLA world absolute coordinates. "
                        "Use this for egomotion generated from CARLA route CSVs.")
    p.add_argument("--z-offset", type=float, default=0.3,
                   help="Z offset [m] added to every pose z_m (default: 0.3)")
    p.add_argument("--image-width", type=int, default=1280,
                   help="Camera image width (default: 1280)")
    p.add_argument("--image-height", type=int, default=720,
                   help="Camera image height (default: 720)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Maximum number of CSV rows to replay")
    p.add_argument("--duration-sec", type=float, default=None,
                   help="Maximum replay duration [s]; converted to frames via "
                        "fps=1/FIXED_DELTA_SECONDS. Combined with --max-frames: "
                        "whichever gives fewer frames wins.")
    p.add_argument("--weather", default="clear",
                   choices=["clear", "hard_rain", "wet_cloudy", "hard_rain_fog"],
                   help="Weather preset (default: clear)")
    p.add_argument("--tire-friction", type=float, default=None,
                   help="Set ego wheel tire_friction (for metadata / future apply_control "
                        "experiments). Has no effect on set_transform replay trajectory.")
    p.add_argument("--friction-trigger", type=float, default=None,
                   help="Place a static.trigger.friction box covering the route "
                        "(optional; for metadata alignment with apply_control experiments)")
    p.add_argument("--strict-frame-sync", action="store_true",
                   help="Exit with error if any camera fails to sync a frame "
                        "(default: warn and skip the frame)")
    return p.parse_args()


# ---------- helpers ----------

def build_weather(preset):
    if preset == "clear":
        return carla.WeatherParameters.ClearNoon
    if preset == "hard_rain":
        return carla.WeatherParameters.HardRainNoon
    if preset == "wet_cloudy":
        return carla.WeatherParameters.WetCloudyNoon
    # hard_rain_fog
    return carla.WeatherParameters(
        cloudiness=100,
        precipitation=100,
        precipitation_deposits=100,
        wetness=100,
        fog_density=35,
        fog_distance=15,
        fog_falloff=1,
        wind_intensity=80,
        sun_altitude_angle=20,
    )


def set_tire_friction(vehicle, friction):
    physics = vehicle.get_physics_control()
    wheels = list(physics.wheels)
    for w in wheels:
        w.tire_friction = float(friction)
    physics.wheels = wheels
    vehicle.apply_physics_control(physics)


def get_image_for_frame(q, target_frame, timeout=2.0):
    """Return camera image matching target_frame; discard stale frames.

    Raises Empty on timeout or if a future frame arrives first (frame skip).
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Empty(f"Timed out waiting for frame {target_frame}")
        image = q.get(True, max(0.01, remaining))
        if image.frame == target_frame:
            return image
        if image.frame < target_frame:
            continue  # stale frame, discard
        raise Empty(f"Skipped target frame {target_frame}, got {image.frame}")


def make_carla_transform(x_m, y_m, z_m, yaw_rad, flip_y, z_offset=0.0,
                         absolute_coords=False, origin_transform=None):
    """Convert egomotion pose to a CARLA world Transform.

    absolute_coords=True : x_m/y_m/z_m/yaw_rad are CARLA world absolute coords.
    absolute_coords=False: pose is relative to origin_transform (spawn point).
    z_offset             : added to z in both modes.
    flip_y               : negate Y and yaw (relative mode only).
    """
    if absolute_coords:
        return carla.Transform(
            carla.Location(x=float(x_m), y=float(y_m), z=float(z_m) + z_offset),
            carla.Rotation(yaw=math.degrees(float(yaw_rad))),
        )

    if flip_y:
        rel_y = -float(y_m)
        rel_yaw_deg = -math.degrees(float(yaw_rad))
    else:
        rel_y = float(y_m)
        rel_yaw_deg = math.degrees(float(yaw_rad))

    rel_loc = carla.Location(x=float(x_m), y=rel_y, z=float(z_m))

    if origin_transform is not None:
        wl = origin_transform.transform(rel_loc)
        world_loc = carla.Location(x=wl.x, y=wl.y, z=wl.z + z_offset)
        world_yaw = origin_transform.rotation.yaw + rel_yaw_deg
    else:
        world_loc = carla.Location(x=float(x_m), y=rel_y, z=float(z_m) + z_offset)
        world_yaw = rel_yaw_deg

    return carla.Transform(world_loc, carla.Rotation(yaw=world_yaw))


def drain_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except Empty:
            break


def setup_output_dirs(base_dir, sample_id):
    out = Path(base_dir) / sample_id
    cam_dirs = {}
    for name, _, _ in CAMERA_SPECS:
        d = out / "images" / name
        d.mkdir(parents=True, exist_ok=True)
        cam_dirs[name] = d
    ego_dir = out / "ego_motion"
    ego_dir.mkdir(parents=True, exist_ok=True)
    return out, cam_dirs, ego_dir


def get_vehicle_blueprint(world):
    lib = world.get_blueprint_library()
    candidates = sorted(lib.filter("vehicle.lincoln.*"), key=lambda bp: bp.id)
    if not candidates:
        candidates = sorted(lib.filter("vehicle.*"), key=lambda bp: bp.id)
    if not candidates:
        sys.exit("[ERROR] No vehicle blueprints found in blueprint library")
    return candidates[0]


def try_spawn_vehicle_at_safe_point(world, vehicle_bp):
    """Spawn vehicle at the first collision-free spawn point."""
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        sys.exit("[ERROR] No spawn points available in this map")
    for i, sp in enumerate(spawn_points):
        actor = world.try_spawn_actor(vehicle_bp, sp)
        if actor is not None:
            print(f"[INFO] Spawned {vehicle_bp.id} at spawn_point[{i}]: {sp.location}")
            return actor
    sys.exit(f"[ERROR] Failed to spawn vehicle at all {len(spawn_points)} spawn points "
             "(all occupied or out of bounds)")


def build_camera_blueprint(world, width, height, fov):
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    cam_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))
    return cam_bp


def place_friction_trigger(world, df, friction, z_offset, absolute_coords, origin_transform):
    """Place a static.trigger.friction box covering the route (optional)."""
    bp = world.get_blueprint_library().find("static.trigger.friction")
    if bp is None:
        print("[WARN] static.trigger.friction blueprint not found, skipping trigger")
        return None

    if absolute_coords:
        xs = df["x_m"].to_numpy(dtype=float)
        ys = df["y_m"].to_numpy(dtype=float)
        zs = df["z_m"].to_numpy(dtype=float)
        cx = float((xs.min() + xs.max()) / 2)
        cy = float((ys.min() + ys.max()) / 2)
        cz = float(zs.mean()) + z_offset
        half_x = float((xs.max() - xs.min()) / 2) + 10.0
        half_y = float((ys.max() - ys.min()) / 2) + 10.0
    else:
        if origin_transform is not None:
            cx = origin_transform.location.x
            cy = origin_transform.location.y
            cz = origin_transform.location.z + z_offset
        else:
            cx, cy, cz = 0.0, 0.0, z_offset
        # conservative extent: max absolute displacement + margin
        max_disp = float(df[["x_m", "y_m"]].abs().max().max()) + 50.0
        half_x = half_y = max_disp

    # CARLA trigger extents are in cm
    bp.set_attribute("friction", str(float(friction)))
    bp.set_attribute("extent_x", str(half_x * 100.0))
    bp.set_attribute("extent_y", str(half_y * 100.0))
    bp.set_attribute("extent_z", str(200.0))  # 2 m half-height

    actor = world.try_spawn_actor(
        bp, carla.Transform(carla.Location(x=cx, y=cy, z=cz))
    )
    if actor is None:
        print("[WARN] Failed to place friction trigger actor")
    return actor


# ---------- main ----------

def main():
    args = parse_args()

    # ---------- output directories ----------
    out_base, cam_dirs, ego_dir = setup_output_dirs(args.output_dir, args.sample_id)

    # ---------- load CSV ----------
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"[ERROR] CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] CSV is missing columns: {sorted(missing)}")
    input_rows = len(df)
    print(f"[INFO] Loaded {input_rows} rows from {csv_path}")

    # ---------- frame limit from --max-frames / --duration-sec ----------
    max_frames = args.max_frames
    if args.duration_sec is not None:
        from_duration = int(args.duration_sec / FIXED_DELTA_SECONDS)
        max_frames = from_duration if max_frames is None else min(max_frames, from_duration)
    if max_frames is not None:
        df = df.head(max_frames)
    used_rows = len(df)
    if used_rows < input_rows:
        print(f"[INFO] Limited to {used_rows} rows "
              f"(max_frames={args.max_frames}, duration_sec={args.duration_sec})")

    # ---------- CARLA client (use already-running world) ----------
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    map_name = world.get_map().name
    print(f"[INFO] Using map: {map_name}")

    if args.map is not None and args.map not in map_name:
        msg = (f"Expected map containing '{args.map}' but server has '{map_name}'")
        if args.strict_map_check:
            sys.exit(f"[ERROR] {msg}")
        print(f"[WARN] {msg} — proceeding anyway")

    # ---------- coordinate mode ----------
    if args.absolute_coords:
        origin_transform = None
        coord_mode = "absolute_carla_world"
        print("[INFO] Coordinate mode: absolute CARLA world (no origin_transform composition)")
    else:
        origin_transform = world.get_map().get_spawn_points()[0]
        coord_mode = "relative_to_origin_transform"
        print(f"[INFO] Coordinate mode: relative to spawn origin {origin_transform.location}")

    original_settings = world.get_settings()
    original_weather = world.get_weather()
    ego_vehicle = None
    cameras = []       # list of (name, actor, queue)
    friction_trigger = None
    ego_records = []
    cam_names = [name for name, _, _ in CAMERA_SPECS]

    try:
        # ---------- synchronous mode ----------
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        # ---------- weather ----------
        world.set_weather(build_weather(args.weather))
        print(f"[INFO] Weather set to: {args.weather}")

        # ---------- spawn ego vehicle at a safe spawn point ----------
        vehicle_bp = get_vehicle_blueprint(world)
        ego_vehicle = try_spawn_vehicle_at_safe_point(world, vehicle_bp)
        ego_vehicle.set_autopilot(False)
        ego_vehicle.set_simulate_physics(False)

        # ---------- tire friction (metadata / future apply_control experiments) ----------
        if args.tire_friction is not None:
            set_tire_friction(ego_vehicle, args.tire_friction)
            print(f"[INFO] tire_friction={args.tire_friction} applied to ego wheels, "
                  f"but physics is disabled for set_transform replay.")

        # ---------- move to first egomotion pose ----------
        first = df.iloc[0]
        first_transform = make_carla_transform(
            first.x_m, first.y_m, first.z_m, first.yaw_rad,
            args.flip_y, args.z_offset, args.absolute_coords, origin_transform,
        )
        ego_vehicle.set_transform(first_transform)
        print(f"[INFO] Moved ego to first pose: {first_transform.location}")

        # ---------- 4 cameras ----------
        for name, cam_offset, fov in CAMERA_SPECS:
            cam_bp = build_camera_blueprint(world, args.image_width, args.image_height, fov)
            actor = world.spawn_actor(cam_bp, cam_offset, attach_to=ego_vehicle)
            q = Queue()
            actor.listen(lambda data, _q=q: _q.put(data))
            cameras.append((name, actor, q))
            print(f"[INFO] Camera '{name}': {args.image_width}x{args.image_height} "
                  f"fov={fov} sensor_tick={FIXED_DELTA_SECONDS}")

        # ---------- warm-up ----------
        for _ in range(WARMUP_TICKS):
            world.tick()
            for _, _, q in cameras:
                drain_queue(q)

        # ---------- friction trigger (optional) ----------
        if args.friction_trigger is not None:
            friction_trigger = place_friction_trigger(
                world, df, args.friction_trigger, args.z_offset,
                args.absolute_coords, origin_transform,
            )
            if friction_trigger is not None:
                print(f"[INFO] Friction trigger placed: friction={args.friction_trigger}")

        # ---------- main loop ----------
        n_rows = len(df)
        saved = 0
        skipped = 0

        for frame_idx, row in enumerate(df.itertuples(index=False)):
            t = make_carla_transform(
                row.x_m, row.y_m, row.z_m, row.yaw_rad,
                args.flip_y, args.z_offset, args.absolute_coords, origin_transform,
            )
            ego_vehicle.set_transform(t)
            target_frame = world.tick()

            # collect images from all 4 cameras
            images = {}
            sync_ok = True
            for name, _, q in cameras:
                try:
                    images[name] = get_image_for_frame(q, target_frame)
                except Empty as e:
                    print(f"[WARN] Frame {frame_idx:06d} camera '{name}': {e}, skipping frame")
                    if args.strict_frame_sync:
                        sys.exit(f"[ERROR] Strict frame sync failed at frame {frame_idx:06d}.")
                    sync_ok = False
                    break

            if not sync_ok:
                skipped += 1
                for _, _, q in cameras:
                    drain_queue(q)
                continue

            # save images
            cam_paths = {}
            for name in cam_names:
                rel_path = f"images/{name}/{frame_idx:06d}.png"
                images[name].save_to_disk(str(out_base / rel_path))
                cam_paths[name] = rel_path

            carla_loc = t.location
            carla_yaw_deg = t.rotation.yaw

            rec = {
                "frame": frame_idx,
                "t_s": row.t_s,
                "timestamp_us": row.timestamp_us,
                "x_m": row.x_m,
                "y_m": row.y_m,
                "z_m": row.z_m,
                "yaw_rad": row.yaw_rad,
                "qx": row.qx,
                "qy": row.qy,
                "qz": row.qz,
                "qw": row.qw,
                "carla_frame": images[cam_names[0]].frame,
                "carla_timestamp": images[cam_names[0]].timestamp,
                "carla_x": carla_loc.x,
                "carla_y": carla_loc.y,
                "carla_z": carla_loc.z,
                "carla_yaw_deg": carla_yaw_deg,
            }
            for name in cam_names:
                rec[f"{name}_path"] = cam_paths[name]
            ego_records.append(rec)
            saved += 1

            if frame_idx % 50 == 0 or frame_idx == n_rows - 1:
                print(f"[{frame_idx:06d}/{n_rows-1:06d}] t={row.t_s:.3f}s  "
                      f"carla_frame={images[cam_names[0]].frame}  "
                      f"carla=({carla_loc.x:.2f}, {carla_loc.y:.2f}, {carla_loc.z:.2f})  "
                      f"yaw={carla_yaw_deg:.1f}deg")

        print(f"[INFO] Captured {saved} frames, skipped {skipped}")

        # ---------- save ego_history.csv ----------
        ego_df = pd.DataFrame(ego_records)
        ego_csv_path = ego_dir / "ego_history.csv"
        ego_df.to_csv(ego_csv_path, index=False)
        print(f"[INFO] ego_history.csv saved: {ego_csv_path}")

        # ---------- save metadata.json ----------
        orig_info = None
        if origin_transform is not None:
            ol = origin_transform.location
            or_ = origin_transform.rotation
            orig_info = {
                "x": ol.x, "y": ol.y, "z": ol.z,
                "roll": or_.roll, "pitch": or_.pitch, "yaw_deg": or_.yaw,
            }

        cameras_meta = []
        for name, cam_offset, fov in CAMERA_SPECS:
            cameras_meta.append({
                "name": name,
                "image_width": args.image_width,
                "image_height": args.image_height,
                "fov": fov,
                "transform": {
                    "x": cam_offset.location.x,
                    "y": cam_offset.location.y,
                    "z": cam_offset.location.z,
                    "roll": cam_offset.rotation.roll,
                    "pitch": cam_offset.rotation.pitch,
                    "yaw": cam_offset.rotation.yaw,
                },
            })

        metadata = {
            "map": map_name,
            "input_csv": str(csv_path),
            "fps": 1.0 / FIXED_DELTA_SECONDS,
            "fixed_delta_seconds": FIXED_DELTA_SECONDS,
            "coordinate_mode": coord_mode,
            "flip_y": args.flip_y,
            "z_offset": args.z_offset,
            "cameras": cameras_meta,
            "origin_transform": orig_info,
            "weather": args.weather,
            "tire_friction": args.tire_friction,
            "friction_trigger": args.friction_trigger,
            "input_rows": input_rows,
            "used_rows": used_rows,
            "requested_max_frames": args.max_frames,
            "requested_duration_sec": args.duration_sec,
            "saved_frames": saved,
            "skipped_frames": skipped,
            "warmup_ticks": WARMUP_TICKS,
        }
        meta_path = out_base / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] metadata.json saved: {meta_path}")

        print(f"\n[DONE] {saved} frames saved → {out_base}")

    finally:
        for name, actor, _ in cameras:
            actor.stop()
            actor.destroy()
        if friction_trigger is not None:
            friction_trigger.destroy()
        if ego_vehicle is not None:
            ego_vehicle.destroy()
        world.apply_settings(original_settings)
        try:
            world.set_weather(original_weather)
        except Exception:
            pass
        print("[INFO] Cleaned up actors and restored world settings")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
