#!/usr/bin/env python3
"""
Find CARLA map route segments whose curvature profile best matches a target road profile.

Usage (default — loads each map in --maps sequentially):
    python scripts/find_matching_carla_routes.py \
        --target-road-profile ./outputs_physicsnemo/ref_routes/road_profile_right.csv \
        --maps Town01 Town03 Town04 \
        --distance 200 \
        --ds 1.0 \
        --top-k 10 \
        --output-dir carla_maps

Use already-loaded map (legacy behaviour with --use-loaded-map):
    python scripts/find_matching_carla_routes.py \
        --target-road-profile ./outputs_physicsnemo/ref_routes/road_profile_right.csv \
        --maps Town10HD_Opt \
        --distance 200 \
        --ds 1.0 \
        --top-k 10 \
        --output-dir carla_maps \
        --use-loaded-map

Branch exploration (new):
    python scripts/find_matching_carla_routes.py \
        --target-road-profile ./outputs_physicsnemo/ref_routes/road_profile_right.csv \
        --maps Town01 \
        --distance 220 \
        --ds 1.0 \
        --top-k 10 \
        --output-dir carla_maps \
        --max-branches 1 \
        --max-branch-options 3

  --max-branches 1 --max-branch-options 3 means: at the first junction
  encountered in each route, explore up to 3 directions; subsequent
  junctions always take next_wps[0].
"""

import argparse
import math
import sys
from pathlib import Path

import carla
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-road-profile", required=True,
                   help="Target road profile CSV with columns: s_m, curvature_1pm")
    p.add_argument("--maps", nargs="+", default=["Town01"],
                   help="CARLA map names to search (default: Town01)")
    p.add_argument("--distance", type=float, default=200.0,
                   help="Minimum route extraction length in meters (default: 200)")
    p.add_argument("--ds", type=float, default=1.0,
                   help="Waypoint sampling interval in meters (default: 1.0)")
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of top candidates to save (default: 10)")
    p.add_argument("--output-dir", default="output/route_search",
                   help="Output directory (default: output/route_search)")
    p.add_argument("--host", default="localhost", help="CARLA server host")
    p.add_argument("--port", type=int, default=2000, help="CARLA server port")
    p.add_argument("--max-branches", type=int, default=0,
                   help="Maximum number of junction branch points to expand with "
                        "multiple next_wps options per route. "
                        "0 = disabled, always take next_wps[0] (default: 0)")
    p.add_argument("--max-branch-options", type=int, default=1,
                   help="How many next_wps candidates to explore at each expanded "
                        "branch point. 1 = next_wps[0] only (same as default). "
                        "e.g. 3 means next_wps[:3] (default: 1)")
    p.add_argument("--use-loaded-map", action="store_true",
                   help="Use the map already loaded in the CARLA server instead of "
                        "calling load_world() for each entry in --maps. "
                        "Equivalent to the legacy single-map behaviour.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def _normalize_angle(angle):
    """Normalize a single angle to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _normalize_angle_vec(angles):
    """Normalize a numpy array of angles to (-pi, pi]."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Route building
# ---------------------------------------------------------------------------

def _walk_route(wps, current_wp, total_s, target_distance, ds,
                max_branches, max_branch_options, branches_used):
    """Walk waypoints, forking at junctions when branch budget remains.

    Walks the route in-place (extending `wps`) until a junction with multiple
    next waypoints is reached and the branch budget allows expansion.
    At that point, copies of the accumulated waypoints are made for each
    candidate next waypoint, and the function recurses into each branch.

    Parameters
    ----------
    wps : list[carla.Waypoint]
        Waypoints accumulated so far; extended in-place until a fork.
    current_wp : carla.Waypoint
        The last waypoint in `wps`.
    total_s : float
        Arc length accumulated so far.
    target_distance : float
        Target route length to reach.
    ds : float
        Waypoint step size.
    max_branches : int
        Maximum number of junctions to expand (per-route budget).
    max_branch_options : int
        Maximum next_wps candidates to follow at each expanded junction.
    branches_used : int
        How many expansions have been used in this route so far.

    Returns
    -------
    list[list[carla.Waypoint]]
        One or more completed routes.
    """
    while total_s < target_distance:
        next_wps = current_wp.next(ds)
        if not next_wps:
            break

        n_next = len(next_wps)
        should_branch = (n_next > 1
                         and max_branch_options > 1
                         and branches_used < max_branches)

        if should_branch:
            n_expand = min(n_next, max_branch_options)
            loc = current_wp.transform.location
            print(f"[BRANCH] ({loc.x:.1f}, {loc.y:.1f}): "
                  f"{n_next} options → expanding {n_expand} "
                  f"(branch {branches_used + 1}/{max_branches})")
            results = []
            for i in range(n_expand):
                # Copy prefix so each branch has its own independent list
                sub_wps = list(wps) + [next_wps[i]]
                results.extend(_walk_route(
                    sub_wps, next_wps[i], total_s + ds, target_distance, ds,
                    max_branches, max_branch_options, branches_used + 1,
                ))
            return results
        else:
            # Default: take next_wps[0]; emit warning on junction
            if n_next > 1:
                loc = current_wp.transform.location
                print(f"[WARN]   Branch at ({loc.x:.1f}, {loc.y:.1f}): "
                      f"{n_next} options, taking [0]")
            wps.append(next_wps[0])
            current_wp = next_wps[0]
            total_s += ds

    return [wps]


def build_route(carla_map, spawn_transform, target_distance, ds,
                max_branches=0, max_branch_options=1):
    """Walk waypoints from spawn_transform for at least target_distance meters.

    Parameters
    ----------
    carla_map : carla.Map
    spawn_transform : carla.Transform
    target_distance : float
        Minimum arc length to walk.
    ds : float
        Waypoint step size in metres.
    max_branches : int
        Maximum number of junction branch points to expand per route.
        0 (default) reproduces the original next_wps[0]-only behaviour.
    max_branch_options : int
        Candidates to explore per expanded junction.
        1 (default) reproduces the original behaviour.

    Returns
    -------
    list[list[carla.Waypoint]]
        A list of routes. In default mode (max_branches=0 or
        max_branch_options=1) this always contains exactly one route,
        matching the original single-route return value.
        Returns an empty list if the start position has no drivable lane.
    """
    wp = carla_map.get_waypoint(
        spawn_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return []

    return _walk_route(
        wps=[wp],
        current_wp=wp,
        total_s=0.0,
        target_distance=target_distance,
        ds=ds,
        max_branches=max_branches,
        max_branch_options=max_branch_options,
        branches_used=0,
    )


# ---------------------------------------------------------------------------
# Profile computation and scoring
# ---------------------------------------------------------------------------

def compute_route_profile(wps):
    """Compute s_m, x_m, y_m, z_m, yaw_rad, curvature_1pm from a waypoint list.

    Curvature uses central differences for interior points and
    forward/backward differences at endpoints. Angle wrap is handled via
    normalization to (-pi, pi].
    """
    xs = np.array([wp.transform.location.x for wp in wps], dtype=float)
    ys = np.array([wp.transform.location.y for wp in wps], dtype=float)
    zs = np.array([wp.transform.location.z for wp in wps], dtype=float)
    yaws_rad = np.radians([wp.transform.rotation.yaw for wp in wps])

    # Cumulative arc length from actual inter-waypoint distances
    deltas = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2)
    s_vals = np.concatenate([[0.0], np.cumsum(deltas)])

    n = len(wps)
    curvatures = np.zeros(n)

    if n >= 3:
        # Central differences for interior points (vectorized)
        dyaw_cd = _normalize_angle_vec(yaws_rad[2:] - yaws_rad[:-2])
        ds_cd = s_vals[2:] - s_vals[:-2]
        curvatures[1:-1] = np.where(ds_cd > 1e-6, dyaw_cd / ds_cd, 0.0)

    if n >= 2:
        dyaw0 = _normalize_angle(float(yaws_rad[1] - yaws_rad[0]))
        ds0 = float(s_vals[1] - s_vals[0])
        curvatures[0] = dyaw0 / ds0 if ds0 > 1e-6 else 0.0

        dyaw_last = _normalize_angle(float(yaws_rad[-1] - yaws_rad[-2]))
        ds_last = float(s_vals[-1] - s_vals[-2])
        curvatures[-1] = dyaw_last / ds_last if ds_last > 1e-6 else 0.0

    return s_vals, xs, ys, zs, yaws_rad, curvatures


def compute_rmse(target_s, target_curv, route_s, route_curv):
    """Interpolate route_curv at target_s positions and return RMSE."""
    route_curv_interp = np.interp(target_s, route_s, route_curv)
    return float(np.sqrt(np.mean((target_curv - route_curv_interp) ** 2)))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Whether multi-branch exploration is active
    use_branching = args.max_branches > 0 and args.max_branch_options > 1

    out_dir = Path(args.output_dir)
    routes_dir = out_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)

    # ---------- load target profile ----------
    profile_path = Path(args.target_road_profile)
    if not profile_path.exists():
        sys.exit(f"[ERROR] Target profile not found: {profile_path}")
    profile_df = pd.read_csv(profile_path)
    missing = {"s_m", "curvature_1pm"} - set(profile_df.columns)
    if missing:
        sys.exit(f"[ERROR] Target profile missing columns: {sorted(missing)}")

    target_s = profile_df["s_m"].to_numpy(dtype=float)
    target_s = target_s - target_s[0]  # normalize to start at 0
    target_curv = profile_df["curvature_1pm"].to_numpy(dtype=float)
    target_end_s = float(target_s[-1])
    print(f"[INFO] Target profile: {len(target_s)} points, "
          f"s=[0, {target_end_s:.1f}]m, "
          f"curvature=[{target_curv.min():.4f}, {target_curv.max():.4f}] 1/m")

    if use_branching:
        print(f"[INFO] Branch exploration: max_branches={args.max_branches}, "
              f"max_branch_options={args.max_branch_options}")
    else:
        print(f"[INFO] Branch exploration: disabled "
              f"(max_branches={args.max_branches}, "
              f"max_branch_options={args.max_branch_options})")

    # Route must cover the full target length; add a small margin
    route_distance = max(args.distance, target_end_s) + args.ds * 2

    # ---------- CARLA client ----------
    client = carla.Client(args.host, args.port)
    # Use a generous timeout: load_world() can take 30-60 s on large maps
    client.set_timeout(120.0)

    all_candidates = []

    # ------------------------------------------------------------------
    # Build the list of (world, map_name) to search.
    #
    # --use-loaded-map : legacy mode — use the currently loaded world as-is.
    # default          : call load_world() for each name in --maps.
    # ------------------------------------------------------------------
    if args.use_loaded_map:
        world = client.get_world()
        carla_map = world.get_map()
        loaded_map_name = carla_map.name.split("/")[-1]
        print(f"[INFO] --use-loaded-map: using already loaded map: {carla_map.name}")
        if args.maps and loaded_map_name not in args.maps:
            print(f"[WARN] Loaded map '{loaded_map_name}' is not in --maps={args.maps}")
        worlds_to_search = [(world, loaded_map_name)]
    else:
        if not args.maps:
            sys.exit("[ERROR] --maps is required when not using --use-loaded-map")
        print(f"[INFO] Maps to load and search: {args.maps}")
        worlds_to_search = []
        for map_name in args.maps:
            print(f"[INFO] Loading map: {map_name} ...")
            world = client.load_world(map_name)
            loaded_map_name = world.get_map().name.split("/")[-1]
            print(f"[INFO] Loaded: {world.get_map().name}")
            worlds_to_search.append((world, loaded_map_name))

    # ------------------------------------------------------------------
    # Search each world
    # ------------------------------------------------------------------
    for world, loaded_map_name in worlds_to_search:
        # Disable rendering for faster waypoint queries
        settings = world.get_settings()
        settings.no_rendering_mode = True
        world.apply_settings(settings)

        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        print(f"\n[INFO] {loaded_map_name}: {len(spawn_points)} spawn points")

        map_valid = 0
        map_skip  = 0

        for sp_idx, sp in enumerate(spawn_points):
            routes = build_route(
                carla_map, sp, route_distance, args.ds,
                max_branches=args.max_branches,
                max_branch_options=args.max_branch_options,
            )

            n_routes = len(routes)
            if use_branching and n_routes > 1:
                print(f"[INFO]   spawn={sp_idx:3d}: {n_routes} route variants generated")

            for branch_id, wps in enumerate(routes):
                if len(wps) < 2:
                    map_skip += 1
                    continue

                s_vals, xs, ys, zs, yaw_rads, curvs = compute_route_profile(wps)

                if s_vals[-1] < target_end_s:
                    map_skip += 1
                    continue

                rmse_n = compute_rmse(target_s, target_curv, s_vals, curvs)
                rmse_f = compute_rmse(target_s, target_curv, s_vals, -curvs)
                flipped = rmse_f < rmse_n
                score   = min(rmse_n, rmse_f)

                cand = {
                    "map":            loaded_map_name,
                    "spawn_idx":      sp_idx,
                    "spawn_x":        float(sp.location.x),
                    "spawn_y":        float(sp.location.y),
                    "spawn_z":        float(sp.location.z),
                    "rmse":           score,
                    "rmse_normal":    rmse_n,
                    "rmse_flipped":   rmse_f,
                    "flipped":        flipped,
                    "route_length_m": float(s_vals[-1]),
                    "n_waypoints":    len(wps),
                    "s_vals":         s_vals,
                    "xs":             xs,
                    "ys":             ys,
                    "zs":             zs,
                    "yaw_rads":       yaw_rads,
                    "curvs":          curvs,
                }
                if use_branching:
                    cand["branch_id"] = branch_id

                all_candidates.append(cand)
                map_valid += 1

        print(f"[INFO] {loaded_map_name}: {map_valid} valid route candidates, "
              f"{map_skip} skipped")
        if use_branching:
            map_total = sum(1 for c in all_candidates if c["map"] == loaded_map_name)
            print(f"[INFO] {loaded_map_name}: {map_total} total "
                  f"(including branch variants)")

    total_candidates = len(all_candidates)
    if len(worlds_to_search) > 1:
        print(f"\n[INFO] All maps combined: {total_candidates} valid candidates")

    if not all_candidates:
        sys.exit("[ERROR] No valid route candidates found. "
                 "Try increasing --distance or checking the maps.")

    # ---------- sort globally and select top-k ----------
    all_candidates.sort(key=lambda c: c["rmse"])
    top_k = all_candidates[: args.top_k]

    print(f"\n[INFO] Top {len(top_k)} candidates (of {total_candidates} total):")

    summary_rows = []
    for rank, cand in enumerate(top_k):
        direction = "flipped" if cand["flipped"] else "normal"

        # File naming: include branch_id only in branching mode
        if use_branching:
            branch_id = cand["branch_id"]
            route_csv_name = (f"{cand['map']}_sp{cand['spawn_idx']:03d}"
                              f"_b{branch_id:02d}_{direction}.csv")
        else:
            route_csv_name = f"{cand['map']}_sp{cand['spawn_idx']:03d}_{direction}.csv"

        route_csv_path = routes_dir / route_csv_name

        # Apply curvature sign flip for the saved CSV
        saved_curvs = -cand["curvs"] if cand["flipped"] else cand["curvs"]

        route_df = pd.DataFrame({
            "s_m":           cand["s_vals"],
            "x_m":           cand["xs"],
            "y_m":           cand["ys"],
            "z_m":           cand["zs"],
            "yaw_rad":       cand["yaw_rads"],
            "curvature_1pm": saved_curvs,
        })
        route_df.to_csv(route_csv_path, index=False)

        # Console log
        if use_branching:
            print(f"  [{rank + 1:2d}] {cand['map']}  spawn={cand['spawn_idx']:3d}  "
                  f"branch={cand['branch_id']:2d}  rmse={cand['rmse']:.6f}  {direction}  "
                  f"len={cand['route_length_m']:.1f}m  wps={cand['n_waypoints']}")
        else:
            print(f"  [{rank + 1:2d}] {cand['map']}  spawn={cand['spawn_idx']:3d}  "
                  f"rmse={cand['rmse']:.6f}  {direction}  "
                  f"len={cand['route_length_m']:.1f}m  wps={cand['n_waypoints']}")

        # Summary row — existing columns always present; branch_id added in branch mode
        row = {
            "rank":           rank + 1,
            "map":            cand["map"],
            "spawn_idx":      cand["spawn_idx"],
            "spawn_x":        cand["spawn_x"],
            "spawn_y":        cand["spawn_y"],
            "spawn_z":        cand["spawn_z"],
            "rmse":           cand["rmse"],
            "rmse_normal":    cand["rmse_normal"],
            "rmse_flipped":   cand["rmse_flipped"],
            "flipped":        cand["flipped"],
            "route_length_m": cand["route_length_m"],
            "n_waypoints":    cand["n_waypoints"],
            "route_csv":      str(route_csv_path.relative_to(out_dir)),
        }
        if use_branching:
            row["branch_id"] = cand["branch_id"]

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "route_candidates.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n[INFO] Summary saved: {summary_path}")
    print(f"[INFO] Route CSVs saved in: {routes_dir}")
    print(f"\n[DONE] {len(top_k)} candidates saved → {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
