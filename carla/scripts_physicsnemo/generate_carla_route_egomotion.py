#!/usr/bin/env python3
"""
Generate PhysicsNeMo egomotion CSV from a CARLA route CSV.

Supports three driving scenarios with configurable assumed/actual mu:
  normal_dry      : mu_assumed=0.9, mu_actual=0.9 (dry road, correct assumption)
  safe_low_mu     : mu_assumed=0.4, mu_actual=0.4 (wet/icy, driver aware)
  misjudged_low_mu: mu_assumed=0.9, mu_actual=0.4 (driver misjudges slippery surface)

Output directory: <output-dir>/carla_route/<route_stem>/<scenario>/
  egomotion.csv         - CARLA world absolute coordinates
  simulation_result.csv - full simulation state history (local frame)
  trajectory.png        - XY trajectory vs road centerline
  timeseries.png        - speed / beta / ax / steering vs time
  metadata.json         - run parameters

Usage:
  python scripts_physicsnemo/generate_carla_route_egomotion.py \\
      --model-path outputs_training/vehicle_model.pt \\
      --route-csv carla_maps/routes/Town10HD_Opt_sp147_normal.csv \\
      --scenario safe_low_mu \\
      [--mu-assumed 0.9] [--mu-actual 0.3] \\
      [--speed-limit-kph 40] [--lane-width-m 3.5] \\
      [--n-trials 250] [--dt 0.1] [--t-max 60] \\
      [--output-dir outputs_physicsnemo]"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Allow imports from this directory regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

from inference import (
    CARLARouteRoad,
    DrivingConstraints,
    RoadFollowingPolicy,
    SearchConfig,
    Simulator,
    evaluate_result,
    optimize_policy,
    run_policy_simulation,
    save_egomotion_csv_carla_world,
    save_result_csv,
    save_timeseries_plot,
    save_trajectory_plot,
)
from physicsnemo_can_vehicle_training import (
    PhysicsInformedVehicleModel,
    TrainConfig,
    VehicleSpecPriors,
)

# Default scenario mu values
SCENARIO_MU = {
    "normal_dry":       {"mu_assumed": 0.9, "mu_actual": 0.9},
    "safe_low_mu":      {"mu_assumed": 0.2, "mu_actual": 0.2},
    "misjudged_low_mu": {"mu_assumed": 0.9, "mu_actual": 0.2},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True,
                   help="Path to trained PhysicsNeMo vehicle model (.pt)")
    p.add_argument("--route-csv", required=True,
                   help="CARLA route CSV with columns: s_m, x_m, y_m, z_m, yaw_rad, curvature_1pm")

    p.add_argument("--scenario", default="misjudged_low_mu",
                   choices=list(SCENARIO_MU.keys()),
                   help="Driving scenario (sets default mu values; default: misjudged_low_mu)")
    p.add_argument("--mu-assumed", type=float, default=None,
                   help="Assumed tire friction coefficient for policy optimization "
                        "(overrides scenario default)")
    p.add_argument("--mu-actual", type=float, default=None,
                   help="Actual tire friction coefficient for simulation "
                        "(overrides scenario default)")

    p.add_argument("--speed-limit-kph", type=float, default=40.0,
                   help="Road speed limit [km/h] (default: 40)")
    p.add_argument("--lane-width-m", type=float, default=3.5,
                   help="Lane width [m] (default: 3.5)")
    p.add_argument("--curvature-sigma", type=float, default=2.0,
                   help="Gaussian smoothing sigma for curvature (default: 2.0)")
    p.add_argument("--kappa-turn-threshold", type=float, default=0.01,
                   help="Curvature threshold for turn region detection (default: 0.01)")

    p.add_argument("--n-trials", type=int, default=250,
                   help="Policy search trials (default: 250)")
    p.add_argument("--dt", type=float, default=0.1,
                   help="Simulation timestep [s] (default: 0.1)")
    p.add_argument("--t-max", type=float, default=60.0,
                   help="Max simulation time [s] (default: 60)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for policy search (default: 42)")

    p.add_argument("--wheelbase-m", type=float, default=2.3,
                   help="Vehicle wheelbase [m] (default: 2.3)")
    p.add_argument("--mass-kg", type=float, default=2050.0,
                   help="Vehicle mass [kg] (default: 2050.0)")
    p.add_argument("--hidden-dim", type=int, default=32,
                   help="Model hidden dim — must match training config (default: 32)")

    p.add_argument("--device", default="cpu",
                   help="Torch device (default: cpu)")
    p.add_argument("--output-dir", default="outputs_physicsnemo",
                   help="Base output directory (default: outputs_physicsnemo)")
    p.add_argument("--initial-speed-kph", type=float, default=None,
                   help="Initial vehicle speed [km/h] (default: road speed limit or v_straight)")
    return p.parse_args()


def load_vehicle_model(model_path, wheelbase_m, mass_kg, hidden_dim, device):
    priors = VehicleSpecPriors(
        mass_kg_init=mass_kg,
        wheelbase_m=wheelbase_m,
        front_weight_fraction=0.55,
    )
    cfg = TrainConfig(device=device, hidden_dim=hidden_dim)
    model = PhysicsInformedVehicleModel(
        priors=priors,
        cfg=cfg,
        mass_kg=mass_kg,
        drag_terms={},
    )
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        summary = ckpt.get("summary", "")
    else:
        model.load_state_dict(ckpt)
        summary = ""
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded: {model_path}")
    if summary:
        print(f"[INFO] Summary: {summary}")
    return model


def main():
    args = parse_args()

    # --- resolve mu values ---
    defaults = SCENARIO_MU[args.scenario]
    mu_assumed = args.mu_assumed if args.mu_assumed is not None else defaults["mu_assumed"]
    mu_actual  = args.mu_actual  if args.mu_actual  is not None else defaults["mu_actual"]

    print(f"[INFO] Scenario     : {args.scenario}")
    print(f"[INFO] mu_assumed   : {mu_assumed}")
    print(f"[INFO] mu_actual    : {mu_actual}")

    # --- initial speed ---
    initial_speed_mps = args.initial_speed_kph / 3.6 if args.initial_speed_kph is not None else None
    if initial_speed_mps is not None:
        print(f"[INFO] initial_speed: {args.initial_speed_kph:.1f} km/h = {initial_speed_mps:.3f} m/s")
    else:
        print(f"[INFO] initial_speed: auto (road speed limit or v_straight)")

    # --- output directory ---
    route_stem = Path(args.route_csv).stem
    out_dir = Path(args.output_dir) / "carla_route" / route_stem / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output dir   : {out_dir}")

    # --- build road ---
    road = CARLARouteRoad(
        route_csv=args.route_csv,
        lane_width_m=args.lane_width_m,
        speed_limit_kph=args.speed_limit_kph,
        curvature_sigma=args.curvature_sigma,
        kappa_turn_threshold=args.kappa_turn_threshold,
    )
    print(f"[INFO] Route CSV    : {args.route_csv}")
    print(f"[INFO] total_length : {road.total_length:.1f} m")
    print(f"[INFO] turn_start_s : {road.turn_start_s:.1f} m")
    print(f"[INFO] turn_end_s   : {road.turn_end_s:.1f} m")
    print(f"[INFO] R (effective): {road.R:.1f} m")
    print(f"[INFO] L1 / L2      : {road.L1:.1f} m / {road.L2:.1f} m")
    print(f"[INFO] speed_limit  : {road.speed_limit_mps * 3.6:.1f} km/h")

    # --- load model ---
    model = load_vehicle_model(
        args.model_path,
        wheelbase_m=args.wheelbase_m,
        mass_kg=args.mass_kg,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    simulator = Simulator(model, device=args.device)
    constraints = DrivingConstraints()

    # --- policy optimisation (using mu_assumed) ---
    search_cfg = SearchConfig(
        n_trials=args.n_trials,
        dt=args.dt,
        t_max=args.t_max,
        seed=args.seed,
        outdir=str(out_dir / "search"),
    )
    print(f"\n[INFO] Optimising policy (n_trials={args.n_trials}, mu_assumed={mu_assumed}) ...")
    best_params, best_result_opt, best_metrics_opt, _ = optimize_policy(
        simulator=simulator,
        road=road,
        assumed_mu=mu_assumed,
        constraints=constraints,
        search_cfg=search_cfg,
    )

    print(f"[INFO] Best policy: feasible={best_metrics_opt['feasible']}  "
          f"score={best_metrics_opt['score']:.1f}  "
          f"v_straight={best_params.v_straight:.2f} m/s  "
          f"v_turn={best_params.v_turn:.2f} m/s")

    # --- run scenario simulation (actual_mu may differ) ---
    print(f"\n[INFO] Running simulation (mu_assumed={mu_assumed}, mu_actual={mu_actual}) ...")
    policy = RoadFollowingPolicy(
        road=road,
        wheelbase_m=float(model.priors.wheelbase_m),
        constraints=constraints,
        params=best_params,
    )
    result = run_policy_simulation(
        simulator=simulator,
        road=road,
        policy=policy,
        actual_mu=mu_actual,
        control_mu=mu_assumed,
        dt=args.dt,
        t_max=args.t_max,
        initial_speed_mps=initial_speed_mps,
    )
    metrics = evaluate_result(result, road, constraints)

    print(f"[INFO] Simulation done: goal_reached={result['goal_reached']}  "
          f"final_time={result['final_time']:.1f}s  "
          f"max_beta={metrics['max_abs_beta']:.4f} rad  "
          f"max_ey={metrics['max_abs_ey']:.3f} m")

    # --- save outputs ---
    egomotion_path = out_dir / "egomotion.csv"
    save_egomotion_csv_carla_world(result, road, egomotion_path)
    print(f"[INFO] egomotion.csv: {egomotion_path} ({len(result['t'])} rows)")

    sim_csv_path = out_dir / "simulation_result.csv"
    save_result_csv(result, sim_csv_path)
    print(f"[INFO] simulation_result.csv: {sim_csv_path}")

    traj_png = out_dir / "trajectory.png"
    save_trajectory_plot(
        result, road, traj_png,
        title=f"{route_stem} / {args.scenario}  "
              f"mu_assumed={mu_assumed}  mu_actual={mu_actual}",
    )
    print(f"[INFO] trajectory.png: {traj_png}")

    ts_png = out_dir / "timeseries.png"
    save_timeseries_plot(
        result, constraints, ts_png,
        title=f"{route_stem} / {args.scenario}",
    )
    print(f"[INFO] timeseries.png: {ts_png}")

    # --- metadata.json ---
    meta = {
        "route_csv": str(args.route_csv),
        "route_stem": route_stem,
        "scenario": args.scenario,
        "mu_assumed": mu_assumed,
        "mu_actual": mu_actual,
        "speed_limit_kph": args.speed_limit_kph,
        "lane_width_m": args.lane_width_m,
        "n_trials": args.n_trials,
        "dt": args.dt,
        "t_max": args.t_max,
        "seed": args.seed,
        "road": {
            "total_length_m": road.total_length,
            "turn_start_s": road.turn_start_s,
            "turn_end_s": road.turn_end_s,
            "turn_len_m": road.turn_len,
            "R_m": road.R,
            "L1_m": road.L1,
            "L2_m": road.L2,
        },
        "best_policy": {k: float(v) for k, v in best_params.__dict__.items()},
        "metrics": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                    for k, v in metrics.items()},
        "goal_reached": bool(result["goal_reached"]),
        "final_time_s": float(result["final_time"]),
        "n_timesteps": int(len(result["t"])),
        "initial_speed_kph": args.initial_speed_kph,
        "initial_speed_mps": initial_speed_mps,
        "model_path": str(args.model_path),
        "device": args.device,
        "wheelbase_m": args.wheelbase_m,
        "mass_kg": args.mass_kg,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] metadata.json: {meta_path}")

    print(f"\n[DONE] All outputs saved → {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
