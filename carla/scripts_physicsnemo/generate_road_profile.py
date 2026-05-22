#!/usr/bin/env python3
"""
Generate road curvature profile CSV for:
100m straight -> 90 degree turn -> 100m straight

Example usage:
python generate_road_profile.py \
  --turn-direction right \
  --output road_profile_right.csv

Output columns:
- s_m
- curvature_1pm
- section
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_straight_turn_straight_profile(
    straight1_m: float = 100.0,
    turn_radius_m: float = 20.0,
    straight2_m: float = 100.0,
    sample_ds: float = 0.5,
    turn_direction: str = "right",
) -> pd.DataFrame:
    """
    Generate s-curvature profile for:
    straight -> right 90-degree circular arc -> straight.

    Parameters
    ----------
    straight1_m:
        Length of the first straight segment [m].
    turn_radius_m:
        Radius of the 90-degree turn [m].
    straight2_m:
        Length of the second straight segment [m].
    sample_ds:
        Sampling interval along arc length [m].
    turn_direction:
        Turn direction.
        "right" means negative curvature.
        "left" means positive curvature.

    Returns
    -------
    pd.DataFrame
        Columns:
        - s_m
        - curvature_1pm
        - section
    """

    L1 = float(straight1_m)
    R = float(turn_radius_m)
    L2 = float(straight2_m)
    ds = float(sample_ds)

    if R <= 0:
        raise ValueError("turn_radius_m must be positive")
    if ds <= 0:
        raise ValueError("sample_ds must be positive")
    if L1 < 0 or L2 < 0:
        raise ValueError("straight lengths must be non-negative")

    turn_len = 0.5 * np.pi * R
    turn_start_s = L1
    turn_end_s = L1 + turn_len
    total_length = L1 + turn_len + L2

    # Include the final point.
    s_values = np.arange(0.0, total_length + 0.5 * ds, ds)

    if turn_direction == "right":
        kappa_turn = -1.0 / R
    elif turn_direction == "left":
        kappa_turn = 1.0 / R
    else:
        raise ValueError("turn_direction must be 'right' or 'left'")

    rows = []
    for s in s_values:
        if s < turn_start_s:
            curvature = 0.0
            section = "straight1"
        elif s <= turn_end_s:
            curvature = kappa_turn
            section = f"{turn_direction}_turn"
        else:
            curvature = 0.0
            section = "straight2"

        rows.append(
            {
                "s_m": float(s),
                "curvature_1pm": float(curvature),
                "section": section,
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--straight1-m", type=float, default=100.0)
    parser.add_argument("--turn-radius-m", type=float, default=20.0)
    parser.add_argument("--straight2-m", type=float, default=100.0)
    parser.add_argument("--sample-ds", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="physicsnemo_road_profile.csv")
    parser.add_argument(
        "--turn-direction",
        choices=["right", "left"],
        default="right",
        help="Turn direction. right=negative curvature, left=positive curvature.",
    )

    args = parser.parse_args()

    df = generate_straight_turn_straight_profile(
        straight1_m=args.straight1_m,
        turn_radius_m=args.turn_radius_m,
        straight2_m=args.straight2_m,
        sample_ds=args.sample_ds,
        turn_direction=args.turn_direction,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    turn_len = 0.5 * np.pi * args.turn_radius_m
    total_length = args.straight1_m + turn_len + args.straight2_m
    turn_section = f"{args.turn_direction}_turn"


    print(f"[INFO] Saved road profile: {output_path}")
    print(f"[INFO] straight1_m      = {args.straight1_m:.3f}")
    print(f"[INFO] turn_radius_m   = {args.turn_radius_m:.3f}")
    print(f"[INFO] turn_len_m      = {turn_len:.3f}")
    print(f"[INFO] straight2_m      = {args.straight2_m:.3f}")
    print(f"[INFO] total_length_m   = {total_length:.3f}")
    print(f"[INFO] sample_ds        = {args.sample_ds:.3f}")
    print(f"[INFO] turn_curvature   = "f"{df.loc[df['section'] == turn_section, 'curvature_1pm'].iloc[0]:.6f} [1/m]")

if __name__ == "__main__":
    main()