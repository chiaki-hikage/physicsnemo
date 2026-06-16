import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =========================
# 1. モデル読み込み
# =========================
def load_model(model_class, priors, cfg, model_path, device="cpu"):
    """
    学習コードで保存した vehicle_model.pt を読み込む。

    想定する保存形式:
        torch.save({"state_dict": model.state_dict(), "summary": summary}, path)
    """
    model = model_class(priors, cfg, mass_kg=priors.mass_kg_init, drag_terms={})

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        summary = ckpt.get("summary", "No summary found")
    else:
        # torch.save(model.state_dict(), path) 形式にも一応対応
        model.load_state_dict(ckpt)
        summary = "No summary found"

    model.to(device)
    model.eval()

    print("Loaded model. Summary:")
    print(summary)

    return model


# =========================
# 2. CAN入力生成（学習時の6chを維持）
# =========================
def make_can(
    vx,
    ax=0.0,
    yaw_proxy=0.0,
    yaw_rel=1.0,
    throttle=0.0,
    rpm=0.0,
    device="cpu",
):
    """
    学習時の base channel 順序に合わせる。
      0 ax_mps2
      1 throttle_pct
      2 rpm
      3 vx_mps
      4 yaw_like_rate_radps
      5 is_yaw_reliable

    推論では throttle/rpm はダミー、vx/ax/yaw_proxy はシナリオと状態から生成する。
    """
    return torch.tensor(
        [[ax, throttle, rpm, vx, yaw_proxy, yaw_rel]],
        dtype=torch.float32,
        device=device,
    )


# =========================
# 4. シミュレータ
# =========================
class Simulator:
    def __init__(
        self,
        model,
        device="cpu",
        history_steps=3,
        ax_min=-5.0,
        ax_max=3.0,
        speed_kp=1.5,
        speed_ff=1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.history_steps = history_steps
        self.ax_min = ax_min
        self.ax_max = ax_max
        self.speed_kp = speed_kp
        self.speed_ff = speed_ff

    def set_mu(self, mu):
        lo, hi = self.model.priors.mu_bounds
        mu = np.clip(mu, lo + 1e-5, hi - 1e-5)
        z = (mu - lo) / (hi - lo)
        raw = np.log(z / (1.0 - z))
        with torch.no_grad():
            self.model.raw_mu.copy_(torch.tensor(raw, dtype=torch.float32, device=self.device))
        print(f"[set_mu] requested_mu={mu}, actual_model_mu={self.model.mu().item()}")

    def _compute_ax_command(self, scenario, t, dt, vx_current):
        """
        速度の連続性を保つため、目標速度プロファイルから加速度を作る。

        ax ≈ feed-forward d(v_target)/dt + feedback (v_target - vx_current)
        これにより、vx(t+dt) ≈ vx(t) + ax(t+0.5dt) * dt の関係に近づける。
        """
        v_ref_now = float(scenario.target_v(t))
        v_ref_next = float(scenario.target_v(t + dt))
        ax_ff = (v_ref_next - v_ref_now) / dt
        ax_fb = self.speed_kp * (v_ref_now - vx_current)
        ax = self.speed_ff * ax_ff + ax_fb
        return float(np.clip(ax, self.ax_min, self.ax_max)), v_ref_now

    def run(self, scenario, T=10.0, dt=0.1, mu=0.9, initial_speed=None):
        self.model.eval()

        if initial_speed is None:
            initial_speed = float(scenario.target_v(0.0))

        state = torch.tensor(
            [[0.0, 0.0, 0.0, initial_speed, 0.0, 0.0]],
            dtype=torch.float32,
            device=self.device,
        )

        # CAN履歴初期化：初期は等速直線、yaw_proxy=0
        can_buffer = [
            make_can(initial_speed, ax=0.0, yaw_proxy=0.0, yaw_rel=1.0, device=self.device)
            for _ in range(self.history_steps)
        ]

        rows = []
        n_steps = int(np.round(T / dt))

        for i in range(n_steps):
            t = i * dt

            # 現在状態から速度・yaw rateを取り出す。
            vx_current = float(state[0, 3].detach().cpu().item())
            r_current = float(state[0, 5].detach().cpu().item())

            # 目標速度から整合的な加速度を生成。
            ax_cmd, v_ref = self._compute_ax_command(scenario, t, dt, vx_current)

            # 操舵角。右折は負。
            delta = float(scenario.delta(t))

            # CAN履歴を更新。
            # yaw_proxy は外部指定の操舵ではなく、直前までの推論状態 r を観測相当として入れる。
            new_can = make_can(
                vx=vx_current,
                ax=ax_cmd,
                yaw_proxy=r_current,
                yaw_rel=1.0,
                device=self.device,
            )
            can_buffer.pop(0)
            can_buffer.append(new_can)
            can_hist = torch.cat(can_buffer, dim=-1)  # [1, 6 * history_steps]

            dt_tensor = torch.tensor([[dt]], dtype=torch.float32, device=self.device)
            delta_tensor = torch.tensor([delta], dtype=torch.float32, device=self.device)

            with torch.no_grad():

                mu_tensor = torch.tensor(mu, dtype=torch.float32, device=self.device)

                state, aux = self.model.step(
                    state,
                    can_hist,
                    dt_tensor,
                    delta_input=delta_tensor,
                    mu_override=mu_tensor,
                )

            s_np = state[0].detach().cpu().numpy()
            rows.append({
                "t": t,
                "x": s_np[0],
                "y": s_np[1],
                "psi": s_np[2],
                "vx": s_np[3],
                "beta": s_np[4],
                "r": s_np[5],
                "v_ref": v_ref,
                "ax_cmd": ax_cmd,
                "delta": delta,
                "mu": mu,
                "delta_eff": float(aux["delta_eff"].detach().cpu().reshape(-1)[0]),
                "dvx_pred": float(aux["dvx_pred"].detach().cpu().reshape(-1)[0]),
                "r_dot_res": float(aux["r_dot_res"].detach().cpu().reshape(-1)[0]),
                "beta_dot_res": float(aux["beta_dot_res"].detach().cpu().reshape(-1)[0]),
            })

        # dict of numpy arrays として返す（既存plot互換）
        result = {k: np.array([row[k] for row in rows]) for k in rows[0].keys()}
        return result


# ============================================================
# 追加ユーティリティ
# ============================================================
def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def kph_to_mps(kph):
    return kph / 3.6


# ============================================================
# 道路定義
# 例：200m直進 → 右90度カーブ → 100m直進
# ============================================================
class StraightRightTurnStraightRoad:
    def __init__(
        self,
        straight1_m=200.0,
        turn_radius_m=20.0,
        straight2_m=100.0,
        lane_width_m=3.5,
        speed_limit_kph=40.0,
        sample_ds=0.5,
    ):
        self.L1 = float(straight1_m)
        self.R = float(turn_radius_m)
        self.L2 = float(straight2_m)
        self.lane_width_m = float(lane_width_m)
        self.speed_limit_mps = kph_to_mps(float(speed_limit_kph))
        self.sample_ds = float(sample_ds)

        self.turn_start_s = self.L1
        self.turn_len = 0.5 * np.pi * self.R
        self.turn_end_s = self.L1 + self.turn_len
        self.total_length = self.L1 + self.turn_len + self.L2

        self._build_reference()

    def centerline_pose(self, s):
        s = float(np.clip(s, 0.0, self.total_length))

        # 1) 直進区間
        if s <= self.L1:
            x = s
            y = 0.0
            psi = 0.0
            kappa = 0.0
            return x, y, psi, kappa

        # 2) 右90度カーブ
        if s <= self.turn_end_s:
            ds = s - self.L1
            phi = 0.5 * np.pi - ds / self.R   # pi/2 -> 0
            x = self.L1 + self.R * np.cos(phi)
            y = -self.R + self.R * np.sin(phi)
            psi = phi - 0.5 * np.pi           # 0 -> -pi/2
            kappa = -1.0 / self.R
            return x, y, psi, kappa

        # 3) カーブ後直進（下向き）
        ds2 = s - self.turn_end_s
        x = self.L1 + self.R
        y = -self.R - ds2
        psi = -0.5 * np.pi
        kappa = 0.0
        return x, y, psi, kappa

    def _build_reference(self):
        s_grid = np.arange(0.0, self.total_length + self.sample_ds, self.sample_ds)
        xs, ys, psis, kappas = [], [], [], []
        for s in s_grid:
            x, y, psi, kappa = self.centerline_pose(s)
            xs.append(x)
            ys.append(y)
            psis.append(psi)
            kappas.append(kappa)

        self.s_ref = np.asarray(s_grid)
        self.x_ref = np.asarray(xs)
        self.y_ref = np.asarray(ys)
        self.psi_ref = np.asarray(psis)
        self.kappa_ref = np.asarray(kappas)

        # 車線境界
        nx = -np.sin(self.psi_ref)
        ny = np.cos(self.psi_ref)
        half = 0.5 * self.lane_width_m

        self.left_x = self.x_ref + half * nx
        self.left_y = self.y_ref + half * ny
        self.right_x = self.x_ref - half * nx
        self.right_y = self.y_ref - half * ny

    def project(self, x, y, psi=None):
        dx = x - self.x_ref
        dy = y - self.y_ref
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))

        x0 = self.x_ref[idx]
        y0 = self.y_ref[idx]
        psi0 = self.psi_ref[idx]
        s0 = self.s_ref[idx]
        kappa0 = self.kappa_ref[idx]

        dx0 = x - x0
        dy0 = y - y0

        # 道路座標系での横ずれ
        ey = -np.sin(psi0) * dx0 + np.cos(psi0) * dy0

        if psi is None:
            epsi = 0.0
        else:
            epsi = wrap_to_pi(psi - psi0)

        return {
            "idx": idx,
            "s": s0,
            "ey": ey,
            "epsi": epsi,
            "x_ref": x0,
            "y_ref": y0,
            "psi_ref": psi0,
            "kappa_ref": kappa0,
        }

    def is_inside(self, x, y):
        proj = self.project(x, y)
        return abs(proj["ey"]) <= 0.5 * self.lane_width_m

    def goal_reached(self, x, y, psi=None, margin_m=2.0):
        proj = self.project(x, y, psi)
        return proj["s"] >= (self.total_length - margin_m)


# ============================================================
# 道路定義（CARLAルートCSVから構築）
# ============================================================
class CARLARouteRoad:
    """
    Road definition built from a CARLA route CSV.

    Duck-type compatible with StraightRightTurnStraightRoad for use with
    RoadFollowingPolicy, optimize_policy, and run_policy_simulation.

    CSV columns: s_m, x_m, y_m, z_m, yaw_rad, curvature_1pm
    (CARLA world absolute coordinates; curvature sign: negative = right turn from driver's view)

    Internal reference arrays are in a 2-D local frame:
      - Origin at the first route point (x0, y0)
      - +x_local along the initial heading (yaw0)
      - +y_local is 90° CCW from +x_local (left side of road)
      - Curvature sign is preserved directly from the CSV

    CARLA world <-> local frame:
      x_carla = x0 + cos(yaw0)*x_local - sin(yaw0)*y_local
      y_carla = y0 + sin(yaw0)*x_local + cos(yaw0)*y_local
      yaw_carla_rad = yaw0 + psi_local
    """

    def __init__(
        self,
        route_csv,
        lane_width_m: float = 3.5,
        speed_limit_kph: float = 40.0,
        curvature_sigma: float = 2.0,
        kappa_turn_threshold: float = 0.01,
    ):
        self.lane_width_m = float(lane_width_m)
        self.speed_limit_mps = kph_to_mps(float(speed_limit_kph))

        df = pd.read_csv(route_csv)
        missing = {"s_m", "x_m", "y_m", "z_m", "yaw_rad", "curvature_1pm"} - set(df.columns)
        if missing:
            raise ValueError(f"Route CSV missing columns: {sorted(missing)}")

        # CARLA world origin pose
        self.x0 = float(df["x_m"].iloc[0])
        self.y0 = float(df["y_m"].iloc[0])
        self.z0 = float(df["z_m"].iloc[0])
        self.yaw0 = float(df["yaw_rad"].iloc[0])

        s_csv = df["s_m"].to_numpy(dtype=float)
        s_csv = s_csv - s_csv[0]

        x_world = df["x_m"].to_numpy(dtype=float)
        y_world = df["y_m"].to_numpy(dtype=float)
        z_world = df["z_m"].to_numpy(dtype=float)
        yaw_world = df["yaw_rad"].to_numpy(dtype=float)
        kappa_raw = df["curvature_1pm"].to_numpy(dtype=float)

        # Transform to local frame
        c0, s0 = np.cos(self.yaw0), np.sin(self.yaw0)
        dx = x_world - self.x0
        dy = y_world - self.y0
        x_local = c0 * dx + s0 * dy
        y_local = -s0 * dx + c0 * dy
        # Unwrap for smooth angular interpolation
        psi_local = np.unwrap(yaw_world - self.yaw0)

        # Smooth curvature
        try:
            from scipy.ndimage import gaussian_filter1d
            kappa_smooth = gaussian_filter1d(kappa_raw.astype(float), sigma=curvature_sigma)
        except ImportError:
            kappa_smooth = kappa_raw.copy()

        self.s_ref = s_csv
        self.x_ref = x_local
        self.y_ref = y_local
        self.psi_ref = psi_local
        self.kappa_ref = kappa_smooth
        self.z_ref = z_world

        self.total_length = float(s_csv[-1])

        # Lane boundaries (local frame)
        nx = -np.sin(self.psi_ref)
        ny = np.cos(self.psi_ref)
        half = 0.5 * self.lane_width_m
        self.left_x = self.x_ref + half * nx
        self.left_y = self.y_ref + half * ny
        self.right_x = self.x_ref - half * nx
        self.right_y = self.y_ref - half * ny

        # Detect turn region (contiguous segment where |kappa| > threshold)
        abs_k = np.abs(kappa_smooth)
        turn_mask = abs_k > kappa_turn_threshold
        if turn_mask.any():
            idxs = np.where(turn_mask)[0]
            self.turn_start_s = float(s_csv[idxs[0]])
            self.turn_end_s = float(s_csv[idxs[-1]])
        else:
            self.turn_start_s = self.total_length / 3.0
            self.turn_end_s = 2.0 * self.total_length / 3.0

        self.turn_len = self.turn_end_s - self.turn_start_s
        self.L1 = self.turn_start_s
        self.L2 = self.total_length - self.turn_end_s

        max_k = float(np.max(abs_k))
        self.R = 1.0 / max(max_k, 1e-3)

    def centerline_pose(self, s):
        s = float(np.clip(s, 0.0, self.total_length))
        x = float(np.interp(s, self.s_ref, self.x_ref))
        y = float(np.interp(s, self.s_ref, self.y_ref))
        psi = float(np.interp(s, self.s_ref, self.psi_ref))
        kappa = float(np.interp(s, self.s_ref, self.kappa_ref))
        return x, y, psi, kappa

    def project(self, x, y, psi=None):
        dx = x - self.x_ref
        dy = y - self.y_ref
        idx = int(np.argmin(dx * dx + dy * dy))

        x0, y0 = self.x_ref[idx], self.y_ref[idx]
        psi0 = self.psi_ref[idx]
        s0 = self.s_ref[idx]
        kappa0 = self.kappa_ref[idx]

        dx0 = x - x0
        dy0 = y - y0
        ey = -np.sin(psi0) * dx0 + np.cos(psi0) * dy0

        epsi = 0.0 if psi is None else wrap_to_pi(psi - psi0)

        return {"idx": idx, "s": s0, "ey": ey, "epsi": epsi,
                "x_ref": x0, "y_ref": y0, "psi_ref": psi0, "kappa_ref": kappa0}

    def is_inside(self, x, y):
        return abs(self.project(x, y)["ey"]) <= 0.5 * self.lane_width_m

    def goal_reached(self, x, y, psi=None, margin_m=2.0):
        return self.project(x, y, psi)["s"] >= (self.total_length - margin_m)

    def local_to_carla(self, x_local, y_local, psi_local):
        """Convert local-frame (x, y, psi) arrays to CARLA world coordinates."""
        c0, s0 = np.cos(self.yaw0), np.sin(self.yaw0)
        xl = np.asarray(x_local, dtype=float)
        yl = np.asarray(y_local, dtype=float)
        x_carla = self.x0 + c0 * xl - s0 * yl
        y_carla = self.y0 + s0 * xl + c0 * yl
        yaw_carla = self.yaw0 + np.asarray(psi_local, dtype=float)
        return x_carla, y_carla, yaw_carla

    def interp_z(self, s_arr):
        """Interpolate CARLA z from arc-length array."""
        return np.interp(np.asarray(s_arr, dtype=float), self.s_ref, self.z_ref)


# ============================================================
# 制約・探索設定
# ============================================================
@dataclass
class DrivingConstraints:
    beta_max: float = 0.08                 # [rad]
    ax_min: float = -2.5                   # [m/s^2] 強めの減速下限
    ax_max: float = 2.0                    # [m/s^2] 加速上限
    jerk_max: float = 2.0                  # [m/s^3]
    delta_max: float = np.deg2rad(25.0)    # [rad]
    steer_rate_max: float = np.deg2rad(25.0)  # [rad/s]
    speed_margin_mps: float = 0.0


@dataclass
class SearchConfig:
    n_trials: int = 250
    dt: float = 0.1
    t_max: float = 60.0
    seed: int = 42
    outdir: str = "planner_outputs"


@dataclass
class PolicyParams:
    v_straight: float
    v_turn: float
    brake_buffer_m: float
    accel_buffer_m: float
    lookahead_m: float
    ky: float
    kpsi: float
    kv: float
    speed_tolerance_mps: float = 0.5


# ============================================================
# 道路追従＋速度計画ポリシー
# ============================================================
class RoadFollowingPolicy:
    def __init__(self, road, wheelbase_m, constraints: DrivingConstraints, params: PolicyParams):
        self.road = road
        self.wheelbase_m = float(wheelbase_m)
        self.constraints = constraints
        self.params = params

        self.prev_delta = 0.0
        self.prev_ax = 0.0

    def reset(self):
        self.prev_delta = 0.0
        self.prev_ax = 0.0
        self.v_cmd = None

    def target_speed(self, s):
        """
        進捗 s に応じて目標速度を設定
          - 手前直線: v_straight
          - カーブ前 brake_buffer_m で v_turn へ減速
          - カーブ中: v_turn
          - カーブ後 accel_buffer_m で v_straight へ戻す
        """
        v_st = min(self.params.v_straight, self.road.speed_limit_mps)
        v_tr = min(self.params.v_turn, self.road.speed_limit_mps)

        s_brake_start = max(0.0, self.road.turn_start_s - self.params.brake_buffer_m)
        s_accel_end = min(self.road.total_length, self.road.turn_end_s + self.params.accel_buffer_m)

        if s < s_brake_start:
            return v_st

        if s < self.road.turn_start_s:
            a = (s - s_brake_start) / max(1e-6, self.road.turn_start_s - s_brake_start)
            return v_st * (1.0 - a) + v_tr * a

        if s <= self.road.turn_end_s:
            return v_tr

        if s <= s_accel_end:
            a = (s - self.road.turn_end_s) / max(1e-6, s_accel_end - self.road.turn_end_s)
            return v_tr * (1.0 - a) + v_st * a

        return v_st
    
    def driving_phase(self, s):
        """
        道路進捗 s に応じて運転フェーズを返す。

        approach : カーブ手前の通常走行
        braking  : カーブ前の減速区間
        turning  : カーブ中
        exit     : カーブ後の加速区間以降
        """
        s_brake_start = max(0.0, self.road.turn_start_s - self.params.brake_buffer_m)

        if s < s_brake_start:
            return "approach"
        elif s < self.road.turn_start_s:
            return "braking"
        elif s <= self.road.turn_end_s:
            return "turning"
        else:
            return "exit"

    def control(self, t, state_np, dt):
        x, y, psi, vx, beta, r = state_np
        proj = self.road.project(x, y, psi)

        s = proj["s"]
        ey = proj["ey"]
        epsi = proj["epsi"]

        s_la = min(self.road.total_length, s + self.params.lookahead_m)
        x_la, y_la, psi_la, kappa_la = self.road.centerline_pose(s_la)

        # feed-forward steering
        delta_ff = np.arctan(self.wheelbase_m * kappa_la)

        # feedback steering
        delta_raw = delta_ff - self.params.ky * ey - self.params.kpsi * epsi

        # steering angle / steering rate 制限
        delta_raw = float(np.clip(delta_raw, -self.constraints.delta_max, self.constraints.delta_max))
        d_lim = self.constraints.steer_rate_max * dt
        delta_cmd = float(np.clip(delta_raw, self.prev_delta - d_lim, self.prev_delta + d_lim))

        # target speed
        v_tgt = self.target_speed(s)
        v_tgt = min(v_tgt, self.road.speed_limit_mps + self.constraints.speed_margin_mps)

        # driving phase
        phase = self.driving_phase(s)

        # ==================================================
        # 人間らしい速度指令 v_cmd を作る
        #   approach : 直線速度を維持
        #   braking  : v_turn へ徐々に減速
        #   turning  : v_turn を維持。基本は加速しない
        #   exit     : v_straight へ徐々に加速
        # ==================================================
        if self.v_cmd is None:
            self.v_cmd = float(vx)


        # v_cmd 自体を加速度上限の範囲で徐々に目標速度へ近づける
        if v_tgt < self.v_cmd:
            dv_allowed = abs(self.constraints.ax_min) * dt
            self.v_cmd = max(v_tgt, self.v_cmd - dv_allowed)
        else:
            dv_allowed = self.constraints.ax_max * dt
            self.v_cmd = min(v_tgt, self.v_cmd + dv_allowed)

        # v_cmd に対する速度追従
        ax_raw = self.params.kv * (self.v_cmd - vx)
        ax_raw = float(np.clip(ax_raw, self.constraints.ax_min, self.constraints.ax_max))

        # ==================================================
        # フェーズ別の加速度符号制約
        # ==================================================
        if phase == "approach":
            if vx > v_tgt + self.params.speed_tolerance_mps:
                # 速度超過時は ax_min まで減速を許可
                ax_raw = float(np.clip(ax_raw, self.constraints.ax_min, 0.0))
            else:
                # 安全速度付近では緩やかな維持
                ax_raw = float(np.clip(ax_raw, -0.3, 0.3))

        elif phase == "braking":
            # カーブ前はブレーキのみ
            ax_raw = min(ax_raw, 0.0)

        elif phase == "turning":
            # 旋回中は加速しない。
            # 必要なら軽い減速は許可。
            ax_raw = min(ax_raw, 0.0)

        elif phase == "exit":
            # カーブ後はアクセルのみ。
            # すでに目標速度以上ならブレーキせず0にする。
            ax_raw = max(ax_raw, 0.0)

        # jerk 制限
        da_lim = self.constraints.jerk_max * dt
        ax_cmd = float(np.clip(ax_raw, self.prev_ax - da_lim, self.prev_ax + da_lim))

        # jerk制限後に符号が少し戻る可能性があるため、最後に再度フェーズ制約
        if phase == "approach":
            if vx > v_tgt + self.params.speed_tolerance_mps:
                ax_cmd = float(np.clip(ax_cmd, self.constraints.ax_min, 0.0))
            else:
                ax_cmd = float(np.clip(ax_cmd, -0.3, 0.3))
        elif phase == "braking":
            ax_cmd = min(ax_cmd, 0.0)
        elif phase == "turning":
            ax_cmd = min(ax_cmd, 0.0)
        elif phase == "exit":
            ax_cmd = max(ax_cmd, 0.0)

        self.prev_delta = delta_cmd
        self.prev_ax = ax_cmd

        return {
            "s": s,
            "ey": ey,
            "epsi": epsi,
            "phase": phase,
            "v_target": v_tgt,
            "v_cmd": self.v_cmd,
            "delta_cmd": delta_cmd,
            "ax_cmd": ax_cmd,
            "x_ref": proj["x_ref"],
            "y_ref": proj["y_ref"],
            "psi_ref": proj["psi_ref"],
            "kappa_ref": proj["kappa_ref"],
            "x_la": x_la,
            "y_la": y_la,
            "psi_la": psi_la,
            "kappa_la": kappa_la,
        }


# ============================================================
# Simulator 拡張：policy + road で走らせる
# ============================================================
def run_policy_simulation(
    simulator,
    road: StraightRightTurnStraightRoad,
    policy: RoadFollowingPolicy,
    actual_mu: float,
    control_mu: float = None,
    dt: float = 0.1,
    t_max: float = 60.0,
    initial_speed_mps: float = None,
    ):
    """
    既存の Simulator / 学習済みモデルを使って、
    road と policy に基づくシミュレーションを行う。
    """

    if control_mu is None:
        control_mu = actual_mu

    simulator.model.eval()

    if initial_speed_mps is None:
        initial_speed_mps = min(road.speed_limit_mps, policy.params.v_straight)

    state = torch.tensor(
        [[0.0, 0.0, 0.0, initial_speed_mps, 0.0, 0.0]],
        dtype=torch.float32,
        device=simulator.device,
    )

    # history_steps 分の CAN 履歴
    can_buffer = [
        make_can(
            vx=initial_speed_mps,
            ax=0.0,
            yaw_proxy=0.0,
            yaw_rel=1.0,
            device=simulator.device,
        )
        for _ in range(simulator.history_steps)
    ]

    actual_mu_tensor = torch.tensor(actual_mu, dtype=torch.float32, device=simulator.device)
    policy.reset()

    rows = []
    n_steps = int(np.round(t_max / dt))
    goal = False

    for i in range(n_steps):
        t = i * dt

        state_np = state[0].detach().cpu().numpy()
        x, y, psi, vx, beta, r = state_np

        ctrl = policy.control(t, state_np, dt)
        delta_cmd = ctrl["delta_cmd"]
        ax_cmd = ctrl["ax_cmd"]

        # =========================
        # 縦方向も低μで制限する
        # friction circle approximation:
        # ax^2 + ay^2 <= (mu g)^2
        # =========================
        vx_now = float(state_np[3])
        r_now = float(state_np[5])
        ay_now = vx_now * r_now

        # 制御側が想定しているμで縦方向制限
        #mu_g_control = control_mu * 9.81
        #ax_mu_limit = np.sqrt(max(mu_g_control ** 2 - ay_now ** 2, 0.0))

        #ax_cmd = float(np.clip(ax_cmd, -ax_mu_limit, ax_mu_limit))
        #ctrl["ax_cmd"] = ax_cmd

        # =========================
        # 縦方向の摩擦制限
        # 1) control_mu: 制御側が「出せる」と思っている加速度
        # 2) actual_mu : 実際の路面で本当に出せる加速度
        # =========================

        ax_cmd_requested = float(ctrl["ax_cmd"])

        vx_now = float(state_np[3])
        r_now = float(state_np[5])
        ay_now = vx_now * r_now

        # --- 制御側の想定μでの加速度制限 ---
        mu_g_control = control_mu * 9.81
        ax_limit_control = np.sqrt(max(mu_g_control ** 2 - ay_now ** 2, 0.0))

        ax_cmd_control_limited = float(
            np.clip(ax_cmd_requested, -ax_limit_control, ax_limit_control)
        )

        # --- 実際のμでの加速度制限 ---
        mu_g_actual = actual_mu * 9.81
        ax_limit_actual = np.sqrt(max(mu_g_actual ** 2 - ay_now ** 2, 0.0))

        ax_actual = float(
            np.clip(ax_cmd_control_limited, -ax_limit_actual, ax_limit_actual)
        )

        # 以後、車両モデルに入力する加速度は actual_mu で制限されたものを使う
        ctrl["ax_cmd_requested"] = ax_cmd_requested
        ctrl["ax_cmd_control_limited"] = ax_cmd_control_limited
        ctrl["ax_cmd"] = ax_actual
        ctrl["ax_actual_limit"] = ax_limit_actual
        ctrl["ax_control_limit"] = ax_limit_control
        ctrl["friction_usage_actual"] = np.sqrt(ax_actual ** 2 + ay_now ** 2) / (mu_g_actual + 1e-6)

        # 直前の推論結果 r を yaw_proxy として CAN に入れる
        #new_can = make_can(
        #    vx=vx,
        #   ax=ax_cmd,
        #    yaw_proxy=r,
        #    yaw_rel=1.0,
        #    device=simulator.device,
        #)
        ax_actual = ctrl["ax_cmd"]
        new_can = make_can(
            vx=vx,
            ax=ax_actual,
            yaw_proxy=r,
            yaw_rel=1.0,
            device=simulator.device,
        )

        can_buffer.pop(0)
        can_buffer.append(new_can)
        can_hist = torch.cat(can_buffer, dim=-1)

        dt_tensor = torch.tensor([[dt]], dtype=torch.float32, device=simulator.device)
        delta_tensor = torch.tensor([delta_cmd], dtype=torch.float32, device=simulator.device)

        with torch.no_grad():
            state, aux = simulator.model.step(
                state,
                can_hist,
                dt_tensor,
                delta_input=delta_tensor,
                mu_override=actual_mu_tensor,
            )

        next_np = state[0].detach().cpu().numpy()
        x2, y2, psi2, vx2, beta2, r2 = next_np

        proj2 = road.project(x2, y2, psi2)
        ay_est = vx2 * r2
        inside = abs(proj2["ey"]) <= (0.5 * road.lane_width_m)
        goal = road.goal_reached(x2, y2, psi2)

        rows.append({
            "t": t,
            "x": x2,
            "y": y2,
            "psi": psi2,
            "vx": vx2,
            "beta": beta2,
            "r": r2,
            "ay_est": ay_est,
            "s": proj2["s"],
            "ey": proj2["ey"],
            "epsi": proj2["epsi"],
            "v_target": ctrl["v_target"],
            "delta_cmd": delta_cmd,
            "ax_cmd_requested": ctrl["ax_cmd_requested"],
            "ax_cmd_control_limited": ctrl["ax_cmd_control_limited"],
            "ax_cmd": ctrl["ax_cmd"],  # actual_mu制限後
            "ax_control_limit": ctrl["ax_control_limit"],
            "ax_actual_limit": ctrl["ax_actual_limit"],
            "friction_usage_actual": ctrl["friction_usage_actual"],
            "control_mu": control_mu,
            "actual_mu": actual_mu,
            "inside": float(inside),
            "goal": float(goal),
            "speed_limit": road.speed_limit_mps,
            "x_ref": proj2["x_ref"],
            "y_ref": proj2["y_ref"],
            "delta_eff": float(aux["delta_eff"].detach().cpu().reshape(-1)[0]),
            "dvx_pred": float(aux["dvx_pred"].detach().cpu().reshape(-1)[0]),
            "beta_dot_res": float(aux["beta_dot_res"].detach().cpu().reshape(-1)[0]),
            "r_dot_res": float(aux["r_dot_res"].detach().cpu().reshape(-1)[0]),
            "v_cmd": ctrl.get("v_cmd", ctrl["v_target"]),
            "phase": ctrl.get("phase", ""),
        })

        if goal:
            break

    result = {k: np.array([row[k] for row in rows]) for k in rows[0].keys()}
    result["goal_reached"] = bool(goal)
    result["final_time"] = float(result["t"][-1]) if len(result["t"]) > 0 else np.inf
    return result


# ============================================================
# 評価関数
# ============================================================

@dataclass
class CostWeights:
    """
    evaluate_result() で使う評価関数の重み。

    基本思想:
      - goal_penalty_weight:
          ゴール未到達を非常に重く罰する
      - lane_violation_weight:
          車線逸脱は最重要の安全違反
      - centerline_error_weight:
          車線内でも中心線から大きく外れることを嫌う
      - beta_violation_weight:
          横滑り角の安全制約違反
      - speed_violation_weight:
          速度制限超過
      - friction_violation_weight:
          摩擦円制約違反
      - jerk_violation_weight:
          jerk制約違反
      - steer_rate_violation_weight:
          操舵速度制約違反
    """
    goal_penalty_weight: float = 1.0e6

    lane_violation_weight: float = 1.0e5
    centerline_error_weight: float = 1.0e4
    beta_violation_weight: float = 1.0e4
    speed_violation_weight: float = 1.0e3
    friction_violation_weight: float = 1.0e4
    jerk_violation_weight: float = 1.0e2
    steer_rate_violation_weight: float = 1.0e2


def evaluate_result(result, road, constraints: DrivingConstraints, cost_weights: CostWeights = CostWeights(),):
    ey = result["ey"]
    beta = result["beta"]
    vx = result["vx"]
    ax_cmd = result["ax_cmd"]
    delta_cmd = result["delta_cmd"]
    dt = np.median(np.diff(result["t"])) if len(result["t"]) >= 2 else 0.1

    # 中心線ずれのソフト許容幅
    ey_soft_limit = 0.3
    center_violation = np.maximum(np.abs(ey) - ey_soft_limit, 0.0)

    # 車線逸脱
    lane_limit = 0.5 * road.lane_width_m
    lane_violation = np.maximum(np.abs(ey) - lane_limit, 0.0)

    # 横滑り
    beta_violation = np.maximum(np.abs(beta) - constraints.beta_max, 0.0)

    # 速度超過
    speed_limit = road.speed_limit_mps + constraints.speed_margin_mps
    speed_violation = np.maximum(vx - speed_limit, 0.0)

    # jerk
    jerk = np.diff(ax_cmd, prepend=ax_cmd[0]) / max(1e-6, dt)
    jerk_violation = np.maximum(np.abs(jerk) - constraints.jerk_max, 0.0)

    # 操舵速度
    steer_rate = np.diff(delta_cmd, prepend=delta_cmd[0]) / max(1e-6, dt)
    steer_rate_violation = np.maximum(np.abs(steer_rate) - constraints.steer_rate_max, 0.0)

    # =========================
    # 摩擦円制約
    # ax^2 + ay^2 <= (mu * g)^2
    # =========================
    g = 9.81

    # result["mu"] は配列で入っている想定
    mu_arr = result["actual_mu"] if "actual_mu" in result else result["mu"]
    ay_est = result["ay_est"]

    friction_usage = np.sqrt(ax_cmd ** 2 + ay_est ** 2) / (mu_arr * g + 1e-6)
    friction_violation = np.maximum(friction_usage - 1.0, 0.0)

    # =========================
    # 無次元化したpenalty
    # =========================
    center_pen = float(np.sum((center_violation / max(ey_soft_limit, 1e-6)) ** 2))
    lane_pen = float(np.sum((lane_violation / max(lane_limit, 1e-6)) ** 2))
    beta_pen = float(np.sum((beta_violation / max(constraints.beta_max, 1e-6)) ** 2))
    speed_pen = float(np.sum((speed_violation / max(speed_limit, 1e-6)) ** 2))
    jerk_pen = float(np.sum((jerk_violation / max(constraints.jerk_max, 1e-6)) ** 2))
    steer_rate_pen = float(np.sum((steer_rate_violation / max(constraints.steer_rate_max, 1e-6)) ** 2))
    friction_pen = float(np.sum(friction_violation ** 2))

    goal_pen = 0.0 if result["goal_reached"] else 1.0
    time_cost = result["final_time"] if result["goal_reached"] else (result["final_time"] + 100.0)

    score = (
        time_cost
        + cost_weights.goal_penalty_weight * goal_pen
        + cost_weights.lane_violation_weight * lane_pen
        + cost_weights.centerline_error_weight * center_pen
        + cost_weights.beta_violation_weight * beta_pen
        + cost_weights.speed_violation_weight * speed_pen
        + cost_weights.friction_violation_weight * friction_pen
        + cost_weights.jerk_violation_weight * jerk_pen
        + cost_weights.steer_rate_violation_weight * steer_rate_pen
    )

    feasible = (
        result["goal_reached"]
        and np.max(lane_violation) <= 1e-6
        and np.max(beta_violation) <= 1e-6
        and np.max(speed_violation) <= 1e-6
        and np.max(friction_violation) <= 1e-6
        and np.max(jerk_violation) <= 1e-6
        and np.max(steer_rate_violation) <= 1e-6
    )

    metrics = {
        "score": score,
        "feasible": feasible,
        "goal_reached": result["goal_reached"],
        "final_time": float(result["final_time"]),
        "max_abs_ey": float(np.max(np.abs(ey))),
        "mean_abs_ey": float(np.mean(np.abs(ey))),
        "p95_abs_ey": float(np.percentile(np.abs(ey), 95)),
        "center_pen": center_pen,
        "max_abs_beta": float(np.max(np.abs(beta))),
        "max_vx": float(np.max(vx)),
        "max_abs_ax": float(np.max(np.abs(ax_cmd))),
        "max_abs_jerk": float(np.max(np.abs(jerk))),
        "max_abs_delta": float(np.max(np.abs(delta_cmd))),
        "max_abs_steer_rate": float(np.max(np.abs(steer_rate))),
        "max_friction_usage": float(np.max(friction_usage)),
        "friction_pen": friction_pen,
        "lane_pen": lane_pen,
        "beta_pen": beta_pen,
        "speed_pen": speed_pen,
        "jerk_pen": jerk_pen,
        "steer_rate_pen": steer_rate_pen,
    }

    return metrics


# ============================================================
# パラメータ探索
# ============================================================

def sample_policy_params(rng, road, constraints, assumed_mu):
    g = 9.81
    v_max = road.speed_limit_mps
    R = road.R

    # 1. 直線速度：速度制限の範囲内でやや高め
    v_straight = float(rng.uniform(0.8 * v_max, v_max))

    # 2. 旋回速度：摩擦円から上限を決める
    safety_factor = float(rng.uniform(0.65, 0.85))
    v_turn_phys = safety_factor * np.sqrt(max(1e-6, assumed_mu * g * R))
    v_turn_upper = min(v_max, v_turn_phys)

    # 低すぎる上限への保険
    v_turn_lower = max(1.0, 0.5 * v_turn_upper)
    v_turn = float(rng.uniform(v_turn_lower, v_turn_upper))

    # 3. 減速距離：運動方程式から概算
    a_brake = abs(constraints.ax_min)
    d_brake = max(
        1.0,
        (v_straight ** 2 - v_turn ** 2) / max(1e-6, 2.0 * a_brake),
    )

    # 安全余裕込みで探索
    brake_buffer_m = float(rng.uniform(1.0 * d_brake, 2.5 * d_brake))

    # 4. 加速距離：加速度上限から概算
    a_accel = constraints.ax_max
    d_accel = max(
        1.0,
        (v_straight ** 2 - v_turn ** 2) / max(1e-6, 2.0 * a_accel),
    )

    accel_buffer_m = float(rng.uniform(1.0 * d_accel, 2.5 * d_accel))

    # 5. lookahead：曲率半径に比例
    lookahead_min = max(1.5, 0.25 * R)
    lookahead_max = max(lookahead_min + 0.5, 0.9 * R)
    lookahead_m = float(rng.uniform(lookahead_min, lookahead_max))

    # 6. 横方向ゲイン：まずは探索で調整
    # ここは完全な物理式よりも、道路追従性能で選ぶパラメータ
    ky = float(rng.uniform(0.1, 1.2))
    kpsi = float(rng.uniform(0.6, 3.0))

    # 7. 速度追従ゲイン：時定数 0.5〜1.5秒程度
    # kv ≈ 1/tau
    tau = float(rng.uniform(0.5, 1.5))
    kv = 1.0 / tau

    return PolicyParams(
        v_straight=v_straight,
        v_turn=v_turn,
        brake_buffer_m=brake_buffer_m,
        accel_buffer_m=accel_buffer_m,
        lookahead_m=lookahead_m,
        ky=ky,
        kpsi=kpsi,
        kv=kv,
    )


def optimize_policy(
    simulator,
    road,
    assumed_mu,
    constraints: DrivingConstraints,
    search_cfg: SearchConfig,
):
    rng = np.random.default_rng(search_cfg.seed)
    outdir = Path(search_cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    history = []
    best = None
    best_result = None
    best_metrics = None

    wheelbase_m = simulator.model.priors.wheelbase_m

    for trial in range(search_cfg.n_trials):
        params = sample_policy_params(rng, road, constraints, assumed_mu)

        # 物理的に明らかにおかしいものを排除
        if params.v_turn > params.v_straight:
            continue

        policy = RoadFollowingPolicy(road, wheelbase_m, constraints, params)

        # 最適化時は「想定μ = 実際μ」として運転方策を評価する
        result = run_policy_simulation(
            simulator=simulator,
            road=road,
            policy=policy,
            actual_mu=assumed_mu,
            control_mu=assumed_mu,
            dt=search_cfg.dt,
            t_max=search_cfg.t_max,
            initial_speed_mps=params.v_straight,
        )

        metrics = evaluate_result(result, road, constraints)

        item = {
            "trial": trial,
            "assumed_mu": assumed_mu,
            "actual_mu": assumed_mu,
            "control_mu": assumed_mu,
            "score": metrics["score"],
            "feasible": int(metrics["feasible"]),
            "goal_reached": int(metrics["goal_reached"]),
            "final_time": metrics["final_time"],
            "max_abs_ey": metrics["max_abs_ey"],
            "max_abs_beta": metrics["max_abs_beta"],
            "max_vx": metrics["max_vx"],
            "max_abs_ax": metrics["max_abs_ax"],
            "max_abs_jerk": metrics["max_abs_jerk"],
            "max_abs_delta": metrics["max_abs_delta"],
            "max_abs_steer_rate": metrics["max_abs_steer_rate"],
            "max_friction_usage": metrics.get("max_friction_usage", np.nan),
            "friction_pen": metrics.get("friction_pen", np.nan),
            "v_straight": params.v_straight,
            "v_turn": params.v_turn,
            "brake_buffer_m": params.brake_buffer_m,
            "accel_buffer_m": params.accel_buffer_m,
            "lookahead_m": params.lookahead_m,
            "ky": params.ky,
            "kpsi": params.kpsi,
            "kv": params.kv,
        }
        history.append(item)

        if best is None:
            best = params
            best_result = result
            best_metrics = metrics
        else:
            # feasible最優先、その中でscore最小
            best_key = (0 if best_metrics["feasible"] else 1, best_metrics["score"])
            cur_key = (0 if metrics["feasible"] else 1, metrics["score"])
            if cur_key < best_key:
                best = params
                best_result = result
                best_metrics = metrics

    # search history をCSV保存
    hist_csv = outdir / "search_history.csv"
    if len(history) > 0:
        with open(hist_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    return best, best_result, best_metrics, history

# ============================================================
# 結果保存
# ============================================================
def save_result_csv(result, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    keys = [k for k in result.keys() if isinstance(result[k], np.ndarray)]
    n = len(result[keys[0]])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([result[k][i] for k in keys])


def save_trajectory_plot(result, road, out_png, title="trajectory"):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.plot(road.x_ref, road.y_ref, "k--", lw=1.5, label="centerline")
    plt.plot(road.left_x, road.left_y, color="gray", lw=1.0, label="lane boundary")
    plt.plot(road.right_x, road.right_y, color="gray", lw=1.0)

    plt.plot(result["x"], result["y"], lw=2.0, label="vehicle trajectory")
    plt.scatter(result["x"][0], result["y"][0], s=50, label="start")
    plt.scatter(result["x"][-1], result["y"][-1], s=50, label="end")

    plt.axis("equal")
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.legend()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def save_timeseries_plot(result, constraints: DrivingConstraints, out_png, title="timeseries"):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    t = result["t"]
    jerk = np.diff(result["ax_cmd"], prepend=result["ax_cmd"][0]) / max(1e-6, np.median(np.diff(t)) if len(t) >= 2 else 0.1)

    fig = plt.figure(figsize=(10, 12))

    # 1) speed
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(t, result["vx"], label="vx")
    ax1.plot(t, result["v_target"], "--", label="v_target")
    if "v_cmd" in result:
        ax1.plot(t, result["v_cmd"], "-.", label="v_cmd")
    ax1.plot(t, result["speed_limit"], ":", label="speed_limit")
    ax1.set_ylabel("speed [m/s]")
    ax1.grid()
    ax1.legend()

    # 2) beta / ay
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(t, result["beta"], label="beta")
    ax2.axhline(constraints.beta_max, linestyle="--")
    ax2.axhline(-constraints.beta_max, linestyle="--")
    ax2.plot(t, result["ay_est"], label="ay_est = vx*r")
    ax2.set_ylabel("beta / ay")
    ax2.grid()
    ax2.legend()

    # 3) ax / jerk
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(t, result["ax_cmd"], label="ax_cmd")
    ax3.axhline(constraints.ax_max, linestyle="--")
    ax3.axhline(constraints.ax_min, linestyle="--")
    ax3.plot(t, jerk, label="jerk")
    ax3.axhline(constraints.jerk_max, linestyle=":")
    ax3.axhline(-constraints.jerk_max, linestyle=":")
    ax3.set_ylabel("ax / jerk")
    ax3.grid()
    ax3.legend()

    # 4) steering / lateral error
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.plot(t, result["delta_cmd"], label="delta_cmd")
    ax4.axhline(constraints.delta_max, linestyle="--")
    ax4.axhline(-constraints.delta_max, linestyle="--")
    ax4.plot(t, result["ey"], label="lateral error ey")
    ax4.set_ylabel("delta / ey")
    ax4.set_xlabel("time [s]")
    ax4.grid()
    ax4.legend()

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def save_summary_text(best_params, best_metrics, road, mu, constraints, out_txt):
    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=== Planning Summary ===")
    lines.append(f"mu = {mu:.3f}")
    lines.append(f"road: straight1={road.L1} m, right turn radius={road.R} m, straight2={road.L2} m")
    lines.append(f"speed_limit = {road.speed_limit_mps:.3f} m/s ({road.speed_limit_mps*3.6:.1f} km/h)")
    lines.append("")
    lines.append("=== Constraints ===")
    lines.append(f"beta_max = {constraints.beta_max:.4f} rad")
    lines.append(f"ax_min = {constraints.ax_min:.3f} m/s^2")
    lines.append(f"ax_max = {constraints.ax_max:.3f} m/s^2")
    lines.append(f"jerk_max = {constraints.jerk_max:.3f} m/s^3")
    lines.append(f"delta_max = {constraints.delta_max:.4f} rad")
    lines.append(f"steer_rate_max = {constraints.steer_rate_max:.4f} rad/s")
    lines.append("")
    lines.append("=== Best Params ===")
    for k, v in best_params.__dict__.items():
        lines.append(f"{k} = {v}")
    lines.append("")
    lines.append("=== Best Metrics ===")
    for k, v in best_metrics.items():
        lines.append(f"{k} = {v}")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_egomotion_csv_for_alpamayo(result, out_csv, z_m=0.0):
    """
    Alpamayo等に渡すための簡易egomotion CSVを保存する。

    出力:
      t_s
      timestamp_us
      x_m
      y_m
      z_m
      yaw_rad
      qx, qy, qz, qw  # roll=pitch=0, yawのみのクォータニオン
      vx_mps
      beta_rad
      yaw_rate_radps

    前提:
      - result["t"] は秒
      - result["x"], result["y"] は絶対座標[m]
      - result["psi"] はyaw角[rad]
      - dt=0.1なら10Hz相当
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    t = result["t"]
    x = result["x"]
    y = result["y"]
    psi = result["psi"]
    vx = result["vx"]
    beta = result["beta"]
    r = result["r"]

    # yawのみの quaternion
    # roll=pitch=0 のとき:
    # qx=0, qy=0, qz=sin(yaw/2), qw=cos(yaw/2)
    qx = np.zeros_like(psi)
    qy = np.zeros_like(psi)
    qz = np.sin(0.5 * psi)
    qw = np.cos(0.5 * psi)

    timestamp_us = np.round(t * 1_000_000).astype(np.int64)

    rows = []
    for i in range(len(t)):
        rows.append({
            "t_s": float(t[i]),
            "timestamp_us": int(timestamp_us[i]),
            "x_m": float(x[i]),
            "y_m": float(y[i]),
            "z_m": float(z_m),
            "yaw_rad": float(psi[i]),
            "qx": float(qx[i]),
            "qy": float(qy[i]),
            "qz": float(qz[i]),
            "qw": float(qw[i]),
            "vx_mps": float(vx[i]),
            "beta_rad": float(beta[i]),
            "yaw_rate_radps": float(r[i]),
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_egomotion_csv_carla_world(result, road, out_csv):
    """
    Save egomotion CSV in CARLA world absolute coordinates.

    Converts simulation result (local frame: +x forward, +y left) to CARLA
    world coordinates via road.local_to_carla().

    Output columns: t_s, timestamp_us, x_m, y_m, z_m, yaw_rad,
                    qx, qy, qz, qw, vx_mps, beta_rad, yaw_rate_radps
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    t = result["t"]
    x_carla, y_carla, yaw_carla = road.local_to_carla(
        result["x"], result["y"], result["psi"]
    )

    # Interpolate z from route at projected arc length
    s_arr = result["s"] if "s" in result else np.zeros(len(t))
    z_arr = road.interp_z(s_arr)

    qx = np.zeros_like(yaw_carla)
    qy = np.zeros_like(yaw_carla)
    qz = np.sin(0.5 * yaw_carla)
    qw = np.cos(0.5 * yaw_carla)

    timestamp_us = np.round(t * 1_000_000).astype(np.int64)

    rows = []
    for i in range(len(t)):
        rows.append({
            "t_s": float(t[i]),
            "timestamp_us": int(timestamp_us[i]),
            "x_m": float(x_carla[i]),
            "y_m": float(y_carla[i]),
            "z_m": float(z_arr[i]),
            "yaw_rad": float(yaw_carla[i]),
            "qx": float(qx[i]),
            "qy": float(qy[i]),
            "qz": float(qz[i]),
            "qw": float(qw[i]),
            "vx_mps": float(result["vx"][i]),
            "beta_rad": float(result["beta"][i]),
            "yaw_rate_radps": float(result["r"][i]),
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =========================
# 5. 可視化
# =========================
def plot_result(result, title="", outdir="simulation_outputs", filename_prefix="result"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 軌跡
    plt.figure(figsize=(6, 6))
    plt.plot(result["x"], result["y"])
    plt.axis("equal")
    plt.title(title)
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.savefig(outdir / f"{filename_prefix}_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 状態量
    plt.figure(figsize=(8, 4))
    plt.plot(result["beta"], label="beta")
    plt.plot(result["r"], label="yaw rate")
    plt.legend()
    plt.grid()
    plt.title(f"{title} - beta / yaw rate")
    plt.savefig(outdir / f"{filename_prefix}_states.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_compare(results_dict, title="Trajectory Comparison", outdir="simulation_outputs", filename="trajectory_comparison.png"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))

    for name, res in results_dict.items():
        plt.plot(res["x"], res["y"], label=name)

    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    plt.savefig(outdir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_compare_timeseries(results_dict, key="vx", title=None):
    plt.figure(figsize=(8, 4))
    for name, res in results_dict.items():
        plt.plot(res["t"], res[key], label=name)
    plt.grid()
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel(key)
    plt.title(title or key)
    plt.show()


def make_vehicle_polygon(x, y, psi, length=4.5, width=1.8):
    """
    車体を表す長方形ポリゴンを返す。
    x, y : 車体中心
    psi  : yaw [rad]
    """
    hl = 0.5 * length
    hw = 0.5 * width

    # 車体ローカル座標（前方が +x, 左が +y）
    pts = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw],
    ])

    c = np.cos(psi)
    s = np.sin(psi)
    R = np.array([[c, -s], [s, c]])

    pts_world = pts @ R.T
    pts_world[:, 0] += x
    pts_world[:, 1] += y
    return pts_world

def save_topdown_movie(
    result,
    road,
    out_path,
    fps=10,
    vehicle_length=4.5,
    vehicle_width=1.8,
    tail_length=100,
):
    """
    上空視点(top-down)の動画を保存する。
    result["x"], result["y"], result["psi"] を使用。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = result["x"]
    y = result["y"]
    psi = result["psi"]
    t = result["t"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # 道路
    ax.plot(road.x_ref, road.y_ref, "k--", lw=1.5, label="centerline")
    ax.plot(road.left_x, road.left_y, color="gray", lw=1.0, label="lane boundary")
    ax.plot(road.right_x, road.right_y, color="gray", lw=1.0)

    traj_line, = ax.plot([], [], lw=2.0, label="trajectory")
    veh_patch = Polygon(
        make_vehicle_polygon(x[0], y[0], psi[0], vehicle_length, vehicle_width),
        closed=True,
        fill=False,
    )
    ax.add_patch(veh_patch)

    ax.set_aspect("equal")
    ax.grid()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top-down vehicle motion")
    ax.legend()

    # 表示範囲
    margin = 10.0
    ax.set_xlim(min(np.min(road.left_x), np.min(road.right_x)) - margin,
                max(np.max(road.left_x), np.max(road.right_x)) + margin)
    ax.set_ylim(min(np.min(road.left_y), np.min(road.right_y)) - margin,
                max(np.max(road.left_y), np.max(road.right_y)) + margin)

    def update(i):
        i0 = max(0, i - tail_length)
        traj_line.set_data(x[i0:i+1], y[i0:i+1])
        veh_patch.set_xy(make_vehicle_polygon(x[i], y[i], psi[i], vehicle_length, vehicle_width))
        ax.set_title(f"Top-down vehicle motion  t={t[i]:.1f}s")
        return traj_line, veh_patch

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=1000 / fps,
        blit=False,
    )

    if out_path.suffix.lower() == ".gif":
        ani.save(out_path, writer="pillow", fps=fps)
    else:
        ani.save(out_path, writer="ffmpeg", fps=fps)

    plt.close(fig)


def world_to_ego(xw, yw, x0, y0, psi0):
    """
    世界座標 -> 車体基準座標
    車体前方を +x, 左を +y とする。
    """
    dx = xw - x0
    dy = yw - y0
    c = np.cos(psi0)
    s = np.sin(psi0)

    xe = c * dx + s * dy
    ye = -s * dx + c * dy
    return xe, ye

def ego_to_view(xe, ye, view_yaw_deg=20.0):
    """
    ego座標:
      xe: 車体前方 +x
      ye: 車体左 +y

    view座標:
      画面上方向を車体前方に近づける。
      view_yaw_deg > 0 で少し斜め後方から見るように回転。
    """
    theta = np.deg2rad(view_yaw_deg)

    # まず forward(xe) を画面上方向(y_view)へ、
    # left(ye) を画面左方向(-x_view)へ対応させる。
    x0 = -ye
    y0 = xe

    # さらに固定ビュー角だけ回転
    c = np.cos(theta)
    s = np.sin(theta)

    xv = c * x0 - s * y0
    yv = s * x0 + c * y0
    return xv, yv

def save_egocentric_movie(
    result,
    road,
    out_path,
    fps=10,
    vehicle_length=4.5,
    vehicle_width=1.8,
    forward_range=40.0,
    backward_range=10.0,
    lateral_range=12.0,
    tail_length=100,
    view_yaw_deg=20.0,
):
    """
    車体基準の簡易動画を保存する。

    ego座標では車体前方が +x だが、描画時には
    車体前方が画面上方向に見えるように変換する。
    view_yaw_deg で、少し斜め後方から見るような固定ビュー角を付ける。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = result["x"]
    y = result["y"]
    psi = result["psi"]
    t = result["t"]

    def ego_to_view(xe, ye):
        theta = np.deg2rad(view_yaw_deg)

        # ego: forward=+x, left=+y
        # view: forwardを画面上(+y)へ、leftを画面左(-x)へ
        x0 = -ye
        y0 = xe

        # 固定視点角を付与
        c = np.cos(theta)
        s = np.sin(theta)
        xv = c * x0 - s * y0
        yv = s * x0 + c * y0
        return xv, yv

    def make_vehicle_polygon_view(length=4.5, width=1.8):
        """
        車体をview座標で固定表示する。
        車体前方が画面上方向、少し斜め角付き。
        """
        hl = 0.5 * length
        hw = 0.5 * width

        # ego座標での車体形状
        xe = np.array([ hl,  hl, -hl, -hl])
        ye = np.array([ hw, -hw, -hw,  hw])

        xv, yv = ego_to_view(xe, ye)
        return np.stack([xv, yv], axis=1)

    fig, ax = plt.subplots(figsize=(8, 8))

    centerline_plot, = ax.plot([], [], "k--", lw=1.5, label="centerline")
    left_plot, = ax.plot([], [], color="gray", lw=1.0, label="lane boundary")
    right_plot, = ax.plot([], [], color="gray", lw=1.0)
    traj_plot, = ax.plot([], [], lw=2.0, label="past trajectory")

    # 車体はview座標で固定
    car_fixed = Polygon(
        make_vehicle_polygon_view(vehicle_length, vehicle_width),
        closed=True,
        fill=False,
        edgecolor="tab:red",
        linewidth=2.0,
    )
    ax.add_patch(car_fixed)

    # 前方方向を示す矢印
    ax.arrow(
        0.0,
        0.0,
        0.0,
        vehicle_length * 0.8,
        width=0.08,
        head_width=0.6,
        head_length=0.8,
        length_includes_head=True,
        color="tab:red",
        alpha=0.7,
    )

    # view座標では y が前方、x が左右
    ax.set_xlim(-lateral_range, lateral_range)
    ax.set_ylim(-backward_range, forward_range)
    ax.set_aspect("equal")
    ax.grid()
    ax.set_xlabel("left / right [m]")
    ax.set_ylabel("forward [m]")
    ax.set_title("Ego-centric vehicle motion")
    ax.legend()

    def update(i):
        x0, y0, psi0 = x[i], y[i], psi[i]

        # 道路をego座標へ変換
        cx_e, cy_e = world_to_ego(road.x_ref, road.y_ref, x0, y0, psi0)
        lx_e, ly_e = world_to_ego(road.left_x, road.left_y, x0, y0, psi0)
        rx_e, ry_e = world_to_ego(road.right_x, road.right_y, x0, y0, psi0)

        # ego座標 -> view座標
        cx_v, cy_v = ego_to_view(cx_e, cy_e)
        lx_v, ly_v = ego_to_view(lx_e, ly_e)
        rx_v, ry_v = ego_to_view(rx_e, ry_e)

        centerline_plot.set_data(cx_v, cy_v)
        left_plot.set_data(lx_v, ly_v)
        right_plot.set_data(rx_v, ry_v)

        # 過去軌跡
        i0 = max(0, i - tail_length)
        tx_e, ty_e = world_to_ego(x[i0:i+1], y[i0:i+1], x0, y0, psi0)
        tx_v, ty_v = ego_to_view(tx_e, ty_e)
        traj_plot.set_data(tx_v, ty_v)

        ax.set_title(
            f"Ego-centric vehicle motion  t={t[i]:.1f}s  view_yaw={view_yaw_deg:.0f}deg"
        )

        return centerline_plot, left_plot, right_plot, traj_plot, car_fixed

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=1000 / fps,
        blit=False,
    )

    if out_path.suffix.lower() == ".gif":
        ani.save(out_path, writer="pillow", fps=fps)
    else:
        ani.save(out_path, writer="ffmpeg", fps=fps)

    plt.close(fig)


def save_third_person_2d_movie(
    result,
    road,
    out_path,
    fps=10,
    vehicle_length=4.5,
    vehicle_width=1.8,
    forward_range=50.0,
    backward_range=8.0,
    lateral_range=14.0,
    tail_length=100,
    car_y_offset=-4.0,
):
    """
    2Dの疑似三人称ビュー。
    車体を画面下寄りに固定し、前方道路を画面上方向に表示する。
    地面から20度程度のゲーム風ビューの簡易版。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = result["x"]
    y = result["y"]
    psi = result["psi"]
    t = result["t"]

    def ego_to_screen(xe, ye):
        # ego座標:
        #   xe: 車体前方
        #   ye: 車体左
        # screen座標:
        #   screen x: 左右
        #   screen y: 前方
        xs = ye
        ys = xe + car_y_offset
        return xs, ys

    def make_vehicle_polygon_screen(length=4.5, width=1.8):
        hl = 0.5 * length
        hw = 0.5 * width

        xe = np.array([ hl,  hl, -hl, -hl])
        ye = np.array([ hw, -hw, -hw,  hw])

        xs, ys = ego_to_screen(xe, ye)
        return np.stack([xs, ys], axis=1)

    fig, ax = plt.subplots(figsize=(8, 8))

    centerline_plot, = ax.plot([], [], "k--", lw=1.5, label="centerline")
    left_plot, = ax.plot([], [], color="gray", lw=1.0, label="lane boundary")
    right_plot, = ax.plot([], [], color="gray", lw=1.0)
    traj_plot, = ax.plot([], [], lw=2.0, label="past trajectory")

    car_fixed = Polygon(
        make_vehicle_polygon_screen(vehicle_length, vehicle_width),
        closed=True,
        fill=False,
        edgecolor="tab:red",
        linewidth=2.0,
        label="vehicle",
    )
    ax.add_patch(car_fixed)

    # 車体前方を示す矢印
    ax.arrow(
        0.0,
        car_y_offset,
        0.0,
        vehicle_length * 0.9,
        width=0.08,
        head_width=0.6,
        head_length=0.8,
        length_includes_head=True,
        color="tab:red",
        alpha=0.7,
    )

    ax.set_xlim(-lateral_range, lateral_range)
    ax.set_ylim(-backward_range, forward_range)
    ax.set_aspect("equal")
    ax.grid()
    ax.set_xlabel("left / right [m]")
    ax.set_ylabel("forward [m]")
    ax.set_title("Third-person 2D chase view")
    ax.legend()

    def update(i):
        x0, y0, psi0 = x[i], y[i], psi[i]

        cx_e, cy_e = world_to_ego(road.x_ref, road.y_ref, x0, y0, psi0)
        lx_e, ly_e = world_to_ego(road.left_x, road.left_y, x0, y0, psi0)
        rx_e, ry_e = world_to_ego(road.right_x, road.right_y, x0, y0, psi0)

        cx_s, cy_s = ego_to_screen(cx_e, cy_e)
        lx_s, ly_s = ego_to_screen(lx_e, ly_e)
        rx_s, ry_s = ego_to_screen(rx_e, ry_e)

        centerline_plot.set_data(cx_s, cy_s)
        left_plot.set_data(lx_s, ly_s)
        right_plot.set_data(rx_s, ry_s)

        i0 = max(0, i - tail_length)
        tx_e, ty_e = world_to_ego(x[i0:i+1], y[i0:i+1], x0, y0, psi0)
        tx_s, ty_s = ego_to_screen(tx_e, ty_e)
        traj_plot.set_data(tx_s, ty_s)

        ax.set_title(f"Third-person 2D chase view  t={t[i]:.1f}s")
        return centerline_plot, left_plot, right_plot, traj_plot, car_fixed

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=1000 / fps,
        blit=False,
    )

    if out_path.suffix.lower() == ".gif":
        ani.save(out_path, writer="pillow", fps=fps)
    else:
        ani.save(out_path, writer="ffmpeg", fps=fps)

    plt.close(fig)

def make_vehicle_box_chase_frame(length=4.5, width=1.8, height=1.5, y_offset=0.0):
    """
    chase frame 用の簡易車体ボックス。
    座標系:
      x: 右方向
      y: 前方
      z: 上方向
    車体は原点付近に固定し、前方は +y 方向。
    """
    hl = 0.5 * length
    hw = 0.5 * width
    h = height

    verts = np.array([
        [ hw,  hl + y_offset, 0.0],   # front-right-bottom
        [-hw,  hl + y_offset, 0.0],   # front-left-bottom
        [-hw, -hl + y_offset, 0.0],   # rear-left-bottom
        [ hw, -hl + y_offset, 0.0],   # rear-right-bottom
        [ hw,  hl + y_offset, h],     # front-right-top
        [-hw,  hl + y_offset, h],     # front-left-top
        [-hw, -hl + y_offset, h],     # rear-left-top
        [ hw, -hl + y_offset, h],     # rear-right-top
    ])

    faces = [
        [verts[i] for i in [0, 1, 2, 3]],  # bottom
        [verts[i] for i in [4, 5, 6, 7]],  # top
        [verts[i] for i in [0, 1, 5, 4]],  # front
        [verts[i] for i in [1, 2, 6, 5]],  # left
        [verts[i] for i in [2, 3, 7, 6]],  # rear
        [verts[i] for i in [3, 0, 4, 7]],  # right
    ]
    return faces

def make_vehicle_box_3d(x, y, psi, length=4.5, width=1.8, height=1.5):
    """
    簡易車体3Dボックス。
    底面中心が z=0、車体中心 x,y。
    """
    hl = 0.5 * length
    hw = 0.5 * width
    h = height

    # local vertices: forward x, left y, up z
    verts = np.array([
        [ hl,  hw, 0.0],
        [ hl, -hw, 0.0],
        [-hl, -hw, 0.0],
        [-hl,  hw, 0.0],
        [ hl,  hw, h],
        [ hl, -hw, h],
        [-hl, -hw, h],
        [-hl,  hw, h],
    ])

    c = np.cos(psi)
    s = np.sin(psi)
    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    verts_w = verts @ R.T
    verts_w[:, 0] += x
    verts_w[:, 1] += y

    faces = [
        [verts_w[i] for i in [0, 1, 2, 3]],  # bottom
        [verts_w[i] for i in [4, 5, 6, 7]],  # top
        [verts_w[i] for i in [0, 1, 5, 4]],
        [verts_w[i] for i in [1, 2, 6, 5]],
        [verts_w[i] for i in [2, 3, 7, 6]],
        [verts_w[i] for i in [3, 0, 4, 7]],
    ]
    return faces

def save_chase_camera_3d_movie(
    result,
    road,
    out_path,
    fps=10,
    vehicle_length=4.5,
    vehicle_width=1.8,
    vehicle_height=1.5,
    camera_elevation_deg=20.0,
    lateral_range=12.0,
    forward_range=40.0,
    backward_range=8.0,
    z_max=10.0,
    tail_length=100,
    y_vehicle_offset=0.0,
    start_t=5.0,   # ★追加
):
    """
    車体追従の3D chase camera movie。
    毎フレーム world -> ego 変換し、
    車体は原点固定、前方を常に +y に向ける。
    
    描画座標系:
      x_plot: 右方向
      y_plot: 前方
      z_plot: 上方向

    カメラ:
      車体後方から前方を見る固定視点
      (地面から camera_elevation_deg 度の高さ)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_all = result["t"]
    start_idx = int(np.searchsorted(t_all, start_t, side="left"))

    x = result["x"][start_idx:]
    y = result["y"][start_idx:]
    psi = result["psi"][start_idx:]
    t = result["t"][start_idx:]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("persp")

    def ego_to_chase(xe, ye):
        """
        ego座標:
          xe: 前方
          ye: 左
        chase描画座標:
          x_plot: 右
          y_plot: 前方
        """
        x_plot = -ye
        y_plot = xe
        return x_plot, y_plot

    def update(i):
        ax.cla()

        x0 = x[i]
        y0 = y[i]
        psi0 = psi[i]

        # -----------------------------
        # 道路中心線・境界線を ego frame へ
        # -----------------------------
        cx_e, cy_e = world_to_ego(road.x_ref, road.y_ref, x0, y0, psi0)
        lx_e, ly_e = world_to_ego(road.left_x, road.left_y, x0, y0, psi0)
        rx_e, ry_e = world_to_ego(road.right_x, road.right_y, x0, y0, psi0)

        cx_p, cy_p = ego_to_chase(cx_e, cy_e)
        lx_p, ly_p = ego_to_chase(lx_e, ly_e)
        rx_p, ry_p = ego_to_chase(rx_e, ry_e)

        ax.plot(cx_p, cy_p, np.zeros_like(cx_p), "k--", lw=1.2, label="centerline")
        ax.plot(lx_p, ly_p, np.zeros_like(lx_p), color="gray", lw=1.0, label="lane boundary")
        ax.plot(rx_p, ry_p, np.zeros_like(rx_p), color="gray", lw=1.0)

        # -----------------------------
        # 過去軌跡
        # -----------------------------
        i0 = max(0, i - tail_length)
        tx_e, ty_e = world_to_ego(x[i0:i+1], y[i0:i+1], x0, y0, psi0)
        tx_p, ty_p = ego_to_chase(tx_e, ty_e)
        ax.plot(tx_p, ty_p, np.zeros_like(tx_p), lw=2.0, label="trajectory")

        # -----------------------------
        # 車体（原点固定）
        # -----------------------------
        faces = make_vehicle_box_chase_frame(
            length=vehicle_length,
            width=vehicle_width,
            height=vehicle_height,
            y_offset=y_vehicle_offset,
        )
        car = Poly3DCollection(
            faces,
            alpha=0.45,
            edgecolor="black",
            facecolor="tab:red",
        )
        ax.add_collection3d(car)

        # 車体前方方向を示す矢印
        ax.quiver(
            0.0, y_vehicle_offset, vehicle_height * 0.6,
            0.0, vehicle_length * 0.8, 0.0,
            arrow_length_ratio=0.15,
            linewidth=2.0,
        )

        # -----------------------------
        # 軸範囲
        # -----------------------------
        ax.set_xlim(-lateral_range, lateral_range)
        ax.set_ylim(-backward_range, forward_range)
        ax.set_zlim(0.0, z_max)

        # 軸・目盛り・ラベル・グリッドを消す
        ax.set_axis_off()
        ax.grid(False)

        # 3Dの背景面も透明化
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)

        # 軸線も透明化
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # -----------------------------
        # カメラ設定
        # -----------------------------
        # azim=-90 で「後方から前方を見る」向きになりやすい
        ax.view_init(elev=camera_elevation_deg, azim=-90)

        ax.set_title(f"3D chase camera  t={t[i]:.1f}s")
        ax.legend(loc="upper left")

        return []

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=1000 / fps,
        blit=False,
    )

    if out_path.suffix.lower() == ".gif":
        ani.save(out_path, writer="pillow", fps=fps)
    else:
        ani.save(out_path, writer="ffmpeg", fps=fps)

    plt.close(fig)


# =========================
# 6. 実行例
# =========================
if __name__ == "__main__":
    from physicsnemo_can_vehicle_training import (
        PhysicsInformedVehicleModel,
        VehicleSpecPriors,
        TrainConfig,
    )

    device = "cpu"

    priors = VehicleSpecPriors()
    cfg = TrainConfig()

    model = load_model(
        PhysicsInformedVehicleModel,
        priors,
        cfg,
        model_path="out_physics_vehicle/vehicle_model.pt",
        device=device,
    )

    sim = Simulator(
        model,
        device=device,
        history_steps=cfg.history_steps,
        ax_min=-5.0,
        ax_max=3.0,
        speed_kp=1.5,
        speed_ff=1.0,
    )

    # ------------------------------------
    # 1) 道路設定
    # 100m直進 → 右90度カーブ → 100m直進
    # ------------------------------------
    road = StraightRightTurnStraightRoad(
        straight1_m=100.0,
        turn_radius_m=8.0,
        straight2_m=100.0,
        lane_width_m=3.5,
        speed_limit_kph=40.0,   # 40 km/h
        sample_ds=0.5,
    )


    # ------------------------------------
    # 3) 安全・快適制約
    # ------------------------------------

    constraints = DrivingConstraints(
        beta_max=0.10,                  # 少し許容を広げる
        ax_min=-5.0,                    # 強めのブレーキを許可
        ax_max=3.0,                     # 右折後の加速を許可
        jerk_max=6.0,                   # 加減速変化を少し許容
        delta_max=np.deg2rad(30.0),     # 急カーブ用に少し広げる
        steer_rate_max=np.deg2rad(45.0),
        speed_margin_mps=0.0,
    )

    # ------------------------------------
    # 想定μと実μ
    # ------------------------------------
    actual_mu = 0.20      # 実際の路面は低μ
    aware_mu = actual_mu  # 低μを正しく想定
    unaware_mu = 0.90     # 高μだと誤認

    outdir = Path("low_mu_awareness_comparison_outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------
    # 1) 低μを想定して最適化
    # ------------------------------------
    best_params_aware, _, best_metrics_aware_plan, _ = optimize_policy(
        simulator=sim,
        road=road,
        assumed_mu=aware_mu,
        constraints=constraints,
        search_cfg=SearchConfig(
            n_trials=800,
            dt=0.1,
            t_max=60.0,
            seed=42,
            outdir=str(outdir / "optimization_aware_low_mu"),
        ),
    )

    # ------------------------------------
    # 2) 高μを想定して最適化
    # ------------------------------------
    best_params_unaware, _, best_metrics_unaware_plan, _ = optimize_policy(
        simulator=sim,
        road=road,
        assumed_mu=unaware_mu,
        constraints=constraints,
        search_cfg=SearchConfig(
            n_trials=800,
            dt=0.1,
            t_max=60.0,
            seed=43,
            outdir=str(outdir / "optimization_unaware_high_mu"),
        ),
    )

    print("=== low-mu aware plan ===")
    print(best_params_aware)
    print(best_metrics_aware_plan)

    print("=== high-mu assumed plan ===")
    print(best_params_unaware)
    print(best_metrics_unaware_plan)

    # ------------------------------------
    # policy作成
    # ------------------------------------
    policy_aware = RoadFollowingPolicy(
        road=road,
        wheelbase_m=sim.model.priors.wheelbase_m,
        constraints=constraints,
        params=best_params_aware,
    )

    policy_unaware = RoadFollowingPolicy(
        road=road,
        wheelbase_m=sim.model.priors.wheelbase_m,
        constraints=constraints,
        params=best_params_unaware,
    )

    # ------------------------------------
    # A) 低μを想定して、実際も低μ
    # ------------------------------------
    result_aware = run_policy_simulation(
        simulator=sim,
        road=road,
        policy=policy_aware,
        actual_mu=actual_mu,
        control_mu=aware_mu,
        dt=0.1,
        t_max=60.0,
        initial_speed_mps=best_params_aware.v_straight,
    )

    metrics_aware = evaluate_result(result_aware, road, constraints)

    # ------------------------------------
    # B) 高μを想定して、実際は低μ
    # ------------------------------------
    result_unaware = run_policy_simulation(
        simulator=sim,
        road=road,
        policy=policy_unaware,
        actual_mu=actual_mu,
        control_mu=unaware_mu,
        dt=0.1,
        t_max=60.0,
        initial_speed_mps=best_params_unaware.v_straight,
    )

    metrics_unaware = evaluate_result(result_unaware, road, constraints)

    print("=== aware: planned for low mu, actual low mu ===")
    print(metrics_aware)

    print("=== unaware: planned for high mu, actual low mu ===")
    print(metrics_unaware)

    save_result_csv(result_aware, outdir / "aware_low_mu_result.csv")
    save_result_csv(result_unaware, outdir / "unaware_high_mu_on_low_mu_result.csv")

    save_trajectory_plot(
        result_aware,
        road,
        outdir / "trajectory_aware_low_mu.png",
        title=f"Aware: planned low mu={aware_mu:.2f}, actual mu={actual_mu:.2f}",
    )

    save_trajectory_plot(
        result_unaware,
        road,
        outdir / "trajectory_unaware_high_mu_on_low_mu.png",
        title=f"Unaware: planned mu={unaware_mu:.2f}, actual mu={actual_mu:.2f}",
    )

    save_timeseries_plot(
        result_aware,
        constraints,
        outdir / "timeseries_aware_low_mu.png",
        title=f"Aware low-mu driving (actual mu={actual_mu:.2f})",
    )

    save_timeseries_plot(
        result_unaware,
        constraints,
        outdir / "timeseries_unaware_high_mu_on_low_mu.png",
        title=f"Unaware high-mu driving on low-mu road (actual mu={actual_mu:.2f})",
    )

    save_egomotion_csv_for_alpamayo(
        result_aware,
        outdir / "egomotion_aware_low_mu_10hz.csv",
    )

    save_egomotion_csv_for_alpamayo(
        result_unaware,
        outdir / "egomotion_unaware_high_mu_on_low_mu_10hz.csv",
    )

    save_topdown_movie(
        result_aware,
        road,
        outdir / "aware_low_mu_topdown.mp4",
        fps=10,
    )

    save_topdown_movie(
        result_unaware,
        road,
        outdir / "unaware_high_mu_on_low_mu_topdown.mp4",
        fps=10,
    )

    save_egocentric_movie(
        result_aware,
        road,
        outdir / "aware_low_mu_egocentric.mp4",
        fps=10,
        view_yaw_deg=0.0,
    )

    save_egocentric_movie(
        result_unaware,
        road,
        outdir / "unaware_high_mu_on_low_mu_egocentric.mp4",
        fps=10,
        view_yaw_deg=0.0,
    )


    save_chase_camera_3d_movie(
        result_aware,
        road,
        outdir / "aware_low_mu_chase_3d.mp4",
        fps=10,
        camera_elevation_deg=20.0,
    )

    save_chase_camera_3d_movie(
        result_unaware,
        road,
        outdir / "unaware_high_mu_on_low_mu_chase_3d.mp4",
        fps=10,
        camera_elevation_deg=20.0,
    )

