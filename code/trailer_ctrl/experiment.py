"""Experiment logic without ROS: poses in, velocity command out, logging.

The ROS 2 node (ros_node.py) only moves messages in and out of this class, so
everything that matters (goal frame, angle unwrapping, interval grid, watchdog,
safety stops, log files) can be tested offline (tests/test_experiment.py).

Bodies and what is measured
---------------------------
    car : the TurtleBot (rigid body 1)           -> x1, x2, phi
    t1  : trailer 1 (rigid body 2)               -> theta1  (heading only)
    t2  : trailer 2 (rigid body 3, 2-trailer run) -> theta2  (heading only)
All headings are ABSOLUTE yaw angles in the world frame, as in the model
x = [x1, x2, phi, theta1, theta2]. (x1, x2) is the midpoint of the car axle.

Goal frame
----------
The goal is the pose (x_goal, y_goal, psi_goal) of the car with all trailers
aligned behind it. The controller works in the goal frame: positions are
shifted by (x_goal, y_goal) and rotated by -psi_goal, all headings get
-psi_goal. (The kinematics are invariant under this rigid motion.)
"""
from __future__ import annotations

import csv
import json
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .controller import Params, SampledController, TrailerController


def wrap(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


@dataclass
class Step:
    status: str            # 'wait' | 'run' | 'stale' | 'goal' | 'abort'
    u1: float = 0.0
    u2: float = 0.0
    reason: str = ""


class Experiment:
    def __init__(self, params: Params, goal=(0.0, 0.0, 0.0), car_axle_offset: float = 0.0,
                 stale_s: float = 0.2, hitch_limit: float = 1.35,
                 pos_tol: float = 0.01, ang_tol: float = 0.05, t_max: float = 1800.0,
                 settle_s: float = 0.0):
        """car_axle_offset: distance from the tracked point of the car to the midpoint
        of its axle, measured along the heading, positive = axle is AHEAD of the
        tracked point [m]. stale_s: a pose older than this stops the robot.
        hitch_limit: abort if |phi - theta1| or |theta1 - theta2| exceeds it [rad]
        (the URDF hitch limit is 1.40 rad). pos_tol/ang_tol: goal reached when the
        car is closer than pos_tol and ALL heading errors are below ang_tol.
        t_max: abort after this much time [s]."""
        self.p = params
        self.n = params.n_trailers
        self.ctrl = TrailerController(params)
        self.sc = SampledController(self.ctrl)
        self.goal = tuple(float(v) for v in goal)
        self.off = float(car_axle_offset)
        self.stale_s = stale_s
        self.hitch_limit = hitch_limit
        self.pos_tol = pos_tol
        self.ang_tol = ang_tol
        self.t_max = t_max
        self.names = ["car", "t1"] + (["t2"] if self.n == 2 else [])
        self._lock = threading.Lock()
        self._raw: Dict[str, dict] = {}        # name -> {x, y, yaw(unwrapped), stamp}
        self._branch_done = False
        self.t_start: Optional[float] = None
        self.rows: List[list] = []
        self.status = "wait"
        self.reason = ""
        self.start_x: Optional[np.ndarray] = None
        self.n_stale = 0

    # ---------------------------------------------------------------- input
    def set_pose(self, name: str, x: float, y: float, yaw: float, stamp: float):
        """Store the newest pose of one body (thread safe). yaw in (-pi, pi] or
        already continuous; it is unwrapped here against the previous sample."""
        with self._lock:
            old = self._raw.get(name)
            if old is None:
                u = yaw
            else:
                u = old["yaw"] + wrap(yaw - old["yaw"])
            self._raw[name] = dict(x=x, y=y, yaw=u, stamp=stamp)

    def _state(self, now: float):
        """Return (x_world_state, oldest_age) or None if a body is missing."""
        with self._lock:
            if any(n not in self._raw for n in self.names):
                return None
            d = {n: dict(self._raw[n]) for n in self.names}
            if not self._branch_done:
                # Choose the 2*pi branches once: the car's heading closest to the goal
                # heading, each trailer's heading closest to the car's. The stored
                # unwrapped values are shifted so later samples follow these branches.
                two_pi = 2 * math.pi
                cs = two_pi * round((self.goal[2] - d["car"]["yaw"]) / two_pi)
                self._raw["car"]["yaw"] += cs
                d["car"]["yaw"] += cs
                for n in self.names[1:]:
                    shift = two_pi * round((d["car"]["yaw"] - self._raw[n]["yaw"]) / two_pi)
                    self._raw[n]["yaw"] += shift
                    d[n]["yaw"] += shift
                self._branch_done = True
        age = max(now - v["stamp"] for v in d.values())
        c = d["car"]
        ax = c["x"] + self.off * math.cos(c["yaw"])
        ay = c["y"] + self.off * math.sin(c["yaw"])
        xg, yg, pg = self.goal
        cg, sg = math.cos(pg), math.sin(pg)
        dx, dy = ax - xg, ay - yg
        x = [cg * dx + sg * dy, -sg * dx + cg * dy, c["yaw"] - pg]
        x += [d[n]["yaw"] - pg for n in self.names[1:]]
        return np.array(x), age

    def ready(self) -> bool:
        """True once every body has delivered at least one pose."""
        with self._lock:
            return all(n in self._raw for n in self.names)

    # ---------------------------------------------------------------- step
    def step(self, now: float) -> Step:
        """Call at the control rate with the current time [s]. Returns the
        velocity command (already saturated)."""
        if self.status in ("goal", "abort"):
            return Step(self.status, 0.0, 0.0, self.reason)
        got = self._state(now)
        if got is None:
            return Step("wait")
        x, age = got
        if self.t_start is None:
            self.t_start = now
            self.sc.start(now)
            self.start_x = x.copy()
        t = now - self.t_start
        if age > self.stale_s:
            self.n_stale += 1
            self._log(t, x, 0.0, 0.0, None, stale=1)
            return Step("stale", 0.0, 0.0, "pose older than %.2f s" % self.stale_s)
        # safety stops
        hitch = [abs(x[2] - x[3])] + ([abs(x[3] - x[4])] if self.n == 2 else [])
        if max(hitch) > self.hitch_limit:
            return self._stop("abort", "hitch angle %.2f rad above limit %.2f" % (max(hitch), self.hitch_limit), t, x)
        if t > self.t_max:
            return self._stop("abort", "time limit %.0f s reached" % self.t_max, t, x)
        if math.hypot(x[0], x[1]) <= self.pos_tol and max(abs(a) for a in x[2:]) <= self.ang_tol:
            return self._stop("goal", "goal reached", t, x)
        cmd = self.sc.command(now, x)
        self._log(t, x, cmd.u1, cmd.u2, cmd, stale=0)
        return Step("run", cmd.u1, cmd.u2)

    def _stop(self, status, reason, t, x) -> Step:
        self.status, self.reason = status, reason
        self._log(t, x, 0.0, 0.0, None, stale=0)
        return Step(status, 0.0, 0.0, reason)

    # ---------------------------------------------------------------- log
    def _log(self, t, x, u1, u2, cmd, stale):
        z = self.ctrl.x_to_z(x)
        self.rows.append([t, *x.tolist(), *z.tolist(), float(np.linalg.norm(z)), u1, u2,
                          -1 if cmd is None else cmd.j,
                          0 if cmd is None else int(cmd.clipped),
                          0 if cmd is None else int(cmd.new_interval), stale])

    def header(self):
        n = self.p.n_trailers
        xs = ["x1", "x2", "phi", "theta1"] + (["theta2"] if n == 2 else [])
        zs = ["z%d" % (i + 1) for i in range(len(xs))]
        return ["t"] + xs + zs + ["norm_z", "u1", "u2", "interval", "clipped", "new_interval", "stale"]

    # ---------------------------------------------------------------- export
    def export(self, out_dir: str, meta: Optional[dict] = None, plots: bool = True) -> str:
        """Write log.csv, params.json, summary.txt (and PNG plots if matplotlib
        is there) into out_dir/<timestamp>/. Returns that folder."""
        run = os.path.join(out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run, exist_ok=True)
        with open(os.path.join(run, "log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.header())
            w.writerows(self.rows)
        info = dict(goal=self.goal, car_axle_offset=self.off, stale_s=self.stale_s,
                    hitch_limit=self.hitch_limit, pos_tol=self.pos_tol, ang_tol=self.ang_tol,
                    t_max=self.t_max, status=self.status, reason=self.reason,
                    n_trailers=self.n, gamma=self.p.gamma, epsilon=self.p.epsilon,
                    k=list(self.p.k), lyapunov=self.p.lyapunov, clip_fraction=self.p.clip_fraction,
                    skipped_intervals=self.sc.skipped, stale_ticks=self.n_stale)
        if meta:
            info.update(meta)
        with open(os.path.join(run, "params.json"), "w") as f:
            json.dump(info, f, indent=2)
        with open(os.path.join(run, "summary.txt"), "w") as f:
            f.write(self.summary() + "\n")
        if plots and self.rows:
            try:
                self._plots(run)
            except Exception as e:  # matplotlib missing etc.: the CSV is the record
                with open(os.path.join(run, "summary.txt"), "a") as f:
                    f.write("plots not written: %r\n" % (e,))
        return run

    def summary(self) -> str:
        if not self.rows:
            return "no data (status: %s)" % self.status
        a = np.array(self.rows, float)
        h = self.header()
        col = {k: i for i, k in enumerate(h)}
        run = a[a[:, col["stale"]] == 0]
        lines = ["status: %s (%s)" % (self.status, self.reason),
                 "duration: %.1f s, samples: %d, stale ticks: %d, skipped intervals: %d"
                 % (a[-1, 0], len(a), self.n_stale, self.sc.skipped),
                 "start x: " + np.array2string(self.start_x, precision=4),
                 "end   x: " + np.array2string(a[-1, 1:1 + len(self.start_x)], precision=4),
                 "norm z: start %.4f, end %.4f (ratio %.4f)"
                 % (a[0, col["norm_z"]], a[-1, col["norm_z"]], a[-1, col["norm_z"]] / max(a[0, col["norm_z"]], 1e-12))]
        if len(run):
            u1, u2 = run[:, col["u1"]], run[:, col["u2"]]
            lines.append("max |u1| = %.4f (limit %.3f), max |u2| = %.4f (limit %.3f)"
                         % (np.max(np.abs(u1)), self.p.u1_max, np.max(np.abs(u2)), self.p.u2_max))
            lines.append("clipped samples: %d of %d" % (int(run[:, col["clipped"]].sum()), len(run)))
        return "\n".join(lines)

    def _plots(self, run):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        a = np.array(self.rows, float)
        col = {k: i for i, k in enumerate(self.header())}
        t = a[:, 0]
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(a[:, col["x1"]], a[:, col["x2"]], "b-", label="car axle (goal frame)")
        ax.plot(a[0, col["x1"]], a[0, col["x2"]], "ro", label="start")
        ax.plot(0, 0, "go", label="goal")
        ax.set_xlabel("x1 [m]"); ax.set_ylabel("x2 [m]"); ax.axis("equal"); ax.grid(True); ax.legend()
        fig.savefig(os.path.join(run, "trajectory.png"), dpi=200); plt.close(fig)
        fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(7, 8))
        axs[0].semilogy(t, np.maximum(a[:, col["norm_z"]], 1e-9)); axs[0].set_ylabel("||z||"); axs[0].grid(True)
        axs[1].plot(t, a[:, col["u1"]]); axs[1].axhline(self.p.u1_max, color="r", ls="--")
        axs[1].axhline(-self.p.u1_max, color="r", ls="--"); axs[1].set_ylabel("u1 [m/s]"); axs[1].grid(True)
        axs[2].plot(t, a[:, col["u2"]]); axs[2].axhline(self.p.u2_max, color="r", ls="--")
        axs[2].axhline(-self.p.u2_max, color="r", ls="--"); axs[2].set_ylabel("u2 [rad/s]")
        axs[2].set_xlabel("t [s]"); axs[2].grid(True)
        fig.savefig(os.path.join(run, "norm_and_inputs.png"), dpi=200); plt.close(fig)
        fig, ax = plt.subplots(constrained_layout=True)
        for k in ["phi", "theta1"] + (["theta2"] if self.n == 2 else []):
            ax.plot(t, np.degrees(a[:, col[k]]), label=k)
        ax.set_xlabel("t [s]"); ax.set_ylabel("heading [deg]"); ax.grid(True); ax.legend()
        fig.savefig(os.path.join(run, "angles.png"), dpi=200); plt.close(fig)
