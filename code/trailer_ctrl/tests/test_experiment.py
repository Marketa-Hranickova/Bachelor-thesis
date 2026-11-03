"""Closed-loop test of experiment.py with a simulated plant (no ROS).

The plant is the exact kinematic model, integrated with RK4 while the command is
held constant between control ticks (zero-order hold, as on the robot). Poses are
fed in as a motion capture would give them: world frame, yaw wrapped to (-pi, pi],
the tracked point of the car NOT at the axle.

Run:  python3 trailer_ctrl/tests/test_experiment.py     (or python3 -m ...)
"""
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trailer_ctrl import Params, TrailerController                    # noqa: E402
from trailer_ctrl.experiment import Experiment, wrap                   # noqa: E402
from trailer_ctrl.sim import fields_exact_1tr, fields_exact_2tr, simulate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, "..", "configs", name)) as f:
        return Params.from_dict(json.load(f))


def plant_rhs(x, u, p):
    g = p.geometry
    f1, f2 = fields_exact_1tr(x, g) if p.n_trailers == 1 else fields_exact_2tr(x, g)
    return f1 * u[0] + f2 * u[1]


def rk4(x, u, dt, p, n=4):
    h = dt / n
    for _ in range(n):
        k1 = plant_rhs(x, u, p)
        k2 = plant_rhs(x + h / 2 * k1, u, p)
        k3 = plant_rhs(x + h / 2 * k2, u, p)
        k4 = plant_rhs(x + h * k3, u, p)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def to_world(xg_rel, goal):
    """goal-frame state -> world state (inverse of what Experiment does)."""
    xg, yg, pg = goal
    c, s = math.cos(pg), math.sin(pg)
    x = np.array(xg_rel, float).copy()
    x[0] = xg + c * xg_rel[0] - s * xg_rel[1]
    x[1] = yg + s * xg_rel[0] + c * xg_rel[1]
    x[2:] = x[2:] + pg
    return x


def run_loop(p, x0_rel, goal, t_end, rate=100.0, axle_offset=0.0, out_dir=None, exp_kwargs=None):
    """Closed loop. The plant state is kept in the goal frame; what the experiment
    gets is the world pose with wrapped yaw and the car tracked point shifted back
    by axle_offset (so the axle is axle_offset AHEAD of the tracked point)."""
    exp = Experiment(p, goal=goal, car_axle_offset=axle_offset, **(exp_kwargs or {}))
    x = np.array(x0_rel, float)
    dt = 1.0 / rate
    t = 0.0
    u = (0.0, 0.0)
    n = int(round(t_end * rate))
    for _ in range(n + 1):
        w = to_world(x, goal)
        yaw_car = w[2]
        tracked = (w[0] - axle_offset * math.cos(yaw_car), w[1] - axle_offset * math.sin(yaw_car))
        exp.set_pose("car", tracked[0], tracked[1], wrap(w[2]), t)
        exp.set_pose("t1", 0.0, 0.0, wrap(w[3]), t)
        if p.n_trailers == 2:
            exp.set_pose("t2", 0.0, 0.0, wrap(w[4]), t)
        st = exp.step(t)
        if st.status in ("goal", "abort"):
            break
        u = (st.u1, st.u2) if st.status == "run" else (0.0, 0.0)
        x = rk4(x, u, dt, p)
        t += dt
    return exp, x, t


X0 = {1: np.array([-0.2, -0.2, math.radians(45), math.radians(45)]),
      2: np.array([-0.2, -0.2, math.radians(45), math.radians(45), math.radians(45)])}


def check(label, ok, detail, out):
    out.append((ok, "  [%s] %-46s %s" % ("ok  " if ok else "FAIL", label, detail)))
    return ok


def run_all():
    out = []
    for n, cfg in ((1, "1tr.json"), (2, "2tr.json")):
        p = load(cfg)
        ctrl = TrailerController(p)
        T = 30.0
        # reference: continuous-u offline simulation on the exact model
        _, xs, _, _ = simulate(ctrl, X0[n], T, "exact", dt=0.01)
        zref = np.linalg.norm(ctrl.x_to_z(xs[-1]))
        z0 = np.linalg.norm(ctrl.x_to_z(X0[n]))
        # (a) goal at the origin, tracked point = axle
        # hitch_limit is raised here on purpose: the design path itself folds the hitches
        # beyond 1.40 rad (see README); this test is about the logic, not about that
        free = dict(hitch_limit=10.0)
        exp, x, t = run_loop(p, X0[n], (0.0, 0.0, 0.0), T, exp_kwargs=free)
        z = np.linalg.norm(ctrl.x_to_z(x))
        check("%dtr: ||z|| after %d s vs offline (ZOH 100 Hz)" % (n, T),
              abs(z - zref) / z0 < 1e-2, "online %.5f offline %.5f (start %.4f)" % (z, zref, z0), out)
        # (b) goal rotated by 3.0 rad (angles wrap), shifted, axle offset 0.05 m
        goal = (1.0, -0.5, 3.0)
        exp2, x2, _ = run_loop(p, X0[n], goal, T, axle_offset=0.05, exp_kwargs=free)
        z2 = np.linalg.norm(ctrl.x_to_z(x2))
        check("%dtr: rotated goal + axle offset + wrapping" % n, abs(z2 - z) / z0 < 1e-6,
              "||z|| %.6f vs %.6f" % (z2, z), out)
        # logged state equals plant state in goal frame
        last = np.array(exp2.rows[-1][1:1 + p.n_trailers + 3])
        check("%dtr: logged goal-frame state = plant" % n, np.max(np.abs(last - x2)) < 0.035,
              "max diff %.1e (<= one control tick)" % np.max(np.abs(last - x2)), out)
        # The difference to the offline run is the zero-order hold: it shrinks like 1/rate
        # (n=1: 2.7 % of ||z0|| at 20 Hz, 0.5 % at 100 Hz, 0.1 % at 400 Hz, checked by hand).
        # (c) saturation never above the limit
        u1 = np.array([r[exp.header().index("u1")] for r in exp.rows])
        check("%dtr: |u1| <= %.3f" % (n, p.u1_max), np.max(np.abs(u1)) <= p.u1_max + 1e-12,
              "max %.4f" % np.max(np.abs(u1)), out)

    # (d) safety stops, 1 trailer
    p = load("1tr.json")
    exp, _, _ = run_loop(p, [-0.2, -0.2, 0.0, 1.5], (0, 0, 0), 5.0)
    check("hitch limit aborts", exp.status == "abort" and "hitch" in exp.reason, exp.reason, out)
    exp, _, _ = run_loop(p, X0[1], (0, 0, 0), 5.0, exp_kwargs=dict(t_max=2.0, hitch_limit=10.0))
    check("time limit aborts", exp.status == "abort" and "time" in exp.reason, exp.reason, out)
    exp, _, _ = run_loop(p, [0.001, 0.0, 0.01, 0.0], (0, 0, 0), 5.0)
    check("goal reached stops", exp.status == "goal", exp.reason, out)

    # (e) stale pose -> zero command, resumes afterwards
    e = Experiment(p)
    for nm in ("car", "t1"):
        e.set_pose(nm, -0.2, -0.2, 0.8, 0.0)
    s0 = e.step(0.0)
    s1 = e.step(0.5)             # pose is 0.5 s old
    for nm in ("car", "t1"):
        e.set_pose(nm, -0.2, -0.2, 0.8, 0.6)
    s2 = e.step(0.6)
    check("stale pose stops, fresh pose resumes", s0.status == "run" and s1.status == "stale"
          and s1.u1 == 0.0 and s2.status == "run", "%s %s %s" % (s0.status, s1.status, s2.status), out)

    # (f) export
    d = tempfile.mkdtemp()
    try:
        exp, _, _ = run_loop(p, X0[1], (0, 0, 0), 8.0)
        run = exp.export(d)
        files = sorted(os.listdir(run))
        check("export writes log, params, summary", {"log.csv", "params.json", "summary.txt"} <= set(files),
              ", ".join(files), out)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out


def test_experiment():
    out = run_all()
    assert all(ok for ok, _ in out), "\n".join(l for _, l in out)


if __name__ == "__main__":
    res = run_all()
    print("\n".join(l for _, l in res))
    if all(ok for ok, _ in res):
        print("\n=== EXPERIMENT TEST PASSED ===")
    else:
        print("\n=== EXPERIMENT TEST FAILED ===")
        sys.exit(1)
