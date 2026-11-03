"""Does the Python port compute what the MATLAB reference computes?

The reference numbers are copied from the MATLAB self-tests (selftest in
trailer1.m / trailer2.m; one trailer: 18 Oct 2026 design, two trailers: 27 Oct 2026 design
with the start 0.6*(...); Octave 8.4). Run from the folder that CONTAINS trailer_ctrl/:

    python3 -m pytest trailer_ctrl/tests -q
or  python3 -m trailer_ctrl.tests.test_vs_matlab
or  python3 trailer_ctrl/tests/test_vs_matlab.py   (any folder)
"""
import json
import math
import os
import sys

import numpy as np

# Allow running this file directly (python3 test_vs_matlab.py) from any folder:
# put the folder that contains trailer_ctrl/ on the import path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trailer_ctrl import Params, SampledController, TrailerController
from trailer_ctrl.sim import simulate

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, "..", "configs", name)) as f:
        return TrailerController(Params.from_dict(json.load(f)))


X0_1TR = np.array([-0.2, -0.2, math.radians(45), math.radians(45)])
# 27 Oct 2026: 0.6 * the earlier start, so that ||x0|| = 0.834 < 1 (Victoria); the 1 trailer start is unchanged
X0_2TR = 0.6 * np.array([-0.2, -0.2, math.radians(45), math.radians(45), math.radians(45)])

REF_1TR = dict(
    alpha=1.132750000000000e-02,
    z0=[-2.000000000000000e-01, 7.853981633974483e-01, 2.000000000000000e-01, -2.168101725068444e-02],
    bound=[2.119078228329150e-01, 3.900679149740466e-01],
    gmax=3.550395847357964e-03,
    uctrl=[1.121128800760134e-01, 6.258538533368191e-02],          # t = 0.37 s
    z_approx=[-2.474523528968e-01, 7.832446761761e-01, 2.374795922937e-01, -8.751414946495e-03],
    u_approx=[2.132600407082e-01, 3.902267054966e-01],
    x_exact=[-2.331515570860e-01, -2.334255561516e-01, 7.831352153506e-01, -6.760406833783e-01],
    u_exact=[2.119078228329e-01, 3.900679149740e-01],
)
REF_2TR = dict(
    coeffs=[7.355682212287460e-03, 2.329616385637499e-02, 6.111464836543101e-02, -1.085630195308095e-04],
    z0=[-1.200000000000000e-01, 4.712388980384704e-01, 1.200000000000000e-01,
        -3.086039513965813e-02, 1.475207058413563e-03],
    bound=[2.199969385618743e-01, 5.169873542556976e-01],
    gmax=3.498156874610878e-05,
    uctrl=[-1.818128736544115e-02, 1.962972159915425e-01],         # t = 0.37 s
    z_approx=[-1.198996273460e-01, 4.708447355481e-01, 1.198952285424e-01,
              -3.083479803913e-02, 1.469553981366e-03],
    u_approx=[2.199969385619e-01, 5.018360310199e-01],
    x_exact=[-1.199387678509e-01, -1.199043945509e-01, 4.708447385593e-01,
             5.063290919463e-01, 2.721019209988e-01],
    u_exact=[2.199969385619e-01, 5.018360310199e-01],
)


def close(got, want, tol):
    err = float(np.max(np.abs(np.asarray(got, float) - np.asarray(want, float))))
    return err <= tol, err


def check(label, got, want, tol, out):
    ok, err = close(got, want, tol)
    out.append((ok, "  [%s] %-28s max |difference| = %.2e  (tol %.0e)"
                % ("ok  " if ok else "FAIL", label, err, tol)))
    return ok


# ------------------------------------------------------------------ one trailer
def run_1tr():
    c = load("1tr.json")
    R, out = REF_1TR, []
    z0 = c.x_to_z(X0_1TR)
    check("alpha", c.alpha, R["alpha"], 1e-12, out)
    check("z0 = x_to_z(x0)", z0, R["z0"], 1e-12, out)
    check("control bound at z0", c.bound(z0), R["bound"], 1e-12, out)
    check("gamma_max at z0", c.gamma_max(z0), R["gmax"], 1e-10, out)
    check("u(0.37 s) at z0", c.u(0.37, z0), R["uctrl"], 1e-12, out)
    # SampledController must give the same u as u(tau) at an interval start + 0.37
    sc = SampledController(c); sc.start(0.0)
    cmd = sc.command(0.37, X0_1TR)
    check("SampledController u(0.37 s)", [cmd.u1, cmd.u2], R["uctrl"], 1e-12, out)
    t, z, u1, u2 = simulate(c, X0_1TR, 12.0, "approx", dt=2e-3)
    check("approx z(12 s)", z[-1], R["z_approx"], 1e-7, out)
    # the MATLAB maximum is taken on ode45's own (finer) output grid, so the peak of
    # |u1| in the approximation (which is not on an interval start here) differs by
    # ~3e-6 from the one on a 2 ms grid; with dt -> 0 the two agree (checked)
    check("approx max|u| (grid)", [np.max(np.abs(u1)), np.max(np.abs(u2))], R["u_approx"], 1e-5, out)
    t, x, u1, u2 = simulate(c, X0_1TR, 12.0, "exact", dt=2e-3)
    check("exact  x(12 s)", x[-1], R["x_exact"], 1e-7, out)
    check("exact  max|u|", [np.max(np.abs(u1)), np.max(np.abs(u2))], R["u_exact"], 1e-7, out)
    return out


# ----------------------------------------------------------------- two trailers
def run_2tr():
    from trailer_ctrl.geometry import offhooked_coeffs
    c = load("2tr.json")
    R, out = REF_2TR, []
    a, k, th, be, _ = offhooked_coeffs(c.p.geometry)
    z0 = c.x_to_z(X0_2TR)
    check("alpha, kappa, theta, beta", [a, k, th, be], R["coeffs"], 1e-12, out)
    check("z0 = x_to_z(x0)", z0, R["z0"], 1e-12, out)
    check("control bound at z0", c.bound(z0), R["bound"], 1e-12, out)
    check("gamma_max at z0", c.gamma_max(z0), R["gmax"], 1e-10, out)
    check("u(0.37 s) at z0", c.u(0.37, z0), R["uctrl"], 1e-12, out)
    sc = SampledController(c); sc.start(0.0)
    cmd = sc.command(0.37, X0_2TR)
    check("SampledController u(0.37 s)", [cmd.u1, cmd.u2], R["uctrl"], 1e-12, out)
    t, z, u1, u2 = simulate(c, X0_2TR, 12.0, "approx", dt=2e-3)
    check("approx z(12 s)", z[-1], R["z_approx"], 1e-7, out)
    check("approx max|u|", [np.max(np.abs(u1)), np.max(np.abs(u2))], R["u_approx"], 1e-7, out)
    t, x, u1, u2 = simulate(c, X0_2TR, 12.0, "exact", dt=2e-3)
    check("exact  x(12 s)", x[-1], R["x_exact"], 1e-7, out)
    check("exact  max|u|", [np.max(np.abs(u1)), np.max(np.abs(u2))], R["u_exact"], 1e-7, out)
    return out


def test_one_trailer():
    out = run_1tr()
    assert all(ok for ok, _ in out), "\n".join(l for _, l in out)


def test_two_trailers():
    out = run_2tr()
    assert all(ok for ok, _ in out), "\n".join(l for _, l in out)


def test_interval_grid():
    """Interval starts are on the fixed grid t0 + j*eps; a late call still starts
    the right interval and is counted as skipped."""
    c = load("1tr.json")
    sc = SampledController(c); sc.start(10.0)
    eps = c.p.epsilon
    a = sc.command(10.0 + 0.1, X0_1TR)
    assert a.new_interval and a.j == 0 and abs(a.tau - 0.1) < 1e-12
    b = sc.command(10.0 + eps - 1e-3, X0_1TR)
    assert (not b.new_interval) and b.j == 0
    d = sc.command(10.0 + 3 * eps + 0.2, X0_1TR)       # intervals 1 and 2 missed
    assert d.new_interval and d.j == 3 and abs(d.tau - 0.2) < 1e-9 and sc.skipped == 2


def test_clipping():
    c = load("1tr.json")
    c.p.clip_fraction = 0.5                           # saturate at half the limits
    sc = SampledController(c); sc.start(0.0)
    cmd = sc.command(0.0, X0_1TR)                     # u1 peak of ~0.21 > 0.11
    assert cmd.clipped and abs(cmd.u1) <= 0.5 * c.p.u1_max + 1e-12


if __name__ == "__main__":
    ok_all = True
    for name, fn in (("one trailer", run_1tr), ("two trailers", run_2tr)):
        print("\n=== %s ===" % name)
        for ok, line in fn():
            print(line); ok_all &= ok
    test_interval_grid(); test_clipping()
    print("\n=== PORT TEST %s ===" % ("PASSED" if ok_all else "FAILED"))
