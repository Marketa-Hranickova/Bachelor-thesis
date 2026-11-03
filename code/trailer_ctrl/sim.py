"""Offline closed-loop simulation (needs scipy). For tests and for comparing
with the MATLAB simulators; the ROS node does not import this."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .controller import TrailerController
from .geometry import Geometry


def fields_approx_1tr(z, alpha):
    g1 = np.array([1.0, 0.0, -z[1], -z[2]])
    g2 = np.array([0.0, 1.0, 0.0, -alpha * z[1] ** 2])
    return g1, g2


def fields_approx_2tr(z, alpha, kappa, theta):
    g1 = np.array([1.0, 0.0, -z[1], -z[2], kappa * z[1] ** 3 - z[3]])
    g2 = np.array([0.0, 1.0, 0.0, -alpha * z[1] ** 2, -theta * z[1] * z[2]])
    return g1, g2


def fields_exact_1tr(x, g: Geometry):
    f1 = np.array([np.cos(x[2]), np.sin(x[2]), 0.0, np.sin(x[2] - x[3]) / g.l1])
    f2 = np.array([0.0, 0.0, 1.0, -g.d0 * np.cos(x[2] - x[3]) / g.l1])
    return f1, f2


def fields_exact_2tr(x, g: Geometry):
    d0, d1, l1, l2 = g.d0, g.d1, g.l1, g.l2
    x3, x4, x5 = x[2], x[3], x[4]
    f15 = np.sin(x3 - x5) / l2 + (d1 + l1) * np.sin(x4 - x3) * np.cos(x4 - x5) / (l1 * l2)
    f25 = d0 * (d1 + l1) * np.cos(x3 - x4) * np.cos(x4 - x5) / (l1 * l2) \
        - d0 * np.cos(x3 - x5) / l2
    f1 = np.array([np.cos(x3), np.sin(x3), 0.0, np.sin(x3 - x4) / l1, f15])
    f2 = np.array([0.0, 0.0, 1.0, -d0 * np.cos(x3 - x4) / l1, f25])
    return f1, f2


def simulate(ctrl: TrailerController, x0, t_end: float, mode: str = "exact",
             dt: float = 0.02, rtol: float = 1e-10, atol: float = 1e-12):
    """Sampled closed loop. mode 'exact': x' = u1 f1 + u2 f2 (state x);
    mode 'approx': z' = u1 g1 + u2 g2 (state z). Returns t, state, u1, u2
    on a dt grid (peaks of u1 at interval starts are on the grid)."""
    p = ctrl.p
    eps = p.epsilon
    n = p.n_trailers
    g = p.geometry
    n_int = int(np.ceil(t_end / eps - 1e-9))
    state = np.asarray(x0, float).copy() if mode == "exact" else ctrl.x_to_z(x0).copy()
    T, S, U1, U2 = [], [], [], []
    for j in range(n_int):
        t0, t1 = j * eps, min((j + 1) * eps, t_end)
        zf = ctrl.x_to_z(state) if mode == "exact" else state.copy()

        def rhs(t, s):
            u1, u2 = ctrl.u(t - t0, zf)
            if mode == "exact":
                f1, f2 = fields_exact_1tr(s, g) if n == 1 else fields_exact_2tr(s, g)
            else:
                f1, f2 = fields_approx_1tr(s, ctrl.alpha) if n == 1 else \
                    fields_approx_2tr(s, ctrl.alpha, ctrl.kappa, ctrl.theta)
            return f1 * u1 + f2 * u2

        ts = t0 + dt * np.arange(int(np.floor((t1 - t0) / dt + 1e-9)) + 1)
        ts = ts[ts < t1 - 1e-12]
        ts = np.append(ts, t1)
        sol = solve_ivp(rhs, (t0, t1), state, t_eval=ts, rtol=rtol, atol=atol,
                        method="DOP853", max_step=dt)
        for i, t in enumerate(sol.t):
            if j > 0 and i == 0:
                continue
            u1, u2 = ctrl.u(t - t0, zf)
            T.append(t); S.append(sol.y[:, i]); U1.append(u1); U2.append(u2)
        state = sol.y[:, -1]
    return np.array(T), np.array(S), np.array(U1), np.array(U2)
