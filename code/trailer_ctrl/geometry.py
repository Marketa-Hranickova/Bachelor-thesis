"""Geometry-only quantities and the state map x -> z (privileged coordinates).

Port of the MATLAB files  1 trailer/trailer1.m (x_to_z, one_trailer_coeff)
and  2 trailers/offhooked_coeffs.m, x_to_z.m.  Angles are in RADIANS.

State vectors
    one trailer : x = [x1, x2, phi, theta1]            (r = 3)
    two trailers: x = [x1, x2, phi, theta1, theta2]    (r = 4)
The angles are NOT wrapped: the privileged coordinates are linear in the
angles, so the caller must pass continuous (unwrapped) angles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geometry:
    """d0: car axle -> 1st hitch, l1: 1st hitch -> trailer-1 axle,
    d1: trailer-1 axle -> 2nd hitch, l2: 2nd hitch -> trailer-2 axle  [m].
    For one trailer d1 and l2 are ignored."""
    d0: float
    l1: float
    d1: float = 0.0
    l2: float = 0.0


# ---------------------------------------------------------------- one trailer
def alpha_1tr(g: Geometry) -> float:
    """alpha = d0 (d0 + l1) / 2   (nilpotent approximation, r = 3)."""
    return g.d0 * (g.d0 + g.l1) / 2.0


def x_to_z_1tr(x, g: Geometry) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    d0, l1 = g.d0, g.l1
    return np.array([
        x[0],
        x[2],
        -x[1],
        l1 * x[1] - d0 * l1**2 / (d0 + l1) * x[2] - l1**3 / (d0 + l1) * x[3],
    ])


def z_to_x_1tr(z, g: Geometry) -> np.ndarray:
    z = np.asarray(z, dtype=float).ravel()
    d0, l1 = g.d0, g.l1
    a = d0 * l1**2 / (d0 + l1)
    b = l1**3 / (d0 + l1)
    return np.array([z[0], -z[2], z[1], (-l1 * z[2] - a * z[1] - z[3]) / b])


# --------------------------------------------------------------- two trailers
def offhooked_coeffs(g: Geometry):
    """Returns (alpha, kappa, theta, beta, M0) exactly as offhooked_coeffs.m."""
    d0, d1, l1, l2 = g.d0, g.d1, g.l1, g.l2
    alpha = (d0**3 * (l1**3 + l1**2 * l2 - l1 * l2**2 - d1 * l2 * (d1 + l2))
             + d0 * l1**3 * (d0 * (l1 + 2 * l2) + l2 * (l1 + l2))) \
        / (2 * (d0 + l2) * l1**3)
    kappa = d0**2 * (1.5 * l2**2 * (d0 + l1) * (d1 + l1)
                     - l2 * (l1**3 + 0.5 * d0 * l1**2 - d1 * l1 * (d0 + 1.5 * d1)
                             - 1.5 * d0 * d1**2)
                     + 0.5 * d0 * d1**2 * l1) / (3 * (d0 + l2) * l1**3)
    theta = d0**2 * (l2**2 * (d1 + l1) * (d0 + l1)
                     + (d0 * d1 + (d0 + d1) * l1) * d1 * l2
                     + d0 * d1**2 * l1) / ((d0 + l2) * l1**3)
    beta = (d0 * l2 * (d0**2 * (d1 + l1) * (l1 - l2 - d1) + l1**3 * (d0 + l2))) \
        / (6 * (d0 + l2) * l1**2)

    M0 = np.zeros((5, 5))
    M0[0, 0] = 1
    M0[1, 2] = -1
    M0[2, 1] = 1
    M0[3, 1] = -d0 / l1
    M0[3, 2] = -(d0 + l1) / l1**2
    M0[3, 3] = -(d0 + l1) / l1**3
    M0[3, 4] = -(d0 + l1) / l1**4
    M0[4, 1] = d0 * d1 / (l1 * l2)
    M0[4, 2] = (d0 * d1 * (l1 + l2) + l1 * l2 * (d0 + d1)) / (l1**2 * l2**2)
    M0[4, 3] = (d0 * d1 * (l1**2 + l1 * l2 + l2**2)
                + l1 * l2 * (d0 + d1) * (l1 + l2) + l1**2 * l2**2) / (l1**3 * l2**3)
    M0[4, 4] = (d0 * d1 * (l1**3 + l1**2 * l2 + l1 * l2**2 + l2**3)
                + l1 * l2 * (d0 + d1) * (l1**2 + l1 * l2 + l2**2)
                + l1**2 * l2**2 * (l1 + l2)) / (l1**4 * l2**4)
    return alpha, kappa, theta, beta, M0


def x_to_z_2tr(x, g: Geometry) -> np.ndarray:
    _, _, _, beta, M0 = offhooked_coeffs(g)
    x = np.asarray(x, dtype=float).ravel()
    z = np.linalg.solve(M0, x)
    z[4] = z[4] - beta * x[2] ** 3
    return z


def z_to_x_2tr(z, g: Geometry) -> np.ndarray:
    _, _, _, beta, M0 = offhooked_coeffs(g)
    z = np.asarray(z, dtype=float).ravel().copy()
    z[4] = z[4] + beta * z[1] ** 3
    return M0 @ z
