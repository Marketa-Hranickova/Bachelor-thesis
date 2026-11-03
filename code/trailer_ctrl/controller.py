"""Oscillating Lie-bracket controller for the TurtleBot3 with off-hooked trailers.

Pure Python + numpy, NO ROS and no plotting: the ROS 2 node only calls this
module. Port of the MATLAB reference files trailer1.m (one trailer, r = 3)
and trailer2.m (two trailers, r = 4); the numbers are checked against the
MATLAB self-tests in tests/test_vs_matlab.py.

Sampled scheme (Chapter 5/7 of the thesis)
------------------------------------------
Time is cut into intervals of length epsilon = 2*pi/omega that start at
t_j = t0 + j*epsilon. At the start of an interval the measured state x is
converted to the privileged coordinates z and FROZEN; the amplitudes are
computed from this z, and u1, u2 are a function of the time tau = t - t_j
inside the interval only. Keeping the interval starts on a fixed grid is
essential: all cosines are 1 at tau = 0 (the peak of |u1|, which the bound of
Chapter 6 refers to) and every oscillation completes whole periods per
interval.  (A scheme that restarts "when at least T has elapsed" drifts off
this grid and loses both properties.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .geometry import (Geometry, alpha_1tr, offhooked_coeffs, x_to_z_1tr,
                       x_to_z_2tr)


# ------------------------------------------------------------------ parameters
@dataclass
class Params:
    n_trailers: int                      # 1 (r = 3) or 2 (r = 4)
    geometry: Geometry
    gamma: float                         # gain (fixed)
    epsilon: float                       # sampling interval = base period [s]
    k: Tuple[int, ...]                   # (k12, k112) or (k12, k112, k1112)
    lyapunov: Dict = field(default_factory=lambda: {"type": "sqrt", "delta": 0.02})
    u1_max: float = 0.22                 # TurtleBot3 Burger [m/s]
    u2_max: float = 2.84                 # [rad/s]
    clip_fraction: float = 1.0           # saturate at this fraction of the limits

    def __post_init__(self):
        if self.n_trailers not in (1, 2):
            raise ValueError("n_trailers must be 1 or 2")
        if len(self.k) != self.n_trailers + 1:
            raise ValueError("k needs %d entries" % (self.n_trailers + 1))
        if self.epsilon <= 0 or self.gamma <= 0:
            raise ValueError("epsilon and gamma must be positive")

    @property
    def omega(self) -> float:
        return 2.0 * math.pi / self.epsilon

    @staticmethod
    def from_dict(d: dict) -> "Params":
        g = d["geometry"]
        geo = Geometry(d0=g["d0"], l1=g["l1"], d1=g.get("d1", 0.0), l2=g.get("l2", 0.0))
        return Params(n_trailers=int(d["n_trailers"]), geometry=geo,
                      gamma=float(d["gamma"]), epsilon=float(d["epsilon"]),
                      k=tuple(int(v) for v in d["k"]),
                      lyapunov=dict(d.get("lyapunov", {"type": "sqrt", "delta": 0.02})),
                      u1_max=float(d.get("u1_max", 0.22)),
                      u2_max=float(d.get("u2_max", 2.84)),
                      clip_fraction=float(d.get("clip_fraction", 1.0)))


def make_h(lyapunov: dict) -> Callable[[np.ndarray], float]:
    """State factor h(z) of a radial Lyapunov function: a = -gamma*h(z)*G^-1(z)*z.
        quadratic : h = 1
        sqrt      : V = gamma*(sqrt(r^2+d^2)-d)       h = 1/sqrt(r^2 + delta^2)
        log       : V = gamma/(2c) log(1 + c r^2)     h = 1/(1 + c r^2)   (Grushkovskaya)
    """
    t = lyapunov.get("type", "sqrt")
    if t == "quadratic":
        return lambda z: 1.0
    if t == "sqrt":
        d2 = float(lyapunov["delta"]) ** 2
        return lambda z: 1.0 / math.sqrt(float(np.dot(z, z)) + d2)
    if t == "log":
        c = float(lyapunov["c"])
        return lambda z: 1.0 / (1.0 + c * float(np.dot(z, z)))
    raise ValueError("unknown Lyapunov function type: %r" % t)


# ------------------------------------------------------------------ controller
class TrailerController:
    """Stateless controller: amplitudes at a frozen z, and u(tau)."""

    def __init__(self, p: Params):
        self.p = p
        self.h = make_h(p.lyapunov)
        g = p.geometry
        if p.n_trailers == 1:
            self.alpha = alpha_1tr(g)
            self.kappa = self.theta = None
            self._x_to_z = lambda x: x_to_z_1tr(x, g)
        else:
            self.alpha, self.kappa, self.theta, _, _ = offhooked_coeffs(g)
            self._x_to_z = lambda x: x_to_z_2tr(x, g)

    # -- state map ---------------------------------------------------------
    def x_to_z(self, x) -> np.ndarray:
        return self._x_to_z(x)

    # -- amplitudes a(z) = -gamma h(z) G^-1(z) z ----------------------------
    def amps(self, z, gamma: Optional[float] = None):
        gam = self.p.gamma if gamma is None else gamma
        h = self.h(z)
        s = -gam * h
        a1 = s * z[0]
        a2 = s * z[1]
        a12 = s * (z[0] * z[1] + z[2])
        if self.p.n_trailers == 1:
            a112 = s * (self.alpha * z[1] ** 3 + z[0] * z[2] + z[3])
            return a1, a2, a12, a112
        a112 = s * (z[0] * z[2] + z[3] + self.alpha * z[1] ** 3)
        a1112 = s * ((self.alpha + 2 * self.kappa - self.theta) * z[0] * z[1] ** 3
                     + (self.alpha + 3 * self.kappa) * z[1] ** 2 * z[2]
                     + z[0] * z[3] + z[4])
        return a1, a2, a12, a112, a1112

    def terms(self, z, gamma: Optional[float] = None):
        """Constant parts, signs and sizes T of the oscillating terms
        (osc_terms in the MATLAB files)."""
        w = self.p.omega
        k = self.p.k
        am = self.amps(z, gamma)
        T12 = math.sqrt(2 * k[0] * w * abs(am[2]))
        T112 = (2 * k[1] * w) ** (2.0 / 3.0) * abs(am[3]) ** (1.0 / 3.0)
        out = dict(a1=am[0], a2=am[1], s12=np.sign(am[2]), s112=np.sign(am[3]),
                   T12=T12, T112=T112)
        if self.p.n_trailers == 2:
            out["s1112"] = np.sign(am[4])
            out["T1112"] = (k[2] * w) ** 0.75 * abs(am[4]) ** 0.25
        return out

    # -- control signal inside one interval ----------------------------------
    def u(self, tau: float, z, gamma: Optional[float] = None):
        """(u1, u2) at time tau after the start of the interval whose frozen
        state is z. No saturation here."""
        w = self.p.omega
        k = self.p.k
        t = self.terms(z, gamma)
        u1 = t["a1"] + t["T12"] * math.cos(k[0] * w * tau) \
            + t["T112"] * math.cos(k[1] * w * tau)
        u2 = t["a2"] + t["T12"] * t["s12"] * math.sin(k[0] * w * tau) \
            - 2 * t["T112"] * t["s112"] * math.cos(2 * k[1] * w * tau)
        if self.p.n_trailers == 2:
            u1 += 2 * t["T1112"] * math.cos(k[2] * w * tau)
            u2 -= 6 * t["T1112"] * t["s1112"] * math.sin(3 * k[2] * w * tau)
        return u1, u2

    # -- worst case (peaks coincide), and the largest admissible gain -----------
    def bound(self, z, gamma: Optional[float] = None):
        t = self.terms(z, gamma)
        if self.p.n_trailers == 1:
            b1 = abs(t["a1"]) + t["T12"] + t["T112"]
            b2 = abs(t["a2"]) + t["T12"] + 2 * t["T112"]
        else:
            b1 = abs(t["a1"]) + t["T12"] + t["T112"] + 2 * t["T1112"]
            b2 = abs(t["a2"]) + t["T12"] + 2 * t["T112"] + 6 * t["T1112"]
        return b1, b2

    def gamma_max(self, z, u1max: Optional[float] = None, u2max: Optional[float] = None) -> float:
        """Largest gain for which both limits still hold at state z
        (200 bisection steps in log space, as in MATLAB gamma_max)."""
        u1m = self.p.u1_max if u1max is None else u1max
        u2m = self.p.u2_max if u2max is None else u2max
        lo, hi = 1e-16, 10.0
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            b1, b2 = self.bound(z, mid)
            if b1 <= u1m and b2 <= u2m:
                lo = mid
            else:
                hi = mid
        return lo


# --------------------------------------------------------- sampled-time wrapper
@dataclass
class Command:
    u1: float
    u2: float
    j: int                 # interval index
    tau: float             # time inside the interval [s]
    new_interval: bool     # True on the first call of an interval
    clipped: bool          # True if the saturation was active
    z: np.ndarray          # frozen state of this interval
    bound: Tuple[float, float]   # worst-case (|u1|, |u2|) of this interval


class SampledController:
    """Interval bookkeeping: freezes z at fixed grid times t0 + j*epsilon.

    Use:   sc = SampledController(ctrl); sc.start(t_now)
           cmd = sc.command(t_now, x_measured)     # call as often as you like
    The state is read (and the amplitudes are recomputed) only on the first
    call of a new interval; calls in between reuse the frozen z. If the
    caller is so late that whole intervals were skipped, the new interval is
    still started on the grid and `skipped` counts how many were missed.
    """

    def __init__(self, ctrl: TrailerController):
        self.ctrl = ctrl
        self.t0: Optional[float] = None
        self.j = -1
        self.z: Optional[np.ndarray] = None
        self._bound = (0.0, 0.0)
        self.skipped = 0

    def start(self, t0: float):
        self.t0 = t0
        self.j = -1
        self.z = None
        self.skipped = 0

    def command(self, t: float, x) -> Command:
        if self.t0 is None:
            raise RuntimeError("call start(t0) first")
        eps = self.ctrl.p.epsilon
        j = int(math.floor((t - self.t0) / eps + 1e-9))
        new = j != self.j
        if new:
            if self.j >= 0 and j > self.j + 1:
                self.skipped += j - self.j - 1
            self.z = self.ctrl.x_to_z(x)
            self._bound = self.ctrl.bound(self.z)
            self.j = j
        tau = t - self.t0 - j * eps
        u1, u2 = self.ctrl.u(tau, self.z)
        f = self.ctrl.p.clip_fraction
        l1, l2 = f * self.ctrl.p.u1_max, f * self.ctrl.p.u2_max
        c1, c2 = min(max(u1, -l1), l1), min(max(u2, -l2), l2)
        return Command(c1, c2, j, tau, new, (c1 != u1) or (c2 != u2), self.z, self._bound)
