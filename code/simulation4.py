"""
simulation4.py -- closed-loop simulation of the ONE off-axle trailer system.

Two things are tested that the analysis alone cannot answer:

  1) does the controller actually drive the TRUE (trigonometric) system to the
     target, not just the nilpotent approximation it was designed on?
  2) does a state-dependent gain beat a fixed one? The required amplitude falls
     as the vehicle approaches the target, so a fixed gain -- which must be
     chosen for the worst moment, the start -- wastes authority for the whole
     run.

The plant integrated here is the exact system

    x1' = u1 cos x3
    x2' = u1 sin x3
    x3' = u2
    x4' = ( u1 sin(x3-x4) - d0 u2 cos(x3-x4) ) / l1

The controller only ever sees x, maps it to privileged coordinates, and emits
u -- exactly what the ROS node will do.

    python3 simulation4.py
"""

import os
from math import cos, sin, sqrt, hypot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import controller4 as c4

U1_MAX, U2_MAX = 0.22, 2.84
OUT = "figures4"

# ---------------------------------------------------------------- house style
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DASH   = [(None, None), (5, 2), (1.6, 1.6), (7, 2, 1.6, 2)]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
plt.rcParams.update({
    "figure.figsize": (5.6, 3.4), "figure.dpi": 130,
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6, "savefig.bbox": "tight",
})

def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_axisbelow(True)

def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("  wrote %s/%s.pdf" % (OUT, name))


# ---------------------------------------------------------------- the plant
def plant(x, u1, u2, d0, l1):
    return [u1*cos(x[2]),
            u1*sin(x[2]),
            u2,
            (u1*sin(x[2]-x[3]) - d0*u2*cos(x[2]-x[3]))/l1]


def gamma_max(z, p_template, lo=1e-14, hi=10.0, iters=60):
    """Largest gain that keeps both limits at this state."""
    for _ in range(iters):
        mid = sqrt(lo*hi)
        p = c4.Params(p_template.d0, p_template.l1, gamma=mid,
                      epsilon=p_template.epsilon,
                      k12=p_template.k12, k112=p_template.k112)
        b1, b2 = c4.control_bound(z, p)
        lo, hi = (mid, hi) if (b1 <= U1_MAX and b2 <= U2_MAX) else (lo, mid)
    return lo


def run(x0, d0, l1, epsilon, k12, k112, t_end,
        adaptive=True, gamma_fixed=None, steps_per_period=400,
        gamma_cap=None, cap_product=0.8, stop_pos=None, stop_ang=None):
    """Sampling scheme: amplitudes frozen on each interval of length epsilon."""
    # The cap belongs on the PRODUCT eps*gamma, not on gamma: that product is
    # the fraction of the remaining error removed per sampling step, so it is
    # what decides whether the step overshoots.
    if gamma_cap is None:
        gamma_cap = cap_product/epsilon
    tmpl = c4.Params(d0, l1, gamma=1.0, epsilon=epsilon, k12=k12, k112=k112)
    x = list(x0); t = 0.0
    ts, xs, us, gs = [0.0], [list(x)], [], []
    dt = epsilon/steps_per_period

    while t < t_end - 1e-12:
        z = c4.x_to_z(x, d0, l1)
        g = min(gamma_max(z, tmpl), gamma_cap) if adaptive else gamma_fixed
        p = c4.Params(d0, l1, gamma=g, epsilon=epsilon, k12=k12, k112=k112)
        z_frozen = list(z)
        t_next = min(t + epsilon, t_end)
        while t < t_next - 1e-12:
            h = min(dt, t_next - t)
            def rhs(tt, xx):
                u1, u2 = c4.controls(tt, z_frozen, p)
                return plant(xx, u1, u2, d0, l1)
            k1 = rhs(t, x)
            k2 = rhs(t+h/2, [a+h/2*b for a, b in zip(x, k1)])
            k3 = rhs(t+h/2, [a+h/2*b for a, b in zip(x, k2)])
            k4 = rhs(t+h,   [a+h*b   for a, b in zip(x, k3)])
            x = [a + h/6*(b+2*c+2*d+e)
                 for a, b, c, d, e in zip(x, k1, k2, k3, k4)]
            t += h
            u1, u2 = c4.controls(t, z_frozen, p)
            ts.append(t); xs.append(list(x)); us.append((u1, u2)); gs.append(g)
        if stop_pos is not None:
            pe, ae = err(x)
            if pe <= stop_pos and ae <= stop_ang:
                break
    return ts, xs, us, gs


def err(x):
    """position error [m] and the larger of the two angle errors [rad]"""
    return hypot(x[0], x[1]), max(abs(x[2]), abs(x[3]))


def settle_time(ts, xs, pos_tol, ang_tol):
    """last time the trajectory is still outside the tolerance box"""
    t_last = None
    for t, x in zip(ts, xs):
        p, a = err(x)
        if p > pos_tol or a > ang_tol:
            t_last = t
    return t_last


# ---------------------------------------------------------------- scenarios
def main():
    os.makedirs(OUT, exist_ok=True)
    D0, L1 = 0.10, 0.15
    K12, K112 = 2, 3
    EPS = 4.0
    X0 = [-0.05, -0.05, 0.15, 0.10]      # 7 cm off target, angles in radians
    T_END = 600.0
    POS_TOL, ANG_TOL = 0.005, 0.05

    z0 = c4.x_to_z(X0, D0, L1)
    tmpl = c4.Params(D0, L1, gamma=1.0, epsilon=EPS, k12=K12, k112=K112)
    g_fixed = gamma_max(z0, tmpl)
    print("Scenario: d0=%.2f l1=%.2f, eps=%.0f s, k=(%d,%d)" % (D0, L1, EPS, K12, K112))
    print("  start %.3f m off target, angles %.2f / %.2f rad" % (hypot(*X0[:2]), X0[2], X0[3]))
    print("  fixed gain admissible at the start: %.4e" % g_fixed)

    print("\nrunning fixed-gain ...")
    tf, xf, uf, gf = run(X0, D0, L1, EPS, K12, K112, T_END,
                         adaptive=False, gamma_fixed=g_fixed)
    print("running adaptive-gain ...")
    ta, xa, ua, ga = run(X0, D0, L1, EPS, K12, K112, T_END, adaptive=True)

    for name, ts, xs, us, gs in (("fixed", tf, xf, uf, gf),
                                 ("adaptive", ta, xa, ua, ga)):
        p, a = err(xs[-1])
        st = settle_time(ts, xs, POS_TOL, ANG_TOL)
        print("  %-9s final error %.4f m / %.4f rad | settle %s | max|u1| %.3f | max|u2| %.3f | gamma %.2e..%.2e"
              % (name, p, a, ("%.0f s" % st) if st and st < T_END-1 else "not reached",
                 max(abs(u[0]) for u in us), max(abs(u[1]) for u in us),
                 min(gs), max(gs)))

    # ---- fig 1: convergence -------------------------------------------------
    fig, ax = plt.subplots()
    ax.semilogy(tf, [max(err(x)[0], 1e-6) for x in xf], color=SERIES[0],
                label="fixed gain")
    ax.semilogy(ta, [max(err(x)[0], 1e-6) for x in xa], color=SERIES[2],
                dashes=DASH[2], label="state-dependent gain")
    ax.axhline(POS_TOL, color=INK2, lw=1.0, dashes=(2, 2))
    ax.annotate("tolerance 5 mm", xy=(0, POS_TOL), xytext=(4, 4),
                textcoords="offset points", color=INK2, fontsize=7.5)
    ax.set_ylim(1e-5, 2e-1)
    style(ax, "Position error, true (non-approximated) system",
          "time [s]", "distance to target [m]")
    ax.legend(loc="upper right")
    save(fig, "sim1_convergence")

    # ---- fig 2: control effort ---------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.2), sharex=True)
    axes[0].plot(ta[1:], [u[0] for u in ua], color=SERIES[0], lw=0.7)
    axes[0].axhline(U1_MAX, color=INK2, lw=1.0, dashes=(2, 2))
    axes[0].axhline(-U1_MAX, color=INK2, lw=1.0, dashes=(2, 2))
    style(axes[0], "Control signals, state-dependent gain", "", r"$u_1$  [m/s]")
    axes[1].plot(ta[1:], [u[1] for u in ua], color=SERIES[1], lw=0.7)
    axes[1].axhline(U2_MAX, color=INK2, lw=1.0, dashes=(2, 2))
    axes[1].axhline(-U2_MAX, color=INK2, lw=1.0, dashes=(2, 2))
    style(axes[1], "", "time [s]", r"$u_2$  [rad/s]")
    fig.text(0.5, -0.04, "Dashed lines are the TurtleBot3 limits; both signals stay inside them.",
             fontsize=7.5, color=INK2, ha="center")
    save(fig, "sim2_controls")

    # ---- fig 3: the gain over time -----------------------------------------
    fig, ax = plt.subplots()
    ax.semilogy(ta[1:], ga, color=SERIES[2])
    ax.axhline(g_fixed, color=SERIES[0], lw=1.2, dashes=(5, 2))
    ax.annotate("fixed gain (set for the start)", xy=(ta[-1], g_fixed),
                xytext=(-4, 5), textcoords="offset points", ha="right",
                color=SERIES[0], fontsize=7.5)
    style(ax, "The state-dependent gain grows as the target is approached",
          "time [s]", r"$\gamma$")
    save(fig, "sim3_gain")

    # ---- fig 4: how long may the oscillation period be? --------------------
    print("\nsweeping the oscillation period ...")
    eps_list = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    T_SWEEP = 6000.0
    settles, finals = [], []
    for e in eps_list:
        ts, xs, us, gs = run(X0, D0, L1, e, K12, K112, T_SWEEP, adaptive=True,
                             steps_per_period=200, cap_product=0.8,
                             stop_pos=POS_TOL, stop_ang=ANG_TOL)
        p, a = err(xs[-1])
        reached = (p <= POS_TOL and a <= ANG_TOL)
        settles.append(ts[-1] if reached else float("nan"))
        finals.append(p)
        print("  eps=%5.1f s -> %s  (final %.4f m / %.4f rad, max|u1| %.3f)"
              % (e, ("tolerance reached after %6.0f s" % ts[-1]) if reached
                    else "NOT reached within %.0f s" % T_SWEEP,
                 p, a, max(abs(u[0]) for u in us)))

    fig, ax = plt.subplots()
    ok = [(e, s_) for e, s_ in zip(eps_list, settles) if s_ == s_]
    best = min(ok, key=lambda q: q[1])
    ax.loglog([e for e, _ in ok], [s_ for _, s_ in ok],
              color=SERIES[0], marker="o", ms=4)
    ax.plot([best[0]], [best[1]], marker="o", ms=9, mfc="none",
            mec=SERIES[1], mew=1.6, linestyle="none")
    ax.annotate("optimum  %.0f s  at  $\\varepsilon$ = %.0f s" % (best[1], best[0]),
                xy=best, xytext=(-8, 16), textcoords="offset points",
                ha="right", fontsize=8, color=SERIES[1])
    ax.annotate("amplitude-limited:\nthe velocity limits cap $\\gamma$",
                xy=(1.15, 900), fontsize=7.5, color=INK2, va="center")
    ax.annotate("step-limited:\n$\\varepsilon\\gamma$ must stay below ~1",
                xy=(22, 300), fontsize=7.5, color=INK2, va="center")
    style(ax, "Two regimes, and an optimum where they meet",
          r"oscillation period $\varepsilon$  [s]",
          "time to reach 5 mm / 0.05 rad  [s]")
    save(fig, "sim4_period_sweep")

    print("\nFigures in %s/ (pdf for LaTeX, png for preview)." % OUT)


if __name__ == "__main__":
    main()
