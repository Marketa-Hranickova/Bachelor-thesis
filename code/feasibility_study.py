"""
feasibility_study.py -- can the oscillating controller of
Zuyev & Grushkovskaya (arXiv:2403.03341) run on a TurtleBot3 Burger?

Produces five figures for the thesis and prints the numbers behind them.
Depends only on controller5.py (same folder) and matplotlib.

    python3 feasibility_study.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from controller5 import Params, control_bound, x_to_z, offhooked_coeffs, amplitudes
from math import pi, sqrt

# --------------------------------------------------------------------------
# Robot and experiment definition
# --------------------------------------------------------------------------
D0, D1, L1, L2 = 0.10, 0.10, 0.15, 0.15      # TODO: measured values
X0 = [-0.5, -0.5, 0.3, 0.2, 0.1]             # start pose, physical units
U1_MAX, U2_MAX = 0.22, 2.84                  # TurtleBot3 Burger limits
K = dict(k12=8, k112=6, k1112=5)

OUT = "figures"

# --------------------------------------------------------------------------
# House style -- thin marks, recessive grid, no chart junk
# --------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated, adjacent-pair safe
DASH   = [(None, None), (5, 2), (1.6, 1.6), (7, 2, 1.6, 2)]  # print / greyscale relief
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "figure.figsize": (5.6, 3.5), "figure.dpi": 130,
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
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("  wrote %s/%s.pdf" % (OUT, name))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def max_gamma(z, d0, d1, l1, l2, epsilon):
    """Largest gain for which BOTH velocity limits hold. Bisection in log-space."""
    lo, hi = 1e-18, 10.0
    for _ in range(200):
        mid = sqrt(lo*hi)
        p = Params(d0, d1, l1, l2, gamma=mid, epsilon=epsilon, **K)
        b1, b2 = control_bound(z, p)
        if b1 <= U1_MAX and b2 <= U2_MAX:
            lo = mid
        else:
            hi = mid
    return lo

def terms(z, p):
    """The four contributions to the |u2| bound, separately."""
    a1, a2, a12, a112, a1112 = amplitudes(z, p.alpha, p.kappa, p.theta, p.gamma, 1.0)
    w = p.omega
    t12   = sqrt(2*p.k12*w*abs(a12))
    t112  = (2*p.k112*w)**(2/3) * abs(a112)**(1/3)
    t1112 = (p.k1112*w)**(3/4) * abs(a1112)**(1/4)
    return abs(a2), t12, 2*t112, 6*t1112


def logspace(a, b, n):
    return [a*(b/a)**(i/(n-1)) for i in range(n)]

def linspace(a, b, n):
    return [a + (b-a)*i/(n-1) for i in range(n)]


# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    z0 = x_to_z(X0, D0, D1, L1, L2)
    al, ka, th, be = offhooked_coeffs(D0, D1, L1, L2)

    print("Geometry: d0=%.3f d1=%.3f l1=%.3f l2=%.3f m" % (D0, D1, L1, L2))
    print("  alpha=%.6f kappa=%.6f theta=%.6f beta=%.6f" % (al, ka, th, be))
    print("  z0 = [%s]" % ", ".join("%.5f" % v for v in z0))

    # ---------------- Figure 1: admissible gain vs oscillation period -------
    eps = logspace(0.05, 10.0, 60)
    g   = [max_gamma(z0, D0, D1, L1, L2, e) for e in eps]

    fig, ax = plt.subplots()
    ax.loglog(eps, g, color=SERIES[0])
    ax.annotate(r"$\gamma_{\max}$", xy=(eps[-1], g[-1]), xytext=(-4, 6),
                textcoords="offset points", ha="right", color=SERIES[0], fontsize=8)
    style(ax, "Largest gain that respects both velocity limits",
          r"oscillation period $\varepsilon$  [s]", r"admissible $\gamma$")
    fig.text(0.5, -0.06,
             "Every value lies below $10^{-6}$: no choice of period makes the "
             "published gain usable.", fontsize=7.5, color=INK2, ha="center")
    save(fig, "fig1_gamma_vs_epsilon")
    print("\nFigure 1: gamma_max at eps=0.1 -> %.2e ;  eps=1 -> %.2e ;  eps=10 -> %.2e"
          % (max_gamma(z0, D0, D1, L1, L2, 0.1),
             max_gamma(z0, D0, D1, L1, L2, 1.0),
             max_gamma(z0, D0, D1, L1, L2, 10.0)))

    # ---------------- Figure 2: which term is responsible -------------------
    labels = [r"$|a_2|$", r"bracket 2  ($\omega^{1/2}$)",
              r"bracket 3  ($\omega^{2/3}$)", r"bracket 4  ($\omega^{3/4}$)"]
    curves = [[], [], [], []]
    for e in eps:
        p = Params(D0, D1, L1, L2, gamma=1.0, epsilon=e, **K)
        for c, v in zip(curves, terms(z0, p)):
            c.append(max(v, 1e-12))

    fig, ax = plt.subplots()
    for c, lab, col, dash in zip(curves, labels, SERIES, DASH):
        ax.loglog(eps, c, label=lab, color=col,
                  dashes=dash if dash[0] else (1, 0))
    ax.axhline(U2_MAX, color=INK2, lw=1.0, dashes=(2, 2))
    ax.annotate("TurtleBot3 limit  2.84 rad/s", xy=(eps[0], U2_MAX),
                xytext=(2, 4), textcoords="offset points", color=INK2, fontsize=7.5)
    style(ax, r"Contributions to the $|u_2|$ bound  ($\gamma=1$)",
          r"oscillation period $\varepsilon$  [s]", "contribution  [rad/s]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    save(fig, "fig2_term_breakdown")
    p1 = Params(D0, D1, L1, L2, gamma=1.0, epsilon=1.0, **K)
    t = terms(z0, p1)
    print("Figure 2: at eps=1, gamma=1 the four terms are "
          "%.3f / %.3f / %.3f / %.3f rad/s (sum %.1f)" % (t + (sum(t),)))

    # ---------------- Figure 3: does geometry help? -------------------------
    ls  = linspace(0.05, 0.40, 40)
    d0s = linspace(0.02, 0.30, 40)
    gl  = [max_gamma(x_to_z(X0, D0, D1, l, l), D0, D1, l, l, 1.0) for l in ls]
    gd  = [max_gamma(x_to_z(X0, d, D1, L1, L2), d, D1, L1, L2, 1.0) for d in d0s]

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.9), sharey=True)
    axes[0].plot(ls, gl, color=SERIES[0])
    style(axes[0], "", r"$\ell_1=\ell_2=\ell$  [m]", r"admissible $\gamma$")
    axes[1].plot(d0s, gd, color=SERIES[1], dashes=DASH[1])
    style(axes[1], "", r"$d_0$  [m]", "")
    for a in axes:
        a.set_ylim(0, max(max(gl), max(gd))*1.15)
    fig.suptitle("Geometry is not the lever", x=0.075, y=1.04, ha="left", fontsize=9.5, color=INK)
    fig.text(0.5, -0.08,
             "Both axes stay within one order of magnitude of $10^{-8}$: no "
             "buildable geometry rescues the amplitudes.",
             fontsize=7.5, color=INK2, ha="center")
    save(fig, "fig3_geometry")
    print("Figure 3: gamma_max over l in [0.05,0.40] -> %.2e .. %.2e ; "
          "over d0 in [0.02,0.30] -> %.2e .. %.2e"
          % (min(gl), max(gl), min(gd), max(gd)))

    # ---------------- Figure 4: coefficient collapse as d0 -> 0 -------------
    A, Kp, T = [], [], []
    for d in d0s:
        a, k, t_, _ = offhooked_coeffs(d, D1, L1, L2)
        A.append(abs(a)); Kp.append(abs(k)); T.append(abs(t_))

    fig, ax = plt.subplots()
    for vals, lab, col, dash in zip([A, Kp, T], [r"$\alpha$", r"$\kappa$", r"$\theta$"],
                                    SERIES, DASH):
        ax.loglog(d0s, vals, label=lab, color=col, dashes=dash if dash[0] else (1, 0))
        ax.annotate(lab, xy=(d0s[-1], vals[-1]), xytext=(4, -2),
                    textcoords="offset points", color=col, fontsize=8)
    style(ax, r"Coefficients of the nilpotent approximation vs. hitch offset",
          r"$d_0$  [m]", "coefficient magnitude")
    ax.legend(loc="lower right")
    save(fig, "fig4_coefficients_vs_d0")
    a2_, k2_, t2_, _ = offhooked_coeffs(0.02, D1, L1, L2)
    a3_, k3_, t3_, _ = offhooked_coeffs(0.30, D1, L1, L2)
    print("Figure 4: from d0=0.02 to 0.30, alpha x%.0f, kappa x%.0f, theta x%.0f"
          % (a3_/a2_, k3_/k2_, t3_/t2_))

    # ---------------- Figure 5: why the r=2 case worked ---------------------
    gam = logspace(1e-9, 1e-1, 60)
    b_r4, b_r2 = [], []
    for gv in gam:
        p = Params(D0, D1, L1, L2, gamma=gv, epsilon=1.0, **K)
        a2v, t12, t112, t1112 = terms(z0, p)
        b_r4.append(a2v + t12 + t112 + t1112)
        b_r2.append(a2v + t12)                      # same formula, deep terms dropped

    fig, ax = plt.subplots()
    ax.loglog(gam, b_r4, label=r"degree $r=4$  (two trailers)", color=SERIES[0])
    ax.loglog(gam, b_r2, label=r"degree $r=2$  (unicycle)", color=SERIES[2], dashes=DASH[2])
    ax.axhline(U2_MAX, color=INK2, lw=1.0, dashes=(2, 2))
    ax.annotate("TurtleBot3 limit", xy=(gam[0], U2_MAX), xytext=(2, 4),
                textcoords="offset points", color=INK2, fontsize=7.5)
    style(ax, r"Bound on $|u_2|$ at $\varepsilon = 1$",
          r"gain $\gamma$", r"$|u_2|$ bound  [rad/s]")
    ax.legend(loc="upper left")
    fig.text(0.5, -0.06,
             "Identical control law; the $r=4$ curve carries two extra brackets. "
             "That gap is the whole difficulty.", fontsize=7.5, color=INK2, ha="center")
    save(fig, "fig5_r2_vs_r4")

    def maxg(bound_fn):
        lo, hi = 1e-18, 10.0
        for _ in range(200):
            mid = sqrt(lo*hi)
            p = Params(D0, D1, L1, L2, gamma=mid, epsilon=1.0, **K)
            lo, hi = (mid, hi) if bound_fn(terms(z0, p)) <= U2_MAX else (lo, mid)
        return lo
    g4 = maxg(lambda t: sum(t))
    g2 = maxg(lambda t: t[0] + t[1])
    print("Figure 5 (u2 constraint only): admissible gamma  r=4 -> %.2e ;  r=2 -> %.2e ;  ratio %.0f x"
          % (g4, g2, g2/g4))

    print("\nAll figures written to %s/ (pdf for LaTeX, png for preview)." % OUT)


if __name__ == "__main__":
    main()
