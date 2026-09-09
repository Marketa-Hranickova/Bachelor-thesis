"""
controller5.py -- time-varying oscillating controller for the 5-state
off-hooked trailer system, with no ROS dependency.

Reference:
    A. Zuyev, V. Grushkovskaya, "Stabilization of a Nonholonomic Car Model
    with Off-Hooked Trailers", arXiv:2403.03341v1.
    Equations referred to below are from that paper.

This module is deliberately dependency-free (standard library only) so it can
be unit-tested on any machine and then imported by the ROS 2 node unchanged.

Structure mirrors stab/calculations.m so the two can be compared line by line.

Run directly to execute the self-tests:
    python3 controller5.py
"""

from math import cos, sin, copysign, sqrt, log, pi

# ---------------------------------------------------------------------------
# 1. Mechanical parameters -> coefficients of the nilpotent approximation
# ---------------------------------------------------------------------------

def offhooked_coeffs(d0, d1, l1, l2):
    """Coefficients alpha, kappa, theta of system (14) and beta of eq. (9).

    d0 -- robot wheel axle  -> 1st hitch axis
    l1 -- 1st hitch axis    -> trailer 1 axle
    d1 -- trailer 1 axle    -> 2nd hitch axis
    l2 -- 2nd hitch axis    -> trailer 2 axle
    """
    alpha = (d0**3 * (l1**3 + l1**2*l2 - l1*l2**2 - d1*l2*(d1 + l2))
             + d0*l1**3 * (d0*(l1 + 2*l2) + l2*(l1 + l2))) \
            / (2*(d0 + l2)*l1**3)

    kappa = d0**2 * (1.5*l2**2*(d0 + l1)*(d1 + l1)
                     - l2*(l1**3 + 0.5*d0*l1**2
                           - d1*l1*(d0 + 1.5*d1) - 1.5*d0*d1**2)
                     + 0.5*d0*d1**2*l1) \
            / (3*(d0 + l2)*l1**3)

    theta = d0**2 * (l2**2*(d1 + l1)*(d0 + l1)
                     + (d0*d1 + (d0 + d1)*l1)*d1*l2
                     + d0*d1**2*l1) \
            / ((d0 + l2)*l1**3)

    beta = (d0*l2 * (d0**2*(d1 + l1)*(l1 - l2 - d1) + l1**3*(d0 + l2))) \
           / (6*(d0 + l2)*l1**2)

    return alpha, kappa, theta, beta


# ---------------------------------------------------------------------------
# 2. Physical state -> privileged coordinates
# ---------------------------------------------------------------------------

def x_to_z(x, d0, d1, l1, l2):
    """Map x = [xC0, yC0, phi, theta1, theta2] to privileged z (eqs. 6 and 9)."""
    c4_x2 = (l1 + l2)
    c4_x3 = -(d0 * (d0*(l1**2 + l1*l2 + l2**2) + l1*l2*(l1 + l2))) \
            / ((d0 + l1)*(d0 + l2))
    c4_x4 = -(l2**2*(l1 + l2)*(d1 + l1) / ((d0 + l2)*(d1 + l2))
              + (l1**2*l2*(d0*d1 + (d0 + d1)*l1) + d0*d1*l1**3)
              / ((d0 + l1)*(d0 + l2)*(d1 + l2)))
    c4_x5 = -l2**4 / ((d0 + l2)*(d1 + l2))

    c5_x2 = -l1*l2
    c5_x3 = (d0*l1*l2 * (l1*l2 + d0*(l1 + l2))) / ((d0 + l1)*(d0 + l2))
    c5_x4 = (l1*l2**3*(d1 + l1) / ((d0 + l2)*(d1 + l2))
             + (l1**2*l2 * (l2*(d0*d1 + (d0 + d1)*l1) + d0*d1*l1))
             / ((d0 + l1)*(d0 + l2)*(d1 + l2)))
    c5_x5 = (l1*l2**4) / ((d0 + l2)*(d1 + l2))

    y4 = c4_x2*x[1] + c4_x3*x[2] + c4_x4*x[3] + c4_x5*x[4]
    y5 = c5_x2*x[1] + c5_x3*x[2] + c5_x4*x[3] + c5_x5*x[4]

    _, _, _, beta = offhooked_coeffs(d0, d1, l1, l2)

    return [x[0], x[2], -x[1], y4, y5 - beta*x[2]**3]


# ---------------------------------------------------------------------------
# 3. Vector fields of the nilpotent approximation (eq. 14)
# ---------------------------------------------------------------------------

def g1(z, alpha, kappa, theta):
    return [1.0, 0.0, -z[1], -z[2], kappa*z[1]**3 - z[3]]

def g2(z, alpha, kappa, theta):
    return [0.0, 1.0, 0.0, -alpha*z[1]**2, -theta*z[1]*z[2]]


# ---------------------------------------------------------------------------
# 4. State-dependent amplitudes  a(z) = -h * G^-1(z) * grad Q(z)   (eq. 17)
# ---------------------------------------------------------------------------

def h_one(z):
    """h = 1: the plain case used in the thesis."""
    return 1.0

def h_log(z):
    """h = |log ||z||^2|: the hyperexponential variant (other project)."""
    r2 = sum(v*v for v in z)
    return abs(log(max(r2, 1e-9)))


def amplitudes(z, alpha, kappa, theta, gamma, h=1.0):
    """Return (a1, a2, a12, a112, a1112).

    Closed form of -gamma*h*G^-1(z)*z, verified symbolically against the
    inverse of matrix (15).
    """
    a1    = -gamma*h * z[0]
    a2    = -gamma*h * z[1]
    a12   = -gamma*h * (z[0]*z[1] + z[2])
    a112  = -gamma*h * (alpha*z[1]**3 + z[0]*z[2] + z[3])
    a1112 = -gamma*h * ((alpha + 2*kappa - theta) * z[0]*z[1]**3
                        + (alpha + 3*kappa) * z[1]**2*z[2]
                        + z[0]*z[3] + z[4])
    return a1, a2, a12, a112, a1112


# ---------------------------------------------------------------------------
# 5. The controls (eq. 16)
# ---------------------------------------------------------------------------

class Params:
    """Everything the control law needs, in one place."""

    def __init__(self, d0, d1, l1, l2,
                 gamma=1.0, epsilon=1.0,
                 k12=8, k112=6, k1112=5, h_fun=h_one):
        self.d0, self.d1, self.l1, self.l2 = d0, d1, l1, l2
        self.alpha, self.kappa, self.theta, self.beta = \
            offhooked_coeffs(d0, d1, l1, l2)
        self.gamma = gamma
        self.epsilon = epsilon
        self.omega = 2*pi/epsilon
        self.k12, self.k112, self.k1112 = k12, k112, k1112
        self.h_fun = h_fun

    def __repr__(self):
        return ("Params(alpha=%.6f, kappa=%.6f, theta=%.6f, beta=%.6f, "
                "gamma=%.4g, eps=%.4g, omega=%.4g)"
                % (self.alpha, self.kappa, self.theta, self.beta,
                   self.gamma, self.epsilon, self.omega))


def controls(t, z_frozen, p):
    """u1, u2 at time t, with the amplitudes frozen at z_frozen (eq. 16)."""
    h = p.h_fun(z_frozen)
    a1, a2, a12, a112, a1112 = amplitudes(
        z_frozen, p.alpha, p.kappa, p.theta, p.gamma, h)

    w = p.omega
    t12   = sqrt(2*p.k12*w*abs(a12))
    t112  = (2*p.k112*w)**(2.0/3.0) * abs(a112)**(1.0/3.0)
    t1112 = (p.k1112*w)**(3.0/4.0) * abs(a1112)**(1.0/4.0)

    s12   = copysign(1.0, a12)
    s112  = copysign(1.0, a112)
    s1112 = copysign(1.0, a1112)

    u1 = (a1
          + t12   * cos(p.k12*w*t)
          + t112  * cos(p.k112*w*t)
          + 2.0*t1112 * cos(p.k1112*w*t))

    u2 = (a2
          + t12   * s12   * sin(p.k12*w*t)
          - 2.0*t112  * s112  * cos(2*p.k112*w*t)
          - 6.0*t1112 * s1112 * sin(3*p.k1112*w*t))

    return u1, u2


def control_bound(z, p):
    """Worst-case |u1|, |u2| over one period (triangle inequality).

    Cheap conservative check against the robot's velocity limits.
    """
    h = p.h_fun(z)
    a1, a2, a12, a112, a1112 = amplitudes(
        z, p.alpha, p.kappa, p.theta, p.gamma, h)
    w = p.omega
    t12   = sqrt(2*p.k12*w*abs(a12))
    t112  = (2*p.k112*w)**(2.0/3.0) * abs(a112)**(1.0/3.0)
    t1112 = (p.k1112*w)**(3.0/4.0) * abs(a1112)**(1.0/4.0)
    return (abs(a1) + t12 + t112 + 2*t1112,
            abs(a2) + t12 + 2*t112 + 6*t1112)


# ---------------------------------------------------------------------------
# 6. Closed-loop simulation with the sampling scheme
# ---------------------------------------------------------------------------

def _rhs(t, z, z_frozen, p):
    u1, u2 = controls(t, z_frozen, p)
    f1 = g1(z, p.alpha, p.kappa, p.theta)
    f2 = g2(z, p.alpha, p.kappa, p.theta)
    return [u1*a + u2*b for a, b in zip(f1, f2)]


def simulate(z0, p, t_end, dt=None):
    """Sampling scheme: amplitudes frozen on each interval of length epsilon.

    Returns (ts, zs, us) with us = list of (u1, u2).
    Fixed-step RK4; dt defaults to epsilon/2000.
    """
    if dt is None:
        dt = p.epsilon/2000.0

    z = list(z0)
    t = 0.0
    ts, zs, us = [t], [list(z)], []

    while t < t_end - 1e-15:
        z_frozen = list(z)                      # freeze at the start of the interval
        t_next = min(t + p.epsilon, t_end)
        while t < t_next - 1e-15:
            step = min(dt, t_next - t)
            k1 = _rhs(t,          z,                                    z_frozen, p)
            k2 = _rhs(t + step/2, [zi + step/2*ki for zi, ki in zip(z, k1)], z_frozen, p)
            k3 = _rhs(t + step/2, [zi + step/2*ki for zi, ki in zip(z, k2)], z_frozen, p)
            k4 = _rhs(t + step,   [zi + step*ki   for zi, ki in zip(z, k3)], z_frozen, p)
            z = [zi + step/6.0*(a + 2*b + 2*c + d)
                 for zi, a, b, c, d in zip(z, k1, k2, k3, k4)]
            t += step
            ts.append(t); zs.append(list(z))
            us.append(controls(t, z_frozen, p))

    return ts, zs, us


# ---------------------------------------------------------------------------
# 7. Self-tests
# ---------------------------------------------------------------------------

def _self_test():
    ok = True

    # (a) coefficients against the two parameter sets used in the paper
    print("=== A. Coefficients vs. the paper ===")
    for (d0, d1, l1, l2), exp in [((0.1, 0.1, 1, 1), (0.10495, 0.0024, 0.0112)),
                                  ((1.0, 1.0, 1, 1), (1.0, 1.5, 4.0))]:
        a, k, th, b = offhooked_coeffs(d0, d1, l1, l2)
        good = (abs(a - exp[0]) < 1e-4 and abs(k - exp[1]) < 1e-4
                and abs(th - exp[2]) < 1e-4)
        ok &= good
        print("  d0=%.1f d1=%.1f l1=%.1f l2=%.1f -> alpha=%.6f kappa=%.6f "
              "theta=%.6f beta=%.6f   %s"
              % (d0, d1, l1, l2, a, k, th, b, "OK" if good else "FAIL"))

    # (b) reference points for a line-by-line comparison against MATLAB.
    #     Paste the same numbers into calculations.m and diff.
    print("\n=== B. Reference values -- compare these with MATLAB ===")
    p = Params(1.0, 1.0, 1.0, 1.0, gamma=1.0, epsilon=0.1,
               k12=8, k112=6, k1112=5)
    print("  " + repr(p))
    z = [1.0, -2.0, 2.0, -1.0, 1.0]
    print("  z = %s" % z)
    a = amplitudes(z, p.alpha, p.kappa, p.theta, p.gamma, 1.0)
    print("  a1=%.10f  a2=%.10f  a12=%.10f" % a[:3])
    print("  a112=%.10f  a1112=%.10f" % a[3:])
    for t in (0.0, 0.01, 0.025, 0.05):
        u1, u2 = controls(t, z, p)
        print("  t=%.3f   u1=%18.10f   u2=%18.10f" % (t, u1, u2))
    b1, b2 = control_bound(z, p)
    print("  worst-case |u1| <= %.4f,  |u2| <= %.4f" % (b1, b2))

    # (c) sanity: with u frozen and h=1 the flow must preserve det G = 1,
    #     i.e. the map is volume preserving -- check the fields are consistent
    print("\n=== C. Structural check: det G(z) == 1 ===")
    det = _det_G([0.3, -0.7, 0.2, 1.1, -0.4], p)
    good = abs(det - 1.0) < 1e-12
    ok &= good
    print("  det G = %.15f   %s" % (det, "OK" if good else "FAIL"))

    print("\n%s" % ("ALL SELF-TESTS PASSED" if ok else "SELF-TESTS FAILED"))
    return ok


def _det_G(z, p):
    """Determinant of matrix (15); must be identically 1."""
    al, ka, th = p.alpha, p.kappa, p.theta
    G = [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [-z[1], 0, 1, 0, 0],
         [-z[2], -al*z[1]**2, 0, 1, 0],
         [ka*z[1]**3 - z[3], -th*z[1]*z[2], -(al + 3*ka - th)*z[1]**2, 0, 1]]
    # Gaussian elimination
    n = 5
    M = [row[:] for row in G]
    det = 1.0
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        if piv != i:
            M[i], M[piv] = M[piv], M[i]
            det = -det
        det *= M[i][i]
        for r in range(i+1, n):
            f = M[r][i]/M[i][i]
            for c in range(i, n):
                M[r][c] -= f*M[i][c]
    return det


if __name__ == "__main__":
    _self_test()
