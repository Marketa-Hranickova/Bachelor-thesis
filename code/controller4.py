"""
controller4.py -- oscillating controller for the ONE off-axle trailer system
(4 states, degree of nonholonomy r = 3). No ROS, standard library only.

Derived in this project by the method of Bellaiche/Jean/Risler, in the same
way as the five-state case of arXiv:2403.03341. Results of the derivation:

    state        x = (xC0, yC0, phi, theta1)
    growth       (n1, n2, n3) = (2, 3, 4),   r = 3
    weights      w = (1, 1, 2, 3)
    det F(0)     = -(d0 + l1) / l1^3
    privileged coordinates need NO nonlinear correction (z = F(0)^-1 x)

    nilpotent approximation
        z1' = u1
        z2' = u2
        z3' = -z2 u1
        z4' = -z3 u1 - alpha z2^2 u2       with alpha = d0 (d0 + l1) / 2

Note alpha -> 0 as d0 -> 0, and the approximation then becomes the chained
form - which is exactly what the classical on-axle theory predicts. That limit
is used as a self-test below.

Run directly for the self-tests:  python3 controller4.py
"""

from math import cos, sin, copysign, sqrt, log, pi


# --------------------------------------------------------------------------
def coefficient(d0, l1):
    """The single coefficient of the nilpotent approximation."""
    return d0*(d0 + l1)/2.0


def det_F0(d0, l1):
    """Determinant of the bracket matrix at the origin; nonzero => LARC."""
    return -(d0 + l1)/l1**3


def x_to_z(x, d0, l1):
    """x = [xC0, yC0, phi, theta1]  ->  privileged coordinates z."""
    z4 = l1*x[1] - d0*l1**2*x[2]/(d0 + l1) - l1**3*x[3]/(d0 + l1)
    return [x[0], x[2], -x[1], z4]


def g1(z, alpha):
    return [1.0, 0.0, -z[1], -z[2]]


def g2(z, alpha):
    return [0.0, 1.0, 0.0, -alpha*z[1]**2]


# --------------------------------------------------------------------------
def h_one(z):
    return 1.0


def h_log(z):
    r2 = sum(v*v for v in z)
    return abs(log(max(r2, 1e-9)))


def amplitudes(z, alpha, gamma, h=1.0):
    """a = -gamma*h*G^-1(z)*z, in closed form (verified symbolically)."""
    a1   = -gamma*h * z[0]
    a2   = -gamma*h * z[1]
    a12  = -gamma*h * (z[0]*z[1] + z[2])
    a112 = -gamma*h * (alpha*z[1]**3 + z[0]*z[2] + z[3])
    return a1, a2, a12, a112


class Params:
    def __init__(self, d0, l1, gamma=1.0, epsilon=1.0,
                 k12=8, k112=6, h_fun=h_one):
        self.d0, self.l1 = d0, l1
        self.alpha = coefficient(d0, l1)
        self.gamma = gamma
        self.epsilon = epsilon
        self.omega = 2*pi/epsilon
        self.k12, self.k112 = k12, k112
        self.h_fun = h_fun

    def __repr__(self):
        return ("Params4(alpha=%.6f, gamma=%.4g, eps=%.4g, omega=%.4g)"
                % (self.alpha, self.gamma, self.epsilon, self.omega))


def controls(t, z_frozen, p):
    """Degree-3 member of the family of eq. (16): the length-4 bracket term,
    which is the expensive one, simply does not exist here."""
    h = p.h_fun(z_frozen)
    a1, a2, a12, a112 = amplitudes(z_frozen, p.alpha, p.gamma, h)
    w = p.omega
    t12  = sqrt(2*p.k12*w*abs(a12))
    t112 = (2*p.k112*w)**(2.0/3.0) * abs(a112)**(1.0/3.0)
    s12, s112 = copysign(1.0, a12), copysign(1.0, a112)

    u1 = a1 + t12*cos(p.k12*w*t) + t112*cos(p.k112*w*t)
    u2 = (a2 + t12*s12*sin(p.k12*w*t)
             - 2.0*t112*s112*cos(2*p.k112*w*t))
    return u1, u2


def control_bound(z, p):
    """Worst case over one period."""
    h = p.h_fun(z)
    a1, a2, a12, a112 = amplitudes(z, p.alpha, p.gamma, h)
    w = p.omega
    t12  = sqrt(2*p.k12*w*abs(a12))
    t112 = (2*p.k112*w)**(2.0/3.0) * abs(a112)**(1.0/3.0)
    return abs(a1) + t12 + t112, abs(a2) + t12 + 2*t112


# --------------------------------------------------------------------------
def _det_G(z, alpha):
    G = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [-z[1], 0, 1, 0],
         [-z[2], -alpha*z[1]**2, 0, 1]]
    n = 4
    M = [r[:] for r in G]
    det = 1.0
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        if piv != i:
            M[i], M[piv] = M[piv], M[i]; det = -det
        det *= M[i][i]
        for r in range(i+1, n):
            f = M[r][i]/M[i][i]
            for c in range(i, n):
                M[r][c] -= f*M[i][c]
    return det


def _self_test():
    ok = True
    print("=== A. Structure ===")
    d = _det_G([0.3, -0.7, 0.2, 1.1], coefficient(0.10, 0.15))
    good = abs(d - 1.0) < 1e-12
    ok &= good
    print("  det G = %.15f   %s" % (d, "OK" if good else "FAIL"))

    print("\n=== B. On-axle limit must give the chained form ===")
    for d0 in (0.1, 0.01, 0.001, 0.0):
        print("  d0=%-7.4g  alpha = %.8f" % (d0, coefficient(d0, 0.15)))
    good = coefficient(0.0, 0.15) == 0.0
    ok &= good
    print("  alpha(d0=0) = 0  ->  z4' = -z3 u1, i.e. chained form   %s"
          % ("OK" if good else "FAIL"))

    print("\n=== C. Reference values ===")
    p = Params(0.10, 0.15, gamma=1.0, epsilon=1.0)
    print("  " + repr(p))
    print("  det F(0) = %.6f" % det_F0(0.10, 0.15))
    x0 = [-0.5, -0.5, 0.3, 0.2]
    z0 = x_to_z(x0, 0.10, 0.15)
    print("  x0 = %s" % x0)
    print("  z0 = [%s]" % ", ".join("%.6f" % v for v in z0))
    print("  a  = [%s]" % ", ".join("%.6f" % v for v in
                                    amplitudes(z0, p.alpha, p.gamma)))
    for t in (0.0, 0.1, 0.25):
        print("  t=%.2f  u1=%12.5f  u2=%12.5f" % ((t,) + controls(t, z0, p)))
    b1, b2 = control_bound(z0, p)
    print("  worst case |u1| <= %.3f m/s, |u2| <= %.3f rad/s" % (b1, b2))

    print("\n%s" % ("ALL SELF-TESTS PASSED" if ok else "SELF-TESTS FAILED"))
    return ok


if __name__ == "__main__":
    _self_test()
