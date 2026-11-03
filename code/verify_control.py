"""Check of Chapter 5 (Oscillating control design).

The nilpotent approximation of the car with one trailer,
    z1' = u1,  z2' = u2,  z3' = -z2 u1,  z4' = -z3 u1 - alpha z2^2 u2,
is integrated EXACTLY over one sampling interval [0, eps] under the controller
    u1 = a1 + T12 cos(k12 w t) + T112 cos(k112 w t)
    u2 = a2 + T12 s12 sin(k12 w t) - 2 T112 s112 cos(2 k112 w t),
    T12 = sqrt(2 k12 w |a12|),  T112 = (2 k112 w)^(2/3) |a112|^(1/3),  w = 2 pi/eps,
with the amplitudes a1, a2, a12, a112 frozen at the state z at t = 0. The script
  (1) derives by hand-checkable integrals the mean drift of the two oscillating terms
      (they must equal a12 and a112),
  (2) integrates the system symbolically over one interval and shows
          z(eps) = z + eps * G(z) a + R,
      with R = 0 when a1 = a2 = 0 (multipliers (3,2)), and gives R_3 in closed form,
  (3) scans the multiplier pairs (k12, k112), k <= 12, numerically for a spurious
      term in the one-interval displacement (non-resonance),
  (4) for the parameters of params.m (1 trailer) compares the displacement over one
      interval with eps * G(z0) a(z0) for several gains.

Here G(z) = (g1, g2, [g1,g2], [g1,[g1,g2]]); G(z) a = (a1, a2, a12 - z2 a1,
a112 - z3 a1 - alpha z2^2 a2)^T.
Requires: sympy, numpy, scipy.   Run:  python3 verify_control.py
"""
import textwrap
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

t, eps = sp.symbols('t eps', positive=True)


def show(*parts):
    for line in ' '.join(str(p) for p in parts).split('\n'):
        print(textwrap.fill(line, width=76, subsequent_indent='      ') if line else '')


# ------------------------------------------------------------------ (1)
show('=== (1) mean drift of the oscillating terms (per unit time) ===')
k, w, T, s = sp.symbols('k w T s', positive=True)
th = k * w * t
# level 2: u1 = T cos(k w t), u2 = T s sin(k w t)
z2 = sp.integrate(T * s * sp.sin(k * w * t), t)
rate2 = sp.simplify(-z2 * T * sp.cos(th))
mean2 = sp.simplify(sp.integrate(rate2.subs(t, t), (t, 0, 2 * sp.pi / (k * w))) * k * w / (2 * sp.pi))
show('level 2: mean of dz3/dt =', sp.simplify(mean2 + 0), ' (= a12 if T^2 = 2 k w |a12|, s = sign a12)')
# level 3: u1 = T cos(k w t), u2 = -2 T s cos(2 k w t)
z2b = sp.integrate(-2 * T * s * sp.cos(2 * th), t)
z3b = sp.integrate(-z2b * T * sp.cos(th), t)
rate3 = sp.simplify(-z3b * T * sp.cos(th))
mean3 = sp.simplify(sp.integrate(rate3, (t, 0, 2 * sp.pi / (k * w))) * k * w / (2 * sp.pi))
show('level 3: mean of dz4/dt =', mean3, ' (= a112 if T^3 = 4 k^2 w^2 |a112|, s = sign a112)')
show('                      and z3 returns to its start after one interval:',
     sp.simplify(z3b.subs(t, 2 * sp.pi / (k * w)) - z3b.subs(t, 0)) == 0)

# ------------------------------------------------------------------ (2)
show('\n=== (2) one-interval map, exact integration, multipliers (k12,k112) = (3,2) ===')
k12, k112 = 3, 2
om = 2 * sp.pi / eps
a1, a2 = sp.symbols('a1 a2', real=True)
B12, B112, al = sp.symbols('B12 B112 alpha', positive=True)
z10, z20, z30, z40 = sp.symbols('z10 z20 z30 z40', real=True)


def one_interval(s12, s112, a1v, a2v):
    T12 = sp.sqrt(2 * k12 * om * B12)
    T112 = (2 * k112 * om)**sp.Rational(2, 3) * B112**sp.Rational(1, 3)
    u1 = a1v + T12 * sp.cos(k12 * om * t) + T112 * sp.cos(k112 * om * t)
    u2 = a2v + T12 * s12 * sp.sin(k12 * om * t) - 2 * T112 * s112 * sp.cos(2 * k112 * om * t)
    z1 = z10 + sp.integrate(u1, (t, 0, t))
    z2 = z20 + sp.integrate(u2, (t, 0, t))
    z3 = z30 - sp.integrate(sp.expand(z2 * u1), (t, 0, t))
    z4 = z40 - sp.integrate(sp.expand(z3 * u1 + al * z2**2 * u2), (t, 0, t))
    return [sp.simplify(zz.subs(t, eps)) for zz in (z1, z2, z3, z4)]


for s12, s112 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
    z = one_interval(s12, s112, 0, 0)
    a12, a112 = s12 * B12, s112 * B112
    target = [z10, z20, z30 + eps * a12, z40 + eps * a112]
    show('a1=a2=0, signs (s12,s112) =', (s12, s112), ': z(eps) - [z + eps*G*a] =',
         [sp.simplify(z[i] - target[i]) for i in range(4)])
z = one_interval(1, 1, a1, a2)
Ga = [a1, a2, -z20 * a1 + B12, -z30 * a1 - al * z20**2 * a2 + B112]
R3 = sp.simplify(z[2] - z30 - eps * Ga[2])
show('a1, a2 general, s12 = 1:  R3 = z3(eps) - z30 - eps*(G a)_3 =', R3)
show('closed form  R3 = -eps^2 a1 a2 / 2 - s12 a1 eps^(3/2) sqrt(|a12|/(pi k12)):',
     sp.simplify(R3 - (-eps**2 * a1 * a2 / 2 - a1 * eps**sp.Rational(3, 2) * sp.sqrt(B12 / (sp.pi * k12)))) == 0)
show('z1, z2 are exact: R1, R2 =', sp.simplify(z[0] - z10 - eps * a1), sp.simplify(z[1] - z20 - eps * a2))
R4 = sp.expand(sp.simplify(z[3] - z40 - eps * Ga[3]))
show('every term of R4 contains a1 or a2:',
     sp.simplify(R4.subs({a1: 0, a2: 0})) == 0)


# ------------------------------------------------------------------ (3)
def one_interval_num(k12, k112, B12=0.7, B112=0.5, alpha=0.9, s12=1, s112=1, epsn=2.0,
                     z0=(0.3, -0.4, 0.8, -0.6)):
    omn = 2 * np.pi / epsn
    T12 = np.sqrt(2 * k12 * omn * B12)
    T112 = (2 * k112 * omn)**(2 / 3) * B112**(1 / 3)

    def f(tt, zz):
        u1 = T12 * np.cos(k12 * omn * tt) + T112 * np.cos(k112 * omn * tt)
        u2 = T12 * s12 * np.sin(k12 * omn * tt) - 2 * T112 * s112 * np.cos(2 * k112 * omn * tt)
        return [u1, u2, -zz[1] * u1, -zz[2] * u1 - alpha * zz[1]**2 * u2]

    sol = solve_ivp(f, [0, epsn], z0, rtol=1e-12, atol=1e-13, method='DOP853')
    zz = sol.y[:, -1]
    return zz[2] - z0[2] - epsn * s12 * B12, zz[3] - z0[3] - epsn * s112 * B112


show('\n=== (3) multiplier pairs with a spurious term (a1 = a2 = 0, k12, k112 <= 12) ===')
bad = set()
for k12n in range(1, 13):
    for k112n in range(1, 13):
        for s in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            r3, r4 = one_interval_num(k12n, k112n, s12=s[0], s112=s[1])
            if abs(r3) > 1e-8 or abs(r4) > 1e-8:
                bad.add((k12n, k112n))
show('pairs (k12,k112) with a non-zero residual:', sorted(bad))
show('all of them satisfy k12 = k112 or k12 = 3*k112:',
     all(a == b or a == 3 * b for a, b in bad))

# ------------------------------------------------------------------ (4)
show('\n=== (4) one interval for the platform parameters (params.m, one trailer) ===')
d0, l1 = 0.115, 0.082
alpha = d0 * (d0 + l1) / 2
delta, k12n, k112n, epsn = 0.02, 3, 2, 5.0
x0 = np.array([-0.2, -0.2, np.deg2rad(45), np.deg2rad(45)])
z0 = np.array([x0[0], x0[2], -x0[1],
               l1 * x0[1] - d0 * l1**2 / (d0 + l1) * x0[2] - l1**3 / (d0 + l1) * x0[3]])
show('z0 =', np.round(z0, 5), ' |z0| =', round(float(np.linalg.norm(z0)), 4))
show('%10s %9s %12s %12s' % ('gamma', 'mu', '|R|/|eps G a|', 'max|u1| [m/s]'))
hh = 1 / np.sqrt(z0 @ z0 + delta**2)
for factor in (0.25, 0.5, 1, 2, 4, 8):
    gam = 3.2028e-3 * factor
    omn = 2 * np.pi / epsn
    a1n, a2n = -gam * hh * z0[0], -gam * hh * z0[1]
    a12n = -gam * hh * (z0[0] * z0[1] + z0[2])
    a112n = -gam * hh * (alpha * z0[1]**3 + z0[0] * z0[2] + z0[3])
    T12 = np.sqrt(2 * k12n * omn * abs(a12n))
    T112 = (2 * k112n * omn)**(2 / 3) * abs(a112n)**(1 / 3)
    s12, s112 = np.sign(a12n), np.sign(a112n)

    def f(tt, zz):
        u1 = a1n + T12 * np.cos(k12n * omn * tt) + T112 * np.cos(k112n * omn * tt)
        u2 = a2n + T12 * s12 * np.sin(k12n * omn * tt) - 2 * T112 * s112 * np.cos(2 * k112n * omn * tt)
        return [u1, u2, -zz[1] * u1, -zz[2] * u1 - alpha * zz[1]**2 * u2]

    tt = np.linspace(0, epsn, 2001)
    sol = solve_ivp(f, [0, epsn], z0, t_eval=tt, rtol=1e-11, atol=1e-13, method='DOP853')
    step = sol.y[:, -1] - z0
    Ga = np.array([a1n, a2n, -z0[1] * a1n + a12n, -z0[2] * a1n - alpha * z0[1]**2 * a2n + a112n])
    u1max = max(abs(a1n + T12 * np.cos(k12n * omn * x) + T112 * np.cos(k112n * omn * x)) for x in tt)
    show('%10.3e %9.4f %12.4f %12.3f' % (gam, epsn * gam * hh,
         np.linalg.norm(step - epsn * Ga) / np.linalg.norm(epsn * Ga), u1max))
