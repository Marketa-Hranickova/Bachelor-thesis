"""Symbolic check of Chapter 4 (Nilpotent approximation).

For the car with one trailer (4 states) and with two trailers (5 states) the
script
  (1) builds the matrix F(0) of brackets at the origin and the linear
      privileged coordinates y = F(0)^{-1} x,
  (2) applies the weighted-order test: substitute z_i -> eps^{w_i} z_i into
      the exact equation for dz_q/dt and expand in eps. The lowest power of
      eps must be w_q - 1; the coefficient of this power is the vector field
      of the nilpotent approximation,
  (3) for two trailers, determines the cubic correction beta from the
      requirement that the eps^2 term of dz_5/dt vanishes, and compares
      alpha, kappa, theta, beta with the closed-form expressions,
  (4) verifies that the approximating fields generate a nilpotent Lie algebra
      with det G(z) = 1.

Weights: (1,1,2,3) for one trailer, (1,1,2,3,4) for two trailers.
Bracket convention (same as in the thesis):  [f, g] = (dg/dx) f - (df/dx) g

Requires sympy (tested with 1.14).  Run:  python3 verify_nilpotent.py
"""
import textwrap
import sympy as sp

R = sp.Rational
eps = sp.Symbol('eps')
u1, u2 = sp.symbols('u1 u2')


def show(*parts):
    """print() with line wrapping, so the output fits a page of the thesis."""
    text = ' '.join(str(p) for p in parts)
    for line in text.split('\n'):
        print(textwrap.fill(line, width=76, subsequent_indent='      ') if line else '')



def bracket(f, g, X):
    return g.jacobian(X) * f - f.jacobian(X) * g


def exact_fields(n, d0, d1, l1, l2):
    """Fields f1, f2 of the n-state system (n = 4 or 5), see Chapter 3."""
    x = sp.symbols('x1:%d' % (n + 1))
    x3, x4 = x[2], x[3]
    f1 = [sp.cos(x3), sp.sin(x3), 0, sp.sin(x3 - x4) / l1]
    f2 = [0, 0, 1, -d0 * sp.cos(x3 - x4) / l1]
    if n == 5:
        x5 = x[4]
        f1.append(sp.sin(x3 - x5) / l2
                  - (d1 + l1) * sp.sin(x3 - x4) * sp.cos(x4 - x5) / (l1 * l2))
        f2.append(d0 * (d1 + l1) * sp.cos(x3 - x4) * sp.cos(x4 - x5) / (l1 * l2)
                  - d0 * sp.cos(x3 - x5) / l2)
    return sp.Matrix(x), sp.Matrix(f1), sp.Matrix(f2)


def weighted_expansion(n, d0, d1, l1, l2):
    """Weighted-order test. Returns (coefficients[q][k], beta symbol, z symbols).

    coefficients[q][k] is the coefficient of eps^k in dz_q/dt after the
    substitution z_i -> eps^{w_i} z_i.
    """
    w = [1, 1, 2, 3] if n == 4 else [1, 1, 2, 3, 4]
    X, f1, f2 = exact_fields(n, d0, d1, l1, l2)
    cols = [f1, f2, bracket(f1, f2, X)]
    for _ in range(n - 3):
        cols.append(bracket(f1, cols[-1], X))
    F0 = sp.Matrix.hstack(*cols).subs({xi: 0 for xi in X})
    y = F0.inv() * X                       # linear privileged coordinates
    beta = sp.Symbol('beta')
    z_of_x = list(y)
    if n == 5:                             # cubic correction, see Section 4.5
        z_of_x[4] = y[4] - beta * X[2]**3
    z = sp.symbols('z1:%d' % (n + 1))
    y_of_z = list(z)
    if n == 5:
        y_of_z[4] = z[4] + beta * z[1]**3
    scaled = [eps**w[i] * y_of_z[i] for i in range(n)]
    x_of_z = F0 * sp.Matrix(scaled)
    sub = {X[i]: x_of_z[i] for i in range(n)}
    coeffs = []
    for q in range(n):
        grad = sp.Matrix([[sp.diff(z_of_x[q], X[i]) for i in range(n)]])
        zdot = (grad * (u1 * f1 + u2 * f2))[0].subs(sub)
        series = sp.expand(sp.series(zdot, eps, 0, w[q] + 1).removeO())
        coeffs.append([sp.simplify(series.coeff(eps, k)) for k in range(w[q] + 1)])
    return coeffs, beta, z, F0


def closed_form_coefficients(d0, d1, l1, l2):
    alpha = (d0**3 * (l1**3 + l1**2 * l2 - l1 * l2**2 - d1 * l2 * (d1 + l2))
             + d0 * l1**3 * (d0 * (l1 + 2 * l2) + l2 * (l1 + l2))) \
        / (2 * (d0 + l2) * l1**3)
    kappa = d0**2 * (R(3, 2) * l2**2 * (d0 + l1) * (d1 + l1)
                     - l2 * (l1**3 + d0 * l1**2 / 2
                             - d1 * l1 * (d0 + R(3, 2) * d1) - R(3, 2) * d0 * d1**2)
                     + d0 * d1**2 * l1 / 2) / (3 * (d0 + l2) * l1**3)
    theta = d0**2 * (l2**2 * (d1 + l1) * (d0 + l1)
                     + (d0 * d1 + (d0 + d1) * l1) * d1 * l2
                     + d0 * d1**2 * l1) / ((d0 + l2) * l1**3)
    beta = d0 * l2 * (d0**2 * (d1 + l1) * (l1 - l2 - d1)
                      + l1**3 * (d0 + l2)) / (6 * (d0 + l2) * l1**2)
    return alpha, kappa, theta, beta


# ------------------------------------------------------------ one trailer
show('=== ONE TRAILER (symbolic in d0, l1) ===')
d0, l1 = sp.symbols('d0 l1', positive=True)
coeffs, _, z, F0 = weighted_expansion(4, d0, 0, l1, 1)
show('F(0) =', F0.tolist())
show('F(0)^{-1} x =', [sp.factor(c) for c in (F0.inv() * sp.Matrix(sp.symbols('x1:5')))])
for q, c in enumerate(coeffs):
    show('dz%d/dt, coefficients of eps^0..%d:' % (q + 1, len(c) - 1),
          [sp.factor(ck) for ck in c])
alpha1 = d0 * (d0 + l1) / 2
show('coefficient of u2*z2^2 in dz4/dt is -alpha with alpha = d0(d0+l1)/2:',
      sp.simplify(coeffs[3][2].coeff(u2, 1).coeff(sp.Symbol('z2'), 2) + alpha1) == 0)

# nilpotent approximation, one trailer
Z4 = sp.Matrix(sp.symbols('z1:5'))
zz1, zz2, zz3, zz4 = Z4
a = sp.Symbol('alpha')
g1 = sp.Matrix([1, 0, -zz2, -zz3])
g2 = sp.Matrix([0, 1, 0, -a * zz2**2])
b12 = bracket(g1, g2, Z4)
b112 = bracket(g1, b12, Z4)
show('[g1,g2] =', list(b12), ' [g1,[g1,g2]] =', list(b112))
show('[g2,[g1,g2]] =', list(bracket(g2, b12, Z4)),
      ' all brackets of length 4 vanish:',
      all(bracket(p, q, Z4) == sp.zeros(4, 1)
          for p in (g1, g2) for q in (b112, bracket(g2, b12, Z4))))
G = sp.Matrix.hstack(g1, g2, b12, b112)
show('det G(z) =', sp.simplify(G.det()))
gam = sp.Symbol('gamma')
show('a(z) = -gamma G^{-1} z =',
      [sp.factor(sp.simplify(c)) for c in (-gam * G.inv() * Z4)])
show('limit d0 -> 0: alpha =', alpha1.subs(d0, 0), '(chained form: dz4/dt = -z3 u1)')

# ------------------------------------------------------------ two trailers
show('\n=== TWO TRAILERS (exact rational parameter sets) ===')
param_sets = [(1, 1, 1, 1),
              (R(1, 10), R(1, 10), 1, 1),
              (R(23, 200), R(21, 250), R(41, 500), R(89, 1000))]   # d0, d1, l1, l2
for vals in param_sets:
    vals = tuple(sp.sympify(v) for v in vals)
    coeffs, beta, z, F0 = weighted_expansion(5, *vals)
    beta_sol = sp.solve(coeffs[4][2], beta)[0]    # remove the eps^2 term of dz5/dt
    zdot = [[sp.simplify(c.subs(beta, beta_sol)) for c in row] for row in coeffs]
    al, ka, th, be = closed_form_coefficients(*vals)
    z2, z3, z4 = z[1], z[2], z[3]
    ok = [
        sp.simplify(zdot[3][2].coeff(u2, 1).coeff(z2, 2) + al) == 0,
        sp.simplify(zdot[4][3].coeff(u1, 1).coeff(z2, 3) - ka) == 0,
        sp.simplify(zdot[4][3].coeff(u2, 1).coeff(z2, 1).coeff(z3, 1) + th) == 0,
        sp.simplify(beta_sol - be) == 0,
        zdot[4][:3] == [0, 0, 0],
    ]
    show('d0,d1,l1,l2 =', vals)
    show('  alpha, kappa, theta, beta =', al, ka, th, be)
    show('  float:', [float(v) for v in (al, ka, th, be)])
    show('  derived z-system agrees with closed forms (alpha,kappa,theta,beta,order):', ok)

Z5 = sp.Matrix(sp.symbols('z1:6'))
ka, th = sp.symbols('kappa theta')
g1 = sp.Matrix([1, 0, -Z5[1], -Z5[2], ka * Z5[1]**3 - Z5[3]])
g2 = sp.Matrix([0, 1, 0, -a * Z5[1]**2, -th * Z5[1] * Z5[2]])
b12 = bracket(g1, g2, Z5)
b112 = bracket(g1, b12, Z5)
b1112 = bracket(g1, b112, Z5)
G5 = sp.Matrix.hstack(g1, g2, b12, b112, b1112)
show('\ntwo-trailer approximation: det G(z) =', sp.simplify(G5.det()))
