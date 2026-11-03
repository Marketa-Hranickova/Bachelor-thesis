"""Symbolic check of Chapter 3 (Kinematic model).

Re-derives the control-affine model of a car with two off-hooked trailers
directly from the rolling-without-slipping constraints, and verifies
  (1) the vector fields f1, f2 (and the one-trailer fields fb1, fb2),
  (2) det F(0) for F = (f1, f2, [f1,f2], [f1,[f1,f2]], [f1,[f1,[f1,f2]]]),
  (3) the growth vector at the origin: (2,3,4,5) and (2,3,4) for one trailer,
  (4) the closed-form brackets and det Fb of the one-trailer system.

Bracket convention (same as in the thesis):
    [f, g] = (dg/dx) f - (df/dx) g

Requires: sympy (tested with 1.14).   Run:  python3 verify_controllability.py
"""
import sympy as sp

x1, x2, x3, x4, x5 = sp.symbols('x1:6', real=True)
d0, d1, l1, l2 = sp.symbols('d0 d1 l1 l2', positive=True)
u1, u2, w4, w5 = sp.symbols('u1 u2 w4 w5')   # w4, w5: unknown x4', x5'
X = sp.Matrix([x1, x2, x3, x4, x5])
ORIGIN = {x1: 0, x2: 0, x3: 0, x4: 0, x5: 0}


def e(a):
    """Unit vector along heading a."""
    return sp.Matrix([sp.cos(a), sp.sin(a)])


def e_perp(a):
    """Unit vector orthogonal to heading a."""
    return sp.Matrix([-sp.sin(a), sp.cos(a)])


def bracket(f, g, vars_):
    """Lie bracket [f, g] = (dg/dx) f - (df/dx) g."""
    return g.jacobian(vars_) * f - f.jacobian(vars_) * g


# ---------------------------------------------------------------- 1. model
C0 = sp.Matrix([x1, x2])
C1 = C0 - d0 * e(x3) - l1 * e(x4)
C2 = C0 - d0 * e(x3) - (l1 + d1) * e(x4) - l2 * e(x5)

xdot = sp.Matrix([u1 * sp.cos(x3), u1 * sp.sin(x3), u2, w4, w5])


def rolling(C, theta):
    """Rolling-without-slipping constraint: C' . e_perp(theta) = 0."""
    return sp.expand((C.jacobian(X) * xdot).dot(e_perp(theta)))


sol4 = sp.solve(rolling(C1, x4), w4)[0]
sol5 = sp.solve(rolling(C2, x5).subs(w4, sol4), w5)[0]
sol4, sol5 = sp.expand(sol4), sp.expand(sol5)

f15 = sp.sin(x3 - x5) / l2 - (d1 + l1) * sp.sin(x3 - x4) * sp.cos(x4 - x5) / (l1 * l2)
f25 = (d0 * (d1 + l1) * sp.cos(x3 - x4) * sp.cos(x4 - x5) / (l1 * l2)
       - d0 * sp.cos(x3 - x5) / l2)
f1 = sp.Matrix([sp.cos(x3), sp.sin(x3), 0, sp.sin(x3 - x4) / l1, f15])
f2 = sp.Matrix([0, 0, 1, -d0 * sp.cos(x3 - x4) / l1, f25])

checks = {
    'x4_dot coefficient of u1': sol4.coeff(u1) - f1[3],
    'x4_dot coefficient of u2': sol4.coeff(u2) - f2[3],
    'x5_dot coefficient of u1': sol5.coeff(u1) - f1[4],
    'x5_dot coefficient of u2': sol5.coeff(u2) - f2[4],
}
for name, expr in checks.items():
    print(f'{name:28s} difference = {sp.simplify(expr)}')

# ---------------------------------------------- 2. det F(0), growth vector
b12 = bracket(f1, f2, X)
b112 = bracket(f1, b12, X)
b1112 = bracket(f1, b112, X)
F = sp.Matrix.hstack(f1, f2, b12, b112, b1112)
detF0 = sp.factor(sp.simplify(F.subs(ORIGIN).det()))
print('\ndet F(0) =', detF0)
claimed = -(d0 + l1) * (d0 + l2) * (d1 + l2) / (l1**4 * l2**4)
print('matches the closed form:', sp.simplify(detF0 - claimed) == 0)

b212 = bracket(f2, b12, X)
b2112 = bracket(f2, b112, X)
b2212 = bracket(f2, b212, X)


def rank0(vectors, vars_=X, origin=ORIGIN):
    return sp.Matrix.hstack(*vectors).subs(origin).rank()


n1 = rank0([f1, f2])
n2 = rank0([f1, f2, b12])
n3 = rank0([f1, f2, b12, b112, b212])
n4 = rank0([f1, f2, b12, b112, b212, b1112, b2112, b2212])
print('growth vector (n1,n2,n3,n4) =', (n1, n2, n3, n4))

# --------------------------------------------------------- 3. one trailer
Y = sp.Matrix([x1, x2, x3, x4])
fb1 = sp.Matrix([sp.cos(x3), sp.sin(x3), 0, sp.sin(x3 - x4) / l1])
fb2 = sp.Matrix([0, 0, 1, -d0 * sp.cos(x3 - x4) / l1])
assert sp.simplify(fb1 - f1[:4, :]) == sp.zeros(4, 1)   # first 4 rows of f1
assert sp.simplify(fb2 - f2[:4, :]) == sp.zeros(4, 1)   # first 4 rows of f2
c12 = bracket(fb1, fb2, Y)
c112 = bracket(fb1, c12, Y)
c212 = bracket(fb2, c12, Y)
Fb = sp.Matrix.hstack(fb1, fb2, c12, c112)
print('\n[fb1,fb2]       =', sp.simplify(c12.T))
print('[fb1,[fb1,fb2]]  =', sp.simplify(c112.T))
print('det Fb(x)      =', sp.simplify(Fb.det()))
print('det Fb(0)      =', sp.factor(sp.simplify(Fb.det().subs({x3: 0, x4: 0}))))
orig4 = {x1: 0, x2: 0, x3: 0, x4: 0}
m1 = rank0([fb1, fb2], Y, orig4)
m2 = rank0([fb1, fb2, c12], Y, orig4)
m3 = rank0([fb1, fb2, c12, c112, c212], Y, orig4)
print('one-trailer growth vector (n1,n2,n3) =', (m1, m2, m3))
