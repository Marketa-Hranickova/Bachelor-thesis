"""Figure for Chapter 5: the oscillating controller over the first sampling intervals.

Simulates the nilpotent approximation of the car with one trailer under the
sampled oscillating controller (parameters as in the MATLAB file params.m of the
"1 trailer" folder) and plots the controls and the privileged coordinates.
Output: ../figures/fig_one_period.pdf when run from the code/ folder of the thesis
(otherwise fig_one_period.pdf in the current directory).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---- parameters (same numbers as params.m, one trailer) -------------------
d0, l1 = 0.115, 0.082
alpha = d0 * (d0 + l1) / 2
gamma, eps, delta = 3.2028e-3, 5.0, 0.02
k12, k112 = 3, 2
omega = 2 * np.pi / eps
x0 = np.array([-0.2, -0.2, np.deg2rad(45), np.deg2rad(45)])   # x1, x2, phi, theta1


def x_to_z(x):
    return np.array([x[0], x[2], -x[1],
                     l1 * x[1] - d0 * l1**2 / (d0 + l1) * x[2] - l1**3 / (d0 + l1) * x[3]])


h = lambda z: 1.0 / np.sqrt(z @ z + delta**2)


def terms(zf):
    hh = h(zf)
    a1, a2 = -gamma * hh * zf[0], -gamma * hh * zf[1]
    a12 = -gamma * hh * (zf[0] * zf[1] + zf[2])
    a112 = -gamma * hh * (alpha * zf[1]**3 + zf[0] * zf[2] + zf[3])
    T12 = np.sqrt(2 * k12 * omega * abs(a12))
    T112 = (2 * k112 * omega)**(2 / 3) * abs(a112)**(1 / 3)
    return a1, a2, np.sign(a12), np.sign(a112), T12, T112


def controls(t, zf):
    a1, a2, s12, s112, T12, T112 = terms(zf)
    u1 = a1 + T12 * np.cos(k12 * omega * t) + T112 * np.cos(k112 * omega * t)
    u2 = a2 + T12 * s12 * np.sin(k12 * omega * t) - 2 * T112 * s112 * np.cos(2 * k112 * omega * t)
    return u1, u2


def run(n_int, z0, pts=400):
    ts, zs, us = [], [], []
    z = z0.copy()
    samples = [z.copy()]
    for j in range(n_int):
        zf = z.copy()

        def f(t, zz):
            u1, u2 = controls(t, zf)
            return [u1, u2, -zz[1] * u1, -zz[2] * u1 - alpha * zz[1]**2 * u2]

        tt = np.linspace(j * eps, (j + 1) * eps, pts)
        sol = solve_ivp(f, [tt[0], tt[-1]], z, t_eval=tt, rtol=1e-10, atol=1e-12)
        ts.append(sol.t)
        zs.append(sol.y.T)
        us.append(np.array([controls(t, zf) for t in sol.t]))
        z = sol.y[:, -1]
        samples.append(z.copy())
    return np.concatenate(ts), np.concatenate(zs), np.concatenate(us), np.array(samples)


if __name__ == '__main__':
    z0 = x_to_z(x0)
    n = 3
    t, z, u, S = run(n, z0)
    fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.6), sharex=True)
    ax[0].plot(t, u[:, 0], lw=1.0)
    ax[0].set_ylabel(r'$u_1$ [m/s]')
    ax[1].plot(t, u[:, 1], lw=1.0, color='tab:red')
    ax[1].set_ylabel(r'$u_2$ [rad/s]')
    names = [r'$z_1$', r'$z_2$', r'$z_3$', r'$z_4$']
    for i in range(4):
        ax[2].plot(t, z[:, i], lw=1.2, label=names[i])
    ts = np.arange(n + 1) * eps
    for i in range(4):
        ax[2].plot(ts, S[:, i], 'k.', ms=5)
    ax[2].set_ylabel('privileged coordinates')
    ax[2].set_xlabel('t [s]')
    ax[2].legend(ncol=4, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.02))
    for a in ax:
        for tj in ts:
            a.axvline(tj, color='0.8', lw=0.6, zorder=0)
        a.grid(alpha=0.2)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
    out = os.path.join(out, 'fig_one_period.pdf') if os.path.isdir(out) else 'fig_one_period.pdf'
    fig.savefig(out)
    print('saved', out)
    print('z0 =', z0, ' |z0| =', np.linalg.norm(z0))
    # check: z(eps) = z0 + eps*G(z0)*a(z0) + remainder
    a1, a2, s12, s112, T12, T112 = terms(z0)
    hh = h(z0)
    a12 = -gamma * hh * (z0[0] * z0[1] + z0[2]); a112 = -gamma * hh * (alpha * z0[1]**3 + z0[0] * z0[2] + z0[3])
    Ga = np.array([a1, a2, -z0[1] * a1 + a12, -z0[2] * a1 - alpha * z0[1]**2 * a2 + a112])
    print('z(eps)-z0      =', S[1] - S[0])
    print('eps*G(z0)a(z0) =', eps * Ga)
    print('radial pred    =', -eps * gamma * hh * z0, ' mu = eps*gamma*h =', eps * gamma * hh)
    print('max|u1|,|u2| over plotted =', np.abs(u).max(axis=0))
