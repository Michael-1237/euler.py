# euler.py
Variational Quantum Linear Solver (VQLS) demo for an implicit Euler step of a cosmology-inspired state update, using Qiskit statevector simulation.

---

## vqls_implicit_euler.py

```python

"""
vqls_flrw.py
============

Variational Quantum Linear Solver (VQLS) prototype for timestep-wise FLRW evolution.

This script implements the workflow:

1) Build a 4x4 linear system A4 x = b4 per timestep (implicit/semi-implicit embedding).
2) Represent the unknown x as a 2-qubit variational state |x(theta)> using EfficientSU2.
3) Minimize the scale-invariant VQLS cost:
       C(theta) = 1 - |<b_hat|A|x(theta)>|^2 / <x(theta)|A†A|x(theta)>
   using noiseless Statevector simulation (exact debugging mode).
4) Decode physical values (rho_{n+1}, H_{n+1}) from the optimized state by a stable
   least-squares scaling rule, then (optionally) enforce the affine component constraint.
5) Compare VQLS against:
   (i) classical solve of the same linear system A4 x=b4 (fair baseline), and
   (ii) RK4 integration of the underlying nonlinear ODE (reference baseline).
6) Plot rho(t) and H(t) for matter, radiation, and dark-energy-like regimes.


"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ============================================================
# 1) Build the embedded 4x4 linear system matrix A4(H_n)
# ============================================================
def build_A4(Hn: float, w: float, kappa: float, Lambda: float, dt: float) -> np.ndarray:
    """
    Construct the 4x4 embedded matrix A4 used in the implicit/semi-implicit step.

    Parameters
    ----------
    Hn : float
        Current Hubble parameter H_n.
    w : float
        Equation-of-state parameter (matter: 0, radiation: 1/3, dark-energy-like: -1).
    kappa : float
        Coupling constant.
    Lambda : float
        Cosmological constant (can be set to nonzero if desired).
    dt : float
        Timestep size.

    Returns
    -------
    A4 : (4,4) complex ndarray
        Embedded linear operator acting on a 4D padded state.
    """
    a = 1 + 3 * dt * Hn * (1 + w)
    b = (kappa * dt / 6) * (1 + 3 * w)
    c = 1 + dt * Hn
    d = (Lambda * dt / 3)

    A4 = np.array(
        [
            [a, 0,  0, 0],
            [b, c, -d, 0],
            [0, 0,  1, 0],
            [0, 0,  0, 1],
        ],
        dtype=complex,
    )
    return A4


# ============================================================
# 2) Classical baseline: solve the SAME linear system A4 x = b4
# ============================================================
def classical_linear_step(A4: np.ndarray, b4: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Solve A4 x = b4 classically.

    Returns
    -------
    rho_next, H_next, x : float, float, ndarray
    """
    x = np.linalg.solve(A4, b4)
    return float(np.real(x[0])), float(np.real(x[1])), x


# ============================================================
# 3) VQLS cost (exact statevector debug mode)
#    C = 1 - |<b_hat|A|x>|^2 / <x|A†A|x>
# ============================================================
def vqls_cost(theta: np.ndarray, ansatz: EfficientSU2, A4: np.ndarray, b4: np.ndarray) -> float:
    """
    Compute VQLS cost using exact statevector expectations (noiseless debug mode).
    This cost matches the VQLS objective, but evaluates overlaps classically.

    Parameters
    ----------
    theta : ndarray
        Ansatz parameters.
    ansatz : QuantumCircuit
        Parameterized circuit defining |x(theta)>.
    A4 : ndarray
        Linear system operator.
    b4 : ndarray
        Right-hand-side vector.

    Returns
    -------
    cost : float
    """
    bound = ansatz.assign_parameters(dict(zip(ansatz.parameters, theta)), inplace=False)
    x = Statevector.from_instruction(bound).data  # normalized |x>

    Ax = A4 @ x
    denom = np.vdot(Ax, Ax)  # <x|A†A|x> = ||A|x>||^2
    if abs(denom) < 1e-15:
        return 1e6

    # Normalize b for stability
    b_hat = b4 / np.linalg.norm(b4)
    overlap = np.vdot(b_hat, Ax)  # <b_hat|A|x>

    return float(np.real(1.0 - (abs(overlap) ** 2) / denom))


# ============================================================
# 4) Stable decoding: least-squares scaling alpha*
# ============================================================
def decode_x_phys(
    theta: np.ndarray,
    ansatz: EfficientSU2,
    A4: np.ndarray,
    b4: np.ndarray,
    enforce_third_is_one: bool = True,
) -> np.ndarray | None:
    """
    Decode a physically-scaled solution vector from the normalized ansatz state.

    We first compute the least-squares optimal scaling:
        alpha* = <Ax|b> / <Ax|Ax>
    then set x_phys = alpha* x.

    Optionally, enforce the affine padding constraint x_phys[2] = 1 by dividing
    the vector by its third component (if not too small).

    Returns
    -------
    x_phys : ndarray or None
        Decoded physical vector, or None if ill-conditioned.
    """
    bound = ansatz.assign_parameters(dict(zip(ansatz.parameters, theta)), inplace=False)
    x = Statevector.from_instruction(bound).data

    Ax = A4 @ x
    denom = np.vdot(Ax, Ax)
    if abs(denom) < 1e-15:
        return None

    alpha = np.vdot(Ax, b4) / denom  # complex scalar
    x_phys = alpha * x

    if enforce_third_is_one:
        if abs(x_phys[2]) < 1e-12:
            return None
        x_phys = x_phys / x_phys[2]

    return x_phys


# ============================================================
# 5) One timestep solve via VQLS optimization
# ============================================================
def vqls_one_step(
    rho_n: float,
    H_n: float,
    w: float,
    kappa: float,
    Lambda: float,
    dt: float,
    ansatz: EfficientSU2,
    theta0: np.ndarray,
    maxiter: int = 120,
) -> tuple[float, float, float, np.ndarray, float]:
    """
    Solve one timestep by minimizing the VQLS cost for A4 x=b4.

    Returns
    -------
    rho_next, H_next : float
    cost_opt : float
    theta_opt : ndarray
    resid : float   (||A4 x_phys - b4||)
    """
    A4 = build_A4(H_n, w, kappa, Lambda, dt)
    b4 = np.array([rho_n, H_n, 1.0, 0.0], dtype=complex)

    obj = lambda th: vqls_cost(th, ansatz, A4, b4)

    if _HAS_SCIPY:
        res = minimize(obj, theta0, method="COBYLA", options={"maxiter": maxiter})
        theta_opt = np.asarray(res.x, dtype=float)
        cost_opt = float(res.fun)
    else:
        # fallback: simple random search
        rng = np.random.default_rng(0)
        theta_opt = np.array(theta0, dtype=float)
        cost_opt = obj(theta_opt)
        for _ in range(maxiter):
            trial = theta0 + rng.normal(scale=0.25, size=theta0.shape)
            val = obj(trial)
            if val < cost_opt:
                cost_opt = val
                theta_opt = trial.copy()

    x_phys = decode_x_phys(theta_opt, ansatz, A4, b4, enforce_third_is_one=True)
    if x_phys is None:
        return np.nan, np.nan, cost_opt, theta_opt, np.inf

    rho_next = float(np.real(x_phys[0]))
    H_next = float(np.real(x_phys[1]))
    resid = float(np.linalg.norm(A4 @ x_phys - b4))

    return rho_next, H_next, cost_opt, theta_opt, resid


# ============================================================
# 6) Trajectory runners
# ============================================================
def run_vqls_trajectory(
    rho0: float,
    H0: float,
    w: float,
    kappa: float,
    Lambda: float,
    dt: float,
    steps: int,
    reps: int = 2,
    maxiter: int = 120,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run VQLS per-timestep solves and return trajectories.
    Uses warm-start by feeding the previous theta into the next step.
    """
    ansatz = EfficientSU2(2, reps=reps)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=ansatz.num_parameters)

    rho = [float(rho0)]
    H = [float(H0)]
    costs = []
    resids = []

    for n in range(steps):
        rho_np1, H_np1, c, theta, r = vqls_one_step(
            rho[-1], H[-1], w, kappa, Lambda, dt,
            ansatz=ansatz,
            theta0=theta,
            maxiter=maxiter,
        )
        if not np.isfinite(rho_np1) or not np.isfinite(H_np1):
            print(f"[STOP] VQLS decoding/optimization failed at step {n}.")
            break

        rho.append(rho_np1)
        H.append(H_np1)
        costs.append(c)
        resids.append(r)

    t = np.arange(len(rho)) * dt
    return t, np.array(rho), np.array(H), np.array(costs), np.array(resids)


def run_classical_linear_trajectory(
    rho0: float,
    H0: float,
    w: float,
    kappa: float,
    Lambda: float,
    dt: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classical baseline solving the same linear system A4 x=b4 at each step.
    """
    rho = [float(rho0)]
    H = [float(H0)]

    for _ in range(steps):
        A4 = build_A4(H[-1], w, kappa, Lambda, dt)
        b4 = np.array([rho[-1], H[-1], 1.0, 0.0], dtype=complex)
        rho_np1, H_np1, _ = classical_linear_step(A4, b4)
        rho.append(rho_np1)
        H.append(H_np1)

    t = np.arange(len(rho)) * dt
    return t, np.array(rho), np.array(H)


# ============================================================
# 7) Optional RK4 reference for the original nonlinear ODE
# ============================================================
def flrw_rhs(t: float, y: np.ndarray, w: float, kappa: float, Lambda: float) -> np.ndarray:
    """
    Nonlinear FLRW ODE right-hand-side for y=(rho, H).
    """
    rho, H = y
    drho = -3.0 * H * rho * (1.0 + w)
    dH = -(kappa / 6.0) * (1.0 + 3.0 * w) * rho + (Lambda / 3.0) - H**2
    return np.array([drho, dH], dtype=float)


def rk4_step(rhs, t: float, y: np.ndarray, dt: float, *args) -> np.ndarray:
    k1 = rhs(t, y, *args)
    k2 = rhs(t + dt / 2, y + dt * k1 / 2, *args)
    k3 = rhs(t + dt / 2, y + dt * k2 / 2, *args)
    k4 = rhs(t + dt, y + dt * k3, *args)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_flrw_rk4(
    rho0: float,
    H0: float,
    w: float,
    kappa: float,
    Lambda: float,
    dt: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RK4 integration of the original nonlinear ODE (reference comparison).
    """
    y = np.array([rho0, H0], dtype=float)
    t = np.arange(steps + 1) * dt
    rho = np.zeros(steps + 1)
    H = np.zeros(steps + 1)
    rho[0], H[0] = y

    for n in range(steps):
        y = rk4_step(flrw_rhs, t[n], y, dt, w, kappa, Lambda)
        rho[n + 1], H[n + 1] = y

    return t, rho, H


# ============================================================
# 8) Main: run 3 regimes and plot
# ============================================================
def main():
    rho0, H0 = 1.0, 0.7
    kappa = 1.0
    Lambda = 0.0
    dt = 0.05
    steps = 20

    reps = 2
    maxiter = 120

    regimes = [
        {"name": "Matter (w=0)", "w": 0.0},
        {"name": "Radiation (w=1/3)", "w": 1.0 / 3.0},
        {"name": "Dark energy-like (w=-1)", "w": -1.0},
    ]

    runs = {}

    for i, r in enumerate(regimes):
        w = r["w"]

        tV, rhoV, HV, costs, resids = run_vqls_trajectory(
            rho0, H0, w, kappa, Lambda, dt, steps,
            reps=reps, maxiter=maxiter, seed=7 + 100 * i
        )

        tL, rhoL, HL = run_classical_linear_trajectory(rho0, H0, w, kappa, Lambda, dt, steps)
        tR, rhoR, HR = solve_flrw_rk4(rho0, H0, w, kappa, Lambda, dt, steps)

        m = min(len(tV), len(tL), len(tR))
        runs[r["name"]] = dict(
            t=tV[:m],
            V=dict(rho=rhoV[:m], H=HV[:m]),
            Lin=dict(rho=rhoL[:m], H=HL[:m]),
            RK4=dict(rho=rhoR[:m], H=HR[:m]),
            diag=dict(costs=costs, resids=resids),
        )

        if len(costs) > 0:
            print(f"\n{r['name']}")
            print(f"  mean cost   = {np.mean(costs):.3e}")
            print(f"  mean resid  = {np.mean(resids):.3e}")

    # --- H(t) plot ---
    plt.figure()
    for name, d in runs.items():
        plt.plot(d["t"], d["V"]["H"], label=f"VQLS {name}")
        plt.plot(d["t"], d["Lin"]["H"], linestyle="--", label=f"Classical A4-solve {name}")
        plt.plot(d["t"], d["RK4"]["H"], linestyle=":", label=f"RK4 ODE {name}")
    plt.xlabel("t")
    plt.ylabel("H(t)")
    plt.title("H(t): VQLS (solid) vs Classical A4-solve (dashed) vs RK4 ODE (dotted)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- rho(t) plot ---
    plt.figure()
    for name, d in runs.items():
        plt.plot(d["t"], d["V"]["rho"], label=f"VQLS {name}")
        plt.plot(d["t"], d["Lin"]["rho"], linestyle="--", label=f"Classical A4-solve {name}")
        plt.plot(d["t"], d["RK4"]["rho"], linestyle=":", label=f"RK4 ODE {name}")
    plt.xlabel("t")
    plt.ylabel(r"$\\rho(t)$")
    plt.title(r"$\\rho(t)$: VQLS (solid) vs Classical A4-solve (dashed) vs RK4 ODE (dotted)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

