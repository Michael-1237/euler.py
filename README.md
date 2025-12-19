# euler.py
Variational Quantum Linear Solver (VQLS) demo for an implicit Euler step of a cosmology-inspired state update, using Qiskit statevector simulation.

---

## vqls_implicit_euler.py

```python
import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector


# ----------------------------
# 1) 
#    y = [rho, H, 1]^T
# ----------------------------
def build_A3_implicit(Hn: float, w: float, kappa: float, Lambda: float, dt: float) -> np.ndarray:
    """
    Build A3 = I - dt*A_n using the A_n structure from the whiteboard:
      A_n =
      [ -3 Hn (1+w),           0,          0 ]
      [ -(kappa/6)(1+3w),    -Hn,       Lambda/3 ]
      [ 0,                    0,          0 ]
    """
    A_n = np.array(
        [
            [-3.0 * Hn * (1.0 + w), 0.0, 0.0],
            [-(kappa / 6.0) * (1.0 + 3.0 * w), -Hn, (Lambda / 3.0)],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    A3 = np.eye(3) - dt * A_n
    return A3


def embed_3_to_4(A3: np.ndarray) -> np.ndarray:
    """
    Embed a 3x3 matrix into 4x4 for 2-qubit amplitude encoding:
        A4 = [[A3, 0],
              [0,  1]]
    """
    A4 = np.eye(4, dtype=complex)
    A4[:3, :3] = A3.astype(complex)
    A4[3, 3] = 1.0
    return A4


def pad_vec3_to_4(v3: np.ndarray) -> np.ndarray:
    """Pad a length-3 vector to length 4 by appending 0."""
    v4 = np.zeros(4, dtype=complex)
    v4[:3] = v3.astype(complex)
    v4[3] = 0.0
    return v4


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-15:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


# ----------------------------
# 2) VQLS alignment cost (scale-invariant)
#    C(theta) = 1 - |<b|A|x(theta)>|^2 / ||A|x(theta)>||^2
#    where |b> is normalized.
# ----------------------------
def vqls_cost_alignment(
    theta: np.ndarray,
    ansatz: EfficientSU2,
    A4: np.ndarray,
    b_state: np.ndarray,
    amp2_min: float = 1e-3,
    penalty_weight: float = 0.1,
) -> float:
    """
    Evaluate the VQLS alignment cost using exact statevector simulation.

    We also add a small penalty to avoid the ansatz accidentally making the
    3rd component (index 2, corresponding to the constant '1' slot) too close to 0,
    because decoding rescales by that amplitude.
    """
    # Prepare |x(theta)>
    bind = dict(zip(list(ansatz.parameters), theta))
    qc = ansatz.assign_parameters(bind, inplace=False)
    x = Statevector.from_instruction(qc).data  # length 4, normalized

    # Penalty to keep amplitude[2] away from 0 so decoding is stable
    amp2 = abs(x[2])
    penalty = 0.0
    if amp2 < amp2_min:
        penalty = penalty_weight * (amp2_min - amp2) ** 2

    # Compute A|x>
    Ax = A4 @ x
    denom = np.vdot(Ax, Ax).real  # ||A|x>||^2

    if denom < 1e-15:
        return 1e6  # ill-posed for this theta

    # b_state is normalized: <b|A|x>
    overlap = np.vdot(b_state, Ax)
    cost = 1.0 - (abs(overlap) ** 2) / denom

    # Numerical safety
    cost = float(np.real(cost))
    return cost + penalty


# ----------------------------
# 3) Classical optimizer wrapper
#    Uses SciPy COBYLA if available; otherwise random search fallback.
# ----------------------------
def minimize_cost(cost_fn, theta0: np.ndarray, maxiter: int = 200, seed: int = 0) -> tuple[np.ndarray, float]:
    try:
        from scipy.optimize import minimize

        res = minimize(cost_fn, theta0, method="COBYLA", options={"maxiter": maxiter})
        return np.array(res.x, dtype=float), float(res.fun)
    except Exception:
        rng = np.random.default_rng(seed)
        best_theta = theta0.copy()
        best_val = float(cost_fn(best_theta))
        for _ in range(maxiter):
            trial = best_theta + rng.normal(scale=0.2, size=best_theta.shape)
            val = float(cost_fn(trial))
            if val < best_val:
                best_val, best_theta = val, trial
        return best_theta, best_val


# ----------------------------
# 4) Decode physical (rho_{n+1}, H_{n+1}) from the normalized |x>
#    We fix the overall scale by forcing the "constant slot" (index 2) to 1.
# ----------------------------
def decode_rho_H_from_state(x: np.ndarray) -> tuple[float, float]:
    """
    x is a normalized 4-amplitude state.
    We interpret it as proportional to (rho_{n+1}, H_{n+1}, 1, 0)
    and fix scale by dividing by x[2].
    """
    if abs(x[2]) < 1e-12:
        return np.nan, np.nan

    x_phys = x / x[2]
    rho_next = float(np.real(x_phys[0]))
    H_next = float(np.real(x_phys[1]))
    return rho_next, H_next


# ----------------------------
# 5) Solve ONE implicit step with VQLS
# ----------------------------
def vqls_solve_one_step(
    rho_n: float,
    H_n: float,
    w: float,
    kappa: float,
    Lambda: float,
    dt: float,
    reps: int = 2,
    maxiter: int = 200,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Returns (rho_{n+1}, H_{n+1}, final_cost).
    """
    # Build A3 and embed to 4D
    A3 = build_A3_implicit(H_n, w, kappa, Lambda, dt)
    A4 = embed_3_to_4(A3)

    # Build b = y_n padded, then normalize to a quantum state |b>
    b3 = np.array([rho_n, H_n, 1.0], dtype=float)
    b4 = pad_vec3_to_4(b3)
    b_state = normalize(b4)

    # Ansatz for |x(theta)>
    ansatz = EfficientSU2(2, reps=reps)
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-np.pi, np.pi, size=len(ansatz.parameters))

    def cost_fn(th):
        return vqls_cost_alignment(th, ansatz, A4, b_state)

    theta_opt, final_cost = minimize_cost(cost_fn, theta0, maxiter=maxiter, seed=seed)

    # Prepare optimized state and decode
    bind = dict(zip(list(ansatz.parameters), theta_opt))
    qc_opt = ansatz.assign_parameters(bind, inplace=False)
    x_opt = Statevector.from_instruction(qc_opt).data

    rho_next, H_next = decode_rho_H_from_state(x_opt)
    return rho_next, H_next, final_cost


# ----------------------------
# 6) Classical reference solution for one implicit step
# ----------------------------
def classical_solve_one_step(
    rho_n: float,
    H_n: float,
    w: float,
    kappa: float,
    Lambda: float,
    dt: float,
) -> tuple[float, float]:
    """
    Solve A3 x3 = b3 exactly with numpy for comparison.
    """
    A3 = build_A3_implicit(H_n, w, kappa, Lambda, dt)
    b3 = np.array([rho_n, H_n, 1.0], dtype=float)
    x3 = np.linalg.solve(A3, b3)
    return float(x3[0]), float(x3[1])


# ----------------------------
# 7) Run multiple steps + plot
# ----------------------------
def run_demo():
    # Example parameters (edit these to your case)
    rho0 = 1.0
    H0 = 0.7
    w = 0.0
    kappa = 1.0
    Lambda = 0.1
    dt = 0.05

    steps = 15

    rho_v = [rho0]
    H_v = [H0]
    costs = [np.nan]

    rho_c = [rho0]
    H_c = [H0]

    for n in range(steps):
        rho_n, H_n = rho_v[-1], H_v[-1]

        # VQLS step
        rho_np1, H_np1, cost = vqls_solve_one_step(
            rho_n, H_n, w, kappa, Lambda, dt, reps=2, maxiter=180, seed=10 + n
        )
        rho_v.append(rho_np1)
        H_v.append(H_np1)
        costs.append(cost)

        # Classical reference step (using same "frozen A_n" implicit matrix)
        rc, Hc = classical_solve_one_step(rho_c[-1], H_c[-1], w, kappa, Lambda, dt)
        rho_c.append(rc)
        H_c.append(Hc)

        print(f"step {n:02d} | VQLS rho={rho_np1:.6f} H={H_np1:.6f} cost={cost:.3e} "
              f"| classical rho={rc:.6f} H={Hc:.6f}")

    t = np.arange(steps + 1) * dt

    plt.figure()
    plt.plot(t, rho_v, label="VQLS (implicit)")
    plt.plot(t, rho_c, "--", label="Classical (implicit)")
    plt.xlabel("t")
    plt.ylabel("rho(t)")
    plt.title("rho(t) trajectory")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(t, H_v, label="VQLS (implicit)")
    plt.plot(t, H_c, "--", label="Classical (implicit)")
    plt.xlabel("t")
    plt.ylabel("H(t)")
    plt.title("H(t) trajectory")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(t, costs, label="VQLS cost")
    plt.xlabel("t")
    plt.ylabel("C(theta)")
    plt.title("VQLS cost per step (should approach ~0)")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    run_demo()
