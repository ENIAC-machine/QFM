import numpy as np
from typing import Dict, Any, List, Tuple


def _validate_unitary(U: np.ndarray, name: str, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check that U is a square unitary matrix within tolerance.
    Returns U as complex ndarray and identity matrix of matching size.
    """
    U = np.asarray(U, dtype=complex)

    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")

    n = U.shape[0]
    I = np.eye(n, dtype=complex)

    err = np.linalg.norm(U.conj().T @ U - I, ord=2)
    if err > tol:
        raise ValueError(
            f"{name} is not unitary within tolerance {tol}. "
            f"Unitary error: {err:.3e}"
        )

    return U, I


def find_valid_alpha_beta(
    U1: np.ndarray,
    U2: np.ndarray,
    tol: float = 1e-10
) -> Dict[str, Any]:
    """
    Find all valid alpha, beta in [0, 1] with alpha + beta = 1
    such that alpha*U1 + beta*U2 is unitary.

    Returns
    -------
    dict
        If U1 and U2 are equal within tolerance:
            status = "all"
            alpha_interval = (0.0, 1.0)

        Otherwise:
            status = "endpoints_only"
            pairs = [(alpha, beta, error), ...]

    Notes
    -----
    For alpha, beta >= 0 and alpha + beta = 1, a nontrivial interior
    solution alpha in (0, 1) exists only if U1 == U2.
    """

    U1, I = _validate_unitary(U1, "U1", tol)
    U2, _ = _validate_unitary(U2, "U2", tol)

    if U1.shape != U2.shape:
        raise ValueError("U1 and U2 must have the same shape.")

    def unitary_error(A: np.ndarray) -> float:
        return np.linalg.norm(A.conj().T @ A - I, ord=2)

    # Relative unitary W = U1 U2^\dagger.
    # The convex combination is unitary for an interior alpha only if W = I.
    W = U1 @ U2.conj().T
    rel_err = np.linalg.norm(W - I, ord=2)

    # If U1 and U2 are the same unitary, all alphas are valid.
    if rel_err <= tol:
        return {
            "status": "all",
            "alpha_interval": (0.0, 1.0),
            "message": (
                "U1 and U2 are equal within tolerance. "
                "Every alpha in [0, 1] with beta = 1 - alpha is valid."
            ),
            "relative_error_U1_U2_dagger": rel_err,
        }

    # Otherwise, only endpoints can be valid.
    valid_pairs = []

    for alpha in (0.0, 1.0):
        beta = 1.0 - alpha
        A = alpha * U1 + beta * U2
        err = unitary_error(A)

        if err <= tol:
            valid_pairs.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "unitary_error": err,
                }
            )

    return {
        "status": "endpoints_only" if valid_pairs else "none",
        "pairs": valid_pairs,
        "message": (
            "For distinct unitaries, there is no interior alpha in (0, 1) "
            "such that alpha*U1 + (1-alpha)*U2 is unitary. "
            "Only alpha=0 or alpha=1 can be valid."
        ),
        "relative_error_U1_U2_dagger": rel_err,
    }


def scan_valid_alpha_beta(
    U1: np.ndarray,
    U2: np.ndarray,
    num_points: int = 1001,
    tol: float = 1e-8
) -> List[Dict[str, float]]:
    """
    Optional numerical scan over alpha in [0, 1].

    This does not prove that all valid alphas are found; it only checks
    a finite grid. Use find_valid_alpha_beta for the analytic result.
    """

    U1, I = _validate_unitary(U1, "U1", tol)
    U2, _ = _validate_unitary(U2, "U2", tol)

    if U1.shape != U2.shape:
        raise ValueError("U1 and U2 must have the same shape.")

    results = []

    for alpha in np.linspace(0.0, 1.0, num_points):
        beta = 1.0 - alpha
        A = alpha * U1 + beta * U2
        err = np.linalg.norm(A.conj().T @ A - I, ord=2)

        if err <= tol:
            results.append(
                {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "unitary_error": float(err),
                }
            )

    return results


if __name__ == "__main__":

    # Example 1: distinct unitaries I and X
    I = np.eye(2, dtype=complex)

    X = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=complex,
    )

    result = find_valid_alpha_beta(I, X)
    print("Distinct unitaries I and X:")
    print(result)
    print()

    # Example 2: identical unitaries
    result_same = find_valid_alpha_beta(I, I)
    print("Identical unitaries I and I:")
    print(result_same)
    print()

    # Optional numerical scan for I and X
    scanned = scan_valid_alpha_beta(I, X, num_points=11, tol=1e-8)
    print("Numerical scan over alpha for I and X:")
    for item in scanned:
        print(item)
