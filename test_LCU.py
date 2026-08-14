import numpy as np
import pennylane as qml

# -----------------------------------------------------
# 1. Define number of system qubits and wires
# -----------------------------------------------------
n = 2
dim = 1 << n

anc = 0
sys_wires = list(range(1, n + 1))
all_wires = [anc] + sys_wires

# -----------------------------------------------------
# 2. Helper: build a unitary that prepares a statevector
# -----------------------------------------------------
def preparation_unitary(psi, seed=0):
    """
    Returns a unitary U such that U|0...0> = psi.

    This is useful for simulation when you only have
    a statevector. For large systems, replace this with
    an actual state-preparation circuit.
    """
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    d = len(psi)

    # Random full-rank matrix with desired first column
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    A[:, 0] = psi

    # QR decomposition gives a unitary whose first column
    # is proportional to psi
    U, _ = np.linalg.qr(A)

    # Fix global phase so that U|0...0> = psi exactly
    phase = np.vdot(U[:, 0], psi)  # <q0|psi>
    if np.abs(phase) > 1e-12:
        U = U * phase

    return U


# -----------------------------------------------------
# 3. Define the two input states
# -----------------------------------------------------

# Example "abstract database" state
psi_A = np.array([1.0, 1.0j, -1.0, 0.0], dtype=complex)
psi_A = psi_A / np.linalg.norm(psi_A)

# Example easily sampled distribution:
# uniform superposition over all computational basis states
psi_B = np.ones(dim, dtype=complex) / np.sqrt(dim)


# -----------------------------------------------------
# 4. Choose coefficients alpha and beta
# -----------------------------------------------------
alpha = np.sqrt(0.7)
beta = np.sqrt(0.3) * np.exp(1j * 0.6)

# Normalize control amplitudes.
# The final postselected state is normalized separately.
norm_ab = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
alpha = alpha / norm_ab
beta = beta / norm_ab

theta = np.arccos(np.clip(np.abs(alpha), 0.0, 1.0))
phi = np.angle(beta) - np.angle(alpha)


# -----------------------------------------------------
# 5. Build controlled preparation unitaries
# -----------------------------------------------------
U_A = preparation_unitary(psi_A, seed=1)
U_B = preparation_unitary(psi_B, seed=2)


# -----------------------------------------------------
# 6. PennyLane device and circuit
# -----------------------------------------------------
dev = qml.device("default.qubit", wires=all_wires)

@qml.qnode(dev)
def linear_combination_circuit(theta, phi):
    # Prepare ancilla:
    # cos(theta)|0> + exp(i phi) sin(theta)|1>
    qml.RY(2 * theta, wires=anc)
    qml.PhaseShift(phi, wires=anc)

    # Controlled preparation:
    # If ancilla = 0, prepare psi_A
    qml.ctrl(qml.QubitUnitary, control=anc, control_values=0)(
        U_A, wires=sys_wires
    )

    # If ancilla = 1, prepare psi_B
    qml.ctrl(qml.QubitUnitary, control=anc, control_values=1)(
        U_B, wires=sys_wires
    )

    # Interfere the two branches
    qml.Hadamard(wires=anc)

    return qml.state()


# -----------------------------------------------------
# 7. Run circuit and postselect ancilla = 0
# -----------------------------------------------------
full_state = linear_combination_circuit(theta, phi)

# Reshape so first tensor index is the ancilla
state_tensor = full_state.reshape([2] * (n + 1))

# Branch where ancilla is |0>
branch0 = state_tensor[0].reshape(-1)

prob_success = np.sum(np.abs(branch0) ** 2)

if prob_success < 1e-14:
    raise ValueError("Postselection probability is too small.")

psi_out = branch0 / np.sqrt(prob_success)


# -----------------------------------------------------
# 8. Compare with expected linear combination
# -----------------------------------------------------
psi_expected = alpha * psi_A + beta * psi_B
psi_expected = psi_expected / np.linalg.norm(psi_expected)

fidelity = np.abs(np.vdot(psi_expected, psi_out)) ** 2

print("Postselection probability:", prob_success)
print("Fidelity with expected state:", fidelity)

print(qml.draw(linear_combination_circuit)(theta, phi))
