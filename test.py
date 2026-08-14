import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

# -----------------------------
# Configuration
# -----------------------------
n_wires = 4
n_layers = 4
MAX_EPOCHS = 200
LR = 0.05
PATIENCE = 30

# PennyLane's default.qubit uses complex128 by default, which requires float64 parameters
dev = qml.device("default.qubit", wires=n_wires)

# -----------------------------
# Ansatz (QuDDPM-style denoiser / state preparer)
# -----------------------------
def ansatz(params):
    for l in range(n_layers):
        for i in range(n_wires):
            qml.RX(params[l, i, 0], wires=i)
            qml.RY(params[l, i, 1], wires=i)
            qml.RZ(params[l, i, 2], wires=i)
        for i in range(n_wires):
            qml.CNOT(wires=[i, (i + 1) % n_wires])

@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(params):
    ansatz(params)
    return qml.state()

# -----------------------------
# Target state: fixed shallow circuit with local structure
# -----------------------------
@qml.qnode(dev, interface="torch")
def target_circuit():
    for i in range(n_wires):
        qml.RX(0.3 * (i + 1), wires=i)
        qml.RY(0.2 * (i + 1), wires=i)
    for i in range(n_wires - 1):
        qml.CNOT(wires=[i, i + 1])
    for i in range(n_wires):
        qml.RZ(0.1 * (i + 1), wires=i)
    return qml.state()

with torch.no_grad():
    target_state = target_circuit().detach()
    # Ensure float64 to match default.qubit state outputs
    target_probs = (torch.abs(target_state) ** 2).to(torch.float64)

# -----------------------------
# Helpers for Pauli expectations from a statevector
# -----------------------------
def all_pauli_expectations(psi, n_wires):
    """Returns a real tensor of shape (n_wires, 3): <X>, <Y>, <Z> per qubit."""
    X = torch.tensor([[0, 1], [1, 0]], dtype=psi.dtype)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=psi.dtype)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=psi.dtype)
    mats = [X, Y, Z]

    psi_t = psi.reshape([2] * n_wires)
    rows = []
    for i in range(n_wires):
        row = []
        for a in range(3):
            P = mats[a]
            out = torch.tensordot(P, psi_t, dims=([1], [i]))
            out = out.movedim(0, i)
            val = torch.sum(torch.conj(psi_t) * out)
            row.append(val.real)
        rows.append(torch.stack(row))
    return torch.stack(rows)

# Precompute target single-qubit Pauli expectations
with torch.no_grad():
    target_pauli = all_pauli_expectations(target_state, n_wires).detach()

# -----------------------------
# Loss functions
# -----------------------------
def local_shadow_loss(psi, target_pauli):
    exp_rho = all_pauli_expectations(psi, n_wires)          # (n, 3)
    overlap = 0.5 * (1.0 + torch.sum(exp_rho * target_pauli, dim=1))  # (n,)
    return torch.mean(1.0 - overlap)

def global_infidelity_loss(psi, target_state):
    fidelity = torch.abs(torch.vdot(psi, target_state)) ** 2
    return 1.0 - fidelity

# Precompute Hamming kernel for MMD 
# Explicitly set dtype to float64 to prevent mixed-precision matrix multiplication errors
idx = torch.arange(2 ** n_wires)
xor_grid = idx.unsqueeze(0) ^ idx.unsqueeze(1)
dist_grid = torch.zeros_like(xor_grid, dtype=torch.float64) 
for i in range(n_wires):
    dist_grid += (xor_grid >> i) & 1
gamma = 1.0 / n_wires
K_mmd = torch.exp(-gamma * dist_grid)

def mmd_loss(psi, target_probs, K):
    p = torch.abs(psi) ** 2
    t = target_probs.to(p.dtype) # Ensure dtype match
    K = K.to(p.dtype)            # Ensure dtype match
    mmd2 = (
        torch.dot(p, K @ p)
        + torch.dot(t, K @ t)
        - 2.0 * torch.dot(p, K @ t)
    )
    return torch.clamp(mmd2, min=0.0)

def wasserstein_loss(psi, target_probs):
    p = torch.abs(psi) ** 2
    t = target_probs.to(p.dtype) # Ensure dtype match
    cdf_p = torch.cumsum(p, dim=0)
    cdf_t = torch.cumsum(t, dim=0)
    return torch.sum(torch.abs(cdf_p - cdf_t))

loss_functions = {
    "Local Shadow": lambda psi: local_shadow_loss(psi, target_pauli),
    "Global Infidelity": lambda psi: global_infidelity_loss(psi, target_state),
    "MMD": lambda psi: mmd_loss(psi, target_probs, K_mmd),
    "Wasserstein": lambda psi: wasserstein_loss(psi, target_probs),
}

# -----------------------------
# Training loop with plateau detection
# -----------------------------
def make_params():
    # Initialize parameters as float64 to match PennyLane's complex128 backend
    return nn.Parameter(torch.randn(n_layers, n_wires, 3, dtype=torch.float64) * 0.1)

def train_and_evaluate(loss_name, loss_fn):
    print(f"Training with {loss_name}...")
    params = make_params()
    optimizer = torch.optim.Adam([params], lr=LR)

    history = []
    plateau_epoch = MAX_EPOCHS

    for epoch in range(MAX_EPOCHS):
        optimizer.zero_grad()
        psi = circuit(params)
        loss = loss_fn(psi)
        loss.backward()
        optimizer.step()

        history.append(loss.item())

        if len(history) >= PATIENCE and epoch > PATIENCE:
            recent_min = min(history[-PATIENCE:])
            past_min = min(history[:-PATIENCE])
            if past_min - recent_min < 1e-4:
                plateau_epoch = epoch
                break

    # Evaluate all losses at the final parameters
    with torch.no_grad():
        psi = circuit(params)
        eval_losses = {name: fn(psi).item() for name, fn in loss_functions.items()}

    return history, plateau_epoch, eval_losses

# -----------------------------
# Run all trainings
# -----------------------------
results = {}
histories = {}
for name, fn in loss_functions.items():
    hist, ep, evals = train_and_evaluate(name, fn)
    results[name] = {"Plateau Epoch": ep, **{f"Final {k}": v for k, v in evals.items()}}
    histories[name] = hist

# -----------------------------
# Table
# -----------------------------
df = pd.DataFrame(results).T
print("\n--- Loss values at each model's plateau point ---")
print(df.to_string())

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 6))
for name, hist in histories.items():
    plt.plot(hist, label=f"Trained on: {name}")
plt.xlabel("Epoch")
plt.ylabel("Training Loss (log scale)")
plt.title("QuDDPM Training Curves (PennyLane)")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()
