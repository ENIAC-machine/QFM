import torch
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Configuration
# ============================================================

INPUT_DIM = 4                 # dimension of input vector
N_SAMPLES = 256               # dataset size
EVAL_SIZE = 64                # subset used to track collisions
BATCH_SIZE = 32

N_LAYERS = 2                  # trainable encoder depth

NUM_SHADOWS = 24              # number of sampled Pauli shadow features
SHADOW_ORDER = 2              # 1 = local, 2 = two-body correlations, etc.

ENCODER_EPOCHS = 40
DECODER_EPOCHS = 100

LR_ENCODER = 0.03
LR_DECODER = 0.01

# Collision tracking thresholds
COLLISION_THRESHOLD = 0.02
FULL_COLLISION_THRESHOLD = 0.02

# Pairs whose whitened input distance is larger than this are treated as distinct
INPUT_MARGIN = 0.25

# Desired minimum shadow distance for distinct inputs
SEPARATION_MARGIN = 0.10

# Weight for the weak isometry loss
ISO_WEIGHT = 0.1

if INPUT_DIM > 7:
    print("Warning: statevector simulation scales exponentially in INPUT_DIM.")

# ============================================================
# Data generation: multivariate normal
# ============================================================

mean = torch.zeros(INPUT_DIM, dtype=torch.float64)

# Create a random positive-definite covariance matrix
A = torch.randn(INPUT_DIM, INPUT_DIM, dtype=torch.float64)
cov = A @ A.T / INPUT_DIM + 0.5 * torch.eye(INPUT_DIM, dtype=torch.float64)

dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)

# IMPORTANT: sample shape should be (N_SAMPLES,)
# This gives X.shape == (N_SAMPLES, INPUT_DIM)
X = dist.sample((N_SAMPLES,))

# Safety checks / reshaping
if X.ndim == 1:
    X = X.unsqueeze(1)

if X.shape == (INPUT_DIM, N_SAMPLES):
    X = X.T

if X.shape != (N_SAMPLES, INPUT_DIM):
    X = X.reshape(N_SAMPLES, INPUT_DIM)

X = X.to(torch.float64)

# ============================================================
# Bijective preprocessing:
# multivariate normal -> whiten -> sigmoid -> (0, pi)
# ============================================================

data_mean = X.mean(dim=0)
centered = X - data_mean

cov_emp = centered.T @ centered / max(N_SAMPLES - 1, 1)
evals, evecs = torch.linalg.eigh(cov_emp)

whiten = evecs @ torch.diag(1.0 / torch.sqrt(evals + 1e-6)) @ evecs.T
unwhiten = evecs @ torch.diag(torch.sqrt(evals + 1e-6)) @ evecs.T

# Whitened coordinates
Z = centered @ whiten.T
Z = Z.reshape(N_SAMPLES, INPUT_DIM)

# Bijective map from R to (0, 1), then to (0, pi)
T = torch.sigmoid(Z)
ANGLES = math.pi * T

# Final safety assertions
assert X.shape == (N_SAMPLES, INPUT_DIM), f"X has shape {X.shape}"
assert Z.shape == (N_SAMPLES, INPUT_DIM), f"Z has shape {Z.shape}"
assert T.shape == (N_SAMPLES, INPUT_DIM), f"T has shape {T.shape}"
assert ANGLES.shape == (N_SAMPLES, INPUT_DIM), f"ANGLES has shape {ANGLES.shape}"

# ============================================================
# Encoder circuit
# ============================================================

n_wires = INPUT_DIM
dev = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def encoder_circuit(params, angles):
    """
    params: trainable parameters of shape (n_layers, n_wires, 3)
    angles: data-dependent RY angles, shape (n_wires,)
    """

    # Data encoding.
    # angles are already in (0, pi), so this base encoding is injective.
    for i in range(n_wires):
        qml.RY(angles[i], wires=i)

    # Trainable unitary post-processing.
    # A unitary cannot create or destroy full-state injectivity,
    # but it can change which information is visible locally.
    for layer in range(params.shape[0]):
        for i in range(n_wires):
            qml.Rot(
                params[layer, i, 0],
                params[layer, i, 1],
                params[layer, i, 2],
                wires=i,
            )

        for i in range(n_wires - 1):
            qml.CNOT(wires=[i, i + 1])

    return qml.state()


encoder_params = torch.nn.Parameter(
    torch.randn(N_LAYERS, n_wires, 3, dtype=torch.float64) * 0.1
)

# ============================================================
# Pauli shadow utilities
# ============================================================

PAULI_I = torch.eye(2, dtype=torch.complex128)
PAULI_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
PAULI_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
PAULI_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

PAULI_MATS = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]


def generate_pauli_pool(n_qubits, max_weight):
    """
    Generate all non-identity Pauli strings up to a given weight.
    Pauli string representation:
        0 = I, 1 = X, 2 = Y, 3 = Z
    """
    pool = []
    max_weight = min(max_weight, n_qubits)

    for weight in range(1, max_weight + 1):
        for qubits in itertools.combinations(range(n_qubits), weight):
            for paulis in itertools.product([1, 2, 3], repeat=weight):
                p = [0] * n_qubits
                for q, pauli_idx in zip(qubits, paulis):
                    p[q] = pauli_idx
                pool.append(tuple(p))

    return pool


pauli_pool = generate_pauli_pool(n_wires, SHADOW_ORDER)

K = min(NUM_SHADOWS, len(pauli_pool))
if K < NUM_SHADOWS:
    print(
        f"Requested {NUM_SHADOWS} shadows, but only {len(pauli_pool)} "
        f"Pauli strings exist for order {SHADOW_ORDER}. Using all of them."
    )

perm = torch.randperm(len(pauli_pool))[:K]
selected_paulis = [pauli_pool[i] for i in perm.tolist()]

print(f"Input dim: {INPUT_DIM}")
print(f"Qubits: {n_wires}")
print(f"Shadow order: {SHADOW_ORDER}")
print(f"Pauli pool size: {len(pauli_pool)}")
print(f"Using {K} shadow features.")


def pauli_expectation(psi, pauli_string):
    """
    Differentiable expectation <psi|P|psi> for a Pauli string P.
    """
    n = len(pauli_string)
    psi_tensor = psi.reshape([2] * n)
    original = psi_tensor
    out = psi_tensor

    for i, p in enumerate(pauli_string):
        if p != 0:
            P = PAULI_MATS[p].to(psi.dtype)
            out = torch.tensordot(P, out, dims=([1], [i]))
            out = out.movedim(0, i)

    val = torch.sum(torch.conj(original) * out)
    return val.real


def compute_shadow_features(states, paulis):
    """
    states: (batch_size, 2**n) complex tensor
    paulis: list of Pauli strings

    returns: real tensor of shape (batch_size, len(paulis))
    """
    rows = []

    for b in range(states.shape[0]):
        psi = states[b]
        vals = []
        for pstr in paulis:
            vals.append(pauli_expectation(psi, pstr))
        rows.append(torch.stack(vals))

    return torch.stack(rows)


def encode_batch(params, angles_batch):
    """
    Encode a batch of data angles.
    """
    states = []
    for i in range(angles_batch.shape[0]):
        states.append(encoder_circuit(params, angles_batch[i]))
    return torch.stack(states)


# ============================================================
# Distance utilities
# ============================================================

def pairwise_shadow_distance(feats):
    """
    feats: (B, K)
    returns: (B, B) mean squared feature distance
    """
    diff = feats.unsqueeze(0) - feats.unsqueeze(1)
    return torch.mean(diff ** 2, dim=2)


def pairwise_euclidean(Z):
    """
    Z: (B, d)
    returns: (B, B) Euclidean distances
    """
    diff = Z.unsqueeze(0) - Z.unsqueeze(1)
    return torch.linalg.norm(diff, dim=2)


def pairwise_full_state_distance(states):
    """
    Pure-state Hilbert-Schmidt distance:
        || |psi_i><psi_i| - |psi_j><psi_j| ||_HS^2
        = 2 - 2 |<psi_i|psi_j>|^2
    """
    gram = torch.einsum("id,jd->ij", torch.conj(states), states)
    fidelity = torch.abs(gram) ** 2
    dist = 2.0 - 2.0 * fidelity
    return torch.clamp(dist.real, min=0.0)


# ============================================================
# Encoder evaluation / collision tracking
# ============================================================

eval_indices = torch.arange(EVAL_SIZE)
Z_eval = Z[eval_indices]
ANGLES_eval = ANGLES[eval_indices]


def evaluate_encoder(params, paulis, Z_eval, angles_eval):
    with torch.no_grad():
        states = encode_batch(params, angles_eval)
        feats = compute_shadow_features(states, paulis)

        D_shadow = pairwise_shadow_distance(feats)
        D_input = pairwise_euclidean(Z_eval)

        B = Z_eval.shape[0]
        mask = ~torch.eye(B, dtype=torch.bool)

        distinct_mask = mask & (D_input > INPUT_MARGIN)

        shadow_collision_mask = distinct_mask & (D_shadow < COLLISION_THRESHOLD)
        shadow_collisions = int(shadow_collision_mask.sum().item())

        min_shadow_dist = float(D_shadow[mask].min().item()) if B > 1 else 0.0

        D_full = pairwise_full_state_distance(states)
        full_collision_mask = distinct_mask & (D_full < FULL_COLLISION_THRESHOLD)
        full_collisions = int(full_collision_mask.sum().item())

        min_full_dist = float(D_full[mask].min().item()) if B > 1 else 0.0

    return {
        "shadow_collisions": shadow_collisions,
        "full_collisions": full_collisions,
        "min_shadow_dist": min_shadow_dist,
        "min_full_dist": min_full_dist,
    }


# ============================================================
# Phase 1: train encoder only
# ============================================================

encoder_optimizer = torch.optim.Adam([encoder_params], lr=LR_ENCODER)

# ============================================================
# Force consistent tensor shapes before TensorDataset
# ============================================================

print("Pre-dataset shapes:")
print("X.shape      :", X.shape)
print("Z.shape      :", Z.shape)
print("T.shape      :", T.shape)
print("ANGLES.shape :", ANGLES.shape)


def _fix_shape(tensor, name, n_samples, input_dim):
    tensor = tensor.detach().to(torch.float64)

    # If it is a flat vector, try to make it 2D
    if tensor.ndim == 1:
        if tensor.numel() == n_samples * input_dim:
            tensor = tensor.reshape(n_samples, input_dim)
        else:
            tensor = tensor.reshape(-1, 1)

    # If it is transposed, fix it
    if tensor.ndim == 2 and tensor.shape == (input_dim, n_samples):
        tensor = tensor.T

    # If it has extra singleton dimensions or wrong 2D shape, try reshape
    if tensor.ndim != 2 or tensor.shape != (n_samples, input_dim):
        if tensor.numel() == n_samples * input_dim:
            tensor = tensor.reshape(n_samples, input_dim)
        else:
            raise RuntimeError(
                f"Cannot fix shape of {name}. "
                f"Got {tuple(tensor.shape)}, expected {(n_samples, input_dim)}."
            )

    return tensor


# First fix X, because we use it as the reference sample count
X = torch.as_tensor(X, dtype=torch.float64)

if X.ndim == 2 and X.shape == (INPUT_DIM, N_SAMPLES):
    X = X.T

if X.numel() == N_SAMPLES * INPUT_DIM:
    X = X.reshape(N_SAMPLES, INPUT_DIM)
else:
    raise RuntimeError(
        f"Cannot fix X shape. Got {tuple(X.shape)}, "
        f"expected {(N_SAMPLES, INPUT_DIM)}."
    )

# Use the actual number of rows from X
N_ACTUAL = X.shape[0]

Z = _fix_shape(Z, "Z", N_ACTUAL, INPUT_DIM)
T = _fix_shape(T, "T", N_ACTUAL, INPUT_DIM)
ANGLES = _fix_shape(ANGLES, "ANGLES", N_ACTUAL, INPUT_DIM)

print("Fixed shapes:")
print("X.shape      :", X.shape)
print("Z.shape      :", Z.shape)
print("T.shape      :", T.shape)
print("ANGLES.shape :", ANGLES.shape)

assert X.shape == (N_ACTUAL, INPUT_DIM)
assert Z.shape == (N_ACTUAL, INPUT_DIM)
assert T.shape == (N_ACTUAL, INPUT_DIM)
assert ANGLES.shape == (N_ACTUAL, INPUT_DIM)

dataset = torch.utils.data.TensorDataset(Z, ANGLES, T, X)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

encoder_history = {
    "loss": [],
    "sep_loss": [],
    "iso_loss": [],
    "shadow_collisions": [],
    "full_collisions": [],
    "min_shadow_dist": [],
    "min_full_dist": [],
}

print("\nTraining encoder...")

for epoch in range(ENCODER_EPOCHS):
    epoch_loss = 0.0
    epoch_sep = 0.0
    epoch_iso = 0.0
    n_batches = 0

    for Z_batch, angles_batch, T_batch, X_batch in loader:
        encoder_optimizer.zero_grad()

        states = encode_batch(encoder_params, angles_batch)
        feats = compute_shadow_features(states, selected_paulis)

        D_shadow = pairwise_shadow_distance(feats)
        D_input = pairwise_euclidean(Z_batch)

        B = Z_batch.shape[0]
        mask = ~torch.eye(B, dtype=torch.bool)

        # Distinct-input mask
        distinct_mask = mask & (D_input > INPUT_MARGIN)

        # Collision / separation loss
        if distinct_mask.any():
            sep_loss = torch.mean(
                torch.relu(SEPARATION_MARGIN - D_shadow[distinct_mask]) ** 2
            )
        else:
            sep_loss = torch.zeros((), dtype=torch.float64)

        # Weak isometry loss.
        # Input distances are bounded to [0, 1) to keep scales comparable.
        input_bounded = D_input / (D_input + 1.0)

        if mask.any():
            iso_loss = torch.mean(
                (D_shadow[mask] - 0.5 * input_bounded[mask]) ** 2
            )
        else:
            iso_loss = torch.zeros((), dtype=torch.float64)

        loss = sep_loss + ISO_WEIGHT * iso_loss

        loss.backward()
        encoder_optimizer.step()

        epoch_loss += float(loss.item())
        epoch_sep += float(sep_loss.item())
        epoch_iso += float(iso_loss.item())
        n_batches += 1

    eval_metrics = evaluate_encoder(
        encoder_params,
        selected_paulis,
        Z_eval,
        ANGLES_eval,
    )

    encoder_history["loss"].append(epoch_loss / max(n_batches, 1))
    encoder_history["sep_loss"].append(epoch_sep / max(n_batches, 1))
    encoder_history["iso_loss"].append(epoch_iso / max(n_batches, 1))
    encoder_history["shadow_collisions"].append(eval_metrics["shadow_collisions"])
    encoder_history["full_collisions"].append(eval_metrics["full_collisions"])
    encoder_history["min_shadow_dist"].append(eval_metrics["min_shadow_dist"])
    encoder_history["min_full_dist"].append(eval_metrics["min_full_dist"])

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch+1:3d} | "
            f"loss {encoder_history['loss'][-1]:.4f} | "
            f"sep {encoder_history['sep_loss'][-1]:.4f} | "
            f"iso {encoder_history['iso_loss'][-1]:.4f} | "
            f"shadow_collisions {eval_metrics['shadow_collisions']:3d} | "
            f"full_collisions {eval_metrics['full_collisions']:3d} | "
            f"min_shadow {eval_metrics['min_shadow_dist']:.4f} | "
            f"min_full {eval_metrics['min_full_dist']:.4f}"
        )


# ============================================================
# Phase 2: train decoder only, encoder frozen
# ============================================================

print("\nPrecomputing frozen encoder shadow features...")


def compute_dataset_features(params, angles, paulis, chunk_size=64):
    feats_list = []

    for start in range(0, len(angles), chunk_size):
        angles_chunk = angles[start:start + chunk_size]

        with torch.no_grad():
            states = encode_batch(params, angles_chunk)
            feats = compute_shadow_features(states, paulis)

        feats_list.append(feats)

    return torch.cat(feats_list, dim=0)


all_feats = compute_dataset_features(
    encoder_params,
    ANGLES,
    selected_paulis,
    chunk_size=64,
)


class ShadowDecoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        hidden = max(64, 2 * input_dim, 2 * output_dim)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, output_dim),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


decoder = ShadowDecoder(K, INPUT_DIM).double()
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LR_DECODER)

decoder_dataset = torch.utils.data.TensorDataset(all_feats, T, X)
decoder_loader = torch.utils.data.DataLoader(
    decoder_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

decoder_history = {
    "decoder_T_mse": [],
    "decoder_X_mse": [],
}

print("\nTraining decoder...")

for epoch in range(DECODER_EPOCHS):
    epoch_loss = 0.0
    n_batches = 0

    for feats_batch, T_batch, X_batch in decoder_loader:
        decoder_optimizer.zero_grad()

        pred_T = decoder(feats_batch)
        loss = torch.nn.functional.mse_loss(pred_T, T_batch)

        loss.backward()
        decoder_optimizer.step()

        epoch_loss += float(loss.item())
        n_batches += 1

    with torch.no_grad():
        pred_T_all = decoder(all_feats)
        decoder_T_mse = float(
            torch.nn.functional.mse_loss(pred_T_all, T).item()
        )

        pred_T_clamped = torch.clamp(pred_T_all, 1e-6, 1.0 - 1e-6)
        pred_Z = torch.log(pred_T_clamped / (1.0 - pred_T_clamped))
        pred_X = pred_Z @ unwhiten.T + data_mean

        decoder_X_mse = float(
            torch.nn.functional.mse_loss(pred_X, X).item()
        )

    decoder_history["decoder_T_mse"].append(decoder_T_mse)
    decoder_history["decoder_X_mse"].append(decoder_X_mse)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:3d} | "
            f"decoder_T_mse {decoder_T_mse:.6f} | "
            f"decoder_X_mse {decoder_X_mse:.6f}"
        )


# ============================================================
# Final summary
# ============================================================

final_metrics = {
    "input_dim": INPUT_DIM,
    "n_qubits": n_wires,
    "shadow_order": SHADOW_ORDER,
    "requested_shadows": NUM_SHADOWS,
    "used_shadows": K,
    "final_encoder_loss": encoder_history["loss"][-1],
    "final_sep_loss": encoder_history["sep_loss"][-1],
    "final_iso_loss": encoder_history["iso_loss"][-1],
    "final_shadow_collisions": encoder_history["shadow_collisions"][-1],
    "final_full_collisions": encoder_history["full_collisions"][-1],
    "final_min_shadow_dist": encoder_history["min_shadow_dist"][-1],
    "final_min_full_dist": encoder_history["min_full_dist"][-1],
    "final_decoder_T_mse": decoder_history["decoder_T_mse"][-1],
    "final_decoder_X_mse": decoder_history["decoder_X_mse"][-1],
}

summary_df = pd.DataFrame([final_metrics]).T
summary_df.columns = ["value"]

print("\nFinal summary:")
print(summary_df)


# ============================================================
# Plots
# ============================================================

fig, axs = plt.subplots(1, 3, figsize=(18, 4))

# Encoder losses
axs[0].plot(encoder_history["loss"], label="total")
axs[0].plot(encoder_history["sep_loss"], label="separation")
axs[0].plot(encoder_history["iso_loss"], label="isometry")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Encoder loss")
axs[0].set_title("Encoder training")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Collisions and distances
axs[1].plot(encoder_history["shadow_collisions"], label="shadow collisions")
axs[1].plot(encoder_history["full_collisions"], label="full-state collisions")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Collision count")
axs[1].set_title("Collisions")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

ax2 = axs[1].twinx()
ax2.plot(encoder_history["min_shadow_dist"], "--", label="min shadow dist")
ax2.plot(encoder_history["min_full_dist"], "--", label="min full dist")
ax2.set_ylabel("Minimum distance")
ax2.legend(loc="center right")

# Decoder losses
axs[2].plot(decoder_history["decoder_T_mse"], label="bounded-coordinate MSE")
axs[2].plot(decoder_history["decoder_X_mse"], label="original-space MSE")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Decoder loss")
axs[2].set_title("Decoder training")
axs[2].set_yscale("log")
axs[2].legend()
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
