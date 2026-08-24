import numpy as np
import scipy.sparse as sps
import pennylane as qml
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml

from scipy.linalg import expm
from itertools import product, combinations
from functools import reduce
from typing import Optional, Iterable, Callable
from math import comb
from tqdm import tqdm

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

DTYPE = torch.complex64

# Define Pauli matrices as PyTorch tensors for differentiability
I = torch.tensor(np.eye(2), dtype=DTYPE)
X = torch.tensor(np.array([[0, 1], [1, 0]]), dtype=DTYPE)
Y = torch.tensor(np.array([[0, -1j], [1j, 0]]), dtype=DTYPE)
Z = torch.tensor(np.array([[1, 0], [0, -1]]), dtype=DTYPE)


class QFM(nn.Module):

    '''
    Implements Quantum Flow Matching algorithm
    '''

    def __init__(self,
                 N: int = 5,
                 locality: int = 3,
                 sigma_min: float = 1e-4,
                 device: str = 'default.qubit',
                 n_layers_haar: int | None = None,
                 init_weights: torch.Tensor | None = None,
                 target_weights: torch.Tensor | None = None
                 ) -> None:

        super().__init__()

        self.N = N
        self.locality = locality
        self.sigma_min = sigma_min
        self.device_name = device
        self.n_layers_haar = n_layers_haar
        
        self.ancilla = 0
        self.register = [x for x in range(self.N + 1) if x != self.ancilla]
        self.dim = 1 << self.N
        
        # We need N+1 wires for the Hadamard test (1 ancilla + N system)
        self.dev = qml.device(self.device_name, wires=self.N + 1)
        
        # Calculate the number of Pauli terms with specific locality
        self.num_terms = comb(self.N, self.locality) * (3 ** self.locality)
        
        # Initialize weights as nn.Parameter for PyTorch compatibility
        if init_weights is None:
            self.weights = nn.Parameter(torch.randn(self.num_terms, dtype=torch.float32))
        else:
            self.weights = nn.Parameter(init_weights)
            
        self.H_train = self.create_Hamiltonian(weights=self.weights)
        
        if target_weights is None:
            self.target_weights = torch.randn(self.num_terms, dtype=torch.float32)
        else:
            self.target_weights = target_weights
            
        self.H_target = self.create_Hamiltonian(weights=self.target_weights)

        # Precompute U_target and U_Haar unitaries (fixed during training)
        # torch.matrix_exp is used instead of scipy.linalg.expm to maintain the computational graph
        H_target_dense = self.H_target.to(torch.complex64)
        H_train_dense = self.H_train.to(torch.complex64)
        
        self.U_target = torch.matrix_exp(-1j * H_target_dense).detach()
        self.U_haar = torch.matrix_exp(-1j * H_train_dense).detach()
        
        # |0> state
        self.zero_state = torch.zeros(self.dim, dtype=torch.complex64)
        self.zero_state[0] = 1.0
        
        # Prior and Target states
        self.v_a = self.U_target @ self.zero_state
        self.v_b = self.U_haar @ self.zero_state
        
        # Precompute geodesic operator (generator of the Fubini-Study geodesic)
        # coeffs=None gives the true geodesic path between v_a and v_b
        v_a_np = self.v_a.detach().cpu().numpy()
        v_b_np = self.v_b.detach().cpu().numpy()
        H_geo_np = self.geodesic_operator(v_a_np, v_b_np, coeffs=None)
        self.H_geodesic = torch.tensor(H_geo_np, dtype=torch.complex64)
        
        # Compile the QNode once during initialization
        self.circuit = self.get_qnode()

    def Haar_sample(self,
                    n_layers: int | None = None
                    ) -> np.ndarray:

        ''' Samples from Haar dist '''
        if self.n_layers_haar is not None:
            n_layers = self.n_layers_haar

        ttl_wires = [x for x in range(self.N + 1) if x != self.ancilla]
        
        # Random parameters for the Haar random circuit
        n_combs = (self.N * (self.N - 1)) // 2
        params = np.random.rand(n_layers, self.N * 3 + n_combs)

        for idl in range(n_layers):
            for i, w in enumerate(ttl_wires):
                qml.RX(params[idl, i*3], wires=w)
                qml.RY(params[idl, i*3+1], wires=w)
                qml.RZ(params[idl, i*3+2], wires=w)
                
            for id_pair, (i, j) in enumerate(combinations(ttl_wires, 2)):
                qml.IsingZZ(params[idl, self.N*3 + id_pair], wires=[i, j])

        return qml.state()

    def inner_prod(self,
                   U_psi: torch.Tensor,
                   U_phi: torch.Tensor,
                   im: bool = False
                   ) -> float:
        '''
        Circuit to calculate the real part of the inner prod of two vectors 
        induced by arbitrary unitaries using the Hadamard test.
        '''
        qml.Hadamard(self.ancilla)

        # Controlled-U_phi^dagger
        qml.ctrl(qml.QubitUnitary, control=self.ancilla)(U_phi.conj().T, wires=self.register)

        # Controlled-U_psi
        qml.ctrl(qml.QubitUnitary, control=self.ancilla)(U_psi, wires=self.register)

        if im:
            qml.S(wires=self.ancilla)

        qml.Hadamard(wires=self.ancilla)

        return qml.expval(qml.PauliZ(wires=self.ancilla))

    def create_Hamiltonian(self,
                           weights: torch.Tensor
                           ) -> torch.Tensor:
        """Creates a parametrised Hamiltonian operator out of Pauli strings."""
        conversion = {'I' : I, 'X' : X, 'Z' : Z, 'Y' : Y}
       
        combs = list(filter(lambda x: x.count('I') == (self.N - self.locality),
                            product(['I', 'X', 'Y', 'Z'], repeat=self.N)))
        
        # Build dense tensors directly to maintain the PyTorch computational graph 
        # for torch.matrix_exp, which does not support sparse matrices.
        dim = 1 << self.N
        H = torch.zeros((dim, dim), dtype=DTYPE)
        
        for idx, (comb, w) in enumerate(zip(combs, weights)):
            mat = reduce(torch.kron, map(lambda x: conversion[x], comb))
            H = H + w * mat

        return H

    @staticmethod
    def geodesic_operator(v_a: np.ndarray,
                          v_b: np.ndarray,
                          coeffs: tuple = None
                          ) -> np.ndarray:
        '''
        H such that expm(-1j*H*t) @ v_a goes from |psi_a> (t=0) to the
        NORMALIZED combination coeffs[0]*|psi_a> + coeffs[1]*|psi_b> (t=1).
        coeffs=None -> Fubini-Study geodesic.
        '''
        c = np.vdot(v_a, v_b)
        mod = abs(c)
        
        # Safety cap to avoid NaNs in sqrt/arccos if states are parallel/anti-parallel
        if mod >= 1.0:
            mod = 0.9999999

        alpha = np.arccos(np.clip(mod, -1, 1))
        beta = np.angle(c)
        e1 = (v_b - c * v_a) / np.sqrt(1 - mod**2)          # eq. (2)

        if coeffs is None:   # geodesic endpoint of eq. (3) at t=1
            w0 = np.cos(alpha) - np.sin(alpha)*np.exp(1j*beta)*c/np.sqrt(1 - mod**2)
            w1 = np.sin(alpha)*np.exp(1j*beta)/np.sqrt(1 - mod**2)
        else:
            w0, w1 = coeffs

        target = w0*v_a + w1*v_b
        norm   = np.linalg.norm(target)
        if norm < 1e-12:
            raise ValueError("coefficients give (almost) the zero vector")
        target /= norm

        # coordinates of target in orthonormal basis {|e0>, |e1>}
        u0 = np.vdot(v_a, target)
        u1 = np.vdot(e1,  target)

        ref   = np.angle(u0) if abs(u0) > 1e-12 else np.angle(u1)  # global phase
        a1    = u1*np.exp(-1j*ref)                                 # a0 = |u0| real >= 0
        theta = np.arctan2(abs(u1), abs(u0))                       # rotation angle
        phi   = np.angle(a1) if abs(a1) > 1e-12 else 0.0           # relative phase

        f = np.exp(1j*phi)*e1
        H = 1j*theta*(np.outer(f, v_a.conj()) - np.outer(v_a, f.conj()))  # eq. (4) form
        H -= ref*np.eye(len(v_a))    # restores the exact global phase
        return H
    
    def get_qnode(self) -> Callable:
        
        @qml.qnode(self.dev, interface='torch')
        def circuit(t: float | torch.Tensor,
                    weights : torch.Tensor
                    ) -> float:
            t_val = t.item() if isinstance(t, torch.Tensor) else t
            
            # 1. Geodesic interpolation (evolve v_a with H_geodesic for time t)
            U_geodesic = torch.matrix_exp(-1j * self.H_geodesic * t_val).to(DTYPE)
            
            # 2. Evolution operator for U_theta (parametrized Hamiltonian)
            H_theta = self.create_Hamiltonian(weights=weights)
            H_theta_dense = H_theta.to(DTYPE)
            U_theta = torch.matrix_exp(-1j * H_theta_dense * t_val)
            
            # The target state at time  is |psi_t> = U_geodesic * U_target |0>
            U_psi = U_geodesic @ self.U_target.to(DTYPE)
            
            # 3. Inner product (Re) via Hadamard Test (Fixed with qml.ctrl)
            qml.Hadamard(wires=self.ancilla)
            
            # Apply controlled U_theta
            qml.ctrl(qml.QubitUnitary, control=self.ancilla)(U_theta.to(DTYPE), wires=self.register)
            
            # Apply controlled U_psi^dagger
            qml.ctrl(qml.QubitUnitary, control=self.ancilla)(U_psi.conj().T, wires=self.register)
            
            qml.Hadamard(wires=self.ancilla)
            
            return qml.expval(qml.PauliZ(wires=self.ancilla))
            
        return circuit

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        x isn't actually used here cause we try to approx a target hamiltonian but I left 
        for forward compatibility and yada yada
        '''

        # Sample t ~ Uniform[0, 1]
        t = torch.rand(1)
        
        # Find inner prod (Re) of U_theta and the geodesic interpolation
        inner_prod = self.circuit(t, self.weights)
        
        # Compute Loss the inner prod above
        loss = inner_prod
        
        return loss


def train_qfm(num_epochs: int = 200,
              batch_size: int = 8,
              lr: float = 0.05,
              N: int = 3,
              locality: int = 2
              ):
    """
    Sample training script for the Quantum Flow Matching (QFM) model.
    """
    # 1. Device setup (QFM currently relies on CPU-based statevector simulation)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Initialize the model
    print("Initializing QFM model...")
    # We use a small N (e.g., 3) for demonstration so the script runs quickly.
    # The Hilbert space dimension scales as 2^N.
    model = QFM(N=N, locality=locality).to(device)
    
    # 3. Setup optimizer
    # Only self.weights are nn.Parameters, so model.parameters() captures them correctly
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # List to store loss history for plotting
    loss_history = []
    
    print(f"Starting training for {num_epochs} epochs with batch_size={batch_size}...")
    print("Note: The theoretical minimum loss is 0.0 (states match up to a global phase).\n")
    
    # 4. Training loop
    for epoch in tqdm(range(num_epochs), desc="Training"):
        optimizer.zero_grad()
        
        # Accumulate gradients over a mini-batch of random time samples 't'
        batch_loss = 0.0
        for _ in range(batch_size):
            # Forward pass samples a random t ~ U[0,1] internally
            loss = model.forward()
            batch_loss += loss
            
        # Average the loss over the batch
        batch_loss = batch_loss / batch_size
        
        # Backward pass
        batch_loss.backward()
        
        # Optimization step
        optimizer.step()
        
        # Logging
        loss_val = batch_loss.item()
        loss_history.append(loss_val)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Loss: {loss_val:.6f}")
            
    # 5. Plotting the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss', color='b', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Quantum Flow Matching (QFM) Training Loss Curve', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return model, loss_history

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    
    # Run training
    trained_model, history = train_qfm(
        num_epochs=1_000, 
        batch_size=16,    # Average gradient over 8 random time samples per step
        lr=0.05, 
        N=5,             # 3 qubits (8-dimensional Hilbert space)
        locality=2       # 2-local Hamiltonian interactions
    )
    
    print("\nTraining complete!")
    
    # Optional: Save the trained weights
    torch.save(trained_model.weights.data, "qfm_trained_weights.pt")
    print("Trained weights saved to 'qfm_trained_weights.pt'")
