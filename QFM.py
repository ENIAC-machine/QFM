import numpy as np
import scipy.sparse as sps
import pennylane as qml
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml


from torch.utils.data import Dataset, DataLoader
from scipy.linalg import expm
from itertools import product, combinations
from functools import reduce
from typing import Optional, Iterable, Callable, Sequence
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


class Unitary_Dataset(Dataset):

    '''
    Custom dataset to hold unitary operators for some dist,
    can be initialized from a list of csr_arrays
    Yes, they gotta be csr_arrays
    '''

    def __init__(self,
                 arr: Sequence[sps.csr_array]
                 ) -> None:
        self.items = arr 

    def __len__(self):
        return self.items.__len__()

    def __getitem__(self,
                    idx: int
                    ) -> sps.csr_array:
        return self.items[idx]


class QFM(nn.Module):

    '''
    Implements Quantum Flow Matching algorithm
    '''

    def __init__(self,
                 N: int = 5,
                 ancillas: Sequence[int] = (0, 1),
                 locality: int = 3,
                 sigma_min: float = 1e-4,
                 device: str = 'default.qubit',
                 n_layers_haar: int | None = None,
                 init_weights: torch.Tensor | None = None,
                 target_weights: torch.Tensor | None = None
                 ) -> None:

        assert len(ancillas) < N,\
                "The total number of ancillas must be less than the total number of qubits"

        super().__init__()

        self.N = N
        self.locality = locality
        self.sigma_min = sigma_min
        self.device_name = device
        self.n_layers_haar = n_layers_haar
        
        self.ancillas = ancillas
        self.ancilla_Hadamard = ancillas[0] #ancilla for the Hadamard test
        self.ancilla_LCU = ancillas[1] #ancilla for the LCU
        self.register = [x for x in range(self.N + 1) if x not in self.ancillas]
        self.dim = 1 << self.N
        
        self.dev = qml.device(self.device_name, wires=self.N + 1)
        
        # Calculate the number of Pauli terms with specific locality
        self.num_terms = comb(len(self.register), self.locality) * (3 ** self.locality)
        
        #init weights for H_theta
        if init_weights is None:
            self.weights = nn.Parameter(torch.randn(self.num_terms, dtype=torch.float32))
        else:
            self.weights = nn.Parameter(init_weights)
        
        self.H_train = self.create_Hamiltonian(weights=self.weights)
        
        '''
        #weights for H_target
        if target_weights is None:
            self.target_weights = torch.randn(self.num_terms, dtype=torch.float32)
        else:
            self.target_weights = target_weights
            
        self.H_target = self.create_Hamiltonian(weights=self.target_weights)
        '''
        
        #create target dataset
        self.dataset = self.create_target_dataset()

        H_train_dense = self.H_train.to(torch.complex64)
        
        self.U_haar = torch.matrix_exp(-1j * H_train_dense).detach()
        
        # |0> state
        self.zero_state = torch.zeros(self.dim, dtype=torch.complex64)
        self.zero_state[0] = 1.0
       
        # Compile the circuit 
        self.circuit = self.get_qnode()

    def Haar_sample(self,
                    n_layers: int | None = None
                    ) -> np.ndarray:
        '''
        Samples from (kinda) Haar dist. Kinda the scrambling circuit from that QFM paper
        
        Inputs:
            n_layers: int - number of layers in the scrambilng circuit  

        Outputs:
           state of the system as a numpy ndarray 
        '''

        if n_layers is None:
            n_layers = self.n_layers_haar
        elif self.n_layers_haar is None:
            raise ValueError('The number of layers should be present either upon initialization of the class or the function call')

        n_combs = (self.N * (self.N - 1)) // 2
        params = np.random.rand(n_layers, self.N * 3 + n_combs)

        for idl in range(n_layers):
            for i, w in enumerate(self.register):
                qml.RX(params[idl, i*3], wires=w)
                qml.RY(params[idl, i*3+1], wires=w)
                qml.RZ(params[idl, i*3+2], wires=w)
                
            for id_pair, (i, j) in enumerate(combinations(self.register, 2)):
                qml.IsingZZ(params[idl, self.N*3 + id_pair], wires=[i, j])

        return qml.state()

    def create_Hamiltonian(self,
                           weights: torch.Tensor
                           ) -> torch.Tensor:
        '''
        Creates a parametrised Hamiltonian operator out of scaled Pauli strings
        H = \sum_{i=1}^{n} alpha_i * S_i , where S_i can be S_i = I \otimes X \otimes I
        '''
        
        conversion = {'I' : I, 'X' : X, 'Z' : Z, 'Y' : Y}
        
        combs = list(
                    filter(lambda x: x.count('I') == (len(self.register) - self.locality),
                           product(['I', 'X', 'Y', 'Z'], repeat=len(self.register))
                           )
                )
        
        # Build dense tensors directly to maintain the PyTorch computational graph 
        # for torch.matrix_exp, which does not support sparse matrices.
        dim = 1 << len(self.register)
        H = torch.zeros((dim, dim), dtype=DTYPE)
       
        for idx, (comb, w) in enumerate(zip(combs, weights)):
            mat = reduce(torch.kron, map(lambda x: conversion[x], comb))
            H = H + w * mat

        return H

    def create_target_dataset(self, size: int = 1_000) -> Unitary_Dataset:
        target_weights = torch.randn((self.num_terms,), dtype=torch.float32)
        self.H_target = self.create_Hamiltonian(weights=target_weights)
        
        unitaries = []
        for i in range(size):
            target_weights[0] = torch.rand((1,))
            unitaries.append(self.create_Hamiltonian(weights=target_weights))
        return Unitary_Dataset(unitaries) 

    def geodesic(self,
                 t: torch.Tensor, #tensor of 1 value
                 U_psi: np.ndarray,
                 U_phi: np.ndarray,
                 delta: float = .0,
                 conj_t: bool = False
                 ) -> None:
        '''
        LCU for constant-speed geodesic with weights t and t-1 for U_psi and U_phi respectively

        Inputs:
            t - weight
            U_psi - the initial unitary (Haar in our case)
            U_phi - the target unitary (from target dist)
            delta - phase alignment value, if states have overlap <psi|phi> = |c|e^{i*delta},
            we need to apply exp(-i*delta)
            conj_t - whether we need the conjugate transpose of the Unitary obtained
                via LCU, defaults to False
        '''

        theta = 2 * torch.arcsin(torch.sqrt(t))

        ops = [
                qml.RY(theta, wires=self.ancilla_LCU),
                qml.ctrl(
                    qml.QubitUnitary(U_psi, wires=self.register), 
                    control=self.ancilla_LCU, control_values=0
                    ),
                qml.ctrl(
                    qml.QubitUnitary(U_phi, wires=self.register), 
                    control=self.ancilla_LCU, control_values=1
                    ),
    ]
    
        if delta != .0:
            ops.append(qml.RZ(-delta, wires=self.ancilla_LCU))
            
        ops.append(qml.RY(-theta, wires=self.ancilla_LCU))

        #reverse operations for the conjugate transpose case
        for op in ops[::1-2*conj_t]:
            qml.apply(op)

    def get_qnode(self) -> Callable:
        
        @qml.qnode(self.dev, interface='torch')
        def circuit(t: float | torch.Tensor,
                    U_target: sps.csr_array,
                    weights : torch.Tensor
                    ) -> float:

            t_val = torch.Tensor([t]) if isinstance(t, float) else t
            
            # 1. Evolution operator for U_theta (parametrized Hamiltonian)
            H_theta = self.create_Hamiltonian(weights=weights)
            H_theta_dense = H_theta.to(DTYPE)
            U_theta = torch.matrix_exp(-1j * H_theta_dense * t_val)
           
            #2. Start from noise 
            qml.QubitUnitary(self.U_haar, wires=self.register)

            #3. Inner product (Re) via Hadamard Test (Fixed with qml.ctrl) to get denoising quality
            qml.Hadamard(wires=self.ancilla_Hadamard)
           
            #Apply controlled U_theta
            qml.ctrl(qml.QubitUnitary, control=self.ancilla_Hadamard)(U_theta.to(DTYPE), wires=self.register)
           
            #Apply the conjugate transpose of the operator that induces geodesic interpolation
            self.geodesic(t_val, self.U_haar, U_target, conj_t=True)

            qml.Hadamard(wires=self.ancilla_Hadamard)
            
            #Всё
            return qml.expval(qml.PauliZ(wires=self.ancilla_Hadamard))
            
        return circuit

    def forward(self,
                x: sps.csr_array
                ) -> torch.Tensor:

        # Sample t ~ Uniform[0, 1]
        t = torch.rand(1)
        
        # Find inner prod (Re) of U_theta and the geodesic interpolation
        inner_prod = self.circuit(t, x, self.weights)
        
        # Compute Loss the inner prod above
        loss = inner_prod
        
        return loss


def train_qfm(num_epochs: int = 200,
              batch_size: int = 8,
              lr: float = 0.05,
              N: int = 3,
              locality: int = 2,
              shuffle: bool = True
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

    dataloader = DataLoader(model.dataset, batch_size=batch_size, shuffle=shuffle)

    # 4. Training loop
    for epoch in tqdm(range(num_epochs), desc="Training"):
        optimizer.zero_grad()
        
        # Accumulate gradients over a mini-batch of random time samples 't'
        batch_loss = 0.0
        for batch in dataloader:
            # Forward pass samples a random t ~ U[0,1] internally
            loss = model.forward(batch).sum()**2
            batch_loss += loss.reshape(-1)
            
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
        num_epochs=100, 
        batch_size=16,    # Average gradient over 8 random time samples per step
        lr=0.05, 
        N=5,             # 3 qubits (8-dimensional Hilbert space)
        locality=2       # 2-local Hamiltonian interactions
    )
    
    print("\nTraining complete!")
    
    # Optional: Save the trained weights
    torch.save(trained_model.weights.data, "qfm_trained_weights.pt")
    print("Trained weights saved to 'qfm_trained_weights.pt'")
