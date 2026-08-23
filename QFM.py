import numpy as np
import scipy.sparse as sps
import pennylane as qml
import torch 
import torch.nn as nn
import yaml

from scipy.linalg import expm
from itertools import product 
from functools import reduce
from typing import Optional, Iterable

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

DTYPE = torch.complex32

# Define Pauli matrices as PyTorch tensors for differentiability
I = torch.tensor(np.eye(2), dtype=DTYPE)
X = torch.tensor(np.array([[0, 1], [1, 0]]), dtype=DTYPE)
Y = torch.tensor(np.array([[0, -1j], [1j, 0]]), dtype=DTYPE)
Z = torch.tensor(np.array([[1, 0], [0, -1]]), dtype=DTYPE)


class QFM():

    '''
    Implements Quantum Flow Matching algorithm
    '''

    def __init__(self,
                 N: int = 5,
                 locality: int = 3,
                 M: int = 50,
                 sigma_min: float = 1e-4,
                 device: str = 'default.qubit',
                 n_layers_haar: int | None = None,
                 U_phi: torch.Tensor | None = None,
                 init_weights: torch.Tensor | None = None
                 target_weights: torch.Tensor | None = None
                 ) -> None:

        kwargs = locals()
        kwargs.pop('self')

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.ancilla = 0
        self.register = filter(lambda x: True if x != self.ancilla else False,
                               range(self.N)
                               )
        self.dim = 1 << self.N
        self.t = np.random.uniform(0, 1)
        # We need N+1 wires for the Hadamard test (1 ancilla + N system)
        self.dev = qml.device(device, wires=self.N + 1)
        self.delta_t = 1 / self.M

        self.H_train = self.create_Hamiltonian()

        self.target_weights = torch.randn(size=self.weights) if self.target_weights is None\
                else self.target_weights 

        self.H_target = self.create_Hamiltonian(weights=self.target_weights)

    def Haar_sample(self,
                    n_layers: int | None = None
                    ) -> np.ndarray:
        '''
        Samples from Haar dist
        '''

        if self.n_layers_haar is not None:
            n_layers = self.n_layers_haar

        ttl_wires = filter(lambda x: True if x != self.ancilla else False,
                           range(self.N)
                           )

        for idl in range(n_layers):

            for i, w in enumerate(ttl_wires):
                qml.RX(np.random.rand(), wires=w)
                qml.RY(np.random.rand(), wires=w)
                qml.RZ(np.random.rand(), wires=w)
                
            n_combs = (self.N - 1) * (self.N - 2)  /  2

            for id_pair, (i, j) in enumerate(combinations(ttl_wires, 2)):
                qml.IsingZZ(params[idl, (self.N - 1)*3 + id_pair], wires=[i, j])

        return qml.state()

    def inner_prod(self,
                   U_psi: torch.Tensor,
                   U_phi: torch.Tensor,
                   im: bool = False,
                   ) -> float:

        '''
        Circuit to calculate the inner prod of two vectors induced by arbitrary unitaries
        '''

        qml.Hadamard(self.ancilla)

        qml.QubitUnitary(U_phi.H,
                         wires=filter(lambda x: True if x != self.ancilla else False,
                                      range(self.N)
                                      )
                         )

        qml.QubitUnitary(U_psi,
                         wires=filter(lambda x: True if x != self.ancilla else False,
                                      range(self.N)
                                      )
                         )

        if im:
            qml.S(wires=self.ancilla)

        qml.Hadamard(wires=self.ancilla)

        return qml.expval(qml.PauliZ(wires=self.ancilla))

    def create_Hamiltonian(self,
                            locality: int | None = None,
                            weights: torch.Tensor | None = None
                            ) -> torch.Tensor:
        """Creates a parametrised Hamiltonian operator out of Pauli strings."""

        if locality is None:
            locality = self.locality

        conversion = {'I' : I, 'X' : X, 'Z' : Z, 'Y' : Y}
       
        combs = filter(lambda x: True if x.count('I') == (self.N - locality) else False,
                       product(conversion.keys(), repeat=self.N)
                       )

        if weights is None:
            if self.weights is None:
                weights = map(lambda x: np.random.rand(), combs)
                self.weights = weights
            else:
                weights = self.weights

        for idx, (comb, w) in enumerate(zip(combs, weights)):
            mat: torch.Tensor = reduce(sps.kron,
                                       map(lambda x: sps.csr_matrix(conversion[x].detach().cpu().to(complex).numpy()),
                                                     comb
                                           )
                                       )
            
            if idx == 0:
                Hamiltonian = w * mat 
            else:
                Hamiltonian += w * mat

        return torch.sparse_csr_tensor(
                torch.from_numpy(Hamiltonian.indptr),
                torch.from_numpy(Hamiltonian.indices),
                torch.from_numpy(Hamiltonian.data),
                size=Hamiltonian.shape
                ).to(DTYPE)

    def geodesic_operator(v_a: complex,
                          v_b: complex,
                          coeffs: tuple(complex) = None
                          ) -> sps.csr_array:
        '''
        H such that expm(-1j*H*t) @ v_a goes from |psi_a> (t=0) to the
        NORMALIZED combination coeffs[0]*|psi_a> + coeffs[1]*|psi_b> (t=1).
        coeffs=None -> Fubini-Study geodesic of eqs. (3)-(4).
        '''

        #zero = sps.csr_array(shape=(U_a.shape[0], 1))
        #zero[0] = 1

        #We know U_a and U_b canonically, so it's easy, we'll have to simulate that anyway
        #v_a, v_b = U_a @ zero, U_b @ zero

        c = np.vdot(v_a, v_b)
        mod = abs(c)
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
    
    @qml.qnode(dev=self.device)
    def forward(x: torch.Tensor) -> torch.Tensor:
   

        

       
