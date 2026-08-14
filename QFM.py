import numpy as np
import pennylane as qml
import torch 
import torch.nn as nn

from itertools import product 
from functools import reduce

from scipy.special import factorial
from scipy.sparse import csr_array
from scipy.linalg import expm

from typing import Iterable, Generator, Iterator

seed = 42
np.random.seed(seed)


I = np.eye(2, dtype=np.complex64)
X = np.eye(2, dtype=np.complex64)[::-1]
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)


class QFM():

    def __init__(self,
                 N: int = 5,
                 M: int = 50,
                 sigma_min: float = 1e-4,
                 device: str = 'default.qubit'
                 ) -> None:

        '''

        '''

        kwargs = locals()
        kwargs.pop('self', None)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.ancilla = 0
        self.dim = 1 << N
        self.t = np.random.uniform(0, 1)
        self.dev = qml.device(device, wires=N)
        self.delta_t = 1 / M

    
    def create_Hamiltonian(locality: int,
                           params: str = 'random'
                           ) -> Generator[tuple(int, Iterator),
                                          Iterable,
                                          tuple(csr_array, Iterable)]:
    
        '''
        Creates a parametrised Hamiltonian operator out of Pauli strings
        '''

        conversion = {'I' : I,
                  'X' : X,
                  'Z' : Z,
                  'Y' : Y}

        assert locality <= n_qubits, 'locality must be less or equal than the number of qubits'
        
        #make combinations of Pauli strings that will be used for Hamiltonian like XIX for locality=2
        combs = map(lambda x: (conversion[x_i] for x_i in x),
                    filter(lambda x: False if ''.join(x).count('I') != n_qubits - locality else True,
                       product(*[('I', 'X', 'Z', 'Y') for i in range(n_qubits)])
                           )
                    )
        
        #make it a csr array cause memory
        Hamiltonian = csr_array(np.zeros((1 << n_qubits, 1 << n_qubits), dtype=np.complex64))
        
        #random params
        if params == 'random':

            weights = []
            for comb in combs:
                weight = np.random.rand()
                Hamiltonian: csr_array = Hamiltonian + csr_array(weight * reduce(np.kron, comb))
                weights.append(weight)
        
        elif params == 'custom':
        
            len_weights = sum(1 for _ in combs)

            weights = yield len_weights, combs 
            
            for comb, weight in zip(combs, weights, strict=True):
                Hamiltonian: csr_array = Hamiltonian + csr_array(weight * reduce(np.kron, comb))


        return Hamiltonian, weights

    @staticmethod
    def Haar(n_layers: int) -> qml.measurements.state.StateMP:


        #reset all
        for wire in range(1, N):
            qml.measure(wires=wire, reset=True)

        if depth is None:
            depth = 4*N

        for d in range(depth):
            
            # we leave out the ancilla
            for q in range(1, N):
                qml.U3(*params[d, q], wires=q)
            
            # we leave out the ancilla
            for q in range(1, N - 1):
                qml.CNOT(wires=[q, q + 1])
        
            # we leave out the ancilla
            for q in range(1, N):
                qml.U3(*params[depth, q], wires=q)

        return qml.state()


    @staticmethod
    @qml.qnode(device)
    def Find_angles() -> Iterator[int]:

        #reset all
        for wire in range(N):
            qml.measure(wires=wire, reset=True)

        qml.Hadamard(wires=ancilla)

        qml.ctrl(qml.QubitUnitary, control=ancilla)(U_target, wires=range(1, N))

        qml.Hadamard(wires=ancilla)

        #Re
        yield qml.expval(qml.Z(ancilla))

        for wire in range(N):
            qml.measure(wires=wire, reset=True)

        qml.Hadamard(wires=ancilla)

        qml.ctrl(qml.QubitUnitary, control=ancilla)(U_target, wires=range(1, N))

        qml.adjoint(qml.S)(wires=ancilla)

        qml.Hadamard(wires=ancilla)

        #Im
        yield qml.expval(qml.Z(ancilla))


    def _get_weights_for_lin_comb(self,
                                  *args
                                  ) -> tuple(float, float, complex):

        '''
        Given the Real and Imaginary parts of the inner product of $psi_1$ and $psi_0$,
        which are the state vectors that are induces by their respective unitary operators,
        we can find the weights to use to find a linear combination of $psi_1$ and $psi_2$ projected
        onto the non-euclidean (geodesic) plane (bloch-sphere for 1 qubit case)
        '''

        phase = np.arctan2(args[0], args[1])

        inner_prod = complex(args[0], args[1])

        a = np.cos(np.arccos(np.abs(inner_prod)) * self.t)
        b = np.sin(np.arccos(np.abs(inner_prod)) * self.t) * phase

        return a, b, phase


    def _get_orthogonal_vectors(a: float,
                                b: float,
                                phase: complex
                                ) -> np.ndarray:

        '''
        Given the weights for the lin. combo and the phase find the $psi_t$
        '''

        e0 = 


#==================================================
#==================================================










N = 5 #number of qubits
assert (N - 1) % 2 == 0, 'The number of qubits in total should be equal to 2*n + 1'
M = 50 #number of steps
sigma_min = 1e-4 #min variance for the flow

ancilla = 0
reg0 = np.arange(1, (N - 1) // 2)
reg1 = np.arange((N - 1) // 2, N+1)
dim = 1 << N
t = np.random.uniform(0, 1)

device = qml.device('default.qubit', wires=N)

I = np.eye(2, dtype=np.complex64)
X = np.eye(2, dtype=np.complex64)[::-1]
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)


def create_Hamiltonian(n_qubits: int,
                       locality: int
                       ) -> tuple(csr_array, list[float]):

    conversion = {'I' : I,
                  'X' : X,
                  'Z' : Z,
                  'Y' : Y}


    assert locality <= n_qubits, 'locality must be less than the number of qubits'

    '''
    Creates a parametrised Hamiltonian operator out of Pauli strings
    '''

    combs = map(lambda x: (conversion[x_i] for x_i in x),
                filter(lambda x: False if ''.join(x).count('I') != n_qubits - locality else True,
                   product(*[('I', 'X', 'Z', 'Y') for i in range(n_qubits)])
                       )
                )
    
    Hamiltonian = csr_array(np.zeros((1 << n_qubits, 1 << n_qubits), dtype=np.complex64))
    params = []
    for comb in combs:
        param = np.random.rand()
        Hamiltonian: csr_array = Hamiltonian + csr_array(param * reduce(np.kron, comb))
        params.append(param)


    return Hamiltonian, params


@qml.qnode(device)
def Haar(depth: int | None = None) -> None:

    #reset all
    for wire in range(1, N):
        qml.measure(wires=wire, reset=True)

    if depth is None:
        depth = 4*N

    for d in range(depth):
        
        # we leave out the ancilla
        for q in range(1, N):
            qml.U3(*params[d, q], wires=q)
        
        # we leave out the ancilla
        for q in range(1, N - 1):
            qml.CNOT(wires=[q, q + 1])
    
        # we leave out the ancilla
        for q in range(1, N):
            qml.U3(*params[depth, q], wires=q)

    return qml.state()

Hamiltonian, params = create_Hamiltonian(3, 2)

delta_t = 1 / M

U_target = expm(-1j*delta_t*Hamiltonian)

Hamiltonian, params = create_Hamiltonian(3, 2)

Hamiltonian_dagger = Hamiltonian.conj().T

@qml.qnode(device)
def Find_angles():

    #reset all
    for wire in range(N):
        qml.measure(wires=wire, reset=True)

    qml.Hadamard(wires=ancilla)

    qml.ctrl(qml.QubitUnitary, control=ancilla)(U_target, wires=range(1, N))

    qml.Hadamard(wires=ancilla)

    #Re
    yield qml.expval(qml.Z(ancilla))

    for wire in range(N):
        qml.measure(wires=wire, reset=True)

    qml.Hadamard(wires=ancilla)

    qml.ctrl(qml.QubitUnitary, control=ancilla)(U_target, wires=range(1, N))

    qml.adjoint(qml.S)(wires=ancilla)

    qml.Hadamard(wires=ancilla)

    #Im
    yield qml.expval(qml.Z(ancilla))

    
def _get_weights_for_lin_comb(*args):

    phase = np.arctan2(args[0], args[1])

    inner_prod = complex(args[0], args[1])

    a = np.cos(np.abs(inner_prod) * t)
    b = np.sin(np.arccos(np.abs(inner_prod)) * t) * phase

    return a, b

U_train = expm(-1j*delta_t*Hamiltonian)

@qml.qnode(device)
def train():

    for wire in range(N):
        qml.measure(wire=wire, reset=True)

    #prepare target state
    qml.QubitUnitary(U_target, wires=range(1, N))
