import pennylane as qml
from scipy.linalg import expm
import numpy as np

dev = qml.device("default.qubit", wires=3)

# ---- plain helper functions (no @qml.qnode) ----
def block_a(wires):
    qml.Hadamard(wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])

def block_b(theta, wires):
    qml.RY(theta, wires[1])
    qml.CRY(theta, wires=[wires[1], wires[2]])

def controlled_prep(anc, ctrl_val, prep_fn, *args):
    qml.ctrl(prep_fn, control=anc, control_values=ctrl_val)(*args)

# ---- top-level QNode ----
@qml.qnode(dev)
def circuit(x):
    qml.Hadamard(0)
    controlled_prep(0, 0, block_a, [1, 2])
    controlled_prep(0, 1, block_b, x, [1, 2])
    qml.Hadamard(0)
    return qml.expval(qml.PauliZ(0))

# ---- display the full expanded circuit ----
print(qml.draw(circuit)(0.5))
