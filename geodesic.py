import numpy as np

from scipy.stats import unitary_group


n_qubits = 4
seed = 42

np.random.seed(42)

dim = 1 << n_qubits

psi1, psi2 = unitary_group.rvs(dim)[:, [1,2]].T

psi1, psi2 = psi1.T, psi2.T

d_FS = lambda vec1, vec2 : np.arccos(np.abs(vec1.T @ vec2))

c = psi1.T @ psi2
alpha = np.arccos(np.abs(c))


e0 = psi1 
e1 = ( psi2 - c* psi1 ) / np.sqrt(1 - np.abs(c)**2)

t = np.random.uniform(0, 1)

psi_t = np.cos(np.arccos(np.abs(psi1.T @ psi2)) * t) * psi1 + np.sin(np.arccos(np.abs(psi1.T @ psi2)) * t) * psi1.T @ psi2  / np.abs(psi1.T @ psi2) * ( psi2 -psi1.T @ psi2 * psi1 ) / np.sqrt(1 - np.abs(psi1.T @ psi2)**2)


print(psi1)
print(psi2)
print(np.arccos(psi1.T @ psi2 / np.linalg.norm(psi1) / np.linalg.norm(psi2)))
print(psi_t)
