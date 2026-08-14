# Algorithm description

## Problem formulation

We have a set of unitary operators $\{U_{dist, i}\}_{i=1}^N$ that can preprare states that correspond to some complex distribution $\mathrm{K}$ . The goal is to train a Hamiltonian operator $\mathrm{H}(\theta, t)$ to approximate this distribution and sample from it 

## Hamiltonian construction

$\mathrm{H}$ is constructed as a sum of pauli strings:

$\mathrm{H} = \sum_{i=1}^{K} \alpha_i S_i, \space \mathcal{S}_i \in \mathcal{W}_k = \left\{ \bigotimes_{j=1}^n P_j \;\middle|\; P_j \in \{I,X,Y,Z\},\; \left|\{j : P_j \neq I\}\right| = k \right\}$

Where:

$\alpha_i$ - trainable parameter for the $i^{th}$ operator

$S_i$ - operator, that is a part of a Hamiltonian, for example XI for the case of 2 qubits

$\mathcal{W}_k$ - the set of all operators that comprise the hamiltonian operator, for example for 2 qubits and operator weight 1 this set is XI, YI, ZI, IZ, IY, IX

## Algorithm

**Input:** A unitary operator $U_{dist, i}$ , $\sigma_{min}$, M

**Output:** Gradient $\nabla \mathrm{H}(\theta, t)$

1. Sample t \~ Uniform[0, 1/M] classically, set $\Delta t = \frac{1}{M}$

2. Prepare a state $\ket{\psi_1}' := t\ket{\psi_1} + ((1-t)+t\sigma_{min})\ket{\psi_0}$

3. Prepare a time evolution operator $U(\Delta t; \theta) = exp(-i\Delta t\sum_jc_j(\theta)P_j)$, where $P_j$ is a tensor prod of pauli strings

4. Find $\ket{\hat\psi_{t+\Delta t}} = U(\Delta t; \theta) \ket{\psi_1}'$

5. Find $\ket{\psi_{t+\Delta t}} = \frac{(t + \Delta t)*\ket{\psi_1} + ((1-t - \Delta t)+(t + \Delta t)\sigma_{min})\ket{\psi_0}}{||(t + \Delta t)*\ket{\psi_1} + ((1-t - \Delta t)+(t + \Delta t)\sigma_{min})\ket{\psi_0}||} $

6. Compute $2 + 2Re(\braket{\psi_{t+\Delta t}|\hat\psi_{t+\Delta t}})$ via the SWAP test
