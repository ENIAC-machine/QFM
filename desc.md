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

footnotes: we first construct the states, find the angle between the states

**Input:** A unitary operator $U_{dist, i}$ , $\sigma_{min}$, M

**Output:** Gradient $\nabla \mathrm{H}(\theta)$

1. Sample t \~ Uniform[0, 1]

2. Find Inner prod (Re) of $U_{target}$ and $U_{Haar}$

3. Find geodesic interpolation of $U_{target}$ and $U_{Haar}$ with weights t and (1-t)

4. Find inner prod (Re) of $U_\theta$ and $U_{Haar}$ 

5. Compute Loss $2 + 2Re(\braket{\psi_{t+\Delta t}|\hat\psi_{t+\Delta t}})$ via the SWAP test and run backprop
