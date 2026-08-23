import numpy as np
import pennylane as qml
import torch
import yaml
from itertools import product

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Pauli matrices as PyTorch tensors for differentiability
I = torch.tensor(np.eye(2), dtype=torch.complex64)
X = torch.tensor(np.array([[0, 1], [1, 0]]), dtype=torch.complex64)
Y = torch.tensor(np.array([[0, -1j], [1j, 0]]), dtype=torch.complex64)
Z = torch.tensor(np.array([[1, 0], [0, -1]]), dtype=torch.complex64)


class QFM:
    def __init__(
        self,
        N: int = 5,
        M: int = 50,
        sigma_min: float = 1e-4,
        device: str = "default.qubit",
        **kwargs
    ) -> None:
        self.N = N
        self.M = M
        self.sigma_min = sigma_min
        self.device_name = device
        self.ancilla = 0
        self.dim = 1 << N

        # diff_method belongs to the QNode, not the device.
        self.dev = qml.device(device, wires=N + 1)

        self.delta_t = 1 / M

        self.zero_state = torch.zeros(self.dim, dtype=torch.complex64)
        self.zero_state[0] = 1.0

        self.eye_dim = torch.eye(self.dim, dtype=torch.complex64)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_target_distributions(
        self,
        U_psi: torch.Tensor,
        U_phi: torch.Tensor
    ):
        """
        Sets the unitaries for the source and target distributions and computes
        the necessary geodesic interpolation constants classically.
        """
        self.U_psi = U_psi
        self.U_phi = U_phi

        self.psi_0 = U_psi @ self.zero_state
        self.psi_1 = U_phi @ self.zero_state

        # c = <psi_0 | psi_1>
        c = torch.vdot(self.psi_0, self.psi_1)

        abs_c = torch.clamp(torch.abs(c), 0.0, 1.0)

        self.c = c
        self.alpha = torch.arccos(abs_c)
        self.beta = torch.angle(c)
        self.denom = torch.sqrt(torch.clamp(1.0 - abs_c * abs_c, min=0.0))

        self.e_0 = self.psi_0

        if self.denom.item() > 1e-6:
            self.e_1 = (self.psi_1 - c * self.psi_0) / self.denom
        else:
            self.e_1 = self.e_0

    def _target_lcu_coefficients(self, t: float):
        """
        Returns the LCU coefficients for the target geodesic

            |psi_t> = a(t)|psi_0> + b(t)|psi_1>.

        With torch.vdot convention <psi_0|psi_1>, the endpoint-correct phase
        is exp(-i beta). If you use the opposite inner-product convention,
        flip the sign of beta.
        """
        t = torch.as_tensor(t, dtype=torch.float32)

        if self.denom.item() > 1e-6:
            coeff_target = (
                torch.exp(-1j * self.beta)
                * torch.sin(self.alpha * t)
                / self.denom
            )
            coeff_source = torch.cos(self.alpha * t) - coeff_target * self.c
        else:
            # Nearly parallel states. The geodesic is degenerate, so we use
            # a stable linear blend that preserves the source state.
            if torch.abs(self.c).item() > 1e-8:
                phase = self.c / torch.abs(self.c)
            else:
                phase = torch.tensor(1.0 + 0.0j, dtype=torch.complex64)

            coeff_source = 1.0 - t
            coeff_target = t * torch.conj(phase)

        return coeff_source, coeff_target

    def get_psi_t(self, t: float) -> torch.Tensor:
        """
        Returns the target geodesic interpolated state at time t.
        Kept for compatibility/debugging.
        """
        coeff_source, coeff_target = self._target_lcu_coefficients(t)
        return coeff_source * self.psi_0 + coeff_target * self.psi_1

    def build_true_lcu_operator(self, t: float) -> torch.Tensor:
        """
        Builds the true LCU operator

            L_true(t) = a(t) U_psi + b(t) U_phi.

        This is generally not unitary by itself.
        """
        coeff_source, coeff_target = self._target_lcu_coefficients(t)
        return coeff_source * self.U_psi + coeff_target * self.U_phi

    def build_estimated_lcu_operator(
        self,
        t: float,
        U_ours: torch.Tensor
    ) -> torch.Tensor:
        """
        Builds the estimated/model LCU operator

            L_est(t) = a(t) U_psi + b(t) U_ours.

        The same target interpolation coefficients are used. This corresponds
        to replacing the target unitary U_phi by our learned/unitary U_ours.

        This is generally not unitary by itself.
        """
        coeff_source, coeff_target = self._target_lcu_coefficients(t)
        return coeff_source * self.U_psi + coeff_target * U_ours

    def _unitary_with_first_column(self, state: torch.Tensor) -> torch.Tensor:
        """
        Constructs a unitary V such that

            V |0> = |state>,

        where |state> is normalized inside this function.

        This is used to convert an LCU-generated state into a valid unitary
        state-preparation operator that can be controlled in the Hadamard test.
        """
        state = state.to(torch.complex64)

        norm = torch.linalg.norm(state)
        if norm.item() < 1e-12:
            # Degenerate fallback. A zero LCU state is not a valid quantum state,
            # but this avoids NaNs.
            return self.eye_dim.clone()

        state = state / (norm + 1e-12)

        # Choose a phase so that the first component of y_prime is real
        # and non-negative. This makes the Householder reflection stable.
        x0 = state[0]
        abs_x0 = torch.abs(x0)

        if abs_x0.item() > 1e-12:
            phase = x0 / abs_x0
        else:
            phase = torch.tensor(1.0 + 0.0j, dtype=torch.complex64)

        y_prime = torch.conj(phase) * state

        v = self.zero_state - y_prime
        v_norm_sq = torch.sum(v.conj() * v).real

        if v_norm_sq.item() > 1e-12:
            H = self.eye_dim - 2.0 * torch.outer(v, v.conj()) / v_norm_sq
        else:
            H = self.eye_dim

        # H |0> = y_prime, therefore phase * H |0> = state.
        V = phase * H
        return V

    def _lcu_operator_to_unitary(self, L: torch.Tensor) -> torch.Tensor:
        """
        Converts an LCU operator L into a valid unitary state preparer V such
        that

            V |0> = L|0> / ||L|0>||.

        This is the object that is treated as a unitary operator inside the
        Hadamard-test circuit.
        """
        psi_L = L @ self.zero_state
        return self._unitary_with_first_column(psi_L)

    def build_true_lcu_unitary(self, t: float) -> torch.Tensor:
        """
        Builds the true LCU operator and converts it into a unitary
        state-preparation operator.
        """
        L_true = self.build_true_lcu_operator(t)
        return self._lcu_operator_to_unitary(L_true)

    def build_estimated_lcu_unitary(
        self,
        t: float,
        U_ours: torch.Tensor
    ) -> torch.Tensor:
        """
        Builds the estimated/model LCU operator and converts it into a unitary
        state-preparation operator.
        """
        L_est = self.build_estimated_lcu_operator(t, U_ours)
        return self._lcu_operator_to_unitary(L_est)

    def create_Hamiltonian(
        self,
        locality: int,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Creates a parametrised Hamiltonian operator out of Pauli strings.
        """
        conversion = {"I": I, "X": X, "Y": Y, "Z": Z}
        n_qubits = self.N
        paulis = ["I", "X", "Y", "Z"]

        combs = [
            p for p in product(paulis, repeat=n_qubits)
            if p.count("I") == n_qubits - locality
        ]

        Hamiltonian = torch.zeros((self.dim, self.dim), dtype=torch.complex64)

        for comb, w in zip(combs, weights):
            mat = conversion[comb[0]]
            for p in comb[1:]:
                mat = torch.kron(mat, conversion[p])
            Hamiltonian = Hamiltonian + w * mat

        return Hamiltonian

    def get_evolution_unitary(
        self,
        H: torch.Tensor,
        delta_t: float
    ) -> torch.Tensor:
        """
        Computes the evolution operator

            U(dt) = exp(-i H dt).
        """
        return torch.linalg.matrix_exp(-1j * H * delta_t)

    def _project_to_unitary(self, U: torch.Tensor) -> torch.Tensor:
        """
        Optional helper: projects a matrix to the nearest unitary in Frobenius
        norm using polar decomposition / SVD.

        This is not used by default because the constructed state-preparation
        unitaries are already unitary, but it can be useful if you want to pass
        a raw LCU matrix directly and force it to be unitary.
        """
        left, _, right_h = torch.linalg.svd(U, full_matrices=False)
        return left @ right_h

    def _measure_overlap(self, U_overlap: torch.Tensor):
        """
        Computes the real and imaginary parts of

            <0|U_overlap|0>

        using the Hadamard test.

        In the training loss we only need the real part:

            Re <psi_true | psi_est>.
        """
        system_wires = list(range(1, self.N + 1))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit_real(U_overlap_mat):
            qml.Hadamard(wires=self.ancilla)
            qml.ctrl(qml.QubitUnitary, control=self.ancilla)(
                U_overlap_mat,
                wires=system_wires
            )
            qml.Hadamard(wires=self.ancilla)
            return qml.expval(qml.PauliZ(self.ancilla))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit_imag(U_overlap_mat):
            qml.Hadamard(wires=self.ancilla)
            qml.ctrl(qml.QubitUnitary, control=self.ancilla)(
                U_overlap_mat,
                wires=system_wires
            )
            qml.adjoint(qml.S)(wires=self.ancilla)
            qml.Hadamard(wires=self.ancilla)
            return qml.expval(qml.PauliZ(self.ancilla))

        return circuit_real(U_overlap), circuit_imag(U_overlap)

    def train(
        self,
        U_psi: torch.Tensor,
        U_phi: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.01,
        locality: int = 2,
        config_path: str = None
    ):
        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                epochs = config.get("epochs", epochs)
                lr = config.get("lr", lr)
                locality = config.get("locality", locality)

        self.set_target_distributions(U_psi, U_phi)

        paulis = ["I", "X", "Y", "Z"]
        combs = [
            p for p in product(paulis, repeat=self.N)
            if p.count("I") == self.N - locality
        ]
        num_weights = len(combs)

        weights = torch.rand(
            num_weights,
            dtype=torch.float32,
            requires_grad=True
        )

        optimizer = torch.optim.Adam([weights], lr=lr)

        print(
            f"Starting training for {epochs} epochs "
            f"with {num_weights} Hamiltonian parameters..."
        )

        for epoch in range(epochs):
            optimizer.zero_grad()

            t = torch.rand(1).item()

            # Use this if you want the loss at psi_t.
            t_eval = t

            # If instead you want the original psi_{t + dt} version, use:
            # t_eval = float(min(t + self.delta_t, 1.0))

            H = self.create_Hamiltonian(locality, weights)
            U_evol = self.get_evolution_unitary(H, self.delta_t)

            # Interpretation:
            # "our unitary" is the Hamiltonian evolution unitary itself.
            #
            # If instead your convention is that the learned endpoint unitary is
            # the evolution applied to the source unitary, use:
            #
            # U_ours = U_evol @ self.U_psi
            #
            U_ours = U_evol

            # Build the two LCUs and convert them into valid unitary
            # state-preparation operators.
            V_true_lcu = self.build_true_lcu_unitary(t_eval)
            V_est_lcu = self.build_estimated_lcu_unitary(t_eval, U_ours)

            # Hadamard-test unitary whose |0> expectation gives the overlap.
            U_overlap = V_true_lcu.conj().T @ V_est_lcu

            # Optional numerical cleanup if your PennyLane version is strict
            # about unitarity:
            #
            # U_overlap = self._project_to_unitary(U_overlap)

            real_overlap, imag_overlap = self._measure_overlap(U_overlap)

            # Minimizing this maximizes Re <psi_true | psi_est>.
            loss = 2.0 - 2.0 * real_overlap

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Re(Overlap): {real_overlap.item():.4f} | "
                    f"Im(Overlap): {imag_overlap.item():.4f}"
                )

        return weights
