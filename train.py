import torch
import numpy as np
import argparse
import yaml
from qfm import QFM # Assuming the class above is saved in qfm.py

def generate_random_unitary(dim):
    """Generates a Haar random unitary matrix"""
    z = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    q = q * ph
    return torch.tensor(q, dtype=torch.complex64)

def main():
    parser = argparse.ArgumentParser(description="Quantum Flow Matching Training")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    N = config.get('N', 5)
    dim = 1 << N
    
    print(f"Initializing QFM with N={N} qubits...")
    qfm = QFM(N=N, 
              M=config.get('M', 50), 
              sigma_min=config.get('sigma_min', 1e-4), 
              device=config.get('device', 'default.qubit'))
              
    print("Generating Haar random unitaries for U_psi and U_phi...")
    U_psi = generate_random_unitary(dim)
    U_phi = generate_random_unitary(dim)
    
    print("Starting training...")
    trained_weights = qfm.train(
        U_psi=U_psi,
        U_phi=U_phi,
        config_path=args.config
    )
    
    print("Training complete!")
    torch.save(trained_weights, 'hamiltonian_weights.pth')

if __name__ == "__main__":
    main()
