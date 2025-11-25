import numpy as np
import pennylane as qml
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian  # Hamiltonians and operators
from pathlib import Path

def load_unitary_matrices():
    path = Path().resolve().parent / "Lindblad_simulation/numerical_simulation/lindbladian_simulation/data/lindblad_operators.pickle"
    dilated_unitary = np.load(path, allow_pickle=True)

    n_qubits = int(np.log2(dilated_unitary[0].shape[0]))

    return dilated_unitary, n_qubits

def tmif4_hamiltonian_pauli(L=4, J=1.0, g=1.2):
    # site-coupling lists (PBC for both spin inversion sectors)
    h_field = [[-g, i] for i in range(L)]
    J_zz = [[-J, i, i + 1] for i in range(L - 1)]  # no PBC

    # define spin static and dynamic lists
    static = [["zz", J_zz], ["x", h_field]]  # static part of H
    dynamic = []  # time-dependent part of H

    # construct spin basis in pos/neg spin inversion sector depending on APBC/PBC
    spin_basis = spin_basis_1d(L=L)

    # build TFIM-4 Hamiltonians
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    Hamiltonian_quspin = hamiltonian(static, dynamic, basis=spin_basis, dtype=np.float64, **no_checks)

    coeffs = []
    ops = []

    # ZZ interactions on (1,2), (2,3), (3,4)
    for i in range(L - 1):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i + 1) @ qml.PauliZ(i + 2))

    # X terms on wires (1,2,3,4)
    for i in range(L):
        coeffs.append(-g)
        ops.append(qml.PauliX(i + 1))

    H_total = qml.Hamiltonian(coeffs, ops)
    
    return Hamiltonian_quspin, H_total

def compare_ground_state():
    Hamiltonian_quspin, H_total = tmif4_hamiltonian_pauli()

    # # calculate spin energy levels
    E_GS_mat, psi_GS_mat  = Hamiltonian_quspin.eigsh(k=1, which="SA") # calculate the ground state so eigenvalue and corresponding eigenvector
    # psi_GS = psi_GS.flatten()
    
    # Find the ground state energy using qml.eigval from the circuit
    E_GS_total = qml.eigvals(H_total).min()  # since eigval returns a list of eigenvalues, we take the min for the ground state
    
    # Compare energies and states
    print(f"Ground state energy from H_mat: {E_GS_mat[0]}")
    print(f"Ground state energy from H_total (Pauli form): {E_GS_total}")
    
    if np.isclose(E_GS_mat[0], E_GS_total):
        print("The ground state energies are the same.")
    else:
        print("The ground state energies are different.")
    
    return E_GS_mat, E_GS_total

compare_ground_state()