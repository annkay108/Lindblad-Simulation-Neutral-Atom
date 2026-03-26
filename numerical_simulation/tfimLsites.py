from quspin.operators import hamiltonian 
from quspin.basis import spin_basis_1d
import numpy as np
import scipy.linalg as la

from lindbladian_simulation.extract_unitary import ExtractUnitary
# from lindbladian_simulation.extract_unitary_mpi import ExtractUnitary
from time import time

L = 9  # system size
J = 1.0  # spin zz interaction
g = 1.2  # z magnetic field strength

h_field = [[-g, i] for i in range(L)]

J_zz = [[-J, i, i + 1] for i in range(L - 1)]  # no PBC

static = [["zz", J_zz], ["x", h_field]]  # static part of H
dynamic = []  # time-dependent part of H
# construct spin basis in pos/neg spin inversion sector depending on APBC/PBC
spin_basis = spin_basis_1d(L=L)
# build TFIM-4 Hamiltonians
no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)

Hamiltonian_quspin = hamiltonian(static, dynamic, basis=spin_basis, dtype=np.float64, **no_checks)


# calculate spin energy levels
E_GS, psi_GS = Hamiltonian_quspin.eigsh(k=1, which="SA") # calculate the ground state so eigenvalue and corresponding eigenvector
psi_GS = psi_GS.flatten()
print("E_GS = ", E_GS)

H_mat = np.array(Hamiltonian_quspin.todense())

E_H, psi_H = la.eigh(H_mat) # calculate the full spectrum of H meaning all the eigenvalues and eigenvectors

gap = E_H[1] - E_H[0]

a = 2.5 * la.norm(H_mat, 2)
da = 0.5 * la.norm(H_mat, 2)
b = gap
db = gap
filter_params = {"a": a, "b": b, "da": da, "db": db}

A = hamiltonian(
    [["z", [[1.0, 0]]]], [], basis=spin_basis, dtype=np.float64, **no_checks
)  # z x 0 x 0 x 0

A_mat = np.array(A.todense()) # 16 x 16

T = 1
num_t = int(T)

S_s = 5.0 / db  # Integral truncation
M_s = int(5 / db / (2 * np.pi / (4 * a)))  # Integral stepsize

num_segment = 3  # discrete segment
num_rep = 1  # average repetition (used to recover \rho_n after tracing out)

extraction = ExtractUnitary(H_mat, A_mat, filter_params, L, num_segment, num_t)

print("Starting Lindblad simulation...")
start = time()
all_gates = (
    extraction.Lindblad_simulation(
        T, num_t, num_segment, num_rep, S_s, M_s
    )
)
end = time()
print(f"Lindblad simulation completed in {end - start} seconds.")

