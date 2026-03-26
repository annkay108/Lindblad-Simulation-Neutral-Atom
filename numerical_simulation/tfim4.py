# %load tfim1d.py
"""
Test Lindblad based method for ground state preparation for TFIM-4 model.
"""

# import sys

# print("Python version:", sys.version)
# print("Sys path:", sys.path)

from quspin.operators import hamiltonian  # Now try the failing import

# print("Hamiltonian moudle of quspin imported successfully.")

from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spin_basis_1d
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
# from qutip import Qobj, mesolve
# import os
# import json
# from pathlib import Path

from lindbladian_simulation.lindblad import LindbladSimulator
# from tfim4_operator import TFIM4Operator
from time import time

##### define model parameters #####
L = 6  # system size
J = 1.0  # spin zz interaction
g = 1.2  # z magnetic field strength
##### define spin model



# site-coupling lists (PBC for both spin inversion sectors)
h_field = [[-g, i] for i in range(L)]
# print(h_field,"<-- h_field")

J_zz = [[-J, i, i + 1] for i in range(L - 1)]  # no PBC
# print(J_zz,"<-- J_zz")

# define spin static and dynamic lists
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
# print(H_mat.shape, "<-- H_mat shape")
E_H, psi_H = la.eigh(H_mat) # calculate the full spectrum of H meaning all the eigenvalues and eigenvectors
# print("E_H = ", E_H)

gap = E_H[1] - E_H[0]
# print("gap = ", gap)

a = 2.5 * la.norm(H_mat, 2)
da = 0.5 * la.norm(H_mat, 2)
b = gap
db = gap
filter_params = {"a": a, "b": b, "da": da, "db": db}

A = hamiltonian(
    [["z", [[1.0, 0]]]], [], basis=spin_basis, dtype=np.float64, **no_checks
)  # z x 0 x 0 x 0

A_mat = np.array(A.todense()) # 16 x 16
# print(A_mat, "<-- A_mat shape")
lb = LindbladSimulator(H_mat, A_mat, filter_params, L=L)

# random initial state
np.random.seed(1)
vt = np.random.randn(lb.Ns)
# worst case: make psi0 and psi_GS orthogonal
psi0 = vt.copy()
psi0 -= psi_GS * np.vdot(psi_GS, psi0)
psi0 = psi0 / la.norm(psi0)
print("Initial state prepared.", psi0.shape)
# print("|<psi0|psiGS>| = ", np.abs(np.vdot(psi_GS, psi0)))

# Exact simulation
T = 200
num_t = int(T)

# exact_start = time()
# print("Exact simulation Started...")
times = np.arange(num_t + 1) * (T / num_t)
# H_obj = Qobj(H_mat)
# rho_GS_obj = Qobj(np.outer(psi_GS, psi_GS.conj()))  # initial state
lb.construct_jump_exact()  # construct Jump operator
# result = mesolve(H_obj, Qobj(psi0), times, [Qobj(lb.A_jump)], [H_obj, rho_GS_obj])
# avg_energy_e = result.expect[0]  # list of energy
# avg_pGS_e = result.expect[1]  # list of overlap
# print("Exact simulation completed.")
# exact_end = time()
# print(f"Exact simulation time: {exact_end - exact_start} seconds.")

zero_block = np.zeros_like(lb.A_jump)
dilated_K = np.block([[zero_block, lb.A_jump.conj().T], [lb.A_jump, zero_block]])
# print(dilated_K.shape, "<-- dilated_K shape")

S_s = 5.0 / db  # Integral truncation
M_s = int(5 / db / (2 * np.pi / (4 * a)))  # Integral stepsize

num_segment = 3  # discrete segment
num_rep = 1  # average repetition (used to recover \rho_n after tracing out)

np.random.seed(seed=1)
flip_dice = np.random.rand(
    num_t, num_rep
)  # used for simulating tracing out in quantum circuit shape=(80, 1)

print("Starting Lindblad simulation...")
start = time()
# time_series, avg_energy, avg_pGS, avg_energy_op, avg_pGS_op, time_H, rho_hist, all_gates
times_l, avg_energy_l, avg_pGS_l, avg_energy_l_op, avg_pGS_l_op, time_H_l, rho_all_l, all_gates = (
    lb.Lindblad_simulation(
        T, num_t, num_segment, psi0, num_rep, S_s, M_s, psi_GS, intorder=2, flip_dice=flip_dice
    )
)
end = time()

lb.save_results(time_series=times_l, avg_energy=avg_energy_l, avg_pGS=avg_pGS_l, time_H=time_H_l, num_t=num_t, T=T, num_segment=num_segment, S_s=S_s, M_s=M_s)
print(f"Lindblad simulation completed in {end - start} seconds.")
# def save_results(results, filename="results"):
#     os.makedirs("data", exist_ok=True)

#     json_path = Path("data") / f"{filename}1.json"
#     with open(json_path, "w") as f:
#         json.dump(results, f, indent=4)
#     print(f"Results saved to {json_path}...")

# save_results(avg_energy_l, filename="lindblad_9sites_250iter_3segNew_energy")
# lb_operator = TFIM4Operator(psi0, psi_GS)
# psi_final, fidelity_list = lb_operator.apply_operator(flip_dice=flip_dice)

# num_segment = 2  # discrete segment
# num_rep = 1  # average repetition (used to recover \rho_n after tracing out)
# times_l2, avg_energy_l2, avg_pGS_l2, time_H_l2, rho_all_l2, all_gates2 = (
#     lb.Lindblad_simulation(
#         T, num_t, num_segment, psi0, num_rep, S_s, M_s, psi_GS, intorder=2
#     )
# )

print(avg_energy_l[-1], "<-- final energy lindblad")

#===================================================================================================
plt.figure(figsize=(12, 10))
# plt.plot(
#     times, avg_energy_e, "g-", label="Lindblad (exact)", linewidth=3, markersize=10
# )
# plt.plot(
#     times_l,
#     avg_energy_l_op,
#     "b--",
#     label=r"Lindblad $(\tau=1,r=2)$",
#     linewidth=1.5,
#     markersize=10,
# )
plt.plot(
    times_l,
    avg_energy_l,
    "r--",
    label=r"Lindblad $(\tau=1,r=1)$",
    linewidth=1.5,
    markersize=10,
)

plt.plot(
    times,
    np.ones_like(times) * E_GS,
    "p-",
    label=r"$\lambda_0$",
    linewidth=1.5,
    markersize=10,
)
plt.legend()
plt.xlabel("time", fontsize=25)
plt.ylabel(r"$<E>$", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=30)
plt.show()

#===================================================================================================
plt.figure(figsize=(12, 10))
# plt.plot(times, avg_pGS_e, "g-", label=r"Lindblad (exact)", linewidth=3, markersize=20)
# plt.plot(
#     times_l,
#     avg_pGS_l_op,
#     "b--",
#     label=r"Lindblad $(\tau=1,r=1)$ operator",
#     linewidth=1.5,
#     markersize=10,
# )
plt.plot(
    times_l,
    avg_pGS_l,
    "r--",
    label=r"Lindblad $(\tau=1,r=1)$",
    linewidth=1.5,
    markersize=10,
)
# plt.plot(
#     times_l,
#     fidelity_list,
#     "y--",
#     marker="o",
#     label=r"Lindblad Operator Simulation unitary",
#     linewidth=1.5,
#     markersize=10,
# )
plt.legend()
plt.xlabel("time", fontsize=25)
plt.ylabel(r"$<p0>$", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=30, loc="lower right")
plt.show()
