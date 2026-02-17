from operator import index
import utils
import numpy as np
import scipy.linalg as la

hamiltonian_quspin, H_total = utils.tmif4_hamiltonian_pauli()
E_GS, psi_GS = hamiltonian_quspin.eigsh(k=1, which="SA") # calculate the ground state so eigenvalue and corresponding eigenvector
psi_GS = psi_GS.flatten()
print("E_GS = ", E_GS)

hamitonian = np.array(hamiltonian_quspin.todense())

zero = np.array([1, 0])

np.random.seed(1)
vt = np.random.randn(hamitonian.shape[0])
# worst case: make psi0 and psi_GS orthogonal
psi0 = vt.copy()
psi0 -= psi_GS * np.vdot(psi_GS, psi0)
psi0 = psi0 / la.norm(psi0)
state = np.kron(zero, psi0)

dilated_unitary, n_qubits = utils.load_unitary_matrices()
no_of_iterations = len(dilated_unitary) // 2

# initial_state = np.zeros(2**n_qubits, dtype=complex)
# initial_state[0] = 1.0
# state = initial_state.copy()

np.random.seed(seed=1)
flip_dice = np.random.rand(
    no_of_iterations, 1
)

for i in range(0,no_of_iterations*2,2):
    U_s = dilated_unitary[i+1] @ dilated_unitary[i]
    state = U_s @ state

    prob_index = 0
    prob = la.norm(state[2**(n_qubits-1):])**2

    if(flip_dice[i//2][0] <= prob):
        prob_index = 1 # select the lower half

    psi_reshaped = state.reshape(2, 16)
    psi_0 = psi_reshaped[prob_index]     
    p0 = np.vdot(psi_0, psi_0).real
    state = np.kron(zero, psi_0 / np.sqrt(p0))

def calculate_final_energy(state):
    state = state.reshape(2, 16)
    state = state[0]
    energy = np.vdot(state, hamitonian @ state).real
    return energy

final_energy = calculate_final_energy(state)
print(f"Final energy after {no_of_iterations} iterations: {final_energy}")