import json
import os
from time import time
import numpy as np
import pennylane as qml
from pathlib import Path
import utils

def run_single_experiment(no_of_iterations, max_bond_dim, no_of_sites):
    # for 23 steps ground -5.002728873837382
    dilated_unitary, n_qubits = utils.load_unitary_matrices()

    initial_state = np.zeros(2**n_qubits, dtype=complex)
    initial_state[0] = 1.0
    state = initial_state

    total_qubits = n_qubits + no_of_iterations

    hamiltonian_quspin, H_total = utils.tmif4_hamiltonian_pauli(no_of_sites)

    kwargs_mps = {
        # Maximum bond dimension of the MPS
        "max_bond_dim": max_bond_dim,
        # Cutoff parameter for the singular value decomposition
        "cutoff": np.finfo(np.complex128).eps,
        # Contraction strategy to apply gates
        "contract": "auto-mps",
    }

    dev = qml.device('default.tensor', method="mps", **kwargs_mps)
    @qml.qnode(dev)
    def apply_unitary_iteration(index, state):
        qml.StatePrep(state, wires=range(n_qubits))
        unitary = dilated_unitary[index+1] @ dilated_unitary[index]
        qml.QubitUnitary(unitary, wires=range(n_qubits))

        result = {"state": qml.state(), "expval": qml.expval(H_total)}
        return result

    startTime = time()
    final_result =[]
    for i in range(0,no_of_iterations*2,2):
        result = apply_unitary_iteration(i, state)

        zero = np.array([1, 0])
        psi_reshaped = result["state"].reshape(2, 2**(n_qubits-1))

        psi_0 = psi_reshaped[0]        # amplitudes where q0 = 0
        p0 = np.vdot(psi_0, psi_0).real

        state = np.kron(zero, psi_0 / np.sqrt(p0))

        energy = result["expval"]

        final_result.append({
        "iterations": i//2 + 1,
        "max_bond_dim": max_bond_dim,
        "execution_time": 1.0,
        "energy": energy,
        })
        # print(f"Iteration {i+1}, state: {state}")
    
    # final_energy = energy
    # endTime = time()

    # exect_time = endTime - startTime
    return final_result

def run_experiments(no_of_iterations, max_bond_dim, no_of_sites):
    result = run_single_experiment(no_of_iterations, max_bond_dim, no_of_sites)
    return result

def save_results(results, filename="results"):
    os.makedirs("data", exist_ok=True)

    json_path = Path("data") / f"{filename}1measure.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_path}...")

if __name__ == "__main__":
    no_of_sites = 8
    no_of_iterations = 250
    max_bond_dim = 15
    results = run_experiments(no_of_iterations, max_bond_dim, no_of_sites)
    save_results(results, filename=f"mps_gate_implementation_results_news_{no_of_sites}sites")