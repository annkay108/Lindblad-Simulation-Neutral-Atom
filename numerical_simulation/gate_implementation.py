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
    def circuit():
        ancilla_idx = 0

        for i in range(no_of_iterations):
            qml.QubitUnitary(dilated_unitary[0], wires=range(n_qubits))
            qml.SWAP(wires=[0, n_qubits + i])
        print("Calculating final energy...")
        return qml.expval(H_total)

    # print(qml.draw(circuit)())
    startTime = time()
    energy = circuit()
    endTime = time()

    exect_time = endTime - startTime
    return {
        "iterations": no_of_iterations,
        "max_bond_dim": max_bond_dim,
        "no_of_sites": no_of_sites,
        "execution_time": exect_time,
        "energy": energy,
    }

def run_experiments(iterations_list, bond_dim_list, no_of_sites):
    results = []
    for no_of_iterations in iterations_list:
        for max_bond_dim in bond_dim_list:
            result = run_single_experiment(no_of_iterations, max_bond_dim, no_of_sites)
            results.append(result)
            print(f"Completed experiment with {no_of_iterations} iterations and max bond dimension {max_bond_dim}. Energy: {result['energy']}, Execution time: {result['execution_time']} seconds.")
    return results

def save_results(results, filename="results"):
    os.makedirs("data", exist_ok=True)

    json_path = Path("data") / f"{filename}1.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_path}...")

if __name__ == "__main__":
    iterations_list = [90]
    bond_dim_list = [350]
    no_of_sites = 12

    results = run_experiments(iterations_list, bond_dim_list, no_of_sites)
    print(results, "<-- final results")
    save_results(results, filename=f"mps_gate_implementation_results_news_{no_of_sites}sites_{iterations_list[0]}iter_3seg_{bond_dim_list[0]}bdlist")


    
