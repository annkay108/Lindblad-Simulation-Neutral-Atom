import time
import numpy as np
import pennylane as qml
from pathlib import Path
import utils


def generate_pennylane_circuit_from_unitary():
    no_of_iterations = 24  #-5.01500273878082 for 24 steps 1254 seconds
    dilated_unitary, n_qubits = utils.load_unitary_matrices()

    total_qubits = n_qubits + no_of_iterations

    hamiltonian_quspin, H_total = utils.tmif4_hamiltonian_pauli()

    dev = qml.device('default.qubit', wires=total_qubits)


    @qml.qnode(dev)
    def circuit():
        iteration_count = 0

        for i, U_s in enumerate(dilated_unitary):
            qml.QubitUnitary(U_s, wires=range(n_qubits))
            if i % 2 == 1:
                qml.measure(0, reset=True)
                iteration_count += 1
            if(iteration_count == no_of_iterations):
                break
        return qml.expval(H_total)

    start_time = time.time()
    result = circuit()
    end_time = time.time()
    print(f"Circuit execution time: {end_time - start_time} seconds for {no_of_iterations} iterations")

    return result

print(generate_pennylane_circuit_from_unitary())

