import numpy as np
import pennylane as qml
from pathlib import Path
import utils

def load_unitary_matrices():
    path = Path().resolve().parent / "Lindblad_simulation/numerical_simulation/lindbladian_simulation/data/lindblad_operators.pickle"
    dilated_unitary = np.load(path, allow_pickle=True)

    n_qubits = int(np.log2(dilated_unitary[0].shape[0]))

    return dilated_unitary, n_qubits

def generate_pennylane_circuit_from_unitary():
    no_of_iterations = 23
    dilated_unitary, n_qubits = load_unitary_matrices()

    total_qubits = n_qubits + no_of_iterations

    hamiltonian_quspin, H_total = utils.tmif4_hamiltonian_pauli()

    dev = qml.device('default.qubit', wires=total_qubits)

    @qml.qnode(dev)
    def circuit():
        ancilla_idx = 0

        for i, U_s in enumerate(dilated_unitary):
            qml.QubitUnitary(U_s, wires=range(n_qubits))
            if i % 2 == 1:
                qml.SWAP(wires=[0, n_qubits + ancilla_idx])
                ancilla_idx += 1
            if(ancilla_idx == no_of_iterations):
                break
        return qml.expval(H_total)

    # print(qml.draw(circuit)())
    result = circuit()
    return result

print(generate_pennylane_circuit_from_unitary())

