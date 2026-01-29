import time
import numpy as np
import pennylane as qml
from pathlib import Path
import utils


def generate_pennylane_circuit_from_unitary():
    no_of_iterations = 150  #-5.01500273878082 for 24 steps 1254 seconds
    dilated_unitary, n_qubits = utils.load_unitary_matrices()

    kwargs_mps = {
        # Maximum bond dimension of the MPS
        "max_bond_dim": 500000,
        # Cutoff parameter for the singular value decomposition
        "cutoff": np.finfo(np.complex128).eps,
        # Contraction strategy to apply gates
        "contract": "auto-mps",
    }

    dev = qml.device('default.tensor', method="mps", **kwargs_mps)

    # total_qubits = n_qubits + no_of_iterations
    # dev = qml.device('default.tensor', wires=total_qubits)

    hamiltonian_quspin, H_total = utils.tmif4_hamiltonian_pauli()


    @qml.qnode(dev)
    def circuit(**kwargs):
        iteration_count = 0

        for i, U_s in enumerate(dilated_unitary):
            qml.QubitUnitary(U_s, wires=range(n_qubits))
            if i % 2 == 1:
                qml.measure(0, **kwargs)
                iteration_count += 1
            if(iteration_count == no_of_iterations):
                break
        return qml.expval(H_total)

    start_time = time.time()
    result = circuit(reset=True)
    end_time = time.time()
    print(f"Circuit reset execution time: {end_time - start_time} seconds for {no_of_iterations} iterations \n Reset result: {result} \n")

    # print(qml.draw(circuit)())

    # start_time_post = time.time()
    # result = circuit(postselect=0)
    # end_time_post = time.time()
    # print(f"Circuit postselect execution time: {end_time_post - start_time_post} seconds for {no_of_iterations} iterations \n Postselect result: {result}")


generate_pennylane_circuit_from_unitary()

