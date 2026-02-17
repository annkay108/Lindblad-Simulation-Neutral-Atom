from time import time
import numpy as np
import pennylane as qml
from pathlib import Path
import utils


def generate_pennylane_circuit_from_unitary():
    no_of_iterations = 80  #-5.01500273878082 for 24 steps 1254 seconds
    dilated_unitary, n_qubits = utils.load_unitary_matrices()

    kwargs_mps = {
        # Maximum bond dimension of the MPS
        "max_bond_dim": 500000,
        # Cutoff parameter for the singular value decomposition
        "cutoff": np.finfo(np.complex128).eps,
        # Contraction strategy to apply gates
        "contract": "auto-mps",
    }

    dev = qml.device('default.qubit')

    # total_qubits = n_qubits + no_of_iterations
    # dev = qml.device('default.tensor', wires=total_qubits)

    hamiltonian_quspin, H_total = utils.tmif4_hamiltonian_pauli()

    initial_state = np.zeros(2**n_qubits, dtype=complex)
    initial_state[0] = 1.0
    state = initial_state

    @qml.qnode(dev)
    def apply_unitary_iteration(index, state):
        qml.StatePrep(state, wires=range(n_qubits))
        unitary = dilated_unitary[index+1] @ dilated_unitary[index]
        qml.QubitUnitary(unitary, wires=range(n_qubits))

        result = {"state": qml.state(), "expval": qml.expval(H_total)}
        return result

    startTime = time()
    for i in range(0,no_of_iterations*2,2):
        result = apply_unitary_iteration(i, state)

        zero = np.array([1, 0])
        psi_reshaped = result["state"].reshape(2, (2**(n_qubits-1)))

        psi_0 = psi_reshaped[0]        # amplitudes where q0 = 0
        p0 = np.vdot(psi_0, psi_0).real

        state = np.kron(zero, psi_0 / np.sqrt(p0))

        energy = result["expval"]
        # print(f"Iteration {i+1}, state: {state}")
    
    final_energy = energy
    endTime = time()
    exect_time = endTime - startTime
    return {
        "iterations": no_of_iterations,
        "execution_time": exect_time,
        "energy": final_energy,
    }
    # @qml.qnode(dev)
    # def circuit(**kwargs):
    #     iteration_count = 0

    #     for i, U_s in enumerate(dilated_unitary):
    #         qml.QubitUnitary(U_s, wires=range(n_qubits))
    #         if i % 2 == 1:
    #             qml.measure(0, **kwargs)
    #             iteration_count += 1
    #         if(iteration_count == no_of_iterations):
    #             break
    #     return qml.expval(H_total)

    # start_time = time.time()
    # result = circuit(reset=True)
    # end_time = time.time()
    # print(f"Circuit reset execution time: {end_time - start_time} seconds for {no_of_iterations} iterations \n Reset result: {result} \n")




    # print(qml.draw(circuit)())

    # start_time_post = time.time()
    # result = circuit(postselect=0)
    # end_time_post = time.time()
    # print(f"Circuit postselect execution time: {end_time_post - start_time_post} seconds for {no_of_iterations} iterations \n Postselect result: {result}")


print(generate_pennylane_circuit_from_unitary())

