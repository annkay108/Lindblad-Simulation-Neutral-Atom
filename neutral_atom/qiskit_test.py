from qiskit import QuantumCircuit
from mqt.core import load
from mqt.qmap.na.state_preparation import get_ops_for_solver
from mqt.qmap.na.state_preparation import NAStatePreparationSolver
from mqt.qmap.na.state_preparation import generate_code
import os
import sys

def setup_paths():
    # Append project paths to sys.path
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
    sim_dir = os.path.join(base_dir, "numerical_simulation")
    sys.path.insert(0, base_dir)
    sys.path.insert(0, sim_dir)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, project_root)
    print("Project root added to sys.path:", project_root)
    print("Working directory:", os.getcwd())


setup_paths()
import numerical_simulation.utils as utils

dialated_unitary, n_qubits = utils.load_unitary_matrices()
no_of_iterations = 2

hamiltonian_quspin, H_total = utils.tmif4_hamiltonian_pauli()

def circuit_example():
    qc = QuantumCircuit(n_qubits + no_of_iterations)
    ancilla_idx = 0

    for i, U_s in enumerate(dialated_unitary):
        qc.unitary(U_s, range(n_qubits))
        if i % 2 == 1:
            qc.swap(0, n_qubits + ancilla_idx)
            ancilla_idx += 1
        if(ancilla_idx == no_of_iterations):
            break

    print(qc)

    return qc

# circuit_example()



    

qc = QuantumCircuit(7)
qc.h(range(7))
qc.cz(0, 3)
qc.cz(0, 4)
qc.cz(1, 2)
qc.cz(1, 5)
qc.cz(1, 6)
qc.cz(2, 3)
qc.cz(2, 4)
qc.cz(3, 5)
qc.cz(4, 6)
qc.h(0)
qc.h(2)
qc.h(5)
qc.h(6)

qc.draw(output="mpl")

circ = load(qc)
ops = get_ops_for_solver(circ, "z", 1)  # We extract the 'Z' gates with '1' control, i.e., CZ gates
print(ops)

solver = NAStatePreparationSolver(3, 7, 2, 3, 2, 2, 2, 2, 2, 4)
result = solver.solve(ops, 7, 4, None, False, True)

code = generate_code(circ, result)
print(code)