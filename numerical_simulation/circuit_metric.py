from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
import utils
import numpy as np
from bqskit.compiler.gateset import GateSet
from bqskit.ir.gates import RXGate, RZGate, CZGate, SwapGate
from bqskit import MachineModel
from bqskit import compile, Circuit
from qiskit import qasm2
from bqskit.ir.lang.qasm2 import OPENQASM2Language


def qiskit_to_bqskit(qc: QuantumCircuit) -> Circuit:
    """Convert Qiskit's QuantumCircuit `qc` to a BQSKit Circuit."""
    circuit = OPENQASM2Language().decode(qasm2.dumps(qc))
    return circuit


def generate_qiskit_circuit_from_unitary_with_swaps():
    verbose = True
    basis_gates = [
        "rx",
        "rz",
        "cz",
        "swap",
    ]
    # dilated_unitary, n_qubits = utils.load_unitary_matrices(
    #     "Lindblad_simulation/numerical_simulation/multiple_jump_operator/operator/lindblad_operators4sites_80iter_1.pickle"
    # )
    dilated_unitary, n_qubits = utils.load_unitary_matrices("Lindblad_simulation/numerical_simulation/lindbladian_simulation/operator/lindblad_operators4sites_80_iter_1seg.pickle")

    qc = QuantumCircuit(n_qubits, n_qubits)

    Ugate = UnitaryGate(dilated_unitary)
    qc.append(Ugate, list(range(0, n_qubits)))
    circuit = transpile(qc, basis_gates=basis_gates, optimization_level=1)
    circuit = qiskit_to_bqskit(circuit)

    my_basis = GateSet([RXGate(), RZGate(), CZGate(), SwapGate()])
    model = MachineModel(6, gate_set=my_basis)

    if verbose:
        print(f"-----------------------------------")
        print(f"Gate counts from the input circuit")
        for gate in circuit.gate_set:
            print(f"{gate} Count:", circuit.count(gate))
        print(f"-----------------------------------")
    compiled_circuit = compile(circuit, model=model, optimization_level=1)

    if verbose:
        print(f"-----------------------------------")
        print(f"Gate counts after BQSKit Synthesis")
        for gate in compiled_circuit.gate_set:
            print(f"{gate} Count:", compiled_circuit.count(gate))
        print(f"-----------------------------------")


generate_qiskit_circuit_from_unitary_with_swaps()
