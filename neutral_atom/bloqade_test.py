# # from bloqade import qasm2
# from bloqade.qasm2.emit import QASM2
# # from bloqade.qasm2.parse import pprint

# # @qasm2.main
# # def main():
# #     q = qasm2.qreg(2)
# #     qasm2.h(q[0])
# #     qasm2.cx(q[0], q[1])

# #     c = qasm2.creg(2)
# #     qasm2.measure(q, c)
# #     return c

# # target = QASM2()
# # qasm2_program = target.emit(main)
# # pprint(qasm2_program)


# # from bloqade.pyqrack import StackMemorySimulator
# # from bloqade import qasm2

# # @qasm2.extended
# # def main():
# #     q = qasm2.qreg(2)

# #     qasm2.h(q[0])
# #     qasm2.cx(q[0], q[1])

# #     return q

# # sim = StackMemorySimulator(min_qubits=2)

# # # get the state vector -- oohh entanglement
# # state = sim.state_vector(main)
# # print(state)

# import math
# from kirin.dialects import ilist
# from bloqade import qasm2

# def ghz_log_simd(n: int):
#     n_qubits = int(2**n)

#     @qasm2.extended
#     def layer(i_layer: int, qreg: qasm2.QReg):
#         step = n_qubits // (2**i_layer)

#         def get_qubit(x: int):
#             return qreg[x]

#         ctrl_qubits = ilist.Map(fn=get_qubit, collection=range(0, n_qubits, step))
#         targ_qubits = ilist.Map(
#             fn=get_qubit, collection=range(step // 2, n_qubits, step)
#         )

#         # Ry(-pi/2)
#         qasm2.parallel.u(qargs=targ_qubits, theta=-math.pi / 2, phi=0.0, lam=0.0)

#         # CZ gates
#         qasm2.parallel.cz(ctrls=ctrl_qubits, qargs=targ_qubits)

#         # Ry(pi/2)
#         qasm2.parallel.u(qargs=targ_qubits, theta=math.pi / 2, phi=0.0, lam=0.0)

#     @qasm2.extended
#     def ghz_log_depth_program():

#         qreg = qasm2.qreg(n_qubits)

#         qasm2.h(qreg[0])
#         for i in range(n):
#             layer(i_layer=i, qreg=qreg)

#     return ghz_log_depth_program

# target = qasm2.emit.QASM2(
#     allow_parallel=True,
# )
# ast = target.emit(ghz_log_simd(4))
# qasm2.parse.pprint(ast)

# import pennylane as qml
# from functools import partial

# dev = qml.device("default.qubit", wires=8)

# @partial(qml.set_shots, shots=1000)
# @qml.qnode(dev)
# def GHZ_state_circuit():
#     qml.Hadamard(wires=0)
#     qml.CNOT(wires=[0, 4])
#     qml.CNOT(wires=[0, 2])
#     qml.CNOT(wires=[4, 6])
#     qml.CNOT(wires=[0, 1])
#     qml.CNOT(wires=[2, 3])
#     qml.CNOT(wires=[4, 5])
#     qml.CNOT(wires=[6, 7])  

#     # qml.Hadamard(wires=0)
#     # for i in range(7):
#     #     qml.CNOT(wires=[i, i + 1])
#     return qml.counts()
    
# print(qml.draw(GHZ_state_circuit)())
# print(GHZ_state_circuit())

from bloqade.analog import  piecewise_linear, piecewise_constant
from bloqade.analog.atom_arrangement import ListOfLocations,  Square, Honeycomb
# from bokeh.io import output_notebook # to plot "show()" on the notebook, without opening a new tab
import numpy as np

# my_register = ListOfLocations([(0.0, 0.0), (0.0, 5.0), (0.0, 9.0), (5.0, 2.0), (6.0, 7.0), (9.0, 10.0)])

# my_register.show()

# Square(4,3,lattice_spacing=5.2).show()

# durations = [0.4,3.2,0.4]
# values_MHz = [0.0,2.5,2.5,0.0]
# values_radsec = [i * 2*np.pi for i in values_MHz]

# waveform1 = piecewise_linear(durations, values_radsec)

# waveform1.show()

durations = [2.0,2.0]
values = [0.0,np.pi]

waveform1 = piecewise_constant(durations, values)

waveform1.show()

# rng = np.random.default_rng(1234)
# Honeycomb(3,3, lattice_spacing=4.5).apply_defect_density(0.3, rng=rng).remove_vacant_sites().show()