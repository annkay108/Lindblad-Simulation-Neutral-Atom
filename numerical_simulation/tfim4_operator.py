import numpy as np
import scipy.linalg as la
from pathlib import Path

class TFIM4Operator:
    def __init__(self, psi_initial, psi_GS):
        self.psi_initial = psi_initial
        self.psi_GS = psi_GS

    def partial_trace(self, psi_var, n_qubits, dice=[]):
        Ns = 2**n_qubits // 2
        psi_reduced = np.zeros((Ns,1), dtype=complex)
        prob = la.norm(psi_var[Ns:, 0])**2
        
        if dice[0] <= prob:
            psi_reduced[:, 0] = psi_var[Ns:, 0]
        else:
            psi_reduced[:, 0] = psi_var[:Ns, 0]
        psi_reduced[:,0] /= la.norm(psi_reduced[:,0])

        return psi_reduced
        
    
    def apply_operator(self, flip_dice=[]):
        fidelity_list = []

        path = Path().resolve().parent / "Lindblad_simulation/numerical_simulation/lindbladian_simulation/data/lindblad_operators.pickle"
        dilated_unitary = np.load(path, allow_pickle=True)
       
        n_qubits = int(np.log2(dilated_unitary[0].shape[0]))

        Ns= 2**(n_qubits) // 2

        psi = self.psi_initial.copy()
        psi_Gs = self.psi_GS.copy()

        psi_var = np.zeros((Ns*2,1), dtype=complex)
        psi_var.fill(0j)
        psi_var[:Ns, :] = psi.reshape(-1, 1)

        counter = 1


        fidelity_list.append(np.abs(np.vdot(psi_Gs, psi))**2)

        for mat in dilated_unitary:
            
            psi_var = mat @ psi_var

            if counter % 2 == 0:
                psi_var_reduced = self.partial_trace(psi_var, n_qubits, dice=flip_dice[counter//2-1])
                psi_var = np.zeros((Ns*2,1), dtype=complex)
                psi_var.fill(0j)
                psi_var[:Ns, :] = psi_var_reduced

                fidelity_list.append(np.abs(np.vdot(psi_Gs, psi_var_reduced))**2)

            counter += 1

        print("Final fidelity (operator method): ", fidelity_list[3])

        return psi_var, fidelity_list