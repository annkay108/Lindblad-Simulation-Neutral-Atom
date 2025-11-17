from functools import reduce
import numpy as np
import scipy.linalg as la
from scipy.special import erf
from numpy import pi
import pickle
import os
from pathlib import Path

class LindbladSimulator:
    def __init__(self, H_op, A_op, filter_params):
        self.H_op = H_op
        self.A_op = A_op
        self.Ns = H_op.shape[0]
        self.filter_a = filter_params["a"]
        self.filter_b = filter_params["b"]
        self.filter_da = filter_params["da"]
        self.filter_db = filter_params["db"]
    
    def filter_time(self, t):
        """Define the function for time filtering."""
        a = self.filter_a
        b = self.filter_b
        da = self.filter_da
        db = self.filter_db
        if np.abs(t) < 1e-10:
            return (-b +a)/(2.0*pi)
        else:
            return(
                np.exp(-((da* t)**2)/4)*np.exp(1j*a*t) - np.exp(-((db* t)**2)/4)*np.exp(1j*b*t)
            )/(2.0*pi*1j*t)
        
    def filter_freq(self, w):
        """Define the function for frequency filtering."""
        a = self.filter_a
        b = self.filter_b
        da = self.filter_da
        db = self.filter_db
        return 0.5*(erf((w + a)/da) - erf((w + b)/db))
    
    def construct_jump_exact(self):
        """Construct the jump operator in frequency domain."""
        H_mat = self.H_op
        A_mat = self.A_op

        E_H, psi_H = la.eigh(H_mat)

        A_ovlp = psi_H.conj().T @ (A_mat.dot(psi_H))
        Ns = self.Ns  # number of eigenvectors
        A_jump = np.zeros((Ns, Ns))

        for i in range(Ns):
            for j in range(Ns):
                A_jump += (
                    self.filter_freq(E_H[i] - E_H[j])
                    * A_ovlp[i, j]
                    * np.outer(psi_H[:, i], psi_H[:, j].conj())
                )

        self.A_jump = A_jump
    
    def time_contour(self, S_s, M_s, isreverse=True):
        """
        Construct the time contour for propagating the Kraus operator in
        time domain.
        2M_s+1 grid points (include s=0)
        """
        tau_s = S_s / M_s
        tgrid = np.zeros(2 * M_s + 1)
        tgrid = -S_s + np.arange(2 * M_s + 1) * tau_s

        if isreverse:
            return np.append(tgrid, tgrid[::-1])  # reverse
        else:
            return tgrid
        
    def trace_out_ancilla(self, psi_t_batch, dice, num_batch, Ns, psi):
        for ir in range(num_batch):  # sampling of the ancillary state
            prob = la.norm(psi_t_batch[Ns:, ir]) ** 2
            if dice[ir] <= prob:
                # flip the |1>| state
                psi[:, ir] = psi_t_batch[Ns:, ir]
            else:
                # keep the |0> state
                psi[:, ir] = psi_t_batch[:Ns, ir]

            # normalize
            psi[:, ir] /= la.norm(psi[:, ir])
        return psi
    
    def save_operator(self, ops):
        path = Path().resolve().parent / "Lindblad_simulation/numerical_simulation/lindbladian_simulation/data/lindblad_operators.pickle"
        if not os.path.exists(path):
            with open(path, "wb") as f:
                pickle.dump(ops, f)

    def step_Lindblad(
        self, psi, psi_op, tau, num_t, num_segment, num_rep, S_s, M_s, dice, intorder
    ):
        """
        Propagate one step of the dilated jump operator in a batch.
        """
        num_batch = psi.shape[1]

        if not intorder in {1, 2}:
            raise ValueError("intorder must be 1 or 2.")
        
        # Simulation preparation
        # first order method does not require reversing the grid
        isreverse = intorder > 1
        tau_s = S_s / M_s
        s_contour = self.time_contour(S_s, M_s, isreverse=isreverse)  # discrete s point (85,)
        Ns_contour = s_contour.shape[0] # number of discrete s point 85
        F_contour = np.zeros((Ns_contour), dtype=complex)  # discrete F value
        VF_contour = np.zeros(
            (Ns_contour, 2, 2), dtype=complex
        )  # discrete dilated F value
        tau_scal = (
            np.ones(num_rep) * np.sqrt(tau) / num_segment
        )  # rescaled tau (for discrete Lindblad)
        eHt = self.eHt
        eHT = self.eHT
        E_A = self.E_A  # eigenvalue of A
        psi_A = self.psi_A  # eigenvector of A
        Ns = self.Ns  # dimension of the system
        ZA_dilate = np.zeros(
            (Ns_contour, 2 * Ns, num_rep), dtype=complex
        )  # local jump operator
        # for discrete integral point

        for i in range(Ns_contour):
            if (s_contour[i] == np.min(s_contour)) or (
                s_contour[i] == np.max(s_contour)
            ):
                F_contour[i] = (
                    self.filter_time(s_contour[i]) / 2
                )  # inverse fourier transformed filter function
            else:
                F_contour[i] = self.filter_time(s_contour[i])

            #--------------------------------------------------------
            fac = np.exp(1j * np.angle(F_contour[i]))
            VF_contour[i, :, :] = (
                1.0 / np.sqrt(2) * np.array([[1, 1], [fac, -fac]])
            )  # eigenvectors of σ_l
            if intorder == 1:  # first order
                expZA = np.exp(
                    -1j * tau_s * np.abs(F_contour[i]) * np.outer(E_A, tau_scal)
                )
            else:  # second order
                expZA = np.exp(
                    -1j * 0.5 * tau_s * np.abs(F_contour[i]) * np.outer(E_A, tau_scal)
                )
            ZA_dilate[i, :Ns, :] = expZA  # AK dilated
            ZA_dilate[i, Ns:, :] = expZA.conj()
            #--------------------------------------------------------
        # ---start simulation

        psi_t_batch = np.zeros((2 * Ns, num_batch), dtype=complex) # 32, 1
        psi_t_batch.fill(0j)
        psi_t_batch[:Ns, :] = psi

        psi_t_batch_op = np.zeros((2 * Ns, num_batch), dtype=complex) # 32, 1
        psi_t_batch_op.fill(0j)
        psi_t_batch_op[:Ns, :] = psi_op

        ops = []  #  extract the unitaries of the circuit here
        for iseg in range(num_segment):
            if isreverse:  # second order
                for i in range(int(Ns_contour / 2)):  # left-ordered product
                    VK = np.kron(VF_contour[i, :, :], psi_A)
                    psi_t_batch = VK.conj().T @ psi_t_batch
                    # pointwise multiplication
                    psi_t_batch *= ZA_dilate[i, :, :]
                    psi_t_batch = VK @ psi_t_batch
                    psi_t_batch = np.kron(np.identity(2), eHt) @ psi_t_batch
                for i in range(int(Ns_contour / 2)):  # right-ordered product
                    psi_t_batch = np.kron(np.identity(2), eHt.conj().T) @ psi_t_batch

                    VK = np.kron(VF_contour[i + int(Ns_contour / 2), :, :], psi_A)
                    psi_t_batch = VK.conj().T @ psi_t_batch

                    # pointwise multiplication
                    psi_t_batch *= ZA_dilate[i + int(Ns_contour / 2), :, :]

                    psi_t_batch = VK @ psi_t_batch
            else:  # first order
                # only #left-ordered product
                for i in range(int(Ns_contour)):
                    VK = np.kron(VF_contour[i, :, :], psi_A)

                    psi_t_batch = VK.conj().T @ psi_t_batch
                    ops.append(VK.conj().T)
                    # pointwise multiplication
                    psi_t_batch *= ZA_dilate[i, :, :]
                    ops.append(np.diagflat(ZA_dilate[i, :, :]))
                    psi_t_batch = VK @ psi_t_batch
                    ops.append(VK)
                    psi_t_batch = np.kron(np.identity(2), eHt) @ psi_t_batch
                    ops.append(np.kron(np.identity(2), eHt))
                # rewind the time. This seems quite important in
                # practice, which is consistent with the (unexplained)
                # importance of adding the coherent contribution.
                is_rewind = True
                if is_rewind:
                    psi_t_batch = (
                        np.kron(np.identity(2), self.eHT.conj().T) @ psi_t_batch
                    )
                    psi_t_batch = (
                        np.kron(np.identity(2), self.eHT.conj().T) @ psi_t_batch
                    )
                    ops.append(np.kron(np.identity(2), self.eHT.conj().T))
                    ops.append(np.kron(np.identity(2), self.eHT.conj().T))
                    # ops.append(["r"])

                overall_matrix_algorithm = reduce(lambda a, b: a @ b, reversed(ops))

                psi_t_batch_op = overall_matrix_algorithm @ psi_t_batch_op
        
        psi_without_op = self.trace_out_ancilla(psi_t_batch, dice, num_batch, Ns, psi)
        psi_all_op = self.trace_out_ancilla(psi_t_batch_op, dice, num_batch, Ns, psi_op)

        # for ir in range(num_batch):  # sampling of the ancillary state
        #     prob = la.norm(psi_t_batch[Ns:, ir]) ** 2
        #     if dice[ir] <= prob:
        #         # flip the |1>| state
        #         psi[:, ir] = psi_t_batch[Ns:, ir]
        #     else:
        #         # keep the |0> state
        #         psi[:, ir] = psi_t_batch[:Ns, ir]

        #     # normalize
        #     psi[:, ir] /= la.norm(psi[:, ir])

        return psi_without_op, psi_all_op, overall_matrix_algorithm

    def Lindblad_simulation(
        self, T, num_t, num_segment, psi0, num_rep, S_s, M_s, psi_GS=[], intorder=2, flip_dice=[]
    ):
        """
        Lindblad simulation

        This uses the deterministic propagation with first or second
        order Trotter (intorder).  In particular, the first order Trotter method
        enables propagation with positive time.
        """
        all_gates = (
            []
        )  # extract the unitaries here of the full circuit, (e^-iHt/T e^-iKt/T)^T

        H = self.H_op #shape=(16, 16)

        # Simulation parameter
        tau = T / num_t # T= 80, num_t = 80
        time_series = np.arange(num_t + 1) * tau # [0., 1., 2., ..., 80.]
        Ns = psi0.shape[0]  # length of the state Ns = 16
        tau_s = S_s / M_s  # time step for integral discretization tau_s = 0.11580703270444591 S_s = 4.863895373589 M_s = 42

        eHtau = la.expm(-1j * tau * H) # e^-iHtau where tau is 1
        self.eHt = la.expm(-1j * tau_s * self.H_op)  # short time Hamiltonian simulation shape=(16, 16)
        self.eHT = la.expm(-1j * S_s * self.H_op)# shape=(16, 16)
        self.E_A, self.psi_A = la.eigh(
            self.A_op
        )  # diagonalize A for later implementation

        # Output Storage
        time_H = np.zeros(num_t + 1)  # List of total Hamiltonian simulation time zeros [0, 1, 2, ..., 80]

        avg_energy_hist_op = np.zeros((num_t + 1, num_rep)) # shape is (81, 1)
        avg_energy_hist_op[0, :].fill(np.vdot(psi0, H @ psi0).real)  # List of energy

        avg_energy_hist = np.zeros((num_t + 1, num_rep)) # shape is (81, 1)
        avg_energy_hist[0, :].fill(np.vdot(psi0, H @ psi0).real)  # List of energy

        avg_pGS_hist = np.zeros(
            (num_t + 1, num_rep)
        )  # List of overlap with ground state

        if len(psi_GS) == 0:
            psi_GS = np.zeros_like(psi0)

        avg_pGS_hist[0, :].fill(np.abs(np.vdot(psi0, psi_GS)) ** 2)  # initial overlap

        avg_pGS_hist_op = np.zeros(
            (num_t + 1, num_rep)
        )  # List of overlap with ground state

        avg_pGS_hist_op[0, :].fill(np.abs(np.vdot(psi0, psi_GS)) ** 2)  # initial overlap

        # this randomness is introduced for modeling the tracing out
        # operation. Cannot be derandomized.


        rho_hist = np.zeros((Ns, Ns, num_t + 1), dtype=complex)  # \rho_n Ns=16, num_t=80 shape=(16, 16, 81)
        psi_all = np.zeros((Ns, num_rep), dtype=complex)  # List of psi_n Ns=16, num_rep=1 shape=(16, 1)
        psi_all_ops = np.zeros((Ns, num_rep), dtype=complex)

        for i in range(num_rep):
            psi_all[:, i] = psi0.copy()
            psi_all_ops[:, i] = psi0.copy()

        rho_hist[:, :, 0] = np.outer(psi_all[:, 0], psi_all[:, 0].conj().T)
       
        for it in range(num_t):
            psi_all = eHtau @ psi_all
            all_gates.append(np.kron(np.identity(2), eHtau))

            psi_all_ops = eHtau @ psi_all_ops

            time_H[it + 1] = time_H[it] + tau
            psi_all, psi_all_ops, ops = self.step_Lindblad(
                psi_all,
                psi_all_ops,
                tau,
                num_t,
                num_segment,
                num_rep,
                S_s,
                M_s,
                flip_dice[it, :],
                1,
            )
            all_gates.append(ops)
            rho_hist[:, :, it + 1] = (
                np.einsum("in,jn->ij", psi_all, psi_all.conj()) / num_rep
            )  # taking average to get \rho_n
            time_H[it + 1] = (
                time_H[it + 1] + 2 * num_segment * S_s
            )  # Calculating total H simulation time ???? num_segment=1
            # measurement
            avg_energy_hist[it + 1, :] = np.einsum(
                "in,in->n", psi_all.conj(), H @ psi_all
            ).real  # Calculating energy
            avg_pGS_hist[it + 1, :] = (
                np.abs(np.einsum("in,i->n", psi_all.conj(), psi_GS)) ** 2
            )  # Calculating overlap

            avg_energy_hist_op[it + 1, :] = np.einsum(
                "in,in->n", psi_all_ops.conj(), H @ psi_all_ops
            ).real  # Calculating energy
            avg_pGS_hist_op[it + 1, :] = (
                np.abs(np.einsum("in,i->n", psi_all_ops.conj(), psi_GS)) ** 2
            )  # Calculating overlap

        self.save_operator(all_gates)
        avg_energy = np.mean(avg_energy_hist, axis=1)
        avg_pGS = np.mean(avg_pGS_hist, axis=1)


        avg_energy_op = np.mean(avg_energy_hist_op, axis=1)
        avg_pGS_op = np.mean(avg_pGS_hist_op, axis=1)
        print("Final energy (operator method): ", avg_pGS_hist[3])
        return time_series, avg_energy, avg_pGS, avg_energy_op, avg_pGS_op, time_H, rho_hist, all_gates
