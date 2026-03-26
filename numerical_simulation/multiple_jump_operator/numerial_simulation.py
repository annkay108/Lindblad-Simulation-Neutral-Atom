from functools import reduce
import numpy as np
import scipy.linalg as la

# from scipy.special import erf
from numpy import pi
import pickle
import os
from pathlib import Path


class ExtractUnitary:
    def __init__(self, H_op, A_op, filter_params, L, num_segment, num_t):
        self.H_op = H_op
        self.A_op = A_op
        self.Ns = H_op.shape[0]
        self.filter_a = filter_params["a"]
        self.filter_b = filter_params["b"]
        self.filter_da = filter_params["da"]
        self.filter_db = filter_params["db"]
        self.L = L
        self.num_segment = num_segment
        self.num_t = num_t

    def filter_time(self, t):
        """Define the function for time filtering."""
        a = self.filter_a
        b = self.filter_b
        da = self.filter_da
        db = self.filter_db
        if np.abs(t) < 1e-10:
            return (-b + a) / (2.0 * pi)
        else:
            return (
                np.exp(-((da * t) ** 2) / 4) * np.exp(1j * a * t)
                - np.exp(-((db * t) ** 2) / 4) * np.exp(1j * b * t)
            ) / (2.0 * pi * 1j * t)

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

    def save_operator(self, ops):
        path = (
            Path().resolve().parent
            / f"Lindblad_simulation/numerical_simulation/lindbladian_simulation/data/lindblad_operators{self.L}sites_{self.num_t}iter_{self.num_segment}segNewPow.pickle"
        )
        if not os.path.exists(path):
            with open(path, "wb") as f:
                pickle.dump(ops, f)
        print(f"Operators saved to {path}...")

    # def trace_out_ancilla(self, psi_t_batch, dice, num_batch, Ns, psi):
    #     for ir in range(num_batch):  # sampling of the ancillary state
    #         prob = la.norm(psi_t_batch[Ns:, ir]) ** 2
    #         if dice[ir] <= prob:
    #             # flip the |1>| state
    #             psi[:, ir] = psi_t_batch[Ns:2*Ns, ir]
    #         else:
    #             # keep the |0> state
    #             psi[:, ir] = psi_t_batch[:Ns, ir]

    #         # psi[:, ir] = psi_t_batch[:Ns, ir]

    #         # normalize
    #         psi[:, ir] /= la.norm(psi[:, ir])
    #     return psi

    def trace_out_ancilla(self, psi_t_batch, dice, num_batch, Ns, psi):
        for ir in range(num_batch):

            # --- split into 4 sectors ---
            psi0 = psi_t_batch[0*Ns:1*Ns, ir]
            psi1 = psi_t_batch[1*Ns:2*Ns, ir]
            psi2 = psi_t_batch[2*Ns:3*Ns, ir]
            psi3 = psi_t_batch[3*Ns:4*Ns, ir]

            # --- compute probabilities ---
            p0 = la.norm(psi0)**2
            p1 = la.norm(psi1)**2
            p2 = la.norm(psi2)**2
            p3 = la.norm(psi3)**2

            probs = np.array([p0, p1, p2, p3])
            probs /= np.sum(probs)  # normalize (important for stability)

            # --- cumulative distribution ---
            cdf = np.cumsum(probs)

            r = dice[ir]

            # --- sample sector ---
            if r <= cdf[0]:
                psi[:, ir] = psi0
            elif r <= cdf[1]:
                psi[:, ir] = psi1
            elif r <= cdf[2]:
                psi[:, ir] = psi2
            else:
                psi[:, ir] = psi3

            # --- normalize ---
            psi[:, ir] /= la.norm(psi[:, ir])

        return psi
    def step_Lindblad(self, tau, num_segment, num_rep, S_s, M_s):
        """
        Propagate one step of the dilated jump operator in a batch.
        """

        # Simulation preparation
        # first order method does not require reversing the grid
        isreverse = True
        tau_s = S_s / M_s

        s_contour = self.time_contour(
            S_s, M_s, isreverse=isreverse
        )  # discrete s point (85,)
        Ns_contour = s_contour.shape[0]  # number of discrete s point 85
        F_contour = np.zeros((Ns_contour), dtype=complex)  # discrete F value
        VF_contour = np.zeros(
            (Ns_contour, 4, 4), dtype=complex
        )  # discrete dilated F value
        tau_scal = (
            np.ones(num_rep) * np.sqrt(tau) / num_segment
        )  # rescaled tau (for discrete Lindblad)
        eHts = self.eHts
        E_A = self.E_A  # eigenvalue of A
        psi_A = (
            self.psi_A
        )  # eigenvector of A shape=(16, 16) each column is an eigenvector of A
        Ns = self.Ns  # dimension of the system
        ZA_dilate = np.zeros(
            (Ns_contour, 4 * Ns, num_rep), dtype=complex
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

            # --------------------------------------------------------
            fac = np.exp(-1j * np.angle(F_contour[i]))
            VF_contour[i, :, :] = (
                1.0
                / np.sqrt(2)
                * np.array(
                    [
                        [fac, 0, 0, -fac],
                        [1 / np.sqrt(3), 1, 1 / np.sqrt(3), 1 / np.sqrt(3)],
                        [1 / np.sqrt(3), -1, 1 / np.sqrt(3), 1 / np.sqrt(3)],
                        [1 / np.sqrt(3), 0, -2 / np.sqrt(3), 1 / np.sqrt(3)],
                    ]
                )
            )  # eigenvectors of σ_l

            expZA = np.exp(
                -1j
                * 0.5
                * tau_s
                * np.sqrt(3)
                * np.abs(F_contour[i])
                * np.outer(E_A, tau_scal)
            )

            ZA_dilate[i, :Ns, :] = expZA  # AK dilated
            ZA_dilate[i, -Ns:, :] = expZA.conj()

            # --------------------------------------------------------
        print(
            "Finished calculating local jump operators for all discrete integral points. Start constructing global operators..."
        )

        operator = np.eye(4 * Ns, dtype=complex)  # initialize the operator as identity
        identityKron = np.kron(np.identity(4), eHts)
        identityKronconj = np.kron(np.identity(4), eHts.conj().T)

        if isreverse:  # second order
            for i in range(int(Ns_contour / 2)):  # right-ordered product
                VK = np.kron(VF_contour[i + int(Ns_contour / 2), :, :], psi_A)
                operator = (
                    identityKronconj
                    @ VK
                    @ np.diagflat(ZA_dilate[i + int(Ns_contour / 2), :, :])
                    @ VK.conj().T
                    @ operator
                )
            print("Finished left-ordered product. Start right-ordered product...")
            for i in range(int(Ns_contour / 2)):  # left-ordered product
                VK = np.kron(VF_contour[i, :, :], psi_A)
                operator = (
                    VK
                    @ np.diagflat(ZA_dilate[i, :, :])
                    @ VK.conj().T
                    @ identityKron
                    @ operator
                )
        else:  # first order
            # only #left-ordered product
            print("is not reverse")

        return np.linalg.matrix_power(operator, num_segment)

    def Lindblad_simulation(
        self, T, num_t, num_segment, num_rep, S_s, M_s, psi0, psi_GS=[], flip_dice=[]
    ):
        Ns = psi0.shape[0]
        time_H = np.zeros(num_t + 1)

        H = self.H_op
        tau = T / num_t
        tau_s = S_s / M_s

        time_series = np.arange(num_t + 1) * tau

        eHtau = la.expm(-1j * tau * H)  # e^-iHtau where tau is 1
        self.eHts = la.expm(
            -1j * tau_s * self.H_op
        )  # short time Hamiltonian simulation shape=(16, 16)
        self.eHT = la.expm(-1j * S_s * self.H_op)

        self.E_A, self.psi_A = la.eigh(self.A_op)

        avg_energy_hist = np.zeros((num_t + 1, num_rep))  # shape is (81, 1)
        avg_energy_hist[0, :].fill(np.vdot(psi0, H @ psi0).real)  # List of energy

        avg_pGS_hist = np.zeros((num_t + 1, num_rep))

        avg_pGS_hist[0, :].fill(np.abs(np.vdot(psi0, psi_GS)) ** 2)

        rho_hist = np.zeros(
            (Ns, Ns, num_t + 1), dtype=complex
        )  # \rho_n Ns=16, num_t=80 shape=(16, 16, 81)
        psi_all = np.zeros(
            (Ns, num_rep), dtype=complex
        )  # List of psi_n Ns=16, num_rep=1 shape=(16, 1)

        for i in range(num_rep):
            psi_all[:, i] = self.eHT.conj().T @ psi0.copy()

        rho_hist[:, :, 0] = np.outer(psi_all[:, 0], psi_all[:, 0].conj().T)

        ops = self.step_Lindblad(tau, num_segment, num_rep, S_s, M_s)
        ops = ops @ np.kron(np.identity(4), eHtau)

        for it in range(num_t):
            # print(it, "iteration ")
            time_H[it + 1] = time_H[it] + tau

            psi_full = np.zeros((4 * Ns, 1), dtype=complex)  # 32, 1
            psi_full.fill(0j)
            psi_full[:Ns, :] = psi_all  # initial state shape=(16, 1)

            psi_full = ops @ psi_full

            psi_all = self.trace_out_ancilla(
                psi_full, flip_dice[it, :], num_rep, Ns, psi_all
            )
            rho_hist[:, :, it + 1] = np.outer(psi_all[:, 0], psi_all[:, 0].conj().T)
            time_H[it + 1] = time_H[it + 1] + 2 * num_segment * S_s
            avg_energy_hist[it + 1, :] = np.einsum(
                "in,in->n", psi_all.conj(), H @ psi_all
            ).real
            avg_pGS_hist[it + 1, :] = (
                np.abs(np.einsum("in,i->n", psi_all.conj(), psi_GS)) ** 2
            )
        avg_energy = np.mean(avg_energy_hist, axis=1)
        avg_pGS = np.mean(avg_pGS_hist, axis=1)

        # self.save_operator(all_gates)

        return avg_energy, avg_pGS, time_series, time_H
