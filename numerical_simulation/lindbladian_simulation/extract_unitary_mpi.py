from functools import reduce
import numpy as np
import scipy.linalg as la
from numpy import pi
import pickle
import os
from pathlib import Path
from mpi4py import MPI


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

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def filter_time(self, t):

        a = self.filter_a
        b = self.filter_b
        da = self.filter_da
        db = self.filter_db

        if np.abs(t) < 1e-10:
            return (-b + a) / (2.0 * pi)

        return (
            np.exp(-((da * t) ** 2) / 4) * np.exp(1j * a * t)
            - np.exp(-((db * t) ** 2) / 4) * np.exp(1j * b * t)
        ) / (2.0 * pi * 1j * t)

    def time_contour(self, S_s, M_s, isreverse=True):

        tau_s = S_s / M_s
        tgrid = -S_s + np.arange(2 * M_s + 1) * tau_s

        if isreverse:
            return np.append(tgrid, tgrid[::-1])
        else:
            return tgrid

    def save_operator(self, ops):

        if self.rank != 0:
            return

        path = Path().resolve().parent / \
            f"Lindblad_simulation/numerical_simulation/lindbladian_simulation/data/lindblad_operators{self.L}sites_{self.num_t}iter_{self.num_segment}segNew.pickle"

        with open(path, "wb") as f:
            pickle.dump(ops, f)

        print(f"Operators saved to {path}")

    def step_Lindblad(self, tau, num_segment, num_rep, S_s, M_s):

        isreverse = True
        tau_s = S_s / M_s

        s_contour = self.time_contour(S_s, M_s, isreverse=isreverse)
        Ns_contour = s_contour.shape[0]

        eHts = self.eHts
        E_A = self.E_A
        psi_A = self.psi_A
        Ns = self.Ns

        tau_scal = np.ones(num_rep) * np.sqrt(tau) / num_segment

        # global arrays
        F_contour = np.zeros(Ns_contour, dtype=np.complex128)
        VF_contour = np.zeros((Ns_contour, 2, 2), dtype=np.complex128)
        ZA_dilate = np.zeros((Ns_contour, 2 * Ns, num_rep), dtype=np.complex128)

        # ---------- DISTRIBUTE WORK ----------

        counts = [Ns_contour // self.size + (1 if r < Ns_contour % self.size else 0)
                  for r in range(self.size)]

        starts = [sum(counts[:r]) for r in range(self.size)]

        local_start = starts[self.rank]
        local_end = local_start + counts[self.rank]
        local_n = counts[self.rank]

        local_F = np.zeros(local_n, dtype=np.complex128)
        local_VF = np.zeros((local_n, 2, 2), dtype=np.complex128)
        local_ZA = np.zeros((local_n, 2 * Ns, num_rep), dtype=np.complex128)

        smin = np.min(s_contour)
        smax = np.max(s_contour)

        # ---------- PARALLEL LOOP ----------

        for idx, i in enumerate(range(local_start, local_end)):

            if (s_contour[i] == smin) or (s_contour[i] == smax):
                local_F[idx] = self.filter_time(s_contour[i]) / 2
            else:
                local_F[idx] = self.filter_time(s_contour[i])

            fac = np.exp(1j * np.angle(local_F[idx]))

            local_VF[idx] = (
                1.0 / np.sqrt(2)
                * np.array([[1, 1], [fac, -fac]])
            )

            expZA = np.exp(
                -1j * 0.5 * tau_s *
                np.abs(local_F[idx]) *
                np.outer(E_A, tau_scal)
            )

            local_ZA[idx, :Ns, :] = expZA
            local_ZA[idx, Ns:, :] = expZA.conj()

        # ---------- GATHER RESULTS ----------

        counts = np.array(counts)

        # gather F
        recvcounts_F = counts
        displs_F = np.insert(np.cumsum(recvcounts_F), 0, 0)[0:-1]

        self.comm.Allgatherv(
            [local_F, MPI.COMPLEX],
            [F_contour, recvcounts_F, displs_F, MPI.COMPLEX]
        )

        # gather VF
        local_VF_flat = local_VF.reshape(-1)
        VF_contour_flat = VF_contour.reshape(-1)

        recvcounts_VF = counts * 4
        displs_VF = np.insert(np.cumsum(recvcounts_VF), 0, 0)[0:-1]

        self.comm.Allgatherv(
            [local_VF_flat, MPI.COMPLEX],
            [VF_contour_flat, recvcounts_VF, displs_VF, MPI.COMPLEX]
        )

        # gather ZA
        local_ZA_flat = local_ZA.reshape(-1)
        ZA_dilate_flat = ZA_dilate.reshape(-1)

        recvcounts_ZA = counts * (2 * Ns * num_rep)
        displs_ZA = np.insert(np.cumsum(recvcounts_ZA), 0, 0)[0:-1]

        self.comm.Allgatherv(
            [local_ZA_flat, MPI.COMPLEX],
            [ZA_dilate_flat, recvcounts_ZA, displs_ZA, MPI.COMPLEX]
        )

        # ---------- PROPAGATION ----------

        operator = np.eye(2 * Ns, dtype=np.complex128)

        for iseg in range(num_segment):

            if isreverse:

                for i in range(int(Ns_contour / 2)):

                    VK = np.kron(VF_contour[i], psi_A)

                    operator = (
                        np.kron(np.identity(2), eHts)
                        @ VK
                        @ np.diagflat(ZA_dilate[i])
                        @ VK.conj().T
                        @ operator
                    )

                for i in range(int(Ns_contour / 2)):

                    idx = i + int(Ns_contour / 2)

                    VK = np.kron(VF_contour[idx], psi_A)

                    operator = (
                        VK
                        @ np.diagflat(ZA_dilate[idx])
                        @ VK.conj().T
                        @ np.kron(np.identity(2), eHts.conj().T)
                        @ operator
                    )

        return operator

    def Lindblad_simulation(self, T, num_t, num_segment, num_rep, S_s, M_s):

        all_gates = []

        H = self.H_op
        tau = T / num_t
        tau_s = S_s / M_s

        eHtau = la.expm(-1j * tau * H)
        self.eHts = la.expm(-1j * tau_s * H)

        self.E_A, self.psi_A = la.eigh(self.A_op)

        for it in range(num_t):

            if self.rank == 0:
                print("Iteration", it)

            ops = self.step_Lindblad(
                tau,
                num_segment,
                num_rep,
                S_s,
                M_s
            )

            ops = ops @ np.kron(np.identity(2), eHtau)

            all_gates.append(ops)

        self.save_operator(all_gates)

        return all_gates
