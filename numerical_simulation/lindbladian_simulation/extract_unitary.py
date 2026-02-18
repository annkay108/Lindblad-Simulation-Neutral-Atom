

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
            return (-b +a)/(2.0*pi)
        else:
            return(
                np.exp(-((da* t)**2)/4)*np.exp(1j*a*t) - np.exp(-((db* t)**2)/4)*np.exp(1j*b*t)
            )/(2.0*pi*1j*t)
    
    
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
        path = Path().resolve().parent / f"Lindblad_simulation/numerical_simulation/lindbladian_simulation/data/lindblad_operators{self.L}sites_{self.num_t}iter_{self.num_segment}segNew.pickle"
        if not os.path.exists(path):
            with open(path, "wb") as f:
                pickle.dump(ops, f)
        print(f"Operators saved to {path}...")

    def step_Lindblad(
        self, tau, num_segment, num_rep, S_s, M_s, dice
    ):
        """
        Propagate one step of the dilated jump operator in a batch.
        """
        
        # Simulation preparation
        # first order method does not require reversing the grid
        isreverse = True
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
        eHts = self.eHts
        E_A = self.E_A  # eigenvalue of A
        psi_A = self.psi_A  # eigenvector of A shape=(16, 16) each column is an eigenvector of A
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

            expZA = np.exp(
                    -1j * 0.5 * tau_s * np.abs(F_contour[i]) * np.outer(E_A, tau_scal)
                )
            ZA_dilate[i, :Ns, :] = expZA  # AK dilated
            ZA_dilate[i, Ns:, :] = expZA.conj()
            #--------------------------------------------------------
        operator = np.eye(2 * Ns, dtype=complex)  # initialize the operator as identity
        for iseg in range(num_segment):
            if isreverse:  # second order
                for i in range(int(Ns_contour / 2)):  # left-ordered product
                    VK = np.kron(VF_contour[i, :, :], psi_A)
                    operator = np.kron(np.identity(2), eHts) @ VK @ np.diagflat(ZA_dilate[i, :, :]) @ VK.conj().T@ operator
                for i in range(int(Ns_contour / 2)):  # right-ordered product
                    VK = np.kron(VF_contour[i + int(Ns_contour / 2), :, :], psi_A)
                    operator = VK @ np.diagflat(ZA_dilate[i + int(Ns_contour / 2), :, :]) @ VK.conj().T @ np.kron(np.identity(2), eHts.conj().T) @ operator
            else:  # first order
                # only #left-ordered product
                print("is not reverse")
        
        return operator

    def Lindblad_simulation(
        self, T, num_t, num_segment, num_rep, S_s, M_s,  flip_dice=[]
    ):
        all_gates = (
            []
        )  # extract the unitaries here of the full circuit, (e^-iHt/T e^-iKt/T)^T

        H = self.H_op #shape=(16, 16)

        # Simulation parameter
        tau = T / num_t # T= 80, num_t = 80
        tau_s = S_s / M_s  # time step for integral discretization tau_s = 0.11580703270444591 S_s = 4.863895373589 M_s = 42

        eHtau = la.expm(-1j * tau * H) # e^-iHtau where tau is 1
        self.eHts = la.expm(-1j * tau_s * self.H_op)  # short time Hamiltonian simulation shape=(16, 16)
        self.E_A, self.psi_A = la.eigh(
            self.A_op
        )  
        # Output Storage
        time_H = np.zeros(num_t + 1)  # List of total Hamiltonian simulation time zeros [0, 1, 2, ..., 80]

        for it in range(num_t):
            print(it, "iteration ")

            time_H[it + 1] = time_H[it] + tau
            ops = self.step_Lindblad(
                tau,
                num_segment,
                num_rep,
                S_s,
                M_s,
                flip_dice[it, :],
            )
            ops = ops @ np.kron(np.identity(2), eHtau)
            all_gates.append(ops)

            time_H[it + 1] = (
                time_H[it + 1] + 2 * num_segment * S_s
            )  

        self.save_operator(all_gates)

        return all_gates
