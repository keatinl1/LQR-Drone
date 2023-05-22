import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    '''controller class'''
    def __init__(self, z_pos, z_goal):
        self.A = np.array([[0]])
        self.B = np.array([[1]])
        self.C = np.array([[1]])
        self.D = np.array([[0]])

        self.Q = np.diag([1])
        self.R = np.array([[0.1]])

        # Compute the LQR gain using the algebraic Riccati equation
        S = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ S

        self.z_pos = z_pos
        self.z_goal = z_goal

        self.z_vel = self.control()

    def control(self):
        # Compute the control input using the LQR gain and the current state
        return self.K * (self.z_goal - self.z_pos)