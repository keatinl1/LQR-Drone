import numpy as np

class KalmanFilter:
    '''filter the data from z-ranger'''
    def __init__(self):
        self.A = np.array([[0]])
        self.B = np.array([[1]])
        self.H = np.array([[1]])        # Observation matrix
        self.Q = np.array([[0.01]])     # Process noise covariance
        self.R = np.array([[0.0025]])   # Measurement noise covariance
        self.P = np.array([[0.04]])     # Initial state covariance
        self.x = 0                      # Initial state estimate

    def update(self, x, u):
        # Prediction step
        x_pred = self.A * self.x + self.B * u
        P_pred = self.A * self.P * self.A.T + self.Q

        # Update step
        innovation = x - self.H * x_pred
        S = self.H * P_pred * self.H.T + self.R
        K = P_pred * self.H.T * np.linalg.inv(S)

        self.x = x_pred + K * innovation
        self.P = (np.eye(self.A.shape[1]) - K * self.H) * P_pred

        return self.x