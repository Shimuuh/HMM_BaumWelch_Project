import numpy as np

class HiddenMarkovModel:
    def __init__(self, A, B, pi):
        """
        A: State transition matrix
        B: Emission matrix
        pi: Initial state probabilities
        """
        self.A = np.array(A)
        self.B = np.array(B)
        self.pi = np.array(pi)
        self.N = self.A.shape[0]  # number of states
        self.M = self.B.shape[1]  # number of observation symbols

    def forward(self, observations):
        T = len(observations)
        alpha = np.zeros((T, self.N))

        # Initialization
        alpha[0] = self.pi * self.B[:, observations[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, observations[t]]

        return alpha

    def backward(self, observations):
        T = len(observations)
        beta = np.zeros((T, self.N))

        # Initialization
        beta[T-1] = np.ones(self.N)

        # Recursion
        for t in reversed(range(T-1)):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i] * self.B[:, observations[t+1]] * beta[t+1])

        return beta