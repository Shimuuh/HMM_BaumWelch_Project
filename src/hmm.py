import numpy as np


class HiddenMarkovModel:

    def __init__(self, N, M):
        """
        N = number of hidden states
        M = number of observation symbols
        """

        self.N = N
        self.M = M

        # Random initialization (normalized)
        self.A = np.random.rand(N, N)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        self.B = np.random.rand(N, M)
        self.B = self.B / self.B.sum(axis=1, keepdims=True)

        self.pi = np.random.rand(N)
        self.pi = self.pi / self.pi.sum()


    def forward(self, observations):
        """
        Forward algorithm
        Returns alpha matrix
        """

        T = len(observations)
        alpha = np.zeros((T, self.N))

        # Initialization
        alpha[0] = self.pi * self.B[:, observations[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(
                    alpha[t - 1] * self.A[:, j]
                ) * self.B[j, observations[t]]

        return alpha


    def backward(self, observations):
        """
        Backward algorithm
        Returns beta matrix
        """

        T = len(observations)
        beta = np.zeros((T, self.N))

        # Initialization
        beta[T - 1] = 1

        # Recursion
        for t in reversed(range(T - 1)):
            for i in range(self.N):
                beta[t, i] = np.sum(
                    self.A[i] *
                    self.B[:, observations[t + 1]] *
                    beta[t + 1]
                )

        return beta