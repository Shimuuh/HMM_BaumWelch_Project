import numpy as np

class HiddenMarkovModel:
    def __init__(self, N, M):
        """
        N = number of hidden states
        M = number of observation symbols
        """
        self.N = N
        self.M = M
        
        # Random initialization with better scaling
        # Transition matrix A
        self.A = np.random.rand(N, N)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        
        # Emission matrix B
        self.B = np.random.rand(N, M)
        self.B = self.B / self.B.sum(axis=1, keepdims=True)
        
        # Initial state distribution pi
        self.pi = np.random.rand(N)
        self.pi = self.pi / self.pi.sum()
    
    def forward(self, observations):
        """
        Forward algorithm - computes alpha probabilities
        Returns: alpha matrix of shape (T, N)
        """
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
        """
        Backward algorithm - computes beta probabilities
        Returns: beta matrix of shape (T, N)
        """
        T = len(observations)
        beta = np.zeros((T, self.N))
        
        # Initialization
        beta[T-1] = 1
        
        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1])
        
        return beta
    
    def viterbi(self, observations):
        """
        Viterbi algorithm for finding the most likely state sequence
        """
        T = len(observations)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        
        # Initialization
        delta[0] = self.pi * self.B[:, observations[0]]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.N):
                temp = delta[t-1] * self.A[:, j] * self.B[j, observations[t]]
                delta[t, j] = np.max(temp)
                psi[t, j] = np.argmax(temp)
        
        # Termination
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        
        # Backtracking
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def log_likelihood(self, observations):
        """Compute log likelihood of observations"""
        alpha = self.forward(observations)
        return np.log(np.sum(alpha[-1]))