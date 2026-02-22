import numpy as np

def baum_welch(model, observations, max_iter=100, tol=1e-6):
    """
    Baum-Welch algorithm for training HMM parameters
    """
    N = model.N
    M = model.M
    T = len(observations)
    
    likelihoods = []
    
    for iteration in range(max_iter):
        # Forward & Backward
        alpha = model.forward(observations)
        beta = model.backward(observations)
        
        # Compute likelihood P(O | lambda)
        likelihood = np.sum(alpha[-1])
        likelihoods.append(likelihood)
        
        # Convergence check
        if iteration > 0 and abs(likelihoods[-1] - likelihoods[-2]) < tol:
            # Remove or comment out this print
            # print(f"Converged at iteration {iteration + 1}")
            break
        
        # Compute gamma (probability of being in state i at time t)
        gamma = np.zeros((T, N))
        for t in range(T):
            denom = np.sum(alpha[t] * beta[t])
            if denom > 0:
                gamma[t] = (alpha[t] * beta[t]) / denom
        
        # Compute xi (probability of being in state i at time t and state j at time t+1)
        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            denom = np.sum(alpha[t][:, None] * model.A * model.B[:, observations[t+1]] * beta[t+1])
            if denom > 0:  # Avoid division by zero
                for i in range(N):
                    numer = alpha[t, i] * model.A[i, :] * model.B[:, observations[t+1]] * beta[t+1]
                    xi[t, i, :] = numer / denom
        
        # Update pi
        model.pi = gamma[0]
        
        # Update A (transition matrix)
        for i in range(N):
            denom = np.sum(gamma[:-1, i])
            if denom > 0:
                for j in range(N):
                    model.A[i, j] = np.sum(xi[:, i, j]) / denom
        
        # Update B (emission matrix)
        for i in range(N):
            denom = np.sum(gamma[:, i])
            if denom > 0:
                for k in range(M):
                    mask = (np.array(observations) == k)
                    model.B[i, k] = np.sum(gamma[mask, i]) / denom
        
        # Remove or comment out this progress print
        # if (iteration + 1) % 10 == 0:
        #     print(f"Iteration {iteration + 1}: Likelihood = {likelihood:.6f}")
    
    return model, likelihoods