import numpy as np

def baum_welch(model, observations, max_iter=50):
    N = model.N
    M = model.M
    T = len(observations)

    for _ in range(max_iter):
        alpha = model.forward(observations)
        beta = model.backward(observations)

        xi = np.zeros((T-1, N, N))
        gamma = np.zeros((T, N))

        for t in range(T-1):
            denom = np.sum(alpha[t] @ model.A * model.B[:, observations[t+1]] * beta[t+1])
            for i in range(N):
                numer = alpha[t, i] * model.A[i] * model.B[:, observations[t+1]] * beta[t+1]
                xi[t, i] = numer / denom

        gamma = np.sum(xi, axis=2)
        gamma = np.vstack((gamma, np.sum(xi[T-2], axis=0)))

        model.pi = gamma[0]

        for i in range(N):
            for j in range(N):
                model.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

        for i in range(N):
            for k in range(M):
                mask = (np.array(observations) == k)
                model.B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])

    return model