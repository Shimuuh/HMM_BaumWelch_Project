import numpy as np

def baum_welch(model, observations, max_iter=50, tol=1e-4):
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
            break

        xi = np.zeros((T-1, N, N))
        gamma = np.zeros((T, N))

        # Compute xi and gamma
        for t in range(T-1):
            denom = np.sum(
                alpha[t][:, None] *
                model.A *
                model.B[:, observations[t+1]] *
                beta[t+1]
            )

            for i in range(N):
                numer = (
                    alpha[t, i] *
                    model.A[i] *
                    model.B[:, observations[t+1]] *
                    beta[t+1]
                )
                xi[t, i] = numer / denom

        gamma[:-1] = np.sum(xi, axis=2)
        gamma[-1] = gamma[-2]  # last gamma approximation

        # Update pi
        model.pi = gamma[0]

        # Update A
        for i in range(N):
            for j in range(N):
                model.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

        # Update B
        for i in range(N):
            for k in range(M):
                mask = (np.array(observations) == k)
                model.B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])

    return model, likelihoods