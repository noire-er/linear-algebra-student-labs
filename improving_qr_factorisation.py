import numpy as np
import pandas as pd

def classical_gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

def compute_errors(A, Q, R):
    error1 = np.linalg.norm(A - Q @ R, 2)
    error2 = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), 2)
    error3 = np.linalg.norm(R - np.triu(R), 2)
    return error1, error2, error3


eps_values = [10**(-k) for k in range(6, 17)]

results = []

for eps in eps_values:
    A = np.array([
        [1, 1 + eps],
        [1 + eps, 1]
    ], dtype=float)

    Q, R = classical_gram_schmidt(A)
    e1, e2, e3 = compute_errors(A, Q, R)

    results.append([eps, e1, e2, e3])


df = pd.DataFrame(
    results,
    columns=[r"$\epsilon$", "error1 = ||A-QR||2", "error2 = ||QᵀQ - I||2", "error3 = ||R - triu(R)||2"]   
)

print(df)