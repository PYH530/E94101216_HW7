import numpy as np
from scipy.sparse.linalg import cg

# 系統矩陣 A 與向量 b
A = np.array([
    [ 4, -1,  0, -1,  0,  0],
    [-1,  4, -1,  0, -1,  0],
    [ 0, -1,  4,  0,  1, -1],
    [-1,  0,  0,  4, -1, -1],
    [ 0, -1,  0, -1,  4, -1],
    [ 0,  0, -1,  0, -1,  4]
])
b = np.array([0, -1, 9, 4, 8, 6])

# Jacobi method
def jacobi(A, b, x0=None, tol=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros_like(b, dtype=float) if x0 is None else x0.astype(float)
    D = np.diag(np.diag(A))
    R = A - D
    for _ in range(max_iterations):
        x_new = np.linalg.inv(D) @ (b - R @ x)
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# Gauss-Seidel method
def gauss_seidel(A, b, x0=None, tol=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros_like(b, dtype=float) if x0 is None else x0.astype(float)
    for _ in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# SOR method
def sor(A, b, omega=1.25, x0=None, tol=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros_like(b, dtype=float) if x0 is None else x0.astype(float)
    for _ in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# Conjugate Gradient method
def conjugate_gradient(A, b):
    x, info = cg(A, b)
    return x

# Solve the system
x_jacobi = jacobi(A, b)
x_gs = gauss_seidel(A, b)
x_sor = sor(A, b)
x_cg = conjugate_gradient(A, b)

# Print results
print("Jacobi solution:         ", x_jacobi)
print("Gauss-Seidel solution:   ", x_gs)
print("SOR solution:            ", x_sor)
print("Conjugate Gradient solution:", x_cg)
