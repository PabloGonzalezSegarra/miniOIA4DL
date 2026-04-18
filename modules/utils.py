import numpy as np

#PISTA: es esta la mejor forma de hacer una matmul?
def matmul_biasses(A, B, C, bias):
    m, p, n = A.shape[0], A.shape[1], B.shape[1]
    
    # for i in range(m):
    #     for j in range(n):
    #         for k in range(p):
    #             C[i][j] += A[i][k] * B[k][j]
    #         C[i][j] += bias[j]

    # Usamos numpy para hacer la multiplicación de matrices. Muy simple.
    C = np.dot(A, B) + bias

    return C

