# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

def gemm_blocked(
    cnp.ndarray[cnp.float32_t, ndim=2] A,
    cnp.ndarray[cnp.float32_t, ndim=2] B,
    cnp.ndarray[cnp.float32_t, ndim=2] C,
    int MC, int NC, int KC
):
    cdef int M = A.shape[0]
    cdef int K = A.shape[1]
    cdef int N = B.shape[1]
    cdef int ic, jc, pc

    # Inicio bloque asistido por la IA

    for jc in range(0, N, NC):
        for pc in range(0, K, KC):
            for ic in range(0, M, MC):
                C[ic:ic+MC, jc:jc+NC] += (
                    A[ic:ic+MC, pc:pc+KC] @
                    B[pc:pc+KC, jc:jc+NC]
                )

    # Fin bloque asistido
    return C