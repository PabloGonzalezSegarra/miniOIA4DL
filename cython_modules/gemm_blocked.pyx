# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

# Inicio bloque generado por la IA

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


    # Recorremos las matrices en bloques, primero por columnas de C/B (NC), luego por la parte común de A/B (KC), y finalmente por filas de C/A (MC)
    # Multiplacndo des esta forma, conseguimos que el bloque de B se mantenga en caché mientras se multiplica con diferentes bloques de A maximizando la reutilización de caché.
    for jc in range(0, N, NC):
        for pc in range(0, K, KC):
            for ic in range(0, M, MC):
                C[ic:ic+MC, jc:jc+NC] += (
                    A[ic:ic+MC, pc:pc+KC] @
                    B[pc:pc+KC, jc:jc+NC]
                )

    return C

# Fin bloque generado por la IA