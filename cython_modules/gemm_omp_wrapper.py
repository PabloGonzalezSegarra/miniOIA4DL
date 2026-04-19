import ctypes
import numpy as np
import os

# Inicio bloque generado por la IA 
# Cargar la biblioteca
_lib_path = os.path.join(os.path.dirname(__file__), "gemm_omp.so")
_lib = ctypes.CDLL(_lib_path)
# Definir tipos de argumentos
_lib.gemm_omp.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # A, lda_A
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # B, lda_B
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # C, lda_C
    ctypes.c_int, ctypes.c_int, ctypes.c_int,      # M, K, N
    ctypes.c_int, ctypes.c_int, ctypes.c_int,      # MC, KC, NC
]
_lib.gemm_omp.restype = None
# Fin bloque generado por la IA

def gemm_omp(A, B, C, MC, KC, NC):
    # Obtenemos las dimensiones de las matrices
    M, K = A.shape
    K2, N = B.shape

    # Comprobamos que las dimensiones son compatibles para la multiplicación de matrices
    assert K == K2

    # Convertimos las matrices en arrays contiguos en memoria de tipo f32
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    C = np.ascontiguousarray(C, dtype=np.float32)

    # Llamamos a la funcion definida en gemm_omp.c
    _lib.gemm_omp(
        # Inicio bloque asistido por la IA
        # Pasamos los punteros a las matrices y sus dimensiones
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), K,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
        # Fin bloque asistido por la IA
        M, K, N, MC, KC, NC
    )

    return C