#include <stdlib.h>
#include <string.h>

// Macros para facilitar el acceso a los elementos de las matrices
// (Matrices almacenadas por filas (row-major), lda = número de columnas)
#define Aref(i,j) A[(i)*lda_A + (j)]
#define Bref(i,j) B[(i)*lda_B + (j)]
#define Cref(i,j) C[(i)*lda_C + (j)]

/**
 * @param A         Matriz A de tamaño MxK
 * @param lda_A     Numero de columnas de A (K)
 * @param B         Matriz B de tamaño KxN
 * @param lda_B     Numero de columnas de B (N)
 * @param C         Matriz resultado C de tamaño MxN
 * @param lda_C     Numero de columnas de C (N)
 * @param M         Número de filas de A y C
 * @param K         Número de columnas de A y filas de B
 * @param N         Número de columnas de B y C
 * @param MC        Tamaño del bloque de filas de A y C.
 * @param KC        Tamaño del bloque de columnas de A y filas de B
 * @param NC        Tamaño del bloque de columnas de B y C
 */
void gemm_omp( float *A, int lda_A, float *B, int lda_B,
    float *C, int lda_C, int M, int K, int N,
    int MC, int KC, int NC
) {

    // Definimos las variables de iteración
    int ic, pc, jc, i, p, j;

    // Inicio bloque asistido por la IA, muy util porque menudo cacao

    #pragma omp parallel for private(ic, pc, jc, i, p, j) schedule(static)
    // Bloque externo: recorre las columnas de C en bloques de tamaño NC
    for (jc = 0; jc < N; jc += NC) {
        // Calculamos el tamaño real del bloque de columnas (puede ser menor que NC en el borde)
        int nc = (jc + NC > N) ? N - jc : NC;
        // Bloque medio: recorre las columnas de A y filas de B en bloques de tamaño KC
        for (pc = 0; pc < K; pc += KC) {
            // Calculamos el tamaño real del bloque de columnas de A y filas de B
            int kc = (pc + KC > K) ? K - pc : KC;
            // Bloque interno: recorre las filas de A y C en bloques de tamaño MC
            for (ic = 0; ic < M; ic += MC) {
                // Calculamos el tamaño real del bloque de filas de A y C
                int mc = (ic + MC > M) ? M - ic : MC;
                // micro-bloque: C[ic:ic+mc, jc:jc+nc] += A[ic:ic+mc, pc:pc+kc] @ B[pc:pc+kc, jc:jc+nc]
                for (i = ic; i < ic + mc; i++) {
                    for (p = pc; p < pc + kc; p++) {
                        #pragma ivdep
                        for (j = jc; j < jc + nc; j++) {
                            Cref(i,j) += Aref(i,p) * Bref(p,j);
                        }
                    }
                }
            }
        }
    }

    // Fin bloque asistido por la IA

}