/* MKL: C(M x 512) = A(M x 512) * B(512 x 512)^T, 256- and 128-column panels vs one GEMM, for several M, lda = 512 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
int main(void) {
    int n = 512, Ms[] = {1, 7, 64, 100, 128, 200, 256, 300, 384, 500, 511, 512}, bad = 0;
    size_t T = (size_t)n * n;
    float *A = mkl_malloc(T * 4, 64), *B = mkl_malloc(T * 4, 64), *W = mkl_malloc(T * 4, 64), *P = mkl_malloc(T * 4, 64);
    for (int s = 0; s < 3; s++) {
        unsigned sd = 99 + s;
        for (size_t i = 0; i < T; i++) { A[i] = (float)rand_r(&sd) / RAND_MAX - 0.5f; B[i] = (float)rand_r(&sd) / RAND_MAX - 0.5f; }
        for (unsigned mi = 0; mi < sizeof Ms / sizeof *Ms; mi++) for (int w = 128; w <= 256; w += 128) {
            int M = Ms[mi];
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, n, n, 1.0f, A, n, B, n, 0.0f, W, n);
            for (int j0 = 0; j0 < n; j0 += w)
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, w, n, 1.0f, A, n, B + j0, n, 0.0f, P + (size_t)j0 * n, n);
            int d = 0;
            for (int j = 0; j < n; j++) d |= memcmp(P + (size_t)j * n, W + (size_t)j * n, (size_t)M * 4) != 0;
            if (d) { bad++; printf("seed %d M %d w %d: DIFFER\n", s, M, w); }
        }
    }
    printf("M sweep (1..512), panels 128/256: %s\n", bad ? "DIFFERENCES" : "all identical");
    return 0;
}
