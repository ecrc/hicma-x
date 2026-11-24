/* Column-panel SGEMM vs one SGEMM over the tile, bitwise, with any CBLAS (C = A * B^T, beta 0) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 512, ws[] = {32, 64, 128, 256}, bad = 0;
    size_t T = (size_t)n * n;
    float *A = malloc(T * 4), *B = malloc(T * 4), *W = malloc(T * 4), *P = malloc(T * 4);
    for (int s = 0; s < 3; s++) {
        unsigned sd = 4321 + s;
        for (size_t i = 0; i < T; i++) { A[i] = (float)rand_r(&sd) / RAND_MAX - 0.5f; B[i] = (float)rand_r(&sd) / RAND_MAX - 0.5f; }
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0f, A, n, B, n, 0.0f, W, n);
        for (int wi = 0; wi < 4; wi++) {
            int w = ws[wi], diff = 0;
            for (int j0 = 0; j0 < n; j0 += w) {
                int ww = n - j0 < w ? n - j0 : w;
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, ww, n, 1.0f, A, n, B + j0, n, 0.0f, P + (size_t)j0 * n, n);
            }
            diff = memcmp(P, W, T * 4) != 0;
            bad += diff && w >= 128;
            printf("seed %d w %3d: %s\n", s, w, diff ? "DIFFER" : "identical");
        }
    }
    printf("panels of 128/256 columns: %s\n", bad ? "DIFFERENCES" : "identical");
    return 0;
}
