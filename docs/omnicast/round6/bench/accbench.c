/* Microbenchmark: C(FP64) -= (double) SGEMM(A, B^T) as in the kind-10 runtime CPU GEMM (ACC_DP=1).
 * Variants (per GEMM, NB x NB tiles, T threads each with its own tiles, C tiles cycle through a pool
 * larger than the LLC so C is cold as in the factorization):
 *   ref     : full SGEMM into W (beta 0), then the scalar loop as in the source (gcc -O3 -> SSE2)
 *   avx     : full SGEMM, then an AVX loop (vcvtps2pd ymm + vsubpd)
 *   pN      : SGEMM in column panels of N columns into a panel buffer, each panel accumulated right away (AVX)
 *   pfN     : like avx, but with software prefetch of C N bytes ahead
 * Bit check: panel SGEMM outputs vs the full SGEMM output, bitwise. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <mkl.h>
#include <immintrin.h>
#include <x86intrin.h>

static int NB = 512;

static void loop_ref(double *C, const float *W, int mb, int nb) {
    for (int j = 0; j < nb; j++)
        for (int i = 0; i < mb; i++)
            C[j*mb+i] -= W[j*mb+i];
}

__attribute__((target("avx")))
static void loop_avx(double *C, const float *W, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 w0 = _mm256_loadu_ps(W + i), w1 = _mm256_loadu_ps(W + i + 8);
        __m256d c0 = _mm256_loadu_pd(C + i),      c1 = _mm256_loadu_pd(C + i + 4);
        __m256d c2 = _mm256_loadu_pd(C + i + 8),  c3 = _mm256_loadu_pd(C + i + 12);
        c0 = _mm256_sub_pd(c0, _mm256_cvtps_pd(_mm256_castps256_ps128(w0)));
        c1 = _mm256_sub_pd(c1, _mm256_cvtps_pd(_mm256_extractf128_ps(w0, 1)));
        c2 = _mm256_sub_pd(c2, _mm256_cvtps_pd(_mm256_castps256_ps128(w1)));
        c3 = _mm256_sub_pd(c3, _mm256_cvtps_pd(_mm256_extractf128_ps(w1, 1)));
        _mm256_storeu_pd(C + i, c0);     _mm256_storeu_pd(C + i + 4, c1);
        _mm256_storeu_pd(C + i + 8, c2); _mm256_storeu_pd(C + i + 12, c3);
    }
    for (; i < n; i++) C[i] -= W[i];
}

__attribute__((target("avx")))
static void loop_avx_pf(double *C, const float *W, size_t n, size_t dist) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm_prefetch((const char *)(C + i) + dist, _MM_HINT_T0);
        _mm_prefetch((const char *)(C + i) + dist + 64, _MM_HINT_T0);
        __m256 w0 = _mm256_loadu_ps(W + i), w1 = _mm256_loadu_ps(W + i + 8);
        __m256d c0 = _mm256_loadu_pd(C + i),      c1 = _mm256_loadu_pd(C + i + 4);
        __m256d c2 = _mm256_loadu_pd(C + i + 8),  c3 = _mm256_loadu_pd(C + i + 12);
        c0 = _mm256_sub_pd(c0, _mm256_cvtps_pd(_mm256_castps256_ps128(w0)));
        c1 = _mm256_sub_pd(c1, _mm256_cvtps_pd(_mm256_extractf128_ps(w0, 1)));
        c2 = _mm256_sub_pd(c2, _mm256_cvtps_pd(_mm256_castps256_ps128(w1)));
        c3 = _mm256_sub_pd(c3, _mm256_cvtps_pd(_mm256_extractf128_ps(w1, 1)));
        _mm256_storeu_pd(C + i, c0);     _mm256_storeu_pd(C + i + 4, c1);
        _mm256_storeu_pd(C + i + 8, c2); _mm256_storeu_pd(C + i + 12, c3);
    }
    for (; i < n; i++) C[i] -= W[i];
}


/* AVX loop, C stored with non-temporal stores (C is not reused soon); peel to 32-byte alignment of C */
__attribute__((target("avx")))
static void loop_avx_nt(double *C, const float *W, size_t n, int pfnta) {
    size_t i = 0;
    while (i < n && ((uintptr_t)(C + i) & 31)) { C[i] -= W[i]; i++; }
    for (; i + 16 <= n; i += 16) {
        if (pfnta) { _mm_prefetch((const char *)(C + i) + pfnta, _MM_HINT_NTA); _mm_prefetch((const char *)(C + i) + pfnta + 64, _MM_HINT_NTA); }
        __m256 w0 = _mm256_loadu_ps(W + i), w1 = _mm256_loadu_ps(W + i + 8);
        __m256d c0 = _mm256_load_pd(C + i),      c1 = _mm256_load_pd(C + i + 4);
        __m256d c2 = _mm256_load_pd(C + i + 8),  c3 = _mm256_load_pd(C + i + 12);
        c0 = _mm256_sub_pd(c0, _mm256_cvtps_pd(_mm256_castps256_ps128(w0)));
        c1 = _mm256_sub_pd(c1, _mm256_cvtps_pd(_mm256_extractf128_ps(w0, 1)));
        c2 = _mm256_sub_pd(c2, _mm256_cvtps_pd(_mm256_castps256_ps128(w1)));
        c3 = _mm256_sub_pd(c3, _mm256_cvtps_pd(_mm256_extractf128_ps(w1, 1)));
        _mm256_stream_pd(C + i, c0);     _mm256_stream_pd(C + i + 4, c1);
        _mm256_stream_pd(C + i + 8, c2); _mm256_stream_pd(C + i + 12, c3);
    }
    for (; i < n; i++) C[i] -= W[i];
    _mm_sfence();
}
/* AVX loop with prefetchnta of C (normal stores) */
__attribute__((target("avx")))
static void loop_avx_nta(double *C, const float *W, size_t n, size_t dist) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm_prefetch((const char *)(C + i) + dist, _MM_HINT_NTA);
        _mm_prefetch((const char *)(C + i) + dist + 64, _MM_HINT_NTA);
        __m256 w0 = _mm256_loadu_ps(W + i), w1 = _mm256_loadu_ps(W + i + 8);
        __m256d c0 = _mm256_loadu_pd(C + i),      c1 = _mm256_loadu_pd(C + i + 4);
        __m256d c2 = _mm256_loadu_pd(C + i + 8),  c3 = _mm256_loadu_pd(C + i + 12);
        c0 = _mm256_sub_pd(c0, _mm256_cvtps_pd(_mm256_castps256_ps128(w0)));
        c1 = _mm256_sub_pd(c1, _mm256_cvtps_pd(_mm256_extractf128_ps(w0, 1)));
        c2 = _mm256_sub_pd(c2, _mm256_cvtps_pd(_mm256_castps256_ps128(w1)));
        c3 = _mm256_sub_pd(c3, _mm256_cvtps_pd(_mm256_extractf128_ps(w1, 1)));
        _mm256_storeu_pd(C + i, c0);     _mm256_storeu_pd(C + i + 4, c1);
        _mm256_storeu_pd(C + i + 8, c2); _mm256_storeu_pd(C + i + 12, c3);
    }
    for (; i < n; i++) C[i] -= W[i];
}

static void sg(int m, int n, int k, const float *A, int lda, const float *B, int ldb, float *W, int ldw) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, A, lda, B, ldb, 0.0f, W, ldw);
}

static void fill(float *x, size_t n, unsigned *s, float scale) {
    for (size_t i = 0; i < n; i++) x[i] = scale * ((float)rand_r(s) / RAND_MAX - 0.5f);
}

/* bit check: panel (and row-panel) SGEMMs vs the full SGEMM */
static int bitcheck(int seeds) {
    int bad = 0, ws[] = {8, 16, 32, 48, 64, 96, 128, 256};
    size_t T = (size_t)NB * NB;
    float *A = mkl_malloc(T * 4, 64), *B = mkl_malloc(T * 4, 64), *W = mkl_malloc(T * 4, 64), *P = mkl_malloc(T * 4, 64);
    for (int s = 0; s < seeds; s++) {
        unsigned sd = 1234 + s;
        float sc = s % 3 == 0 ? 1.0f : (s % 3 == 1 ? 1e-3f : 37.0f);
        fill(A, T, &sd, sc); fill(B, T, &sd, sc);
        sg(NB, NB, NB, A, NB, B, NB, W, NB);
        for (unsigned wi = 0; wi < sizeof ws / sizeof *ws; wi++) {
            int w = ws[wi];
            /* column panels written into a panel buffer of ld NB (as the implementation would) */
            int diff = 0;
            for (int j0 = 0; j0 < NB; j0 += w) {
                int ww = NB - j0 < w ? NB - j0 : w;
                sg(NB, ww, NB, A, NB, B + j0, NB, P, NB);
                if (memcmp(P, W + (size_t)j0 * NB, (size_t)ww * NB * 4)) diff++;
            }
            /* row panels */
            int rdiff = 0;
            for (int i0 = 0; i0 < NB; i0 += w) {
                int hh = NB - i0 < w ? NB - i0 : w;
                sg(hh, NB, NB, A + i0, NB, B, NB, P + i0, NB);
            }
            if (memcmp(P, W, T * 4)) rdiff = 1;
            if (diff || rdiff) bad++;
            printf("seed %d scale %g w %3d: column panels %s (%d differ), row panels %s\n", s, sc, w,
                   diff ? "DIFFER" : "identical", diff, rdiff ? "DIFFER" : "identical");
        }
    }
    mkl_free(A); mkl_free(B); mkl_free(W); mkl_free(P);
    return bad;
}


/* MKL packed API: pack A and/or B once, compare bits with cblas_sgemm */
static int packcheck(int seeds) {
    int bad = 0, n = NB;
    size_t T = (size_t)n * n;
    float *A = mkl_malloc(T * 4, 64), *B = mkl_malloc(T * 4, 64), *W = mkl_malloc(T * 4, 64), *P = mkl_malloc(T * 4, 64);
    size_t sa = cblas_sgemm_pack_get_size(CblasAMatrix, n, n, n), sb = cblas_sgemm_pack_get_size(CblasBMatrix, n, n, n);
    float *Ap = mkl_malloc(sa, 64), *Bp = mkl_malloc(sb, 64);
    printf("packed sizes: A %zu B %zu bytes (plain %zu)\n", sa, sb, T * 4);
    for (int s = 0; s < seeds; s++) {
        unsigned sd = 777 + s;
        fill(A, T, &sd, 1.0f); fill(B, T, &sd, 1.0f);
        sg(n, n, n, A, n, B, n, W, n);
        cblas_sgemm_pack(CblasColMajor, CblasAMatrix, CblasNoTrans, n, n, n, 1.0f, A, n, Ap);
        cblas_sgemm_pack(CblasColMajor, CblasBMatrix, CblasTrans, n, n, n, 1.0f, B, n, Bp);
        cblas_sgemm_compute(CblasColMajor, CblasPacked, CblasTrans, n, n, n, Ap, n, B, n, 0.0f, P, n);
        int da = memcmp(P, W, T * 4) != 0;
        cblas_sgemm_compute(CblasColMajor, CblasNoTrans, CblasPacked, n, n, n, A, n, Bp, n, 0.0f, P, n);
        int db = memcmp(P, W, T * 4) != 0;
        cblas_sgemm_compute(CblasColMajor, CblasPacked, CblasPacked, n, n, n, Ap, n, Bp, n, 0.0f, P, n);
        int dab = memcmp(P, W, T * 4) != 0;
        printf("seed %d: packed A %s, packed B %s, both %s\n", s, da ? "DIFFER" : "identical", db ? "DIFFER" : "identical", dab ? "DIFFER" : "identical");
        bad += da + db + dab;
    }
    return bad;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "bench";
    _mm_setcsr(_mm_getcsr() | 0x8040);   /* FTZ | DAZ as --flush_subnormals */
    if (getenv("NB")) NB = atoi(getenv("NB"));
    if (!strcmp(mode, "packcheck")) { int b = packcheck(argc > 2 ? atoi(argv[2]) : 4); printf("packcheck: %s\n", b ? "DIFFERENCES" : "all identical"); return b != 0; }
    if (!strcmp(mode, "bitcheck")) {
        int bad = bitcheck(argc > 2 ? atoi(argv[2]) : 6);
        printf("bitcheck: %s (mkl cbwr branch %d)\n", bad ? "DIFFERENCES" : "all identical", mkl_cbwr_get_auto_branch());
        return bad != 0;
    }
    /* bench <variant> <iters> <pool>   variants: ref avx p<w> pf<dist> sgemm loopref loopavx */
    const char *var = argc > 2 ? argv[2] : "ref";
    int iters = argc > 3 ? atoi(argv[3]) : 40, pool = argc > 4 ? atoi(argv[4]) : 6;
    int nthr = omp_get_max_threads();
    size_t T = (size_t)NB * NB;
    double tot_gemm = 0, tot_loop = 0, tmax = 0;
    uint64_t csum = 0;
    #pragma omp parallel reduction(+:tot_gemm,tot_loop) reduction(max:tmax) reduction(^:csum)
    {
        _mm_setcsr(_mm_getcsr() | 0x8040);
        unsigned sd = 99 + omp_get_thread_num();
        float *A = mkl_malloc(T * 4, 64), *B = mkl_malloc(T * 4, 64), *W = mkl_malloc(T * 8, 64);
        double **C = malloc(pool * sizeof *C);
        fill(A, T, &sd, 1.0f); fill(B, T, &sd, 1.0f);
        for (int p = 0; p < pool; p++) { C[p] = (double *)((char *)mkl_malloc(T * 8 + 64, 64) + (getenv("COFF") ? atoi(getenv("COFF")) : 0)); for (size_t i = 0; i < T; i++) C[p][i] = (double)rand_r(&sd) / RAND_MAX; }
        memset(W, 0, T * 8);
        int w = 0; size_t dist = 0;
        if (var[0] == 'p' && var[1] == 'f') dist = atol(var + 2);
        else if (var[0] == 'p') w = atoi(var + 1);
        sg(NB, NB, NB, A, NB, B, NB, (float *)W, NB);  /* warm up MKL */
        float *Ap = NULL, *Bp = NULL;
        if (!strncmp(var, "pk", 2)) {
            Ap = mkl_malloc(cblas_sgemm_pack_get_size(CblasAMatrix, NB, NB, NB), 64);
            Bp = mkl_malloc(cblas_sgemm_pack_get_size(CblasBMatrix, NB, NB, NB), 64);
            cblas_sgemm_pack(CblasColMajor, CblasAMatrix, CblasNoTrans, NB, NB, NB, 1.0f, A, NB, Ap);
            cblas_sgemm_pack(CblasColMajor, CblasBMatrix, CblasTrans, NB, NB, NB, 1.0f, B, NB, Bp);
        }
        #pragma omp barrier
        double t0 = omp_get_wtime();
        for (int it = 0; it < iters; it++) {
            double *Ct = C[it % pool];
            double a = omp_get_wtime(), b, c;
            if (w) {
                for (int j0 = 0; j0 < NB; j0 += w) {
                    int ww = NB - j0 < w ? NB - j0 : w;
                    sg(NB, ww, NB, A, NB, B + j0, NB, (float *)W, NB);
                    if (strstr(var, "nt")) loop_avx_nt(Ct + (size_t)j0 * NB, (float *)W, (size_t)ww * NB, 0);
                    else loop_avx(Ct + (size_t)j0 * NB, (float *)W, (size_t)ww * NB);
                }
                b = c = omp_get_wtime();
                tot_gemm += b - a;
            } else {
                if (!strcmp(var, "pkA")) cblas_sgemm_compute(CblasColMajor, CblasPacked, CblasTrans, NB, NB, NB, Ap, NB, B, NB, 0.0f, (float *)W, NB);
                else if (!strcmp(var, "pkAB")) cblas_sgemm_compute(CblasColMajor, CblasPacked, CblasPacked, NB, NB, NB, Ap, NB, Bp, NB, 0.0f, (float *)W, NB);
                else if (strncmp(var, "loop", 4)) sg(NB, NB, NB, A, NB, B, NB, (float *)W, NB);
                b = omp_get_wtime();
                if (!strcmp(var, "ref") || !strcmp(var, "loopref") || !strncmp(var, "pk", 2)) loop_ref(Ct, (float *)W, NB, NB);
                else if (!strcmp(var, "avx") || !strcmp(var, "loopavx")) loop_avx(Ct, (float *)W, T);
                else if (!strcmp(var, "nt") || !strcmp(var, "loopnt")) loop_avx_nt(Ct, (float *)W, T, 0);
                else if (!strncmp(var, "ntpf", 4)) loop_avx_nt(Ct, (float *)W, T, atoi(var + 4));
                else if (!strncmp(var, "nta", 3)) loop_avx_nta(Ct, (float *)W, T, atoi(var + 3));
                else if (dist) loop_avx_pf(Ct, (float *)W, T, dist);
                c = omp_get_wtime();
                tot_gemm += b - a; tot_loop += c - b;
            }
        }
        double t1 = omp_get_wtime();
        tmax = t1 - t0;
        for (int p = 0; p < pool; p++) { uint64_t h = 0; for (size_t i = 0; i < T; i++) { uint64_t u; memcpy(&u, &C[p][i], 8); h = h * 31 + u; } csum ^= h; mkl_free((char *)C[p] - (getenv("COFF") ? atoi(getenv("COFF")) : 0)); }
        mkl_free(A); mkl_free(B); mkl_free(W); free(C);
    }
    printf("%-8s thr %2d NB %d iters %d pool %d: per GEMM %.3f ms (sgemm %.3f + loop %.3f) wall %.3f s  hash %016lx\n",
           var, nthr, NB, iters, pool, 1e3 * (tot_gemm + tot_loop) / (nthr * iters),
           1e3 * tot_gemm / (nthr * iters), 1e3 * tot_loop / (nthr * iters), tmax, (unsigned long)csum);
    return 0;
}
