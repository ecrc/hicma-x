// GEMM time with normal vs subnormal inputs (FP32 sgemm, FP16 hgemm with FP32 compute), C = A * B^T, n x n
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#if defined(USE_HIP)
#define HIPBLAS_V2
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#define gpu(x) hip##x
#define BL(x) hipblas##x
typedef hipblasHandle_t blasH;
#define OP_N HIPBLAS_OP_N
#define OP_T HIPBLAS_OP_T
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#define gpu(x) cuda##x
#define BL(x) cublas##x
typedef cublasHandle_t blasH;
#define OP_N CUBLAS_OP_N
#define OP_T CUBLAS_OP_T
#endif
static uint16_t f2h(float f) { // round-to-nearest-even FP32 -> FP16 bits (no NaN handling needed here)
    uint32_t x; memcpy(&x, &f, 4); uint32_t s = (x >> 16) & 0x8000; float a = std::fabs(f);
    if (a >= 65504.f) return s | 0x7BFF;
    if (a < 6.103515625e-05f) { float q = a / 5.9604644775390625e-08f; uint32_t m = (uint32_t)std::nearbyint(q); return s | m; }
    int e; float fr = std::frexp(a, &e); uint32_t m = (uint32_t)std::nearbyint((fr * 2 - 1) * 1024); if (m == 1024) { m = 0; e++; }
    return s | ((uint32_t)(e + 14) << 10) | m;
}
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2048, reps = 20;
    std::mt19937 g(1); std::uniform_real_distribution<float> U(-1, 1);
    blasH h; BL(Create)(&h);
    for (int sub = 0; sub < 2; sub++) {
        // sgemm: normal entries O(1) or subnormal entries ~1e-39 (FP32 subnormal), products underflow too
        std::vector<float> A((size_t)n * n), B((size_t)n * n);
        for (auto &v : A) v = sub ? U(g) * 1e-39f : U(g);
        for (auto &v : B) v = sub ? U(g) * 1e-39f : U(g);
        float *dA, *dB, *dC; gpu(Malloc)(&dA, A.size() * 4); gpu(Malloc)(&dB, B.size() * 4); gpu(Malloc)(&dC, A.size() * 4);
        gpu(Memcpy)(dA, A.data(), A.size() * 4, gpu(MemcpyHostToDevice)); gpu(Memcpy)(dB, B.data(), B.size() * 4, gpu(MemcpyHostToDevice));
        gpu(Memset)(dC, 0, A.size() * 4);
        float al = -1, be = 1;
        BL(Sgemm)(h, OP_N, OP_T, n, n, n, &al, dA, n, dB, n, &be, dC, n); gpu(DeviceSynchronize)();
        gpu(Event_t) e0, e1; gpu(EventCreate)(&e0); gpu(EventCreate)(&e1); gpu(EventRecord)(e0, 0);
        for (int r = 0; r < reps; r++) BL(Sgemm)(h, OP_N, OP_T, n, n, n, &al, dA, n, dB, n, &be, dC, n);
        gpu(EventRecord)(e1, 0); gpu(EventSynchronize)(e1); float ms; gpu(EventElapsedTime)(&ms, e0, e1);
        printf("sgemm n=%d %-10s: %7.3f ms  %6.2f Tflop/s\n", n, sub ? "subnormal" : "normal", ms / reps, 2.0 * n * n * n / (ms / reps * 1e-3) / 1e12);
        // hgemm-style: FP16 A, B (values ~1e-6: FP16 subnormal) with FP32 accumulation via GemmEx
        std::vector<uint16_t> Ah(A.size()), Bh(B.size());
        for (size_t i = 0; i < A.size(); i++) { Ah[i] = f2h(sub ? U(g) * 3e-6f : U(g)); Bh[i] = f2h(sub ? U(g) * 3e-6f : U(g)); }
        void *dAh, *dBh; gpu(Malloc)(&dAh, Ah.size() * 2); gpu(Malloc)(&dBh, Bh.size() * 2);
        gpu(Memcpy)(dAh, Ah.data(), Ah.size() * 2, gpu(MemcpyHostToDevice)); gpu(Memcpy)(dBh, Bh.data(), Bh.size() * 2, gpu(MemcpyHostToDevice));
#if defined(USE_HIP)
        auto ex = [&]() { return hipblasGemmEx(h, OP_N, OP_T, n, n, n, &al, dAh, HIP_R_16F, n, dBh, HIP_R_16F, n, &be, dC, HIP_R_32F, n, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT); };
#else
        auto ex = [&]() { return cublasGemmEx(h, OP_N, OP_T, n, n, n, &al, dAh, CUDA_R_16F, n, dBh, CUDA_R_16F, n, &be, dC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); };
#endif
        ex(); gpu(DeviceSynchronize)(); gpu(EventRecord)(e0, 0);
        for (int r = 0; r < reps; r++) ex();
        gpu(EventRecord)(e1, 0); gpu(EventSynchronize)(e1); gpu(EventElapsedTime)(&ms, e0, e1);
        printf("fp16 gemm (fp32 acc) n=%d %-10s: %7.3f ms  %6.2f Tflop/s\n", n, sub ? "subnormal" : "normal", ms / reps, 2.0 * n * n * n / (ms / reps * 1e-3) / 1e12);
        gpu(Free)(dA); gpu(Free)(dB); gpu(Free)(dC); gpu(Free)(dAh); gpu(Free)(dBh);
    }
    return 0;
}
