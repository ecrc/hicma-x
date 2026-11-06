#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "hicma_parsec_hip_cuda.h"

#define CHUNKSIZE 32

/*
 * Stochastic rounding (SR), HIP version of hicma_parsec_stochastic_rounding.cu:
 * x is rounded to one of its two neighbours in the target format,
 * lower <= |x| <= upper, choosing lower with probability
 * (upper - |x|) / (upper - lower), so that E[SR(x)] = x.  Rounding is done on
 * |x| and the sign put back.  The uniform random number of element (i, j) is
 * a hash of (i, j) and of the bits of the value being rounded, the same as on
 * the CPU and in CUDA.
 */

__device__ __forceinline__ uint64_t hicma_sr_splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/* Uniform in [0, 1), exactly representable as float, from the position
 * (i, j) and the bits of the value to round */
__device__ __forceinline__ float hicma_sr_u01(int i, int j, uint64_t bits)
{
    uint64_t pos = (uint64_t)(uint32_t)i | ((uint64_t)(uint32_t)j << 32);
    return (float)(hicma_sr_splitmix64(bits ^ hicma_sr_splitmix64(pos)) >> 40) * (1.0f / 16777216.0f);
}

__device__ float stochastic_rounding_fp32(double value, float random)
{
    float nearest = (float)value;
    if (!isfinite(value) || !isfinite(nearest)) {
        return nearest;
    }

    double a = fabs(value);
    float n = fabsf(nearest);
    float lower, upper;
    if (a > (double)n) {
        lower = n;
        upper = nextafterf(n, INFINITY);
    } else if (a < (double)n) {
        lower = nextafterf(n, 0.0f);
        upper = n;
    } else {
        return nearest;
    }
    if (isinf(upper)) {
        return nearest;
    }

    double p_lower = ((double)upper - a) / ((double)upper - (double)lower);
    float r = (random < p_lower) ? lower : upper;
    return (value < 0.0) ? -r : r;
}

__device__ __half stochastic_rounding_fp16(double value, float random)
{
    if (isnan(value)) {
        return __float2half((float)value);
    }

    double a = fabs(value);
    /* lower: largest half <= a.  (float)a may round up past a half value,
     * so step back one if the truncated value ended above a. */
    __half lower = __float2half_rz((float)a);
    if ((double)__half2float(lower) > a) {
        lower = __ushort_as_half(__half_as_ushort(lower) - 1);
    }
    double lv = (double)__half2float(lower);
    if (lv == a) {
        return (value < 0.0) ? __hneg(lower) : lower;
    }

    __half upper = __ushort_as_half(__half_as_ushort(lower) + 1);
    double uv = (double)__half2float(upper);
    if (isinf(uv)) {
        /* Above the largest half: keep round-to-nearest (65504 or inf) */
        return __float2half((float)value);
    }

    double p_lower = (uv - a) / (uv - lv);
    __half r = (random < p_lower) ? lower : upper;
    return (value < 0.0) ? __hneg(r) : r;
}

__global__ void double2float_round_GPU_kernel(int nrows, int ncols,
                const double *D, int ldd,
                float *F, int ldf) {
    const int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int idy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (idx >= nrows || idy >= ncols) { return; }

    F[idy * ldf + idx] = stochastic_rounding_fp32(D[idy * ldd + idx], hicma_sr_u01(idx, idy, (uint64_t)__double_as_longlong(D[idy * ldd + idx])));
}

extern "C"
void double2float_round_GPU(int nrows, int ncols,
                const double *D, int ldd,
                float *F, int ldf,
                cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    double2float_round_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, D, ldd, F, ldf);
}

__global__ void float2half_round_GPU_kernel(int nrows, int ncols,
                const float *F, int ldf,
                __half *H, int ldh) {
    const int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int idy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (idx >= nrows || idy >= ncols) { return; }

    H[idy * ldh + idx] = stochastic_rounding_fp16((double)F[idy * ldf + idx], hicma_sr_u01(idx, idy, (uint64_t)__float_as_uint(F[idy * ldf + idx])));
}

extern "C"
void float2half_round_GPU(int nrows, int ncols,
                const float *F, int ldf,
                void *_H, int ldh,
                cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    __half *H = (__half *)_H;
    float2half_round_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}

__global__ void double2half_round_GPU_kernel(int nrows, int ncols,
                const double *D, int ldd,
                __half *H, int ldh) {
    const int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int idy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (idx >= nrows || idy >= ncols) { return; }

    H[idy * ldh + idx] = stochastic_rounding_fp16(D[idy * ldd + idx], hicma_sr_u01(idx, idy, (uint64_t)__double_as_longlong(D[idy * ldd + idx])));
}

extern "C"
void double2half_round_GPU(int nrows, int ncols,
                const double *D, int ldd,
                void *_H, int ldh,
                cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    __half *H = (__half *)_H;
    double2half_round_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, D, ldd, H, ldh);
}
