#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <hiprand/hiprand_kernel.h>
#include <stdio.h>
#include <math.h>
#include "hicma_parsec_hip_cuda.h"

#define CHUNKSIZE 32


// CUDA-compatible stochastic rounding for FP32
__device__ float stochastic_rounding_fp32(double value, curandState *state) {
    float nearest = static_cast<float>(value);
    double nearestValue = static_cast<double>(nearest);

    float lower, upper;
    if (value > nearestValue) {
        lower = nearest;
        upper = __uint_as_float(__float_as_uint(nearest) + 1);
    } else if (value < nearestValue) {
        lower = __uint_as_float(__float_as_uint(nearest) - 1);
        upper = nearest;
    } else {
        return nearest;
    }
    double d_lower = fabs(value - static_cast<double>(lower));
    double d_upper = fabs(static_cast<double>(upper) - value);
    double p_lower = d_upper / (d_lower + d_upper);
    float random = curand_uniform(state);
    return (random < p_lower) ? lower : upper;
}

__device__ __half stochastic_rounding_fp16(float value, curandState *state) {
    __half lower;
    __half upper;
    float value_float = (float)(value);
    __half nearest = __float2half(value_float);
    float nearestValue = __half2float(nearest);

    if (value_float > nearestValue) {
        lower = nearest;
        upper = __ushort_as_half(__half_as_ushort(nearest) + 1);
    } else if (value_float < nearestValue) {
        lower = __ushort_as_half(__half_as_ushort(nearest) - 1);
        upper = nearest;
    } else {
        lower = nearest;
        upper = nearest;
    }

    float x = __half2float(lower);
    float y = __half2float(upper);
    double d_lower = fabs((double)(value - x));
    double d_higher = fabs((double)(y - value));
    double p_lower = d_higher / (double)(d_lower + d_higher);
    float random = curand_uniform(state);
    double rounded = (random < p_lower) ? lower : upper;
    return __float2half((float)rounded);
}

__global__ void double2float_round_GPU_kernel(int nrows, int ncols,
                const double *D, int ldh,
                float *F, int ldf) {
    const int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int idy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (idx >= nrows || idy >= ncols) { return; }

    curandState state;
    const unsigned long long element_id = (unsigned long long)idy * (unsigned long long)ldf + (unsigned long long)idx;
    curand_init(0ULL, element_id, 0ULL, &state);
    F[idy * ldf + idx] = stochastic_rounding_fp32(D[idy * ldh + idx], &state);
}

extern "C"
void double2float_round_GPU(int nrows, int ncols,
                const double *D, int ldh,
                float *F, int ldf,
                cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    double2float_round_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, D, ldh, F, ldf);
}

__global__ void float2half_round_GPU_kernel(int nrows, int ncols,
                const float *F, int ldf,
                __half *H, int ldh) {
    const int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int idy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (idx >= nrows || idy >= ncols) { return; }

    curandState state;
    const unsigned long long element_id = (unsigned long long)idy * (unsigned long long)ldh + (unsigned long long)idx;
    curand_init(0ULL, element_id, 0ULL, &state);
    H[idy * ldh + idx] = stochastic_rounding_fp16(F[idy * ldf + idx], &state);
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
                const double *F, int ldf,
                __half *H, int ldh) {
    const int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int idy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if (idx >= nrows || idy >= ncols) { return; }

    curandState state;
    const unsigned long long element_id = (unsigned long long)idy * (unsigned long long)ldh + (unsigned long long)idx;
    curand_init(0ULL, element_id, 0ULL, &state);
    H[idy * ldh + idx] = stochastic_rounding_fp16((float)F[idy * ldf + idx], &state);
}

extern "C"
void double2half_round_GPU(int nrows, int ncols,
                const double *F, int ldf,
                void *_H, int ldh,
                cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    __half *H = (__half *)_H;
    double2half_round_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
