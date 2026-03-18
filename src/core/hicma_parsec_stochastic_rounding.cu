#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>

#define CHUNKSIZE 32


// CUDA-compatible stochastic rounding for FP32
__device__ float stochastic_rounding_fp32(double value, curandState *state) {
    float nearest = static_cast<float>(value); // Convert double to float (round-to-nearest)

    // Convert back to double to compute the difference
    double nearestValue = static_cast<double>(nearest);

    float lower, upper;

    if (value > nearestValue) {
        // Value is closer to the next higher float
        lower = nearest;
        upper = __uint_as_float(__float_as_uint(nearest) + 1); // Next representable float
    } else if (value < nearestValue) {
        // Value is closer to the next lower float
        lower = __uint_as_float(__float_as_uint(nearest) - 1); // Previous representable float
        upper = nearest;
    } else {
        // Exactly representable
        return nearest;
    }
    double d_lower = fabs(value - static_cast<double>(lower));
    double d_upper = fabs(static_cast<double>(upper) - value);

    double p_lower = d_upper / (d_lower + d_upper); // Probability of rounding down
    float random = curand_uniform(state);
    float rounded = (random < p_lower) ? lower : upper;

    return rounded;
}


/****************************************************************************************************/
//rounding from paper
// CUDA-compatible stochastic rounding for FP16
__device__ __half stochastic_rounding_fp16(float value, curandState *state) {

	__half lower;
	__half upper;	

	// Step 1: Convert double to float, then to __half
	float value_float = (float)(value);
	__half nearest = __float2half(value_float); // Convert float to __half (round-to-nearest-even)

	// Step 2: Convert __half back to float for comparison
	float nearestValue = __half2float(nearest);

	// Step 3: Determine the lower and upper bounds
	if (value_float > nearestValue) {
		// Value is closer to the next higher __half
		lower = nearest;
		upper = __ushort_as_half(__half_as_ushort(nearest) + 1); // Next representable __half
	} else if (value_float < nearestValue) {
		// Value is closer to the next lower __half
		lower = __ushort_as_half(__half_as_ushort(nearest) - 1); // Previous representable __half
		upper = nearest;
	} else {
		// Value is exactly representable as __half
		lower = nearest;
		upper = nearest; // Both bounds are the same
	}
	float x= __half2float(lower);
	float y= __half2float(upper);

	double d_lower = fabs((double)(value - x)); // Fractional part
	double d_higher = fabs((double)(y - value)); // Fractional part


	double p_lower=d_higher/(double)(d_lower+d_higher);
	float random = curand_uniform(state);
	double rounded = (random < p_lower) ? lower : upper;

	//printf("%.7f, %.7f\n", random, p_lower);
	//printf("%.7f, %.7f, %.7f, %.7f, %.7f\n", d_lower, d_higher, p_lower, p_upper, random);
	return  __float2half((float)rounded);
	//printf("lower: %.5f, upper: %.5f\n", x, y);

}



__global__ void double2float_round_GPU_kernel( int nrows, int ncols,
                const double *D, int ldh,
                float *F, int ldf ) {
        const int tx=threadIdx.x;
        const int ty=threadIdx.y;
        const int idx= blockIdx.x * blockDim.x + tx;
        const int idy= blockIdx.y * blockDim.y + ty;

        if( idx >= nrows || idy >= ncols ) { return; }
        curandState state;
        const unsigned long long element_id = (unsigned long long)idy * (unsigned long long)ldf + (unsigned long long)idx;
        curand_init(0ULL, element_id, 0ULL, &state);
        F[idy*ldf+idx]= stochastic_rounding_fp32( D[idy*ldh+idx], &state); //__double2float_rn( D[idy*ldh+idx] ); 
	//printf("D %d %d : %d %d : %lf\n", idx, idy, nrows, ncols, D[idy*ldh+idx]);
	//printf("F %d %d : %d %d : %f\n", idx, idy, nrows, ncols, F[idy*ldh+idx]);
}


extern "C"
void double2float_round_GPU( int nrows, int ncols,
                const double *D, int ldh,
                float *F, int ldf,
                cudaStream_t stream ) {

        int nBlockx= (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky= (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        dim3 dimBlock(CHUNKSIZE,CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        double2float_round_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, D, ldh, F, ldf);
}


__global__ void float2half_round_GPU_kernel( int nrows, int ncols,
                const float *F, int ldf,
                __half *H, int ldh ) {
        const int tx=threadIdx.x;
        const int ty=threadIdx.y;
        const int idx= blockIdx.x * blockDim.x + tx;
        const int idy= blockIdx.y * blockDim.y + ty;

        if( idx >= nrows || idy >= ncols ) { return; }
        curandState state;
        const unsigned long long element_id = (unsigned long long)idy * (unsigned long long)ldh + (unsigned long long)idx;
        curand_init(0ULL, element_id, 0ULL, &state);
        H[idy*ldh+idx]= stochastic_rounding_fp16( F[idy*ldf+idx], &state);
}

extern "C"
void float2half_round_GPU( int nrows, int ncols,
                const float *F, int ldf,
                void *_H, int ldh,
                cudaStream_t stream ) {
        int nBlockx= (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky= (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        __half *H = (__half *)_H;
        float2half_round_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}


__global__ void double2half_round_GPU_kernel( int nrows, int ncols,
                const double *F, int ldf,
                __half *H, int ldh ) {
        const int tx=threadIdx.x;
        const int ty=threadIdx.y;
        const int idx= blockIdx.x * blockDim.x + tx;
        const int idy= blockIdx.y * blockDim.y + ty;

        if( idx >= nrows || idy >= ncols ) { return; }
        curandState state;
        const unsigned long long element_id = (unsigned long long)idy * (unsigned long long)ldh + (unsigned long long)idx;
        curand_init(0ULL, element_id, 0ULL, &state);
        H[idy*ldh+idx]= stochastic_rounding_fp16( (float)F[idy*ldf+idx], &state);
}

extern "C"
void double2half_round_GPU( int nrows, int ncols,
                const double *F, int ldf,
                void *_H, int ldh,
                cudaStream_t stream ) {
        int nBlockx= (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky= (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        __half *H = (__half *)_H;
        double2half_round_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
