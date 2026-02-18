// CUDA runtime and library includes for GPU computation
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <stdio.h>

// Thread block size for optimal GPU performance
#define CHUNKSIZE 32

/****************************************************************************************************/
/**
 * @brief CUDA kernel for climate emulator matrix reshaping
 * 
 * Reshapes a complex matrix S into a triangular matrix T for climate emulation.
 * Each thread processes one matrix element, converting complex values to real values
 * and storing them in a triangular pattern.
 * 
 * @param T Output triangular matrix (double)
 * @param S Input complex matrix (cuDoubleComplex)
 * @param L Matrix dimension
 * @param ldaS Leading dimension of input matrix S
 */
__global__ void climate_emulator_reshape_GPU_kernel( double *T, cuDoubleComplex *S,
        int L, int ldaS ) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;
    
    // Bounds checking: only process elements in the lower triangular part
    if( idx >= L || idy > idx ) { 
        return; 
    }

    // Store real part of complex number in triangular pattern
    T[idx*idx+idx+idy] = cuCreal(S[idy*ldaS+idx]);
    
    // Store imaginary part in symmetric position (if not on diagonal)
    if(idy != 0) {
        T[idx*idx+idx-idy] = cuCimag(S[idy*ldaS+idx]);
    }
}   
    
/**
 * @brief GPU wrapper function for climate emulator matrix reshaping
 * 
 * Launches the reshape kernel with appropriate grid and block dimensions.
 * Uses CHUNKSIZE×CHUNKSIZE thread blocks for optimal performance.
 * 
 * @param T Output triangular matrix (double)
 * @param S Input complex matrix (cuDoubleComplex)
 * @param L Matrix dimension
 * @param ldaS Leading dimension of input matrix S
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void climate_emulator_reshape_GPU( double *T, cuDoubleComplex *S,
        int L, int ldaS,
        cudaStream_t stream ) {
    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (L+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (L+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    climate_emulator_reshape_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(T, S, L, ldaS);
}
/****************************************************************************************************/

/****************************************************************************************************/
/**
 * @brief CUDA kernel for printing complex matrix elements
 * 
 * Debug kernel that prints complex matrix elements to console.
 * Each thread prints one matrix element with its coordinates and values.
 * 
 * @param A Input complex matrix (cuDoubleComplex)
 * @param M Number of rows
 * @param N Number of columns
 * @param lda Leading dimension of matrix A
 */
__global__ void climate_emulator_print_complex_GPU_kernel( cuDoubleComplex *A,
        int M, int N, int lda ) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    // Bounds checking
    if( idx > M || idy > N ) { 
        return; 
    }

    // Print matrix element with coordinates and complex values
    printf("%d %d : %lf %lf\n", idx, idy, cuCreal(A[idy*lda+idx]), cuCimag(A[idy*lda+idx]));
}

/**
 * @brief GPU wrapper function for printing complex matrix
 * 
 * Launches the print kernel for debugging complex matrices.
 * 
 * @param A Input complex matrix (cuDoubleComplex)
 * @param M Number of rows
 * @param N Number of columns
 * @param lda Leading dimension of matrix A
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void climate_emulator_print_complex_GPU( cuDoubleComplex *A,
        int M, int N, int lda,
        cudaStream_t stream ) {
    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (M+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (N+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    climate_emulator_print_complex_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(A, M, N, lda);
}
/****************************************************************************************************/

/****************************************************************************************************/
/**
 * @brief CUDA kernel for printing double matrix elements
 * 
 * Debug kernel that prints double matrix elements to console.
 * Each thread prints one matrix element with its coordinates and value.
 * 
 * @param A Input double matrix
 * @param M Number of rows
 * @param N Number of columns
 * @param lda Leading dimension of matrix A
 */
__global__ void climate_emulator_print_double_GPU_kernel( double *A,
        int M, int N, int lda ) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    // Bounds checking
    if( idx > M || idy > N ) { 
        return; 
    }

    // Print matrix element with coordinates and value
    printf("%d %d : %lf\n", idx, idy, A[idy*lda+idx]);
}

/**
 * @brief GPU wrapper function for printing double matrix
 * 
 * Launches the print kernel for debugging double matrices.
 * 
 * @param A Input double matrix
 * @param M Number of rows
 * @param N Number of columns
 * @param lda Leading dimension of matrix A
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void climate_emulator_print_double_GPU( double *A,
        int M, int N, int lda,
        cudaStream_t stream ) {
    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (M+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (N+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    climate_emulator_print_double_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(A, M, N, lda);
}
/****************************************************************************************************/


/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting double precision to FP8 format
 * 
 * Converts double precision matrix elements to FP8 (8-bit floating point) format.
 * Each thread processes one matrix element, converting from double to FP8.
 * Note: Transposition is applied as only TN format is supported in cublasLtMatmul.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param S Input matrix (double precision)
 * @param lds Leading dimension of input matrix S
 * @param T Output matrix (FP8 format, uint8_t)
 * @param ldt Leading dimension of output matrix T
 */
__global__ void double2fp8_GPU_kernel( int nrows, int ncols,
                const double *S, int lds,
                uint8_t *T, int ldt ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Convert double to FP8 with transposition (TN format requirement)
        // Use E4M3 format (4 exponent bits, 3 mantissa bits) with saturation
        T[idx*ldt+idy] = (uint8_t)__nv_cvt_float_to_fp8( 
            (float)S[idy*lds+idx], __NV_SATFINITE, __NV_E4M3 
        );
}

/**
 * @brief GPU wrapper function for double to FP8 conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointer to uint8_t for FP8 storage.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param S Input matrix (double precision)
 * @param lds Leading dimension of input matrix S
 * @param T Output matrix (FP8 format, void pointer)
 * @param ldt Leading dimension of output matrix T
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void double2fp8_GPU( int nrows, int ncols,
                const double *S, int lds,
                void *T, int ldt,
                cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointer to uint8_t for FP8 storage
        uint8_t *_T = (uint8_t *)T; 
        
        // Launch kernel with calculated dimensions
        double2fp8_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, S, lds, _T, ldt);
}

__global__ void double2fp8_padded_GPU_kernel( int nrows, int ncols,
                int padded_rows, int padded_cols,
                const double *S, int lds,
                uint8_t *T, int ldt ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        if( idx >= padded_rows || idy >= padded_cols ) {
            return;
        }

        if( idx < nrows && idy < ncols ) {
            T[idx*ldt+idy] = (uint8_t)__nv_cvt_float_to_fp8(
                (float)S[idy*lds+idx], __NV_SATFINITE, __NV_E4M3
            );
        } else {
            T[idx*ldt+idy] = 0;
        }
}

extern "C"
void double2fp8_padded_GPU( int nrows, int ncols,
                const double *S, int lds,
                void *T, int ldt,
                int padded_rows, int padded_cols,
                cudaStream_t stream ) {
        int nBlockx = (padded_rows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (padded_cols+CHUNKSIZE-1)/CHUNKSIZE;

        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);

        uint8_t *_T = (uint8_t *)T;

        double2fp8_padded_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, padded_rows, padded_cols, S, lds, _T, ldt);
}
/****************************************************************************************************/

/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting single precision to FP8 format
 * 
 * Converts single precision (float) matrix elements to FP8 (8-bit floating point) format.
 * Each thread processes one matrix element, converting from float to FP8.
 * Note: Transposition is applied as only TN format is supported in cublasLtMatmul.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param S Input matrix (single precision)
 * @param lds Leading dimension of input matrix S
 * @param T Output matrix (FP8 format, uint8_t)
 * @param ldt Leading dimension of output matrix T
 */
__global__ void float2fp8_GPU_kernel( int nrows, int ncols,
                const float *S, int lds,
                uint8_t *T, int ldt ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Convert float to FP8 with transposition (TN format requirement)
        // Use E4M3 format (4 exponent bits, 3 mantissa bits) with saturation
        T[idx*ldt+idy] = (uint8_t)__nv_cvt_float_to_fp8( 
            S[idy*lds+idx], __NV_SATFINITE, __NV_E4M3 
        );
}

/**
 * @brief GPU wrapper function for float to FP8 conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointer to uint8_t for FP8 storage.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param S Input matrix (single precision)
 * @param lds Leading dimension of input matrix S
 * @param T Output matrix (FP8 format, void pointer)
 * @param ldt Leading dimension of output matrix T
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void float2fp8_GPU( int nrows, int ncols,
                const float *S, int lds,
                void *T, int ldt,
                cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointer to uint8_t for FP8 storage
        uint8_t *_T = (uint8_t *)T; 
        
        // Launch kernel with calculated dimensions
        float2fp8_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, S, lds, _T, ldt);
}

__global__ void float2fp8_padded_GPU_kernel( int nrows, int ncols,
                int padded_rows, int padded_cols,
                const float *S, int lds,
                uint8_t *T, int ldt ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        if( idx >= padded_rows || idy >= padded_cols ) {
            return;
        }

        if( idx < nrows && idy < ncols ) {
            T[idx*ldt+idy] = (uint8_t)__nv_cvt_float_to_fp8(
                S[idy*lds+idx], __NV_SATFINITE, __NV_E4M3
            );
        } else {
            T[idx*ldt+idy] = 0;
        }
}

extern "C"
void float2fp8_padded_GPU( int nrows, int ncols,
                const float *S, int lds,
                void *T, int ldt,
                int padded_rows, int padded_cols,
                cudaStream_t stream ) {
        int nBlockx = (padded_rows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (padded_cols+CHUNKSIZE-1)/CHUNKSIZE;

        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);

        uint8_t *_T = (uint8_t *)T;

        float2fp8_padded_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, padded_rows, padded_cols, S, lds, _T, ldt);
}

/**
 * @brief Host function for converting single float to FP8 format
 * 
 * Converts a single float value to FP8 format on the host.
 * Useful for scalar conversions or when GPU kernel launch overhead is not justified.
 * 
 * @param F Input float value
 * @param _H Output pointer for FP8 value
 */
extern "C"
void float2fp8_host(const float F, void* _H) {
    __nv_fp8_e4m3 F8 = __nv_fp8_e4m3(F);
    memcpy(_H, &F8, sizeof(F8));
}
/****************************************************************************************************/


/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting half precision to FP8 format
 * 
 * Converts half precision (__half) matrix elements to FP8 (8-bit floating point) format.
 * Each thread processes one matrix element, converting from half to FP8.
 * Note: Transposition is applied as only TN format is supported in cublasLtMatmul.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param S Input matrix (half precision)
 * @param lds Leading dimension of input matrix S
 * @param T Output matrix (FP8 format, uint8_t)
 * @param ldt Leading dimension of output matrix T
 */
__global__ void half2fp8_GPU_kernel( int nrows, int ncols,
                const __half *S, int lds,
                uint8_t *T, int ldt ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Convert half to FP8 with transposition (TN format requirement)
        // Use E4M3 format (4 exponent bits, 3 mantissa bits) with saturation
        T[idx*ldt+idy] = (uint8_t)__nv_cvt_halfraw_to_fp8( 
            S[idy*lds+idx], __NV_SATFINITE, __NV_E4M3 
        );
}

/**
 * @brief GPU wrapper function for half to FP8 conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointers to appropriate types.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param S Input matrix (half precision, void pointer)
 * @param lds Leading dimension of input matrix S
 * @param T Output matrix (FP8 format, void pointer)
 * @param ldt Leading dimension of output matrix T
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void half2fp8_GPU( int nrows, int ncols,
                const __half *S, int lds,
                void *T, int ldt,
                cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointers to appropriate types
        __half *_S = (__half *)S;
        uint8_t *_T = (uint8_t *)T;
        
        // Launch kernel with calculated dimensions
        half2fp8_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, _S, lds, _T, ldt);
}

__global__ void half2fp8_padded_GPU_kernel( int nrows, int ncols,
                int padded_rows, int padded_cols,
                const __half *S, int lds,
                uint8_t *T, int ldt ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        if( idx >= padded_rows || idy >= padded_cols ) {
            return;
        }

        if( idx < nrows && idy < ncols ) {
            T[idx*ldt+idy] = (uint8_t)__nv_cvt_halfraw_to_fp8(
                S[idy*lds+idx], __NV_SATFINITE, __NV_E4M3
            );
        } else {
            T[idx*ldt+idy] = 0;
        }
}

extern "C"
void half2fp8_padded_GPU( int nrows, int ncols,
                const __half *S, int lds,
                void *T, int ldt,
                int padded_rows, int padded_cols,
                cudaStream_t stream ) {
        int nBlockx = (padded_rows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (padded_cols+CHUNKSIZE-1)/CHUNKSIZE;

        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);

        uint8_t *_T = (uint8_t *)T;

        half2fp8_padded_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, padded_rows, padded_cols, S, lds, _T, ldt);
}
/****************************************************************************************************/


/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting double precision to bfloat16 format
 * 
 * Converts double precision matrix elements to bfloat16 (Brain Floating Point) format.
 * Each thread processes one matrix element, converting from double to bfloat16.
 * Bfloat16 provides better numerical stability than half precision for deep learning.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (double precision)
 * @param ldf Leading dimension of input matrix F
 * @param H Output matrix (bfloat16 format)
 * @param ldh Leading dimension of output matrix H
 */
__global__ void double2bf_GPU_kernel( int nrows, int ncols,
                const double *F, int ldf,
                __nv_bfloat16 *H, int ldh ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Convert double to bfloat16 format
        H[idy*ldh+idx] = __double2bfloat16( F[idy*ldf+idx] );
}

/**
 * @brief GPU wrapper function for double to bfloat16 conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointer to bfloat16.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (double precision)
 * @param ldf Leading dimension of input matrix F
 * @param _H Output matrix (bfloat16 format, void pointer)
 * @param ldh Leading dimension of output matrix H
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void double2bf_GPU( int nrows, int ncols,
                const double *F, int ldf,
                void *_H, int ldh,
                cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointer to bfloat16
        __nv_bfloat16 *H = (__nv_bfloat16 *)_H;
        
        // Launch kernel with calculated dimensions
        double2bf_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
/****************************************************************************************************/
/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting single precision to bfloat16 format
 * 
 * Converts single precision (float) matrix elements to bfloat16 (Brain Floating Point) format.
 * Each thread processes one matrix element, converting from float to bfloat16.
 * Bfloat16 provides better numerical stability than half precision for deep learning.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (single precision)
 * @param ldf Leading dimension of input matrix F
 * @param H Output matrix (bfloat16 format)
 * @param ldh Leading dimension of output matrix H
 */
__global__ void float2bf_GPU_kernel( int nrows, int ncols,
                const float *F, int ldf,
                __nv_bfloat16 *H, int ldh ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Convert float to bfloat16 format
        H[idy*ldh+idx] = __float2bfloat16( F[idy*ldf+idx] );
}

/**
 * @brief GPU wrapper function for float to bfloat16 conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointer to bfloat16.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (single precision)
 * @param ldf Leading dimension of input matrix F
 * @param _H Output matrix (bfloat16 format, void pointer)
 * @param ldh Leading dimension of output matrix H
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void float2bf_GPU( int nrows, int ncols,
                const float *F, int ldf,
                void *_H, int ldh,
                cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointer to bfloat16
        __nv_bfloat16 *H = (__nv_bfloat16 *)_H;
        
        // Launch kernel with calculated dimensions
        float2bf_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}

/**
 * @brief Host function for converting single float to bfloat16 format
 * 
 * Converts a single float value to bfloat16 format on the host.
 * Useful for scalar conversions or when GPU kernel launch overhead is not justified.
 * 
 * @param F Input float value
 * @param _H Output pointer for bfloat16 value
 */
extern "C"
void float2bf_host(const float F, void* _H) {
    __nv_bfloat16 bf = __nv_bfloat16(F);
    memcpy(_H, &bf, sizeof(bf));
}
/****************************************************************************************************/

/**
 * @brief CUDA kernel for converting bfloat16 to single precision format
 * 
 * Converts bfloat16 (Brain Floating Point) matrix elements to single precision (float) format.
 * Each thread processes one matrix element, converting from bfloat16 to float.
 * This is the inverse operation of float2bf_GPU_kernel.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param H Input matrix (bfloat16 format)
 * @param ldh Leading dimension of input matrix H
 * @param F Output matrix (single precision)
 * @param ldf Leading dimension of output matrix F
 */
__global__ void bf2float_GPU_kernel( int nrows, int ncols,
                const __nv_bfloat16 *H, int ldh,
                float *F, int ldf ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Convert bfloat16 to float format
        F[idy*ldf+idx] = __bfloat162float( H[idy*ldh+idx] );
}

/**
 * @brief GPU wrapper function for bfloat16 to float conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointer to bfloat16.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _H Input matrix (bfloat16 format, void pointer)
 * @param ldh Leading dimension of input matrix H
 * @param F Output matrix (single precision)
 * @param ldf Leading dimension of output matrix F
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void bf2float_GPU( int nrows, int ncols,
                const void *_H, int ldh,
                float *F, int ldf,
                cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointer to bfloat16
        __nv_bfloat16 *H = (__nv_bfloat16 *)_H;
        
        // Launch kernel with calculated dimensions
        bf2float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, H, ldh, F, ldf);
}
/****************************************************************************************************/

/**
 * @brief CUDA kernel for copying bfloat16 matrix data
 * 
 * Copies bfloat16 matrix data from source to destination.
 * Each thread processes one matrix element, performing a memory copy operation.
 * Useful for matrix transposition or data reorganization.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param src Source matrix (bfloat16 format)
 * @param dest Destination matrix (bfloat16 format)
 */
__global__ void memcpy_bf_GPU_kernel( int nrows, int ncols,
                __nv_bfloat16 *src, __nv_bfloat16 *dest ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Copy matrix element from source to destination
        dest[idy*nrows+idx] = src[idy*nrows+idx];
}

/**
 * @brief GPU wrapper function for bfloat16 matrix copy
 * 
 * Launches the copy kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointers to bfloat16.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _src Source matrix (bfloat16 format, void pointer)
 * @param _dest Destination matrix (bfloat16 format, void pointer)
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void memcpy_bf_GPU( int nrows, int ncols, void *_src, void *_dest, cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointers to bfloat16
        __nv_bfloat16 *src = (__nv_bfloat16 *)_src;
        __nv_bfloat16 *dest = (__nv_bfloat16 *)_dest;
        
        // Launch kernel with calculated dimensions
        memcpy_bf_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, dest);
}
/****************************************************************************************************/
/****************************************************************************************************/

/**
 * @brief CUDA kernel for converting single precision to double precision
 * 
 * Converts single precision (float) matrix elements to double precision format.
 * Each thread processes one matrix element, converting from float to double.
 * This operation increases precision but doubles memory usage.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (single precision)
 * @param ldf Leading dimension of input matrix F
 * @param D Output matrix (double precision)
 * @param ldh Leading dimension of output matrix D
 */
__global__ void float2double_GPU_kernel( int nrows, int ncols,
                const float *F, int ldf,
                double *D, int ldh ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Convert float to double with explicit casting
        D[idy*ldh+idx] = (double)F[idy*ldf+idx];
}

/**
 * @brief GPU wrapper function for float to double conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Converts single precision matrix to double precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (single precision)
 * @param ldf Leading dimension of input matrix F
 * @param D Output matrix (double precision)
 * @param ldh Leading dimension of output matrix D
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void float2double_GPU( int nrows, int ncols,
                const float *F, int ldf,
                double *D, int ldh,
                cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Launch kernel with calculated dimensions
        float2double_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, D, ldh);
}

/****************************************************************************************************/

/**
 * @brief CUDA kernel for converting double precision to single precision
 * 
 * Converts double precision matrix elements to single precision (float) format.
 * Each thread processes one matrix element, converting from double to float.
 * Uses round-to-nearest rounding mode for optimal precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param D Input matrix (double precision)
 * @param ldh Leading dimension of input matrix D
 * @param F Output matrix (single precision)
 * @param ldf Leading dimension of output matrix F
 */
__global__ void double2float_GPU_kernel( int nrows, int ncols,
                const double *D, int ldh,
                float *F, int ldf ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Convert double to float with round-to-nearest rounding
        F[idy*ldf+idx] = __double2float_rn( D[idy*ldh+idx] );
}

/**
 * @brief GPU wrapper function for double to float conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Converts double precision matrix to single precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param D Input matrix (double precision)
 * @param ldh Leading dimension of input matrix D
 * @param F Output matrix (single precision)
 * @param ldf Leading dimension of output matrix F
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void double2float_GPU( int nrows, int ncols,
                const double *D, int ldh,
                float *F, int ldf,
                cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Launch kernel with calculated dimensions
        double2float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, D, ldh, F, ldf);
}

/****************************************************************************************************/

/**
 * @brief CUDA kernel for converting half precision to single precision
 * 
 * Converts half precision (__half) matrix elements to single precision (float) format.
 * Each thread processes one matrix element, converting from half to float.
 * Half precision provides memory savings but with reduced numerical range.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param H Input matrix (half precision)
 * @param ldh Leading dimension of input matrix H
 * @param F Output matrix (single precision)
 * @param ldf Leading dimension of output matrix F
 */
__global__ void half2float_GPU_kernel( int nrows, int ncols,
                const __half *H, int ldh,
                float *F, int ldf ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Convert half to float format
        F[idy*ldf+idx] = __half2float( H[idy*ldh+idx] );
}

/**
 * @brief GPU wrapper function for half to float conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointer to half precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _H Input matrix (half precision, void pointer)
 * @param ldh Leading dimension of input matrix H
 * @param F Output matrix (single precision)
 * @param ldf Leading dimension of output matrix F
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void half2float_GPU( int nrows, int ncols,
                const void *_H, int ldh,
                float *F, int ldf,
                cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointer to half precision
        __half *H = (__half *)_H;
        
        // Launch kernel with calculated dimensions
        half2float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, H, ldh, F, ldf);
}

/****************************************************************************************************/

/**
 * @brief CUDA kernel for converting single precision to half precision
 * 
 * Converts single precision (float) matrix elements to half precision (__half) format.
 * Each thread processes one matrix element, converting from float to half.
 * Uses round-to-nearest rounding mode for optimal precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (single precision)
 * @param ldf Leading dimension of input matrix F
 * @param H Output matrix (half precision)
 * @param ldh Leading dimension of output matrix H
 */
__global__ void float2half_GPU_kernel( int nrows, int ncols,
                const float *F, int ldf,
                __half *H, int ldh ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Convert float to half with round-to-nearest rounding
        H[idy*ldh+idx] = __float2half_rn( F[idy*ldf+idx] );
}

/**
 * @brief GPU wrapper function for float to half conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointer to half precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (single precision)
 * @param ldf Leading dimension of input matrix F
 * @param _H Output matrix (half precision, void pointer)
 * @param ldh Leading dimension of output matrix H
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void float2half_GPU( int nrows, int ncols,
                const float *F, int ldf,
                void *_H, int ldh,
                cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointer to half precision
        __half *H = (__half *)_H;
        
        // Launch kernel with calculated dimensions
        float2half_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}

/**
 * @brief Host function for converting single float to half precision
 * 
 * Converts a single float value to half precision format on the host.
 * Useful for scalar conversions or when GPU kernel launch overhead is not justified.
 * 
 * @param F Input float value
 * @param _H Output pointer for half precision value
 */
extern "C"
void float2half_host(const float F, void *_H) {
    __half tmph = __float2half(F);
    memcpy(_H, &tmph, sizeof(tmph));
}

/****************************************************************************************************/

/**
 * @brief CUDA kernel for converting half precision to double precision
 * 
 * Converts half precision (__half) matrix elements to double precision format.
 * Each thread processes one matrix element, converting from half to double.
 * This operation increases precision and numerical range.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param H Input matrix (half precision)
 * @param ldh Leading dimension of input matrix H
 * @param F Output matrix (double precision)
 * @param ldf Leading dimension of output matrix F
 */
__global__ void half2double_GPU_kernel( int nrows, int ncols,
                const __half *H, int ldh,
                double *F, int ldf ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Convert half to double with explicit casting
        F[idy*ldf+idx] = (double)( H[idy*ldh+idx] );
}

/**
 * @brief GPU wrapper function for half to double conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointer to half precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _H Input matrix (half precision, void pointer)
 * @param ldh Leading dimension of input matrix H
 * @param F Output matrix (double precision)
 * @param ldf Leading dimension of output matrix F
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void half2double_GPU( int nrows, int ncols,
                const void *_H, int ldh,
                double *F, int ldf,
                cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointer to half precision
        __half *H = (__half *)_H;
        
        // Launch kernel with calculated dimensions
        half2double_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, H, ldh, F, ldf);
}

/****************************************************************************************************/

/**
 * @brief CUDA kernel for converting double precision to half precision
 * 
 * Converts double precision matrix elements to half precision (__half) format.
 * Each thread processes one matrix element, converting from double to half.
 * This operation reduces memory usage but may lose precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (double precision)
 * @param ldf Leading dimension of input matrix F
 * @param H Output matrix (half precision)
 * @param ldh Leading dimension of output matrix H
 */
__global__ void double2half_GPU_kernel( int nrows, int ncols,
                const double *F, int ldf,
                __half *H, int ldh ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Convert double to half format
        H[idy*ldh+idx] = __double2half( F[idy*ldf+idx] );
}

/**
 * @brief GPU wrapper function for double to half conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointer to half precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (double precision)
 * @param ldf Leading dimension of input matrix F
 * @param _H Output matrix (half precision, void pointer)
 * @param ldh Leading dimension of output matrix H
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void double2half_GPU( int nrows, int ncols,
                const double *F, int ldf,
                void *_H, int ldh,
                cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointer to half precision
        __half *H = (__half *)_H;
        
        // Launch kernel with calculated dimensions
        double2half_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}


/****************************************************************************************************/

/**
 * @brief CUDA kernel for copying half precision matrix data
 * 
 * Copies half precision matrix data from source to destination.
 * Each thread processes one matrix element, performing a memory copy operation.
 * Useful for matrix transposition or data reorganization.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param src Source matrix (half precision)
 * @param dest Destination matrix (half precision)
 */
__global__ void memcpy_half_GPU_kernel( int nrows, int ncols,
                __half *src, __half *dest ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Copy matrix element from source to destination
        dest[idy*nrows+idx] = src[idy*nrows+idx];
}

/**
 * @brief GPU wrapper function for half precision matrix copy
 * 
 * Launches the copy kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointers to half precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _src Source matrix (half precision, void pointer)
 * @param _dest Destination matrix (half precision, void pointer)
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void memcpy_half_GPU( int nrows, int ncols, void *_src, void *_dest, cudaStream_t stream ) { 
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointers to half precision
        __half *src = (__half *)_src;
        __half *dest = (__half *)_dest;
        
        // Launch kernel with calculated dimensions
        memcpy_half_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, dest);
}

/****************************************************************************************************/

/**
 * @brief CUDA kernel for copying double precision matrix data
 *
 * Copies double precision (double) matrix data from source to destination.
 * Each thread processes one matrix element, performing a memory copy operation.
 *
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param src Source matrix (double precision)
 * @param dest Destination matrix (double precision)
 */
__global__ void memcpy_double_GPU_kernel( int nrows, int ncols,
              double *src, double *dest ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) {
            return;
        }

        // Copy matrix element from source to destination
        dest[idy*nrows+idx] = src[idy*nrows+idx];
}

/**
 * @brief GPU wrapper function for double precision matrix copy
 *
 * Launches the copy kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointers to double precision.
 *
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _src Source matrix (double precision, void pointer)
 * @param _dest Destination matrix (double precision, void pointer)
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void memcpy_double_GPU( int nrows, int ncols, void *_src, void *_dest, cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;

        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);

        // Cast void pointers to double precision
        double *src = (double *)_src;
        double *dest = (double *)_dest;

        // Launch kernel with calculated dimensions
        memcpy_double_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, dest);
}

/****************************************************************************************************/

/**
 * @brief CUDA kernel for dest(double) -= src(float)
 *
 * Performs element-wise subtraction:
 *   ((double *)dest)[j*nrows+i] -= ((float *)src)[j*nrows+i]
 *
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param src Source matrix (single precision)
 * @param dest Destination matrix (double precision)
 */
__global__ void sub_float_from_double_GPU_kernel( int nrows, int ncols,
               float *src, double *dest ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) {
            return;
        }

        // dest(j, i) -= src(j, i)
        dest[idy*nrows+idx] -= (double)src[idy*nrows+idx];
}

/**
 * @brief GPU wrapper for dest(double) -= src(float)
 *
 * Launches sub_float_from_double_GPU_kernel with grid/block geometry
 * derived from matrix dimensions.
 *
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _src Source matrix (float, void pointer)
 * @param _dest Destination matrix (double, void pointer)
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void sub_float_from_double_GPU( int nrows, int ncols, void *_src, void *_dest, cudaStream_t stream ) {
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;

        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);

        float *src = (float *)_src;
        double *dest = (double *)_dest;

        sub_float_from_double_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, dest);
}

__global__ void sub_float_from_double_ld_GPU_kernel( int nrows, int ncols,
               const float *src, int ldsrc,
               double *dest, int lddest ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        if( idx >= nrows || idy >= ncols ) {
            return;
        }

        dest[idy*lddest+idx] -= (double)src[idy*ldsrc+idx];
}

extern "C"
void sub_float_from_double_ld_GPU( int nrows, int ncols,
               void *_src, int ldsrc,
               void *_dest, int lddest,
               cudaStream_t stream ) {
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;

        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);

        float *src = (float *)_src;
        double *dest = (double *)_dest;

        sub_float_from_double_ld_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, ldsrc, dest, lddest);
}

__global__ void sub_float_from_float_ld_GPU_kernel( int nrows, int ncols,
               const float *src, int ldsrc,
               float *dest, int lddest ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        if( idx >= nrows || idy >= ncols ) {
            return;
        }

        dest[idy*lddest+idx] -= src[idy*ldsrc+idx];
}

extern "C"
void sub_float_from_float_ld_GPU( int nrows, int ncols,
               void *_src, int ldsrc,
               void *_dest, int lddest,
               cudaStream_t stream ) {
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;

        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);

        float *src = (float *)_src;
        float *dest = (float *)_dest;

        sub_float_from_float_ld_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, ldsrc, dest, lddest);
}

/****************************************************************************************************/

/**
 * @brief CUDA kernel for copying single precision matrix data
 * 
 * Copies single precision (float) matrix data from source to destination.
 * Each thread processes one matrix element, performing a memory copy operation.
 * Useful for matrix transposition or data reorganization.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param src Source matrix (single precision)
 * @param dest Destination matrix (single precision)
 */
__global__ void memcpy_float_GPU_kernel( int nrows, int ncols,
               float *src, float *dest ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;
                           
        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }
                           
        // Copy matrix element from source to destination
        dest[idy*nrows+idx] = src[idy*nrows+idx];
}                          
                           
/**
 * @brief GPU wrapper function for single precision matrix copy
 * 
 * Launches the copy kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointers to single precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _src Source matrix (single precision, void pointer)
 * @param _dest Destination matrix (single precision, void pointer)
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"                 
void memcpy_float_GPU( int nrows, int ncols, void *_src, void *_dest, cudaStream_t stream ) {                         
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointers to single precision
        float *src = (float *)_src;
        float *dest = (float *)_dest;
        
        // Launch kernel with calculated dimensions
        memcpy_float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, dest);
}

/****************************************************************************************************/


/****************************************************************************************************/

/**
 * @brief CUDA kernel for copying 32-bit integer matrix data
 * 
 * Copies 32-bit integer matrix data from source to destination.
 * Each thread processes one matrix element, performing a memory copy operation.
 * Useful for matrix transposition or data reorganization.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param src Source matrix (32-bit integer)
 * @param dest Destination matrix (32-bit integer)
 */
__global__ void memcpy_int32_GPU_kernel( int nrows, int ncols,
               int *src, int *dest ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Copy matrix element from source to destination
        dest[idy*nrows+idx] = src[idy*nrows+idx];
}

/**
 * @brief GPU wrapper function for 32-bit integer matrix copy
 * 
 * Launches the copy kernel with appropriate grid and block dimensions.
 * Handles type casting from void pointers to 32-bit integer.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _src Source matrix (32-bit integer, void pointer)
 * @param _dest Destination matrix (32-bit integer, void pointer)
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void hicma_parsec_memcpy_int32_GPU( int nrows, int ncols, void *_src, void *_dest, cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        
        // Cast void pointers to 32-bit integer
        int *src = (int *)_src;
        int *dest = (int *)_dest;
        
        // Launch kernel with calculated dimensions
        memcpy_int32_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, dest);
}


/****************************************************************************************************/

/**
 * @brief CUDA kernel for printing 8-bit integer matrix elements
 * 
 * Debug kernel that prints 8-bit integer matrix elements to console.
 * Each thread prints one matrix element with its coordinates and value.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param A Input matrix (8-bit integer)
 */
__global__ void matrix_print_int8_GPU_kernel( int nrows, int ncols, int8_t *A ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;         
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Print matrix element with coordinates and value
        printf("INT8_PRINT %d %d : %d\n", idx, idy, (int)A[idy*nrows+idx]);
}   
    
/**
 * @brief GPU wrapper function for printing 8-bit integer matrix
 * 
 * Launches the print kernel for debugging 8-bit integer matrices.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param A Input matrix (8-bit integer)
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void matrix_print_int8_GPU( int nrows, int ncols, int8_t *A, cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
    
        // Launch kernel with calculated dimensions
        matrix_print_int8_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, A);
}
    
/****************************************************************************************************/

/****************************************************************************************************/

/**
 * @brief CUDA kernel for printing 32-bit integer matrix elements
 * 
 * Debug kernel that prints 32-bit integer matrix elements to console.
 * Each thread prints one matrix element with its coordinates and value.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param A Input matrix (32-bit integer)
 */
__global__ void matrix_print_int32_GPU_kernel( int nrows, int ncols, int32_t *A ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Print matrix element with coordinates and value
        printf("INT32_PRINT %d %d : %d\n", idx, idy, A[idy*nrows+idx]);
}

/**
 * @brief GPU wrapper function for printing 32-bit integer matrix
 * 
 * Launches the print kernel for debugging 32-bit integer matrices.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param A Input matrix (32-bit integer)
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void matrix_print_int32_GPU( int nrows, int ncols, int32_t *A, cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);

        // Launch kernel with calculated dimensions
        matrix_print_int32_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, A);
}

/****************************************************************************************************/

/****************************************************************************************************/

/**
 * @brief CUDA kernel for printing single precision matrix elements
 * 
 * Debug kernel that prints single precision matrix elements to console.
 * Each thread prints one matrix element with its coordinates and value.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param A Input matrix (single precision)
 */
__global__ void matrix_print_float_GPU_kernel( int nrows, int ncols, float *A ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Print matrix element with coordinates and value
        printf("SINGLE_PRINT %d %d : %g\n", idx, idy, A[idy*nrows+idx]);
}

/**
 * @brief GPU wrapper function for printing single precision matrix
 * 
 * Launches the print kernel for debugging single precision matrices.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param A Input matrix (single precision)
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void matrix_print_float_GPU( int nrows, int ncols, float *A, cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);

        // Launch kernel with calculated dimensions
        matrix_print_float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, A);
}

/****************************************************************************************************/

/**
 * @brief CUDA kernel for printing double precision matrix elements
 * 
 * Debug kernel that prints double precision matrix elements to console.
 * Each thread prints one matrix element with its coordinates and value.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param A Input matrix (double precision)
 */
__global__ void matrix_print_double_GPU_kernel( int nrows, int ncols, double *A ) {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int idx = blockIdx.x * blockDim.x + tx;
        const int idy = blockIdx.y * blockDim.y + ty;

        // Bounds checking
        if( idx >= nrows || idy >= ncols ) { 
            return; 
        }

        // Print matrix element with coordinates and value
        printf("DOUBLE_PRINT %d %d : %g\n", idx, idy, A[idy*nrows+idx]);
}

/**
 * @brief GPU wrapper function for printing double precision matrix
 * 
 * Launches the print kernel for debugging double precision matrices.
 * 
 * @param nrows Number of rows in the matrix
 * @param nrows Number of rows in the matrix
 * @param A Input matrix (double precision)
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void matrix_print_double_GPU( int nrows, int ncols, double *A, cudaStream_t stream ) {
        // Calculate grid dimensions based on matrix size and chunk size
        int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        
        // Define thread block and grid dimensions
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);

        // Launch kernel with calculated dimensions
        matrix_print_double_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, A);
}

/****************************************************************************************************/
/**
 * @brief Custom cuBLAS GEMM wrapper for half precision operations
 * 
 * Wrapper function for cuBLAS GEMM operations that handles half precision
 * alpha and beta parameters. This function converts float pointers to half
 * precision before calling the underlying cuBLAS function.
 * 
 * Note: There is a known issue where NAN values may appear when matrix
 * size is large. Investigation is ongoing.
 * 
 * @param handle cuBLAS handle
 * @param transa Operation on matrix A
 * @param transb Operation on matrix B
 * @param m Number of rows in matrix A and C
 * @param n Number of columns in matrix B and C
 * @param k Number of columns in matrix A and rows in matrix B
 * @param alpha Scalar multiplier for matrix A*B
 * @param A Input matrix A
 * @param Atype Data type of matrix A
 * @param lda Leading dimension of matrix A
 * @param B Input matrix B
 * @param Btype Data type of matrix B
 * @param ldb Leading dimension of matrix B
 * @param beta Scalar multiplier for matrix C
 * @param C Output matrix C
 * @param Ctype Data type of matrix C
 * @param ldc Leading dimension of matrix C
 * @param computeType Computation data type
 * @param algo Algorithm to use for GEMM
 * @return cuBLAS status
 */
extern "C"
cublasStatus_t my_cublasGemmEx(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const void    *alpha,
                           const void     *A, 
                           cudaDataType_t Atype,
                           int lda,
                           const void     *B,
                           cudaDataType_t Btype,  
                           int ldb,
                           const void    *beta, 
                           void           *C,
                           cudaDataType_t Ctype,
                           int ldc,
                           cudaDataType_t computeType,
                           cublasGemmAlgo_t algo)
{
    // Convert float pointers to half precision for alpha and beta
    __half alpha_h = (__half)(((float *)alpha)[0]);
    __half beta_h = (__half)(((float *)beta)[0]);
    
    // Call the underlying cuBLAS GEMM function with converted parameters
    return cublasGemmEx(handle, transa, transb, m, n, k,  
                        &alpha_h, A, Atype, lda,
                                  B, Btype, ldb,
                        &beta_h,  C, Ctype, ldc,
                        computeType, algo);
}

/**===*/
/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting integer to double precision
 * 
 * Converts integer matrix elements to double precision format.
 * Each thread processes one matrix element, converting from int to double.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input matrix (integer)
 * @param ldi Leading dimension of input matrix T
 * @param F Output matrix (double precision)
 * @param ldf Leading dimension of output matrix F
 */
__global__ void int_2double_array_kernel(int nrows, int ncols,
                                        int *T, int ldi,
                                        double *F, int ldf){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;
    
    // Bounds checking
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Convert integer to double with explicit casting
    F[idy*ldf+idx] = (double)T[idy*ldi+idx];
}

/**
 * @brief GPU wrapper function for integer to double conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Converts integer matrix to double precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input matrix (integer)
 * @param ldi Leading dimension of input matrix T
 * @param F Output matrix (double precision)
 * @param ldf Leading dimension of output matrix F
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void int_2double_array(int nrows, int ncols,
                    int *T, int ldi,
                    double *F, int ldf,
                    cudaStream_t stream){
    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    int_2double_array_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, T, ldi, F, ldf);
}
/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting integer matrix to double precision (unary operation)
 * 
 * Converts integer matrix elements to double precision format in-place.
 * Each thread processes one matrix element, converting from int to double.
 * This operation performs an in-place conversion by casting the integer pointer
 * to double and then converting each element.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input/output matrix (integer, cast to double for output)
 * @param ldi Leading dimension of matrix T
 */
__global__ void int_2double_array_kernel_unary(int nrows, int ncols,
                                        int *T, int ldi){

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;
    
    // Cast integer pointer to double for in-place conversion
    double *data_s = (double *)T;
    
    // Bounds checking
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Convert integer to double with explicit casting
    data_s[idy*ldi+idx] = (double)T[idy*ldi+idx];
}

/**
 * @brief GPU wrapper function for integer to double conversion (unary operation)
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Converts integer matrix to double precision in-place.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input/output matrix (integer, cast to double for output)
 * @param ldi Leading dimension of matrix T
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void int_2double_array_unary(int nrows, int ncols,
                    int *T, int ldi,
                    cudaStream_t stream){

    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    int_2double_array_kernel_unary<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, T, ldi);
}
/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting double precision to integer
 * 
 * Converts double precision matrix elements to integer format.
 * Each thread processes one matrix element, converting from double to int.
 * This operation may lose precision due to truncation.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (double precision)
 * @param ldf Leading dimension of input matrix F
 * @param T Output matrix (integer)
 * @param ldi Leading dimension of output matrix T
 */
__global__ void double_2int_array_kernel(int nrows, int ncols,
		                        double *F, int ldf,
                                        int *T, int ldi){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;
    
    // Bounds checking
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Convert double to integer with explicit casting
    T[idy*ldf+idx] = (int)F[idy*ldi+idx];
}

/**
 * @brief GPU wrapper function for double to integer conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Converts double precision matrix to integer.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (double precision)
 * @param ldf Leading dimension of input matrix F
 * @param T Output matrix (integer)
 * @param ldi Leading dimension of output matrix T
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void double_2int_array(int nrows, int ncols,
		    double *F, int ldf,
                    int *T, int ldi,
                    cudaStream_t stream){
    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    double_2int_array_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, T, ldi);
}
/****************************************************************************************************/

/**
 * @brief CUDA kernel for converting integer matrix to double precision
 * 
 * Converts integer matrix elements to double precision format.
 * Each thread processes one matrix element, converting from int to double.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input matrix (integer)
 * @param ldf Leading dimension of input matrix T
 * @param F Output matrix (double precision)
 * @param ldi Leading dimension of output matrix F
 */
__global__ void int2double_array_kernel(int nrows, int ncols,
                                        int *T, int ldf,
                                        double *F, int ldi){

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;
    
    // Bounds checking
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Convert integer to double with explicit casting
    F[idy*ldf+idx] = (double)T[idy*ldi+idx];
}

/**
 * @brief GPU wrapper function for integer to double conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Converts integer matrix to double precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input matrix (integer)
 * @param ldf Leading dimension of input matrix T
 * @param F Output matrix (double precision)
 * @param ldi Leading dimension of output matrix F
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void int2double_array(int nrows, int ncols,
                    int *T, int ldf,
                   double *F, int ldi,
                    cudaStream_t stream){

    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    int2double_array_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, T, ldf, F, ldi);
}
/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting double precision matrix to integer (unary operation)
 * 
 * Converts double precision matrix elements to integer format in-place.
 * Each thread processes one matrix element, converting from double to int.
 * This operation performs an in-place conversion by casting the double pointer
 * to int and then converting each element. Precision may be lost due to truncation.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input/output matrix (double precision, cast to int for output)
 * @param ldf Leading dimension of matrix F
 */
__global__ void double_2int_array_kernel_unary(int nrows, int ncols,
                                        double *F, int ldf){

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;
    
    // Cast double pointer to int for in-place conversion
    int *data_i = (int*) F;
    
    // Bounds checking
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Convert double to integer with explicit casting (truncation may occur)
    data_i[idy*ldf+idx] = (int)F[idy*ldf+idx];
}

/**
 * @brief GPU wrapper function for double to integer conversion (unary operation)
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Converts double precision matrix to integer in-place.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input/output matrix (double precision, cast to int for output)
 * @param ldf Leading dimension of matrix F
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void double_2int_array_unary(int nrows, int ncols,
                    double *F, int ldf,
                    cudaStream_t stream){

    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    double_2int_array_kernel_unary<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf);
}

/********************************************************************************/
/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting integer to single precision
 * 
 * Converts integer matrix elements to single precision (float) format.
 * Each thread processes one matrix element, converting from int to float.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input matrix (integer)
 * @param ldi Leading dimension of input matrix T
 * @param F Output matrix (single precision)
 * @param ldf Leading dimension of output matrix F
 */
__global__ void int_2float_array_kernel(int nrows, int ncols,
                                        int *T, int ldi,
                                        float *F, int ldf){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;
    
    // Bounds checking
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Convert integer to float with explicit casting
    F[idy*ldf+idx] = (float)T[idy*ldi+idx];
}

/**
 * @brief GPU wrapper function for integer to float conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Converts integer matrix to single precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input matrix (integer)
 * @param ldi Leading dimension of input matrix T
 * @param F Output matrix (single precision)
 * @param ldf Leading dimension of output matrix F
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void int_2float_array(int nrows, int ncols,
                    int *T, int ldi,
                    float *F, int ldf,
                    cudaStream_t stream){
    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    int_2float_array_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, T, ldi, F, ldf);
}
/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting integer matrix to single precision (unary operation)
 * 
 * Converts integer matrix elements to single precision (float) format in-place.
 * Each thread processes one matrix element, converting from int to float.
 * This operation performs an in-place conversion by casting the integer pointer
 * to float and then converting each element.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input/output matrix (integer, cast to float for output)
 * @param ldi Leading dimension of matrix T
 */
__global__ void int_2float_array_kernel_unary(int nrows, int ncols,
                                        int *T, int ldi){

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;
    
    // Cast integer pointer to float for in-place conversion
    float *data_s = (float *)T;
    
    // Bounds checking
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Convert integer to float with explicit casting
    data_s[idy*ldi+idx] = (float)T[idy*ldi+idx];
}

/**
 * @brief GPU wrapper function for integer to float conversion (unary operation)
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Converts integer matrix to single precision in-place.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input/output matrix (integer, cast to float for output)
 * @param ldi Leading dimension of matrix T
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void int_2float_array_unary(int nrows, int ncols,
                    int *T, int ldi,
                    cudaStream_t stream){

    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    int_2float_array_kernel_unary<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, T, ldi);
}
/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting single precision to integer
 * 
 * Converts single precision (float) matrix elements to integer format.
 * Each thread processes one matrix element, converting from float to int.
 * This operation may lose precision due to truncation.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (single precision)
 * @param ldf Leading dimension of input matrix F
 * @param T Output matrix (integer)
 * @param ldi Leading dimension of output matrix T
 */
__global__ void float_2int_array_kernel(int nrows, int ncols,
		                        float *F, int ldf,
                                        int *T, int ldi){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;
    
    // Bounds checking
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Convert float to integer with explicit casting
    T[idy*ldf+idx] = (int)F[idy*ldi+idx];
}

/**
 * @brief GPU wrapper function for float to integer conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Converts single precision matrix to integer.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input matrix (single precision)
 * @param ldf Leading dimension of input matrix F
 * @param T Output matrix (integer)
 * @param ldi Leading dimension of output matrix T
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void float_2int_array(int nrows, int ncols,
		    float *F, int ldf,
                    int *T, int ldi,
                    cudaStream_t stream){
    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    float_2int_array_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, T, ldi);
}
/****************************************************************************************************/

/**
 * @brief CUDA kernel for converting integer matrix to single precision
 * 
 * Converts integer matrix elements to single precision (float) format.
 * Each thread processes one matrix element, converting from int to float.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input matrix (integer)
 * @param ldf Leading dimension of input matrix T
 * @param F Output matrix (single precision)
 * @param ldi Leading dimension of output matrix F
 */
__global__ void int2float_array_kernel(int nrows, int ncols,
                                        int *T, int ldf,
                                        float *F, int ldi){

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;
    
    // Bounds checking
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Convert integer to float with explicit casting
    F[idy*ldf+idx] = (float)T[idy*ldi+idx];
}

/**
 * @brief GPU wrapper function for integer to float conversion
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Converts integer matrix to single precision.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input matrix (integer)
 * @param ldf Leading dimension of input matrix T
 * @param F Output matrix (single precision)
 * @param ldi Leading dimension of output matrix F
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void int2float_array(int nrows, int ncols,
                    int *T, int ldf,
                   float *F, int ldi,
                    cudaStream_t stream){

    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    int2float_array_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, T, ldf, F, ldi);
}
/****************************************************************************************************/
/**
 * @brief CUDA kernel for converting single precision matrix to integer (unary operation)
 * 
 * Converts single precision (float) matrix elements to integer format in-place.
 * Each thread processes one matrix element, converting from float to int.
 * This operation performs an in-place conversion by casting the float pointer
 * to int and then converting each element. Precision may be lost due to truncation.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input/output matrix (single precision, cast to int for output)
 * @param ldf Leading dimension of matrix F
 */
__global__ void float_2int_array_kernel_unary(int nrows, int ncols,
                                        float *F, int ldf){

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;
    
    // Cast float pointer to int for in-place conversion
    int *data_i = (int*) F;
    
    // Bounds checking
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Convert float to integer with explicit casting (truncation may occur)
    data_i[idy*ldf+idx] = (int)F[idy*ldf+idx];
}

/**
 * @brief GPU wrapper function for float to integer conversion (unary operation)
 * 
 * Launches the conversion kernel with appropriate grid and block dimensions.
 * Converts single precision matrix to integer in-place.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input/output matrix (single precision, cast to int for output)
 * @param ldf Leading dimension of matrix F
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void float_2int_array_unary(int nrows, int ncols,
                    float *F, int ldf,
                    cudaStream_t stream){

    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    float_2int_array_kernel_unary<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf);
}
/****************************************************************************************************/
/**
 * @brief CUDA kernel for computing Gaussian kernel matrix
 * 
 * Computes Gaussian kernel values for each matrix element using the formula:
 * exp(-distance/(2*gamma^2)). Optionally adds a diagonal offset for square matrices.
 * Each thread processes one matrix element.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param m Row dimension for diagonal check
 * @param n Column dimension for diagonal check
 * @param gamma Gaussian kernel parameter (bandwidth)
 * @param add_diag Value to add to diagonal elements if m == n
 * @param a Input/output matrix (float) - contains distances, outputs kernel values
 * @param lda Leading dimension of matrix a
 */
__global__ void gaussianKernel(int nrows, int ncols, int m, int n,
                                        float gamma, float add_diag, float *a, int lda){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    // Bounds checking
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Get distance value from input matrix
    double distance = a[idy*lda+idx];
    
    // Compute Gaussian kernel parameters
    double rad2 = (double)gamma * gamma;
    double value = (double)distance / (2 * rad2);
    
    // Apply Gaussian kernel transformation
    a[idy*lda+idx] = (float)exp(-value);
    
    // Add diagonal offset for square matrices if on diagonal
    if(m == n && idx == idy) {
        a[idx*lda+idx] += add_diag; 
    }
}

/**
 * @brief GPU wrapper function for Gaussian kernel computation
 * 
 * Launches the gaussianKernel on GPU with appropriate grid and block dimensions.
 * Computes Gaussian kernel values for distance matrices with optional diagonal offset.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param m Row dimension for diagonal check
 * @param n Column dimension for diagonal check
 * @param gamma Gaussian kernel parameter (bandwidth)
 * @param add_diag Value to add to diagonal elements if m == n
 * @param a Input/output matrix (float) - contains distances, outputs kernel values
 * @param lda Leading dimension of matrix a
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void gaussian_gpu(int nrows, int ncols, int m, int n,
                    float gamma, float add_diag, float *a, int lda,
                    cudaStream_t stream){
    // Debug print statements (commented out)
    // printf("%s %d %f %f\n", __FILE__, __LINE__, gamma, add_diag);

    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows+CHUNKSIZE-1)/CHUNKSIZE;
    int nBlocky = (ncols+CHUNKSIZE-1)/CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    gaussianKernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, m, n, gamma, add_diag, a, lda);
    
    // Debug print statements (commented out)
    // printf("%s %d %f %f\n", __FILE__, __LINE__, gamma, add_diag);
}
/****************************************************************************************************/
/****************************************************************************************************/
/**
 * @brief CUDA kernel for Hamming distance binary conversion
 * 
 * Converts input matrix to binary representation for Hamming distance computation.
 * Each thread processes one matrix element:
 * - If element equals target value: output = 0
 * - If element differs from target value: output = 1
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param aout Output matrix (int8_t) - binary representation
 * @param ain Input matrix (int8_t) - original values
 * @param lda Leading dimension of input matrix
 * @param value Target value for binary conversion
 */
__global__ void hamming_subtract_ones_kernel(int nrows, int ncols, int8_t *aout, int8_t *ain, int lda, int value) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    // Check bounds to ensure thread processes valid matrix element
    if (idx >= nrows || idy >= ncols) {
        return;
    }

    // Convert to binary: 0 if matches target, 1 if different
    if (value == ain[idy * lda + idx]) {
        aout[idy * lda + idx] = 0;  // Element matches target value
    } else {
        aout[idy * lda + idx] = 1;  // Element differs from target value
    }
}

/**
 * @brief GPU wrapper function for Hamming distance binary conversion
 * 
 * Launches the hamming_subtract_ones_kernel on GPU with appropriate grid and block dimensions.
 * Uses CHUNKSIZE×CHUNKSIZE thread blocks for optimal performance.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param aout Output matrix (int8_t) - binary representation
 * @param ain Input matrix (int8_t) - original values
 * @param lda Leading dimension of input matrix
 * @param value Target value for binary conversion
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void hicma_parsec_hamming_subtract_ones_GPU(int nrows, int ncols, int8_t *aout, int8_t *ain, int lda, int value, cudaStream_t stream) {
    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    hamming_subtract_ones_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, aout, ain, lda, value);
}
/****************************************************************************************************/

/****************************************************************************************************/
/**
 * @brief CUDA kernel for Hamming distance identity matrix generation
 * 
 * Creates a binary identity matrix for Hamming distance computation.
 * Each thread processes one matrix element:
 * - If element equals target value: output = 1
 * - If element differs from target value: output = 0
 * 
 * This is the inverse operation of hamming_subtract_ones_kernel.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param aout Output matrix (int8_t) - binary identity matrix
 * @param ain Input matrix (int8_t) - original values
 * @param lda Leading dimension of input matrix
 * @param value Target value for identity matching
 */
__global__ void hamming_get_id_matrix_kernel(int nrows, int ncols, int8_t *aout, int8_t *ain, int lda, int value) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    // Check bounds to ensure thread processes valid matrix element
    if (idx >= nrows || idy >= ncols) {
        return;
    }

    // Create identity matrix: 1 if matches target, 0 if different
    if (value == ain[idy * lda + idx]) {
        aout[idy * lda + idx] = 1;  // Element matches target value
    } else {
        aout[idy * lda + idx] = 0;  // Element differs from target value
    }
}

/**
 * @brief GPU wrapper function for Hamming distance identity matrix generation
 * 
 * Launches the hamming_get_id_matrix_kernel on GPU with appropriate grid and block dimensions.
 * Uses CHUNKSIZE×CHUNKSIZE thread blocks for optimal performance.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param aout Output matrix (int8_t) - binary identity matrix
 * @param ain Input matrix (int8_t) - original values
 * @param lda Leading dimension of input matrix
 * @param value Target value for identity matching
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void hicma_parsec_hamming_get_id_matrix_GPU(int nrows, int ncols, int8_t *aout, int8_t *ain, int lda, int value, cudaStream_t stream) {
    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    hamming_get_id_matrix_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, aout, ain, lda, value);
}
/****************************************************************************************************/


/****************************************************************************************************/
/**
 * @brief CUDA kernel for Hamming distance matrix merging
 * 
 * Merges two Hamming distance matrices by adding corresponding elements.
 * Each thread processes one matrix element, adding the transposed input matrix
 * to the output matrix. This operation is used to combine partial Hamming
 * distance computations or to accumulate distances from multiple binary conversions.
 * 
 * @param nrows Number of rows in the matrices
 * @param ncols Number of columns in the matrices
 * @param aout Output matrix (int) - accumulates the merged result
 * @param ain Input matrix (int) - values to add (transposed)
 * @param lda Leading dimension of both matrices
 */
__global__ void hamming_merge_kernel(int nrows, int ncols, int *aout, int *ain, int lda) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    // Check bounds to ensure thread processes valid matrix element
    if (idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Add transposed input matrix to output matrix
    // This accumulates Hamming distances from different binary conversions
    aout[idy * lda + idx] += ain[idx * lda + idy];
}

/**
 * @brief GPU wrapper function for Hamming distance matrix merging
 * 
 * Launches the hamming_merge_kernel on GPU with appropriate grid and block dimensions.
 * Uses CHUNKSIZE×CHUNKSIZE thread blocks for optimal performance.
 * 
 * @param nrows Number of rows in the matrices
 * @param ncols Number of columns in the matrices
 * @param aout Output matrix (int) - accumulates the merged result
 * @param ain Input matrix (int) - values to add (transposed)
 * @param lda Leading dimension of both matrices
 * @param stream CUDA stream for asynchronous execution
 */
extern "C"
void hicma_parsec_hamming_merge_GPU(int nrows, int ncols, int *aout, int *ain, int lda, cudaStream_t stream) {
    // Calculate grid dimensions based on matrix size and chunk size
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    hamming_merge_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, aout, ain, lda);
}
/****************************************************************************************************/

__global__ void store_float_as_double_kernel(double* dst, float x)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    dst[0] = (double)x;
  }
}

__global__ void store_double_kernel(double* dst, double x)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    dst[0] = x;
  }
}

__global__ void store_device_float_as_double_kernel(double* dst, const float* src)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    dst[0] = (double)src[0];
  }
}

// d_dst: device pointer to 1 double (cudaMalloc)
// x: host float
// stream: stream to enqueue the kernel (0 = default)
extern "C" void copy_float_to_device_double(double* d_dst, float x, cudaStream_t stream)
{
  if (!d_dst) return;
  store_float_as_double_kernel<<<1, 1, 0, stream>>>(d_dst, x);
}

extern "C" void set_device_double(double* d_dst, double x, cudaStream_t stream)
{
  if (!d_dst) return;
  store_double_kernel<<<1, 1, 0, stream>>>(d_dst, x);
}

extern "C" void copy_device_float_to_device_double(double* d_dst, const float* d_src, cudaStream_t stream)
{
  if (!d_dst || !d_src) return;
  store_device_float_as_double_kernel<<<1, 1, 0, stream>>>(d_dst, d_src);
}

/****************************************************************************************************/


#if 0
__global__ void dcmg_array_GPU_kernel(double *A, int m, int n, int m0,
        int n0, double* l1_x_cuda, double* l1_y_cuda, double* l2_x_cuda, double* l2_y_cuda,
        double localtheta0, double localtheta1, double localtheta2, int distance_metric, int lda)
{
	const int tx  = threadIdx.x;
	const int ty  = threadIdx.y;
	const int idx = blockIdx.x * blockDim.x + tx;
	const int idy = blockIdx.y * blockDim.y + ty;
	if(idx>=m || idy >=n){return;}

	//double x0, y0;
	double expr  = 0.0;
	double expr1 = 0.0;

	double sigma_square = localtheta0;// * localtheta[0];

	expr = sqrt(pow((l2_x_cuda[idx] - l1_x_cuda[idy]), 2) +
			pow((l2_y_cuda[idx] - l1_y_cuda[idy]), 2));

	expr1 = pow(expr, localtheta2);
	if(expr == 0)
		A[idx + idy * lda] = sigma_square /*+ 1e-4*/;
	else
		A[idx + idy * lda] = sigma_square *  exp(-(expr1/localtheta1)); // power-exp kernel



}

extern "C"
void dcmg_array_GPU( double *A, int m, int n, int m0,
		int n0, double* l1_x_cuda, double* l1_y_cuda, double* l2_x_cuda, double* l2_y_cuda,
		double *localtheta, int distance_metric, int lda, cudaStream_t stream){

	int nBlockx= (m+CHUNKSIZE-1)/CHUNKSIZE;
	int nBlocky= (n+CHUNKSIZE-1)/CHUNKSIZE;
	dim3 dimBlock(CHUNKSIZE,CHUNKSIZE);
	dim3 dimGrid(nBlockx,nBlocky);

	dcmg_array_GPU_kernel<<<dimGrid,dimBlock,0,stream>>>(A, m, n, m0, n0, l1_x_cuda, l1_y_cuda, l2_x_cuda, l2_y_cuda, localtheta[0],localtheta[1],localtheta[2], distance_metric, lda);
}

#endif
