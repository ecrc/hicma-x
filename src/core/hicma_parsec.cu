// CUDA runtime and library includes for GPU computation
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <stdlib.h>

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
 * @brief Keep the lower triangle of an FP32 tile in place (TRMM diagonal tile)
 *
 * Zeroes the strictly upper triangle of the n x n tile L (leading dimension
 * ldl) and, if unit is set, writes 1 on its diagonal, so that a GEMM with L
 * computes the lower triangular product.
 */
__global__ void hicma_parsec_tril_mask_float_GPU_kernel( int n, float *L, int ldl, int unit ) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;
        if( i >= n || j >= n || i > j ) { return; }
        if( i < j ) {
            L[(size_t)j * ldl + i] = 0.0f;
        } else if( unit ) {
            L[(size_t)j * ldl + i] = 1.0f;
        }
}

extern "C"
void hicma_parsec_tril_mask_float_GPU( int n, float *L, int ldl, int unit, cudaStream_t stream ) {
        int nBlock = (n + CHUNKSIZE - 1) / CHUNKSIZE;
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlock, nBlock);
        hicma_parsec_tril_mask_float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(n, L, ldl, unit);
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


__global__ void store_double_kernel(double* dst, double x)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    dst[0] = x;
  }
}


// d_dst: device pointer to 1 double (cudaMalloc)
// x: host float
// stream: stream to enqueue the kernel (0 = default)
extern "C" void set_device_double(double* d_dst, double x, cudaStream_t stream)
{
  if (!d_dst) return;
  store_double_kernel<<<1, 1, 0, stream>>>(d_dst, x);
}


/****************************************************************************************************/

