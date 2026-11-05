#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include "hicma_parsec_hip_cuda.h"

// Thread block size for 2D kernel launches - optimized for GPU memory coalescing
#define CHUNKSIZE 32

/****************************************************************************************************/
/**
 * @brief HIP kernel for converting integer array to float array in-place
 * 
 * This kernel performs type conversion from integer to float for matrix elements.
 * Each thread processes one matrix element, converting the integer value to float
 * and storing it back in the same memory location (in-place conversion).
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix  
 * @param T Input/output matrix (int*) - converted to float* internally
 * @param ldi Leading dimension of the matrix
 */
__global__ void int_2float_array_kernel_unary(int nrows, int ncols,
                                        int *T, int ldi){

    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;
    
    // Cast input matrix to float pointer for in-place conversion
    float *data_s = (float *)T;
    
    // Bounds checking - exit if thread is outside matrix dimensions
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Perform in-place type conversion: int -> float
    data_s[idy * ldi + idx] = (float)T[idy * ldi + idx];
}

/**
 * @brief HIP wrapper function for integer to float array conversion
 * 
 * Launches the int_2float_array_kernel_unary kernel on GPU with appropriate
 * grid and block dimensions. Uses CHUNKSIZE×CHUNKSIZE thread blocks for
 * optimal memory coalescing and performance.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param T Input/output matrix (int*) - converted in-place to float*
 * @param ldi Leading dimension of the matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void int_2float_array_unary(int nrows, int ncols,
                    int *T, int ldi,
                    hipStream_t stream){

    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    int_2float_array_kernel_unary<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, T, ldi);
}
/****************************************************************************************************/


/****************************************************************************************************/
/**
 * @brief HIP kernel for converting float array to integer array in-place
 * 
 * This kernel performs type conversion from float to integer for matrix elements.
 * Each thread processes one matrix element, converting the float value to integer
 * and storing it back in the same memory location (in-place conversion).
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input/output matrix (float*) - converted to int* internally
 * @param ldf Leading dimension of the matrix
 */
__global__ void float_2int_array_kernel_unary(int nrows, int ncols,
                                        float *F, int ldf){

    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;
    
    // Cast input matrix to int pointer for in-place conversion
    int *data_i = (int*) F;
    
    // Bounds checking - exit if thread is outside matrix dimensions
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Perform in-place type conversion: float -> int
    data_i[idy * ldf + idx] = (int)F[idy * ldf + idx];
}

/**
 * @brief HIP wrapper function for float to integer array conversion
 * 
 * Launches the float_2int_array_kernel_unary kernel on GPU with appropriate
 * grid and block dimensions. Uses CHUNKSIZE×CHUNKSIZE thread blocks for
 * optimal memory coalescing and performance.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input/output matrix (float*) - converted in-place to int*
 * @param ldf Leading dimension of the matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void float_2int_array_unary(int nrows, int ncols,
                    float *F, int ldf,
                    hipStream_t stream){

    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    float_2int_array_kernel_unary<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf);
}


/**
 * @brief Host function for converting single float to half precision
 * 
 * Converts a single float value to half precision (16-bit) format
 * and stores it in the provided memory location.
 * 
 * @param F Input float value to convert
 * @param _H Output memory location for half precision result
 */
extern "C"
void float2half_host(const float F, void *_H) {
    // Convert float to half precision
    __half tmph = __float2half(F);
    
    // Copy the half precision value to output memory
    memcpy(_H, &tmph, sizeof(tmph));
} 
/****************************************************************************************************/
/****************************************************************************************************/
/**
 * @brief HIP kernel for computing Gaussian kernel matrix
 * 
 * Computes a Gaussian kernel matrix where each element is calculated as:
 * K(i,j) = exp(-distance(i,j)^2 / (2 * gamma^2))
 * Additionally adds a diagonal term for square matrices.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param m Original matrix dimension (for diagonal check)
 * @param n Original matrix dimension (for diagonal check)
 * @param gamma Gaussian kernel bandwidth parameter
 * @param add_diag Value to add to diagonal elements
 * @param a Input/output matrix containing distances, converted to Gaussian kernel
 * @param lda Leading dimension of the matrix
 */
__global__ void gaussianKernel(int nrows, int ncols, int m, int n,
                                        float gamma, float add_diag, float *a, int lda){
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if(idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Get the distance value from input matrix
    double distance = a[idy * lda + idx];
    
    // Calculate Gaussian kernel parameters
    double rad2 = (double)gamma * gamma;
    double value = (double)distance / (2 * rad2);
    
    // Apply Gaussian kernel transformation: exp(-distance^2 / (2 * gamma^2))
    a[idy * lda + idx] = (float)exp(-value);
    
    // Add diagonal term for square matrices
    if(m == n && idx == idy) {
        a[idx * lda + idx] += add_diag;
    }
}

/**
 * @brief HIP wrapper function for Gaussian kernel computation
 * 
 * Launches the gaussianKernel on GPU with appropriate grid and block dimensions.
 * Uses CHUNKSIZE×CHUNKSIZE thread blocks for optimal performance.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param m Original matrix dimension (for diagonal check)
 * @param n Original matrix dimension (for diagonal check)
 * @param gamma Gaussian kernel bandwidth parameter
 * @param add_diag Value to add to diagonal elements
 * @param a Input/output matrix containing distances, converted to Gaussian kernel
 * @param lda Leading dimension of the matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void gaussian_gpu(int nrows, int ncols, int m, int n,
                    float gamma, float add_diag, float *a, int lda,
                    cudaStream_t stream){

    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    gaussianKernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, m, n, gamma, add_diag, a, lda);
}
/****************************************************************************************************/
/****************************************************************************************************/
/**
 * @brief HIP kernel for Hamming distance binary conversion
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
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

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
 * @brief HIP wrapper function for Hamming distance binary conversion
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
 * @param stream HIP stream for asynchronous execution
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
 * @brief HIP kernel for Hamming distance identity matrix generation
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
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

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
 * @brief HIP wrapper function for Hamming distance identity matrix generation
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
 * @param stream HIP stream for asynchronous execution
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
 * @brief HIP kernel for Hamming distance matrix merging
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
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Check bounds to ensure thread processes valid matrix element
    if (idx >= nrows || idy >= ncols) {
        return;
    }
    
    // Add transposed input matrix to output matrix
    // This accumulates Hamming distances from different binary conversions
    aout[idy * lda + idx] += ain[idx * lda + idy];
}

/**
 * @brief HIP wrapper function for Hamming distance matrix merging
 * 
 * Launches the hamming_merge_kernel on GPU with appropriate grid and block dimensions.
 * Uses CHUNKSIZE×CHUNKSIZE thread blocks for optimal performance.
 * 
 * @param nrows Number of rows in the matrices
 * @param ncols Number of columns in the matrices
 * @param aout Output matrix (int) - accumulates the merged result
 * @param ain Input matrix (int) - values to add (transposed)
 * @param lda Leading dimension of both matrices
 * @param stream HIP stream for asynchronous execution
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


/**
 * @brief HIP kernel for climate emulator matrix reshaping
 * 
 * Reshapes a complex matrix S into a real matrix T by extracting real and imaginary
 * parts. The kernel processes upper triangular elements and maps them to a specific
 * indexing scheme for the climate emulator application.
 * 
 * @param T Output real matrix (double*)
 * @param S Input complex matrix (hipDoubleComplex*)
 * @param L Matrix dimension
 * @param ldaS Leading dimension of input matrix S
 */
__global__ void climate_emulator_reshape_GPU_kernel( double *T, hipDoubleComplex *S,
        int L, int ldaS ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Process only upper triangular elements (idy <= idx) within bounds
    if( idx >= L || idy > idx ) { 
        return; 
    }

    // Extract real part and store in specific indexing pattern
    T[idx * idx + idx + idy] = hipCreal(S[idy * ldaS + idx]);
    
    // Extract imaginary part for off-diagonal elements
    if(idy != 0) {
        T[idx * idx + idx - idy] = hipCimag(S[idy * ldaS + idx]);
    }
}

/**
 * @brief HIP wrapper function for climate emulator matrix reshaping
 * 
 * Launches the climate_emulator_reshape_GPU_kernel on GPU with appropriate
 * grid and block dimensions for matrix reshaping operation.
 * 
 * @param T Output real matrix (double*)
 * @param S Input complex matrix (hipDoubleComplex*)
 * @param L Matrix dimension
 * @param ldaS Leading dimension of input matrix S
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void climate_emulator_reshape_GPU( double *T, hipDoubleComplex *S,
        int L, int ldaS,
        hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (L + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (L + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    climate_emulator_reshape_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(T, S, L, ldaS);
}
/****************************************************************************************************/

/****************************************************************************************************/
/**
 * @brief HIP kernel for printing complex matrix elements
 * 
 * Debug kernel that prints complex matrix elements to stdout. Each thread
 * prints one matrix element with its coordinates and real/imaginary parts.
 * Note: This is for debugging purposes only and should be used sparingly.
 * 
 * @param A Input complex matrix (hipDoubleComplex*)
 * @param M Number of rows in the matrix
 * @param N Number of columns in the matrix
 * @param lda Leading dimension of the matrix
 */
__global__ void climate_emulator_print_complex_GPU_kernel( hipDoubleComplex *A,
        int M, int N, int lda ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - note: uses > instead of >= for inclusive bounds
    if( idx > M || idy > N ) { 
        return; 
    }

    // Print matrix element with coordinates and complex value
    printf("%d %d : %lf %lf\n", idx, idy, hipCreal(A[idy * lda + idx]), hipCimag(A[idy * lda + idx]));
}

/**
 * @brief HIP wrapper function for printing complex matrix
 * 
 * Launches the climate_emulator_print_complex_GPU_kernel for debugging purposes.
 * Prints all elements of a complex matrix to stdout.
 * 
 * @param A Input complex matrix (hipDoubleComplex*)
 * @param M Number of rows in the matrix
 * @param N Number of columns in the matrix
 * @param lda Leading dimension of the matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void climate_emulator_print_complex_GPU( hipDoubleComplex *A,
        int M, int N, int lda,
        hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (M + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (N + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    climate_emulator_print_complex_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(A, M, N, lda);
}
/****************************************************************************************************/

/****************************************************************************************************/
/**
 * @brief HIP kernel for printing double matrix elements
 * 
 * Debug kernel that prints double matrix elements to stdout. Each thread
 * prints one matrix element with its coordinates and value.
 * Note: This is for debugging purposes only and should be used sparingly.
 * 
 * @param A Input double matrix (double*)
 * @param M Number of rows in the matrix
 * @param N Number of columns in the matrix
 * @param lda Leading dimension of the matrix
 */
__global__ void climate_emulator_print_double_GPU_kernel( double *A,
        int M, int N, int lda ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - note: uses > instead of >= for inclusive bounds
    if( idx > M || idy > N ) { 
        return; 
    }

    // Print matrix element with coordinates and value
    printf("%d %d : %lf\n", idx, idy, A[idy * lda + idx]);
}

/**
 * @brief HIP wrapper function for printing double matrix
 * 
 * Launches the climate_emulator_print_double_GPU_kernel for debugging purposes.
 * Prints all elements of a double matrix to stdout.
 * 
 * @param A Input double matrix (double*)
 * @param M Number of rows in the matrix
 * @param N Number of columns in the matrix
 * @param lda Leading dimension of the matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void climate_emulator_print_double_GPU( double *A,
        int M, int N, int lda,
        hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (M + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (N + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    climate_emulator_print_double_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(A, M, N, lda);
}
/****************************************************************************************************/


/****************************************************************************************************/
/**
 * @brief HIP kernel for converting double precision to bfloat16
 * 
 * Converts a double precision matrix to bfloat16 (Brain Floating Point) format.
 * Each thread processes one matrix element, converting from double to bfloat16
 * with proper rounding.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input double precision matrix (const double*)
 * @param ldf Leading dimension of input matrix
 * @param H Output bfloat16 matrix (hip_bfloat16*)
 * @param ldh Leading dimension of output matrix
 */
__global__ void double2bf_GPU_kernel( int nrows, int ncols,
                const double *F, int ldf,
                hip_bfloat16 *H, int ldh ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }
    
    // Convert double to bfloat16 with proper rounding
    H[idy * ldh + idx] = H[idy * ldh + idx].round_to_bfloat16( (float)(F[idy * ldf + idx]) );
}

/**
 * @brief HIP wrapper function for double to bfloat16 conversion
 * 
 * Launches the double2bf_GPU_kernel on GPU with appropriate grid and block dimensions.
 * Converts a double precision matrix to bfloat16 format for reduced memory usage.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input double precision matrix (const double*)
 * @param ldf Leading dimension of input matrix
 * @param _H Output bfloat16 matrix (void* - cast to hip_bfloat16*)
 * @param ldh Leading dimension of output matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void double2bf_GPU( int nrows, int ncols,
                const double *F, int ldf,
                void *_H, int ldh,
                hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Cast void pointer to bfloat16 pointer
    hip_bfloat16 *H = (hip_bfloat16 *)_H;
    
    // Launch kernel with calculated dimensions
    double2bf_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
/****************************************************************************************************/
/****************************************************************************************************/
/**
 * @brief HIP kernel for converting single precision to bfloat16
 * 
 * Converts a single precision (float) matrix to bfloat16 format.
 * Each thread processes one matrix element, converting from float to bfloat16
 * with proper rounding.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input single precision matrix (const float*)
 * @param ldf Leading dimension of input matrix
 * @param H Output bfloat16 matrix (hip_bfloat16*)
 * @param ldh Leading dimension of output matrix
 */
__global__ void float2bf_GPU_kernel( int nrows, int ncols,
                const float *F, int ldf,
                hip_bfloat16 *H, int ldh ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }
    
    // Convert float to bfloat16 with proper rounding
    H[idy * ldh + idx] = H[idy * ldh + idx].round_to_bfloat16( F[idy * ldf + idx] );
}

/**
 * @brief HIP wrapper function for float to bfloat16 conversion
 * 
 * Launches the float2bf_GPU_kernel on GPU with appropriate grid and block dimensions.
 * Converts a single precision matrix to bfloat16 format for reduced memory usage.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input single precision matrix (const float*)
 * @param ldf Leading dimension of input matrix
 * @param _H Output bfloat16 matrix (void* - cast to hip_bfloat16*)
 * @param ldh Leading dimension of output matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void float2bf_GPU( int nrows, int ncols,
                const float *F, int ldf,
                void *_H, int ldh,
                hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Cast void pointer to bfloat16 pointer
    hip_bfloat16 *H = (hip_bfloat16 *)_H;
    
    // Launch kernel with calculated dimensions
    float2bf_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
/****************************************************************************************************/

/**
 * @brief HIP kernel for converting bfloat16 to single precision
 * 
 * Converts a bfloat16 matrix to single precision (float) format.
 * Each thread processes one matrix element, converting from bfloat16 to float.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param H Input bfloat16 matrix (const hip_bfloat16*)
 * @param ldh Leading dimension of input matrix
 * @param F Output single precision matrix (float*)
 * @param ldf Leading dimension of output matrix
 */
__global__ void bf2float_GPU_kernel( int nrows, int ncols,
                const hip_bfloat16 *H, int ldh,
                float *F, int ldf ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }

    // Convert bfloat16 to float
    F[idy * ldf + idx] = (float)( H[idy * ldh + idx] );
}

/**
 * @brief HIP wrapper function for bfloat16 to float conversion
 * 
 * Launches the bf2float_GPU_kernel on GPU with appropriate grid and block dimensions.
 * Converts a bfloat16 matrix to single precision format.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _H Input bfloat16 matrix (const void* - cast to hip_bfloat16*)
 * @param ldh Leading dimension of input matrix
 * @param F Output single precision matrix (float*)
 * @param ldf Leading dimension of output matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void bf2float_GPU( int nrows, int ncols,
                const void *_H, int ldh,
                float *F, int ldf,
                hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Cast void pointer to bfloat16 pointer
    hip_bfloat16 *H = (hip_bfloat16 *)_H;
    
    // Launch kernel with calculated dimensions
    bf2float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, H, ldh, F, ldf);
}
/****************************************************************************************************/

/**
 * @brief HIP kernel for copying bfloat16 matrix
 * 
 * Copies elements from source bfloat16 matrix to destination bfloat16 matrix.
 * Each thread processes one matrix element, performing element-wise copy operation.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param src Source bfloat16 matrix (hip_bfloat16*)
 * @param dest Destination bfloat16 matrix (hip_bfloat16*)
 */
__global__ void memcpy_bf_GPU_kernel( int nrows, int ncols,
                hip_bfloat16 *src, hip_bfloat16 *dest ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }

    // Copy element from source to destination
    dest[idy * nrows + idx] = src[idy * nrows + idx];
}

/**
 * @brief HIP wrapper function for bfloat16 matrix copy
 * 
 * Launches the memcpy_bf_GPU_kernel on GPU with appropriate grid and block dimensions.
 * Performs element-wise copy of bfloat16 matrix from source to destination.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _src Source bfloat16 matrix (void* - cast to hip_bfloat16*)
 * @param _dest Destination bfloat16 matrix (void* - cast to hip_bfloat16*)
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void memcpy_bf_GPU( int nrows, int ncols, void *_src, void *_dest, hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Cast void pointers to bfloat16 pointers
    hip_bfloat16 *src = (hip_bfloat16 *)_src;
    hip_bfloat16 *dest = (hip_bfloat16 *)_dest;
    
    // Launch kernel with calculated dimensions
    memcpy_bf_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, dest);
}
/****************************************************************************************************/
/****************************************************************************************************/

/**
 * @brief HIP kernel for converting single precision to double precision
 * 
 * Converts a single precision (float) matrix to double precision format.
 * Each thread processes one matrix element, converting from float to double.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input single precision matrix (const float*)
 * @param ldf Leading dimension of input matrix
 * @param D Output double precision matrix (double*)
 * @param ldh Leading dimension of output matrix
 */
__global__ void float2double_GPU_kernel( int nrows, int ncols,
                const float *F, int ldf,
                double *D, int ldh ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }

    // Convert float to double
    D[idy * ldh + idx] = (double)F[idy * ldf + idx];
}

/**
 * @brief HIP wrapper function for float to double conversion
 * 
 * Launches the float2double_GPU_kernel on GPU with appropriate grid and block dimensions.
 * Converts a single precision matrix to double precision format.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input single precision matrix (const float*)
 * @param ldf Leading dimension of input matrix
 * @param D Output double precision matrix (double*)
 * @param ldh Leading dimension of output matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void float2double_GPU( int nrows, int ncols,
                const float *F, int ldf,
                double *D, int ldh,
                hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    float2double_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, D, ldh);
}

/****************************************************************************************************/

/**
 * @brief HIP kernel for converting double precision to single precision
 * 
 * Converts a double precision matrix to single precision (float) format.
 * Each thread processes one matrix element, converting from double to float
 * with proper rounding using __double2float_rn.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param D Input double precision matrix (const double*)
 * @param ldh Leading dimension of input matrix
 * @param F Output single precision matrix (float*)
 * @param ldf Leading dimension of output matrix
 */
__global__ void double2float_GPU_kernel( int nrows, int ncols,
                const double *D, int ldh,
                float *F, int ldf ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }

    // Convert double to float with proper rounding
    F[idy * ldf + idx] = __double2float_rn( D[idy * ldh + idx] ); 
}

/**
 * @brief HIP wrapper function for double to float conversion
 * 
 * Launches the double2float_GPU_kernel on GPU with appropriate grid and block dimensions.
 * Converts a double precision matrix to single precision format with proper rounding.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param D Input double precision matrix (const double*)
 * @param ldh Leading dimension of input matrix
 * @param F Output single precision matrix (float*)
 * @param ldf Leading dimension of output matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void double2float_GPU( int nrows, int ncols,
                const double *D, int ldh,
                float *F, int ldf,
                hipStream_t stream ) {

    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Launch kernel with calculated dimensions
    double2float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, D, ldh, F, ldf);
}

/****************************************************************************************************/

/**
 * @brief HIP kernel for converting half precision to single precision
 * 
 * Converts a half precision (__half) matrix to single precision (float) format.
 * Each thread processes one matrix element, converting from half to float.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param H Input half precision matrix (const __half*)
 * @param ldh Leading dimension of input matrix
 * @param F Output single precision matrix (float*)
 * @param ldf Leading dimension of output matrix
 */
__global__ void half2float_GPU_kernel( int nrows, int ncols,
                const __half *H, int ldh,
                float *F, int ldf ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }

    // Convert half to float
    F[idy * ldf + idx] = __half2float( H[idy * ldh + idx] );
}

/**
 * @brief HIP wrapper function for half to float conversion
 * 
 * Launches the half2float_GPU_kernel on GPU with appropriate grid and block dimensions.
 * Converts a half precision matrix to single precision format.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _H Input half precision matrix (const void* - cast to __half*)
 * @param ldh Leading dimension of input matrix
 * @param F Output single precision matrix (float*)
 * @param ldf Leading dimension of output matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void half2float_GPU( int nrows, int ncols,
                const void *_H, int ldh,
                float *F, int ldf,
                hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Cast void pointer to half pointer
    __half *H = (__half *)_H;
    
    // Launch kernel with calculated dimensions
    half2float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, H, ldh, F, ldf);
}

/****************************************************************************************************/

/**
 * @brief HIP kernel for converting single precision to half precision
 * 
 * Converts a single precision (float) matrix to half precision (__half) format.
 * Each thread processes one matrix element, converting from float to half
 * with proper rounding using __float2half_rn.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input single precision matrix (const float*)
 * @param ldf Leading dimension of input matrix
 * @param H Output half precision matrix (__half*)
 * @param ldh Leading dimension of output matrix
 */
__global__ void float2half_GPU_kernel( int nrows, int ncols,
                const float *F, int ldf,
                __half *H, int ldh ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }

    // Convert float to half with proper rounding
    H[idy * ldh + idx] = __float2half_rn( F[idy * ldf + idx] );
}

/**
 * @brief HIP wrapper function for float to half conversion
 * 
 * Launches the float2half_GPU_kernel on GPU with appropriate grid and block dimensions.
 * Converts a single precision matrix to half precision format with proper rounding.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input single precision matrix (const float*)
 * @param ldf Leading dimension of input matrix
 * @param _H Output half precision matrix (void* - cast to __half*)
 * @param ldh Leading dimension of output matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void float2half_GPU( int nrows, int ncols,
                const float *F, int ldf,
                void *_H, int ldh,
                hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Cast void pointer to half pointer
    __half *H = (__half *)_H;
    
    // Launch kernel with calculated dimensions
    float2half_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}


/****************************************************************************************************/

/**
 * @brief HIP kernel for converting half precision to double precision
 * 
 * Converts a half precision (__half) matrix to double precision format.
 * Each thread processes one matrix element, converting from half to double.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param H Input half precision matrix (const __half*)
 * @param ldh Leading dimension of input matrix
 * @param F Output double precision matrix (double*)
 * @param ldf Leading dimension of output matrix
 */
__global__ void half2double_GPU_kernel( int nrows, int ncols,
                const __half *H, int ldh,
                double *F, int ldf ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }

    // Convert half to double
    F[idy * ldf + idx] = (double)( H[idy * ldh + idx] );
}

/**
 * @brief HIP wrapper function for half to double conversion
 * 
 * Launches the half2double_GPU_kernel on GPU with appropriate grid and block dimensions.
 * Converts a half precision matrix to double precision format.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _H Input half precision matrix (const void* - cast to __half*)
 * @param ldh Leading dimension of input matrix
 * @param F Output double precision matrix (double*)
 * @param ldf Leading dimension of output matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void half2double_GPU( int nrows, int ncols,
                const void *_H, int ldh,
                double *F, int ldf,
                hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Cast void pointer to half pointer
    __half *H = (__half *)_H;
    
    // Launch kernel with calculated dimensions
    half2double_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, H, ldh, F, ldf);
}

/****************************************************************************************************/

/**
 * @brief HIP kernel for converting double precision to half precision
 * 
 * Converts a double precision matrix to half precision (__half) format.
 * Each thread processes one matrix element, converting from double to half
 * via intermediate float conversion for better precision control.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input double precision matrix (const double*)
 * @param ldf Leading dimension of input matrix
 * @param H Output half precision matrix (__half*)
 * @param ldh Leading dimension of output matrix
 */
__global__ void double2half_GPU_kernel( int nrows, int ncols,
                const double *F, int ldf,
                __half *H, int ldh ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }

    // Convert double to half via float for better precision control
    // Note: Direct __double2half may not be available, so use float intermediate
    H[idy * ldh + idx] = __float2half( (float)(F[idy * ldf + idx]) );
}

/**
 * @brief HIP wrapper function for double to half conversion
 * 
 * Launches the double2half_GPU_kernel on GPU with appropriate grid and block dimensions.
 * Converts a double precision matrix to half precision format via float intermediate.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input double precision matrix (const double*)
 * @param ldf Leading dimension of input matrix
 * @param _H Output half precision matrix (void* - cast to __half*)
 * @param ldh Leading dimension of output matrix
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void double2half_GPU( int nrows, int ncols,
                const double *F, int ldf,
                void *_H, int ldh,
                hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Cast void pointer to half pointer
    __half *H = (__half *)_H;
    
    // Launch kernel with calculated dimensions
    double2half_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}


/****************************************************************************************************/

/**
 * @brief HIP kernel for copying half precision matrix
 * 
 * Copies elements from source half precision matrix to destination half precision matrix.
 * Each thread processes one matrix element, performing element-wise copy operation.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param src Source half precision matrix (__half*)
 * @param dest Destination half precision matrix (__half*)
 */
__global__ void memcpy_half_GPU_kernel( int nrows, int ncols,
                __half *src, __half *dest ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }

    // Copy element from source to destination
    dest[idy * nrows + idx] = src[idy * nrows + idx];
}

/**
 * @brief HIP wrapper function for half precision matrix copy
 * 
 * Launches the memcpy_half_GPU_kernel on GPU with appropriate grid and block dimensions.
 * Performs element-wise copy of half precision matrix from source to destination.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _src Source half precision matrix (void* - cast to __half*)
 * @param _dest Destination half precision matrix (void* - cast to __half*)
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void memcpy_half_GPU( int nrows, int ncols, void *_src, void *_dest, hipStream_t stream ) { 
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Cast void pointers to half pointers
    __half *src = (__half *)_src;
    __half *dest = (__half *)_dest;
    
    // Launch kernel with calculated dimensions
    memcpy_half_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, dest);
}

/****************************************************************************************************/

/**
 * @brief HIP kernel for copying double precision matrix data
 */
__global__ void memcpy_double_GPU_kernel( int nrows, int ncols,
                const double *src, double *dest ) {
    const int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int idy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if( idx >= nrows || idy >= ncols ) {
        return;
    }

    dest[idy * nrows + idx] = src[idy * nrows + idx];
}

extern "C"
void memcpy_double_GPU( int nrows, int ncols, void *_src, void *_dest,
                        hipStream_t stream ) {
    const dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    const dim3 dimGrid((nrows + CHUNKSIZE - 1) / CHUNKSIZE,
                       (ncols + CHUNKSIZE - 1) / CHUNKSIZE);

    const double *src = (const double *)_src;
    double *dest = (double *)_dest;

    memcpy_double_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, dest);
}

/****************************************************************************************************/

/**
 * @brief HIP kernel for subtracting a float matrix from a double matrix
 */
__global__ void sub_float_from_double_GPU_kernel( int nrows, int ncols,
                const float *src, double *dest ) {
    const int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int idy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if( idx >= nrows || idy >= ncols ) {
        return;
    }

    dest[idy * nrows + idx] -= (double)src[idy * nrows + idx];
}

extern "C"
void sub_float_from_double_GPU( int nrows, int ncols, void *_src, void *_dest,
                                hipStream_t stream ) {
    const dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    const dim3 dimGrid((nrows + CHUNKSIZE - 1) / CHUNKSIZE,
                       (ncols + CHUNKSIZE - 1) / CHUNKSIZE);

    const float *src = (const float *)_src;
    double *dest = (double *)_dest;

    sub_float_from_double_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, dest);
}

/****************************************************************************************************/

/**
 * @brief HIP kernel for copying single precision matrix
 * 
 * Copies elements from source single precision matrix to destination single precision matrix.
 * Each thread processes one matrix element, performing element-wise copy operation.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param src Source single precision matrix (float*)
 * @param dest Destination single precision matrix (float*)
 */
__global__ void memcpy_float_GPU_kernel( int nrows, int ncols,
               float *src, float *dest ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;
                           
    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }
                           
    // Copy element from source to destination
    dest[idy * nrows + idx] = src[idy * nrows + idx];
}                          
                           
/**
 * @brief HIP wrapper function for single precision matrix copy
 * 
 * Launches the memcpy_float_GPU_kernel on GPU with appropriate grid and block dimensions.
 * Performs element-wise copy of single precision matrix from source to destination.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param _src Source single precision matrix (void* - cast to float*)
 * @param _dest Destination single precision matrix (void* - cast to float*)
 * @param stream HIP stream for asynchronous execution
 */
extern "C"                 
void memcpy_float_GPU( int nrows, int ncols, void *_src, void *_dest, hipStream_t stream ) {                         
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    
    // Cast void pointers to float pointers
    float *src = (float *)_src;
    float *dest = (float *)_dest;
    
    // Launch kernel with calculated dimensions
    memcpy_float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, src, dest);
}

/****************************************************************************************************/

/**
 * @brief HIP kernel for printing single precision matrix elements
 * 
 * Debug kernel that prints single precision matrix elements to stdout. Each thread
 * prints one matrix element with its coordinates and value.
 * Note: This is for debugging purposes only and should be used sparingly.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param A Input single precision matrix (float*)
 */
__global__ void matrix_print_float_GPU_kernel( int nrows, int ncols, float *A ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }

    // Print matrix element with coordinates and value
    printf("SINGLE_PRINT %d %d : %g\n", idx, idy, A[idy * nrows + idx]);
}

/**
 * @brief HIP wrapper function for printing single precision matrix
 * 
 * Launches the matrix_print_float_GPU_kernel for debugging purposes.
 * Prints all elements of a single precision matrix to stdout.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param A Input single precision matrix (float*)
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void matrix_print_float_GPU( int nrows, int ncols, float *A, hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);

    // Launch kernel with calculated dimensions
    matrix_print_float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, A);
}

/****************************************************************************************************/

/**
 * @brief HIP kernel for printing double precision matrix elements
 * 
 * Debug kernel that prints double precision matrix elements to stdout. Each thread
 * prints one matrix element with its coordinates and value.
 * Note: This is for debugging purposes only and should be used sparingly.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param A Input double precision matrix (double*)
 */
__global__ void matrix_print_double_GPU_kernel( int nrows, int ncols, double *A ) {
    // Get thread and block indices for 2D grid
    const int tx = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
    const int idy = hipBlockIdx_y * hipBlockDim_y + ty;

    // Bounds checking - exit if thread is outside matrix dimensions
    if( idx >= nrows || idy >= ncols ) { 
        return; 
    }

    // Print matrix element with coordinates and value
    printf("DOUBLE_PRINT %d %d : %g\n", idx, idy, A[idy * nrows + idx]);
}

/**
 * @brief HIP wrapper function for printing double precision matrix
 * 
 * Launches the matrix_print_double_GPU_kernel for debugging purposes.
 * Prints all elements of a double precision matrix to stdout.
 * 
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param A Input double precision matrix (double*)
 * @param stream HIP stream for asynchronous execution
 */
extern "C"
void matrix_print_double_GPU( int nrows, int ncols, double *A, hipStream_t stream ) {
    // Calculate number of thread blocks needed to cover the matrix
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    
    // Define thread block and grid dimensions
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);

    // Launch kernel with calculated dimensions
    matrix_print_double_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, A);
}

/****************************************************************************************************/
/**
 * @brief HIP wrapper function for mixed-precision GEMM operation
 * 
 * Custom wrapper for hipblasGemmEx that handles mixed-precision matrix multiplication.
 * Converts float alpha and beta parameters to half precision for the GEMM operation.
 * 
 * Note: There is a known issue with NaN results when matrix size is large.
 * TODO: Investigate and fix the NaN issue for large matrices.
 * 
 * @param handle HIPBLAS handle
 * @param transa Operation on matrix A (transpose, conjugate, etc.)
 * @param transb Operation on matrix B (transpose, conjugate, etc.)
 * @param m Number of rows of matrix A and C
 * @param n Number of columns of matrix B and C
 * @param k Number of columns of matrix A and rows of matrix B
 * @param alpha Scalar multiplier for matrix A*B (const void* - cast to float*)
 * @param A Input matrix A (const void*)
 * @param Atype Data type of matrix A
 * @param lda Leading dimension of matrix A
 * @param B Input matrix B (const void*)
 * @param Btype Data type of matrix B
 * @param ldb Leading dimension of matrix B
 * @param beta Scalar multiplier for matrix C (const void* - cast to float*)
 * @param C Input/output matrix C (void*)
 * @param Ctype Data type of matrix C
 * @param ldc Leading dimension of matrix C
 * @param computeType Compute precision for the operation
 * @param algo Algorithm to use for the GEMM operation
 * @return HIPBLAS status code
 */
extern "C"
hipblasStatus_t my_cublasGemmEx(hipblasHandle_t handle,
                           hipblasOperation_t transa,
                           hipblasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const void    *alpha,
                           const void     *A, 
                           hipblasDatatype_t Atype,
                           int lda,
                           const void     *B,
                           hipblasDatatype_t Btype,  
                           int ldb,
                           const void    *beta, 
                           void           *C,
                           hipblasDatatype_t Ctype,
                           int ldc,
                           hipblasDatatype_t computeType,
                           hipblasGemmAlgo_t algo)
{
    // Convert float alpha and beta to half precision
    __half alpha_h = (__half)(((float *)alpha)[0]);
    __half beta_h = (__half)(((float *)beta)[0]);
    
    // Call HIPBLAS GEMM with converted parameters
    return hipblasGemmEx(handle, transa, transb, m, n, k,  
                        &alpha_h, A, Atype, lda,
                                  B, Btype, ldb,
                        &beta_h,  C, Ctype, ldc,
                        computeType, algo);
}

/****************************************************************************************************/

__global__ void store_float_as_double_kernel(double* dst, float x)
{
  if (hipThreadIdx_x == 0 && hipBlockIdx_x == 0) {
    dst[0] = (double)x;
  }
}

__global__ void store_double_kernel(double* dst, double x)
{
  if (hipThreadIdx_x == 0 && hipBlockIdx_x == 0) {
    dst[0] = x;
  }
}

__global__ void store_device_float_as_double_kernel(double* dst, const float* src)
{
  if (hipThreadIdx_x == 0 && hipBlockIdx_x == 0) {
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
	const int tx  = hipThreadIdx_x;
	const int ty  = hipThreadIdx_y;
	const int idx = hipBlockIdx_x * hipBlockDim_x + tx;
	const int idy = hipBlockIdx_y * hipBlockDim_y + ty;
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
		double *localtheta, int distance_metric, int lda, hipStream_t stream){

	int nBlockx= (m+CHUNKSIZE-1)/CHUNKSIZE;
	int nBlocky= (n+CHUNKSIZE-1)/CHUNKSIZE;
	dim3 dimBlock(CHUNKSIZE,CHUNKSIZE);
	dim3 dimGrid(nBlockx,nBlocky);

	dcmg_array_GPU_kernel<<<dimGrid,dimBlock,0,stream>>>(A, m, n, m0, n0, l1_x_cuda, l1_y_cuda, l2_x_cuda, l2_y_cuda, localtheta[0],localtheta[1],localtheta[2], distance_metric, lda);
}

#endif
