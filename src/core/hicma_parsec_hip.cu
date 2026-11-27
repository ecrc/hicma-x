#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include "hicma_parsec_hip_cuda.h"

// Thread block size for 2D kernel launches - optimized for GPU memory coalescing
#define CHUNKSIZE 32


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

/*
 * H = half(scale * F) for FP64 or FP32 F. The FP16 matrix instructions of
 * gfx90a flush FP16 subnormal inputs (|x| < 6.1e-5) to zero; callers pass a
 * power of two (exact) that keeps the operand in the normal range and divide
 * the GEMM's alpha by it.
 */


/**
 * @brief Keep the lower triangle of an FP32 tile in place (TRMM diagonal tile)
 *
 * Zeroes the strictly upper triangle of the n x n tile L (leading dimension
 * ldl) and, if unit is set, writes 1 on its diagonal, so that a GEMM with L
 * computes the lower triangular product.
 */
__global__ void hicma_parsec_tril_mask_float_GPU_kernel( int n, float *L, int ldl, int unit ) {
        const int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        const int j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
        if( i >= n || j >= n || i > j ) { return; }
        if( i < j ) {
            L[(size_t)j * ldl + i] = 0.0f;
        } else if( unit ) {
            L[(size_t)j * ldl + i] = 1.0f;
        }
}

extern "C"
void hicma_parsec_tril_mask_float_GPU( int n, float *L, int ldl, int unit, hipStream_t stream ) {
        int nBlock = (n + CHUNKSIZE - 1) / CHUNKSIZE;
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlock, nBlock);
        hicma_parsec_tril_mask_float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(n, L, ldl, unit);
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

                           


__global__ void store_double_kernel(double* dst, double x)
{
  if (hipThreadIdx_x == 0 && hipBlockIdx_x == 0) {
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


/****************************************************************************************************/
/* Lower Cholesky of a tile without rocSOLVER (hicma_potrf_lower_hip_d/_s)
 *
 * rocSOLVER's ?potrf factors its diagonal blocks (127 columns in double, 180 in single) with the
 * potf2 kernel of roclapack_potf2_specialized_kernels.hpp (potf2_simple). There every thread reads
 * the pivot A(k,k) from shared memory and thread 0 then overwrites it with its square root before
 * the first barrier, so a wave that reads late scales its part of column k by the fourth root of the
 * pivot, and info stays 0. It is in ROCm 6.3 (rocSOLVER 3.27) and later releases; upstream added the
 * barrier in March 2026 (ROCm/rocm-libraries#3794), and ROCm 6.4 and later no longer support gfx906.
 * The waves drift apart when other kernels run on the GPU, which the tile Cholesky always does
 * (TRSMs, SYRKs and GEMMs on the other streams): about one POTRF in a thousand was wrong on a
 * Radeon VII, and the factorization with it, now and then without failing its check.
 *
 * Here a single workgroup factors each block of HICMA_POTF2_NB columns in shared memory, with a
 * barrier between reading the pivot and writing it, and hipBLAS does the TRSM and SYRK in between.
 * Same contract as rocsolver_?potrf (lower, info on the device: 0, or the 1-based column of the
 * first pivot that is not positive), on the stream of the hipBLAS handle. */
#define HICMA_POTF2_NB      64
#define HICMA_POTF2_THREADS 256

template <typename T>
__global__ void hicma_potf2_lower_kernel(int n, T *A, int lda, int *info, int offset)
{
    __shared__ T s[HICMA_POTF2_NB * HICMA_POTF2_NB];
    const int tid = hipThreadIdx_x, nt = hipBlockDim_x;

    for( int idx = tid; idx < n * n; idx += nt ) {
        const int i = idx % n, j = idx / n;
        s[i + j * HICMA_POTF2_NB] = (i >= j) ? A[i + (size_t)j * lda] : T(0);
    }
    __syncthreads();

    for( int k = 0; k < n; k++ ) {
        const T akk = s[k + k * HICMA_POTF2_NB];
        __syncthreads();   /* every thread has read the pivot before thread 0 replaces it */
        if( !(akk > T(0)) || !isfinite(akk) ) {
            /* every thread read the same value: they all leave here */
            if( 0 == tid && 0 == *info ) *info = offset + k + 1;
            return;        /* the block keeps its input values; the factorization has failed (info) */
        }
        const T lkk = sqrt(akk);
        for( int i = k + 1 + tid; i < n; i += nt ) s[i + k * HICMA_POTF2_NB] /= lkk;
        if( 0 == tid ) s[k + k * HICMA_POTF2_NB] = lkk;
        __syncthreads();
        const int m = n - k - 1;
        for( int idx = tid; idx < m * m; idx += nt ) {
            const int i = k + 1 + idx % m, j = k + 1 + idx / m;
            if( i >= j ) s[i + j * HICMA_POTF2_NB] -= s[i + k * HICMA_POTF2_NB] * s[j + k * HICMA_POTF2_NB];
        }
        __syncthreads();
    }

    for( int idx = tid; idx < n * n; idx += nt ) {
        const int i = idx % n, j = idx / n;
        if( i >= j ) A[i + (size_t)j * lda] = s[i + j * HICMA_POTF2_NB];
    }
}

static hipblasStatus_t hicma_potrf_trsm(hipblasHandle_t h, int m, int n, const double *L, int lda, double *B)
{
    const double one = 1.0;
    return hipblasDtrsm(h, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, m, n, &one, L, lda, B, lda);
}
static hipblasStatus_t hicma_potrf_trsm(hipblasHandle_t h, int m, int n, const float *L, int lda, float *B)
{
    const float one = 1.0f;
    return hipblasStrsm(h, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, m, n, &one, L, lda, B, lda);
}
static hipblasStatus_t hicma_potrf_syrk(hipblasHandle_t h, int n, int k, const double *A, int lda, double *C)
{
    const double one = 1.0, mone = -1.0;
    return hipblasDsyrk(h, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_N, n, k, &mone, A, lda, &one, C, lda);
}
static hipblasStatus_t hicma_potrf_syrk(hipblasHandle_t h, int n, int k, const float *A, int lda, float *C)
{
    const float one = 1.0f, mone = -1.0f;
    return hipblasSsyrk(h, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_N, n, k, &mone, A, lda, &one, C, lda);
}

template <typename T>
static hipblasStatus_t hicma_potrf_lower_hip(hipblasHandle_t handle, int n, T *A, int lda, int *info)
{
    hipStream_t stream = NULL;
    hipblasStatus_t st = hipblasGetStream(handle, &stream);
    if( HIPBLAS_STATUS_SUCCESS != st ) return st;
    if( hipSuccess != hipMemsetAsync(info, 0, sizeof(int), stream) ) return HIPBLAS_STATUS_EXECUTION_FAILED;
    for( int j = 0; j < n; j += HICMA_POTF2_NB ) {
        const int jb = (n - j < HICMA_POTF2_NB) ? n - j : HICMA_POTF2_NB;
        const int r = n - j - jb;
        T *Ajj = A + j + (size_t)j * lda;
        hipLaunchKernelGGL(hicma_potf2_lower_kernel<T>, dim3(1), dim3(HICMA_POTF2_THREADS), 0, stream, jb, Ajj, lda, info, j);
        if( hipSuccess != hipGetLastError() ) return HIPBLAS_STATUS_EXECUTION_FAILED;
        if( r <= 0 ) break;
        T *Arj = Ajj + jb, *Arr = Ajj + jb + (size_t)jb * lda;
        st = hicma_potrf_trsm(handle, r, jb, Ajj, lda, Arj);
        if( HIPBLAS_STATUS_SUCCESS != st ) return st;
        st = hicma_potrf_syrk(handle, r, jb, Arj, lda, Arr);
        if( HIPBLAS_STATUS_SUCCESS != st ) return st;
    }
    return HIPBLAS_STATUS_SUCCESS;
}

extern "C" hipblasStatus_t hicma_potrf_lower_hip_d(hipblasHandle_t handle, int n, double *A, int lda, int *info)
{
    return hicma_potrf_lower_hip<double>(handle, n, A, lda, info);
}

extern "C" hipblasStatus_t hicma_potrf_lower_hip_s(hipblasHandle_t handle, int n, float *A, int lda, int *info)
{
    return hicma_potrf_lower_hip<float>(handle, n, A, lda, info);
}
