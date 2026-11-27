/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 * 
 * @version 0.2.0
 */

#ifndef HICMA_PARSEC_GPU_H
#define HICMA_PARSEC_GPU_H

/* ============================================================================
 * HICMA PaRSEC includes
 * ============================================================================ */
#include "hicma_parsec_internal.h"

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * GPU architecture and capability functions
 * ============================================================================ */

/**
 * @brief Check GPU architecture compatibility
 * 
 * Examines the GPU device properties to determine the architecture type
 * and sets the appropriate GPU type in the parameters structure.
 * Supports detection of V100, A100, and H100 architectures.
 * 
 * @param[in] params HICMA PaRSEC parameters
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_check_gpu_arch( hicma_parsec_params_t *params );

/* ============================================================================
 * GPU enable/disable functions
 * ============================================================================ */

/**
 * @brief Disable GPU for computation
 * 
 * Disables GPU acceleration for the specified task pool, forcing all
 * computation to run on CPU. This is useful for debugging or when
 * GPU resources are not available.
 * 
 * @param[in] tp Task pool to disable GPU for
 */
void hicma_parsec_disable_GPU( parsec_taskpool_t * tp );

/**
 * @brief Disable GPU to run only on CPU
 * 
 * Disables GPU execution for the specified task pool, forcing all
 * computation to run on CPU only.
 * 
 * @param[in] tp Task pool to disable GPU for
 */
void disable_GPU( parsec_taskpool_t * tp );

/* ============================================================================
 * GPU load balancing functions
 * ============================================================================ */

/**
 * @brief Calculate load balance for GPU 2D operations
 * 
 * Determines the optimal GPU device for 2D distributed operations
 * using a 2D grid-based load balancing strategy.
 * 
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] params HICMA PaRSEC parameters
 * @return Load balance factor (GPU device index)
 */
int gpu_load_balance_2d( int m, int n, hicma_parsec_params_t *params );

/**
 * @brief GPU that tile (m, n) of the Cholesky matrix is advised to live on
 *
 * One rule for every Cholesky kind and for the warmup, which must agree with
 * the factorization. It works on the tile's position among the tiles of its
 * rank (m / P, n / Q), so that a rank uses all its GPUs whatever the process
 * grid. A dense matrix is spread over the gpu_rows x gpu_cols grid
 * (gpu_load_balance_2d): on four linked V100s at N=40960 the grid was 1 to
 * 10 % faster than placing by row or by column (medians of five interleaved
 * runs; FP64 1.07 s against 1.08 and 1.13, adaptive 0.83 against 0.87 and
 * 0.91, run-time decisions 0.64 against 0.70 and 0.65). With low-rank tiles
 * only a band is dense, and a band narrower than the grid would reach only
 * some of its GPUs (the diagonal of a 2 x 2 grid, two of the four), so the
 * band is placed by column; at a band of 8 tiles the rules did not differ
 * beyond the run-to-run spread.
 */
int hicma_parsec_gpu_of_tile( int m, int n, hicma_parsec_params_t *params );

/* ============================================================================
 * GPU workspace management functions
 * ============================================================================ */

/**
 * @brief Lookup GPU workspace for POTRF operations
 * 
 * Finds the appropriate GPU workspace for POTRF operations based on
 * the CUDA device and execution stream.
 * 
 * @param[in] cuda_device CUDA device module
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] ws POTRF workspace
 * @return Pointer to stream workspace
 */
parsec_potrf_stream_workspace_t *lookup_gpu_workspace( parsec_device_cuda_module_t *cuda_device,
        parsec_cuda_exec_stream_t *cuda_stream,
        parsec_potrf_workspace_t *ws );
int lookup_gpu_workspace_j( parsec_device_cuda_module_t *cuda_device,
        parsec_cuda_exec_stream_t *cuda_stream );

/** Operand cache counters {conversions, reuses, fallbacks, slots, bytes} of this process's GPUs */
void hicma_operand_cache_stats( hicma_parsec_data_t *data, uint64_t stats[5], int reset );

/**
 * @brief Allocate memory for workspace
 * 
 * Allocates memory for GPU workspace structures including device-specific
 * workspaces and stream workspaces.
 * 
 * @param[out] ws Pointer to workspace pointer
 */
void workspace_memory_allocate( parsec_potrf_workspace_t **ws );

/**
 * @brief Free workspace memory
 * 
 * Frees all allocated GPU workspace memory including cuBLAS handles,
 * cuSOLVER handles, and GPU buffers.
 * 
 * @param[in] ws Workspace to free
 */
void workspace_memory_free( parsec_potrf_workspace_t *ws);

/* ============================================================================
 * GPU cache and memory management functions
 * ============================================================================ */

/**
 * @brief Initialize GPU handle
 * 
 * Initializes GPU handles including cuBLAS handles for all
 * available GPU devices and execution streams.
 * 
 * @param[in] data HICMA PaRSEC data structure
 */
void gpu_handle_init( hicma_parsec_data_t *data );

/**
 * @brief Finalize GPU handle
 * 
 * Cleans up GPU handles and frees associated resources
 * for all GPU devices and execution streams.
 * 
 * @param[in] data HICMA PaRSEC data structure
 */
void gpu_handle_fini( hicma_parsec_data_t *data );

/**
 * @brief Initialize GPU temporary buffer
 * 
 * Initializes GPU temporary buffers for matrix operations including
 * cuBLAS, cuSOLVER handles, and workspace memory allocation.
 * 
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] mb Block size for rows
 * @param[in] nb Block size for columns
 * @param[in] maxrank Maximum rank
 * @param[in] kind_of_cholesky Type of Cholesky factorization
 */
void gpu_temporay_buffer_init( hicma_parsec_data_t *data, hicma_parsec_params_t *params,
        int mb, int nb, int maxrank, int kind_of_cholesky );

/**
 * @brief Finalize GPU temporary buffer
 * 
 * Frees GPU temporary buffers and cleans up associated resources
 * for the specified Cholesky factorization type.
 * 
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] kind_of_cholesky Type of Cholesky factorization
 */
void gpu_temporay_buffer_fini( hicma_parsec_data_t *data, int kind_of_cholesky );

/* ============================================================================
 * GPU device discovery functions
 * ============================================================================ */

/**
 * @brief Find all CUDA devices
 * 
 * Discovers all available CUDA/HIP devices in the system and
 * returns their indices for device management.
 * 
 * @param[out] dev_index Array of device indices
 * @param[out] nb Number of devices found
 */
void hicma_parsec_find_cuda_devices( int **dev_index, int *nb);

/* ============================================================================
 * GPU datatype conversions: OmniCast (omc_gpu_* in omnicast/omnicast.h),
 * through the helpers of hicma_parsec_datatype_convert.h
 * ============================================================================ */

/* cuBLAS reads alpha and beta in the compute precision: half for
 * CUBLAS_COMPUTE_16F (--tensor_gemm 0x10), float otherwise. A float read as
 * half is 0 (the low bytes of 1.0f and -1.0f), so passing &float with FP16
 * compute silently dropped every update. hbuf is a char buffer of the caller. */
#define HICMA_GEMM_SCALAR(compute16, fvar, hbuf) \
    ((compute16) ? (hicma_parsec_fp32_to_fp16_host((fvar), &(hbuf)[0]), (const void *)&(hbuf)[0]) : (const void *)&(fvar))

/* ============================================================================
 * GPU matrix generation and manipulation
 * ============================================================================ */

/**
 * @brief Generate Gaussian random matrix on GPU
 * 
 * Generates a Gaussian random matrix on GPU with specified parameters
 * for matrix generation and testing.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] m Number of rows in the original matrix
 * @param[in] n Number of columns in the original matrix
 * @param[in] gamma Gaussian parameter
 * @param[in] add_diag Diagonal value to add
 * @param[in] a Pointer to the matrix data
 * @param[in] lda Leading dimension of a
 * @param[in] stream CUDA stream
 */
void gaussian_gpu(int nrows, int ncols, int m, int n,
                    float gamma, float add_diag, float *a, int lda,
                    cudaStream_t stream);



/* ============================================================================
 * GPU memory operations
 * ============================================================================ */

/**
 * @brief Subtract float matrix from double matrix on GPU
 *
 * Performs element-wise operation:
 *   dest(double) -= src(float)
 *
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] _src Source data pointer (float)
 * @param[in,out] _dest Destination data pointer (double)
 * @param[in] stream CUDA stream
 */
void sub_float_from_double_GPU( int nrows, int ncols, void *_src, void *_dest, cudaStream_t stream );

void sub_float_from_double_ld_GPU( int nrows, int ncols,
               void *_src, int ldsrc,
               void *_dest, int lddest,
               cudaStream_t stream );

void sub_float_from_float_ld_GPU( int nrows, int ncols,
               void *_src, int ldsrc,
               void *_dest, int lddest,
               cudaStream_t stream );




/**
 * @brief Keep the lower triangle of an FP32 tile in place
 *
 * Zeroes the strictly upper triangle of the n x n tile L and, if unit is set,
 * writes 1 on its diagonal (TRMM diagonal tile computed as a GEMM).
 *
 * @param[in] n Order of the tile
 * @param[in,out] L Tile (single precision)
 * @param[in] ldl Leading dimension of L
 * @param[in] unit Nonzero for a unit diagonal
 * @param[in] stream CUDA stream
 */
void hicma_parsec_tril_mask_float_GPU( int n, float *L, int ldl, int unit, cudaStream_t stream );


/* ============================================================================
 * GPU matrix printing functions
 * ============================================================================ */

/**
 * @brief Print int32 matrix on GPU
 * 
 * Prints the contents of an int32 matrix stored on GPU
 * for debugging and verification purposes.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] A Input matrix data
 * @param[in] stream CUDA stream
 */
void matrix_print_int32_GPU( int nrows, int ncols, int *A, cudaStream_t stream );

/**
 * @brief Print int8 matrix on GPU
 * 
 * Prints the contents of an int8 matrix stored on GPU
 * for debugging and verification purposes.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] A Input matrix data
 * @param[in] stream CUDA stream
 */
void matrix_print_int8_GPU( int nrows, int ncols, int *A, cudaStream_t stream );

/* ============================================================================
 * GPU matrix generation
 * ============================================================================ */

/* ============================================================================
 * GPU Hamming operations
 * ============================================================================ */

/**
 * @brief Subtract ones from a matrix on GPU
 * 
 * Performs element-wise subtraction of a constant value from
 * a matrix stored on GPU for Hamming distance calculations.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[out] aout Output matrix
 * @param[in] ain Input matrix
 * @param[in] lda Leading dimension of ain
 * @param[in] value Value to subtract
 * @param[in] stream CUDA stream
 */
void hicma_parsec_hamming_subtract_ones_GPU(int nrows, int ncols, int8_t *aout, int8_t *ain, int lda, int value, cudaStream_t stream);

/**
 * @brief Get identity matrix on GPU
 * 
 * Generates an identity matrix on GPU with specified dimensions
 * and value for Hamming distance calculations.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[out] aout Output matrix
 * @param[in] ain Input matrix
 * @param[in] lda Leading dimension of ain
 * @param[in] value Value to set
 * @param[in] stream CUDA stream
 */
void hicma_parsec_hamming_get_id_matrix_GPU(int nrows, int ncols, int8_t *aout, int8_t *ain, int lda, int value, cudaStream_t stream);

/**
 * @brief Perform bitmask tensergemm on GPU
 * 
 * Performs tensor GEMM operation with bitmask support on GPU
 * for specialized matrix operations.
 * 
 * @param[in] input Input data
 * @param[in] A Input matrix A
 * @param[in] B Input matrix B
 * @param[out] C Output matrix C
 * @param[in] m Number of rows
 * @param[in] k Number of columns
 * @param[in] target_value Target value
 * @param[in] stream CUDA stream
 * @return 0 on success, non-zero on failure
 */
int bitmask_tensergemm_GPU( int8_t *input, void *A,  void *B, int32_t *C, int m, int k, int target_value, cudaStream_t stream );

// d_dst: device pointer to 1 double (cudaMalloc)
// x: host scalar value
// stream: stream to enqueue the kernel (0 = default)
void set_device_double(double* d_dst, double x, cudaStream_t stream);

#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
/* Lower Cholesky of an n x n tile on the stream of a hipBLAS handle, in place of
 * rocsolver_?potrf, whose diagonal-block kernel has a race (hicma_parsec_hip.cu).
 * info (device): 0, or the 1-based column of the first pivot that is not positive. */
hipblasStatus_t hicma_potrf_lower_hip_d(hipblasHandle_t handle, int n, double *A, int lda, int *info);
hipblasStatus_t hicma_potrf_lower_hip_s(hipblasHandle_t handle, int n, float *A, int lda, int *info);
#endif

/* ============================================================================
 * GPU core operations
 * ============================================================================ */

/* ============================================================================
 * cuBLAS and CUDA utility functions
 * ============================================================================ */

/**
 * @brief CUDA error checking macro
 * 
 * Macro for checking CUDA errors and reporting them with
 * file and line information for debugging.
 * 
 * @param[in] STR Error string description
 * @param[in] ERROR CUDA error code
 */
#define HiCMA_CUDA_CHECK_ERROR( STR, ERROR)                     \
        cudaError_t __cuda_error = (cudaError_t) (ERROR);               \
        if( cudaSuccess != __cuda_error ) {                             \
            parsec_warning( "%s:%d %s%s", __FILE__, __LINE__,           \
                            (STR), cudaGetErrorString(__cuda_error) );  \
        }

#ifdef __cplusplus
}
#endif

#endif /* defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT) */

#endif /* HICMA_PARSEC_GPU_H */
