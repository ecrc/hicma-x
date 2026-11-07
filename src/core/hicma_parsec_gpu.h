/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 * 
 * @version 1.0.0
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
 * @brief Calculate load balance for dense operations
 * 
 * Determines the optimal GPU device for dense matrix operations
 * based on matrix dimensions and process distribution.
 * 
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] params HICMA PaRSEC parameters
 * @return Load balance factor (GPU device index)
 */
int gpu_load_balance( int m, int n, hicma_parsec_params_t *params );

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
 * @brief Calculate load balance for TLR operations
 * 
 * Determines the optimal GPU device for TLR (Tile Low Rank)
 * operations based on matrix dimensions and available devices.
 * 
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] nb_cuda_devices Number of CUDA devices
 * @return Load balance factor (GPU device index)
 */
int gpu_load_tlr( int m, int n, int nb_cuda_devices);

/**
 * @brief Calculate load balance for mixed precision operations
 * 
 * Determines the optimal GPU device for mixed precision operations
 * considering band sizes for different precision levels.
 * 
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] NT Number of tiles
 * @param[in] P Number of processes
 * @param[in] nb_cuda_devices Number of CUDA devices
 * @param[in] band_size_double Double precision band size
 * @param[in] band_size_single Single precision band size
 * @return Load balance factor (GPU device index)
 */
int gpu_load_mixed_precision( int m, int n, int NT, int P, int nb_cuda_devices, int band_size_double, int band_size_single);

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
 * GPU datatype conversion functions
 * ============================================================================ */

/**
 * @brief Convert double precision to bfloat16 on GPU
 * 
 * Converts a double precision matrix to bfloat16 format on GPU
 * using optimized conversion kernels.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] F Input double precision matrix
 * @param[in] ldf Leading dimension of F
 * @param[out] _H Output bfloat16 matrix
 * @param[in] ldh Leading dimension of H
 * @param[in] stream CUDA stream
 */
void double2bf_GPU( int nrows, int ncols,
                const double *F, int ldf,
                void *_H, int ldh,
                cudaStream_t stream );

/**
 * @brief Convert single precision to bfloat16 on GPU
 * 
 * Converts a single precision matrix to bfloat16 format on GPU
 * using optimized conversion kernels.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] F Input single precision matrix
 * @param[in] ldf Leading dimension of F
 * @param[out] _H Output bfloat16 matrix
 * @param[in] ldh Leading dimension of H
 * @param[in] stream CUDA stream
 */
void float2bf_GPU( int nrows, int ncols,
                const float *F, int ldf,
                void *_H, int ldh,
                cudaStream_t stream );

/**
 * @brief Convert single precision to bfloat16 on host
 * 
 * Converts a single precision value to bfloat16 format on the host CPU.
 * 
 * @param[in] F Input single precision value
 * @param[out] _H Output bfloat16 value
 */
void float2bf_host(const float F, void* _H);

/**
 * @brief Convert bfloat16 to single precision on GPU
 * 
 * Converts a bfloat16 matrix to single precision format on GPU.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] _H Input bfloat16 matrix
 * @param[in] ldh Leading dimension of H
 * @param[out] F Output single precision matrix
 * @param[in] ldf Leading dimension of F
 * @param[in] stream CUDA stream
 */
void bf2float_GPU( int nrows, int ncols,
                const void *_H, int ldh,
                float *F, int ldf,
                cudaStream_t stream );

/**
 * @brief Convert single precision to double precision on GPU
 * 
 * Converts a single precision matrix to double precision format on GPU.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] F Input single precision matrix
 * @param[in] ldf Leading dimension of F
 * @param[out] H Output double precision matrix
 * @param[in] ldh Leading dimension of H
 * @param[in] stream CUDA stream
 */
void float2double_GPU(int nrows, int ncols,
                const float *F, int ldf,
                double *H, int ldh ,
                cudaStream_t stream);

/**
 * @brief Convert double precision to single precision on GPU
 * 
 * Converts a double precision matrix to single precision format on GPU.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] H Input double precision matrix
 * @param[in] ldh Leading dimension of H
 * @param[out] F Output single precision matrix
 * @param[in] ldf Leading dimension of F
 * @param[in] stream CUDA stream
 */
void double2float_GPU(int nrows, int ncols,
                const double *H, int ldh,
                float *F, int ldf,
                cudaStream_t stream);

/**
 * @brief Convert double precision to single precision using stochastic rounding on GPU
 *
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] H Input double precision matrix
 * @param[in] ldh Leading dimension of H
 * @param[out] F Output single precision matrix
 * @param[in] ldf Leading dimension of F
 * @param[in] seed Seed of the tile, hicma_parsec_sr_seed(m, n)
 * @param[in] stream CUDA stream
 */
void double2float_round_GPU(int nrows, int ncols,
                const double *H, int ldh,
                float *F, int ldf, uint64_t seed,
                cudaStream_t stream);

/**
 * @brief Convert single precision to half precision on GPU
 * 
 * Converts a single precision matrix to half precision format on GPU.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] F Input single precision matrix
 * @param[in] ldf Leading dimension of F
 * @param[out] H Output half precision matrix
 * @param[in] ldh Leading dimension of H
 * @param[in] stream CUDA stream
 */
void float2half_GPU( int nrows, int ncols,
                const float *F, int ldf,
                void *H, int ldh,
                cudaStream_t stream );

/**
 * @brief Convert single precision to half precision using stochastic rounding on GPU
 *
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] F Input single precision matrix
 * @param[in] ldf Leading dimension of F
 * @param[out] H Output half precision matrix
 * @param[in] ldh Leading dimension of H
 * @param[in] seed Seed of the tile, hicma_parsec_sr_seed(m, n)
 * @param[in] stream CUDA stream
 */
void float2half_round_GPU( int nrows, int ncols,
                const float *F, int ldf,
                void *H, int ldh, uint64_t seed,
                cudaStream_t stream );

/**
 * @brief Convert single precision to half precision on host
 * 
 * Converts a single precision value to half precision format on the host CPU.
 * 
 * @param[in] F Input single precision value
 * @param[out] _H Output half precision value
 */
void float2half_host(const float F, void *_H);

/**
 * @brief Convert half precision to single precision on GPU
 * 
 * Converts a half precision matrix to single precision format on GPU.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] H Input half precision matrix
 * @param[in] ldh Leading dimension of H
 * @param[out] F Output single precision matrix
 * @param[in] ldf Leading dimension of F
 * @param[in] stream CUDA stream
 */
void half2float_GPU( int nrows, int ncols,
                const void *H, int ldh,
                float *F, int ldf,
                cudaStream_t stream );

/**
 * @brief Convert half precision to double precision on GPU
 * 
 * Converts a half precision matrix to double precision format on GPU.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] _H Input half precision matrix
 * @param[in] ldh Leading dimension of H
 * @param[out] F Output double precision matrix
 * @param[in] ldf Leading dimension of F
 * @param[in] stream CUDA stream
 */
void half2double_GPU( int nrows, int ncols,
                const void *_H, int ldh,
                double *F, int ldf,
                cudaStream_t stream );

/**
 * @brief Convert double precision to half precision on GPU
 * 
 * Converts a double precision matrix to half precision format on GPU.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] F Input double precision matrix
 * @param[in] ldf Leading dimension of F
 * @param[out] _H Output half precision matrix
 * @param[in] ldh Leading dimension of H
 * @param[in] stream CUDA stream
 */
void double2half_GPU( int nrows, int ncols,
                const double *F, int ldf,
                void *_H, int ldh,
                cudaStream_t stream );

/**
 * @brief Convert double precision to half precision using stochastic rounding on GPU
 *
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] F Input double precision matrix
 * @param[in] ldf Leading dimension of F
 * @param[out] H Output half precision matrix
 * @param[in] ldh Leading dimension of H
 * @param[in] seed Seed of the tile, hicma_parsec_sr_seed(m, n)
 * @param[in] stream CUDA stream
 */
void double2half_round_GPU( int nrows, int ncols,
                const double *F, int ldf,
                void *_H, int ldh, uint64_t seed,
                cudaStream_t stream );

/**
 * @brief Convert half precision to FP8 on GPU
 * 
 * Converts a half precision matrix to FP8 format on GPU.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] S Input half precision matrix
 * @param[in] lds Leading dimension of S
 * @param[out] T Output FP8 matrix
 * @param[in] ldt Leading dimension of T
 * @param[in] stream CUDA stream
 */
void half2fp8_GPU( int nrows, int ncols,
                const void *S, int lds,
                void *T, int ldt,
                cudaStream_t stream );

void half2fp8_padded_GPU( int nrows, int ncols,
                const void *S, int lds,
                void *T, int ldt,
                int padded_rows, int padded_cols,
                cudaStream_t stream );

/**
 * @brief Convert single precision to FP8 on GPU
 * 
 * Converts a single precision matrix to FP8 format on GPU.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] S Input single precision matrix
 * @param[in] lds Leading dimension of S
 * @param[out] T Output FP8 matrix
 * @param[in] ldt Leading dimension of T
 * @param[in] stream CUDA stream
 */
void float2fp8_GPU( int nrows, int ncols,
                const float *S, int lds,
                void *T, int ldt,
                cudaStream_t stream );

void float2fp8_padded_GPU( int nrows, int ncols,
                const float *S, int lds,
                void *T, int ldt,
                int padded_rows, int padded_cols,
                cudaStream_t stream );

/**
 * @brief Convert single precision to FP8 on host
 * 
 * Converts a single precision value to FP8 format on the host CPU.
 * 
 * @param[in] F Input single precision value
 * @param[out] H Output FP8 value
 */
void float2fp8_host(const float F, void* H);

/**
 * @brief Convert double precision to FP8 on GPU
 * 
 * Converts a double precision matrix to FP8 format on GPU.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] S Input double precision matrix
 * @param[in] lds Leading dimension of S
 * @param[out] T Output FP8 matrix
 * @param[in] ldt Leading dimension of T
 * @param[in] stream CUDA stream
 */
void double2fp8_GPU( int nrows, int ncols,
                const double *S, int lds,
                void *T, int ldt,
                cudaStream_t stream );

void double2fp8_padded_GPU( int nrows, int ncols,
                const double *S, int lds,
                void *T, int ldt,
                int padded_rows, int padded_cols,
                cudaStream_t stream );

/**
 * @brief Convert a tile to FP8 (E4M3) scaled by a power of two from its amax
 *
 * Writes the same transposed, zero-padded layout as double2fp8_padded_GPU,
 * but of s * S, where s is the largest power of two with s * max|S| <= 448,
 * and stores 1/s in scale[2] for cuBLASLt's A/B scale pointer. All on stream.
 *
 * @param[in] src_elem_size Element size of S: sizeof(double), sizeof(float) or 2 (half)
 * @param[in] nrows, ncols  Size of S
 * @param[in] S, lds        Source tile, column major
 * @param[out] T, ldt       FP8 output
 * @param[in] padded_rows, padded_cols Size of T, zero filled beyond S
 * @param[in,out] scale     Device array of 3 floats: amax, s, 1/s
 * @param[in] scaling       0: s = 1 (plain conversion)
 * @param[in] stream        CUDA stream
 */
void hicma_fp8_scaled_convert_GPU( int src_elem_size, int nrows, int ncols,
                const void *S, int lds,
                void *T, int ldt,
                int padded_rows, int padded_cols,
                float *scale, int scaling,
                cudaStream_t stream );

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

/**
 * @brief Convert float array to int array on GPU (in-place)
 * 
 * Converts a float array to int array on GPU in-place without
 * allocating additional memory.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] F Input float array
 * @param[in] ldf Leading dimension of F
 * @param[in] stream CUDA stream
 */
void float_2int_array_unary(int nrows, int ncols,
                    float *F, int ldf,
                    cudaStream_t stream);

/**
 * @brief Convert int array to float array on GPU (in-place)
 * 
 * Converts an int array to float array on GPU in-place without
 * allocating additional memory.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] T Input int array
 * @param[in] ldi Leading dimension of T
 * @param[in] stream CUDA stream
 */
void int_2float_array_unary(int nrows, int ncols,
                    int *T, int ldi,
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
 * @brief Copy double precision data from source to destination on GPU
 *
 * Performs memory copy operation for double precision data on GPU between
 * source and destination locations.
 *
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] _src Source data pointer
 * @param[out] _dest Destination data pointer
 * @param[in] stream CUDA stream
 */
void memcpy_double_GPU( int nrows, int ncols, void *_src, void *_dest, cudaStream_t stream );

/**
 * @brief Copy float data from source to destination on GPU
 * 
 * Performs memory copy operation for float data on GPU between
 * source and destination locations.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] _src Source data pointer
 * @param[out] _dest Destination data pointer
 * @param[in] stream CUDA stream
 */
void memcpy_float_GPU( int nrows, int ncols, void *_src, void *_dest, cudaStream_t stream );

/**
 * @brief Copy half-precision data from source to destination on GPU
 * 
 * Performs memory copy operation for half-precision data on GPU between
 * source and destination locations.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] _src Source data pointer
 * @param[out] _dest Destination data pointer
 * @param[in] stream CUDA stream
 */
void memcpy_half_GPU( int nrows, int ncols, void *_src, void *_dest, cudaStream_t stream );

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

/**
 * @brief Copy bfloat16 data from source to destination on GPU
 * 
 * Performs memory copy operation for bfloat16 data on GPU between
 * source and destination locations.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] _src Source data pointer
 * @param[out] _dest Destination data pointer
 * @param[in] stream CUDA stream
 */
void memcpy_bf_GPU( int nrows, int ncols, void *_src, void *_dest, cudaStream_t stream );

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
void hicma_parsec_hamming_subtract_ones_GPU(int nrows, int ncols, int *aout, int *ain, int lda, int value, cudaStream_t stream);

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
void hicma_parsec_hamming_get_id_matrix_GPU(int nrows, int ncols, int *aout, int *ain, int lda, int value, cudaStream_t stream);

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
void copy_device_float_to_device_double(double* d_dst, const float* d_src, cudaStream_t stream);

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
