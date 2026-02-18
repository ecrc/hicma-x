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
 * @brief Disable CPU to run only on GPU
 * 
 * Disables CPU execution for the specified task pool, forcing all
 * computation to run on GPU devices only.
 * 
 * @param[in] tp Task pool to disable CPU for
 */
void disable_CPU( parsec_taskpool_t * tp );

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
 * @brief Calculate load balance for dense operations with device count
 * 
 * Determines the optimal GPU device for dense operations when
 * the number of CUDA devices is explicitly specified.
 * 
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] NT Number of tiles
 * @param[in] nb_cuda_devices Number of CUDA devices
 * @return Load balance factor (GPU device index)
 */
int gpu_load_dense( int m, int n, int NT, int nb_cuda_devices);

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
int lookup_gpu_workspace_i( parsec_device_cuda_module_t *cuda_device );
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
 * @brief Reset GPU cache
 * 
 * Resets GPU memory cache and device load information to free
 * up memory and reset performance counters.
 * 
 * @param[in] parsec PaRSEC context
 */
void hicma_parsec_gpu_cache_reset( parsec_context_t *parsec );

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
 * @brief Convert int array to float array on GPU
 * 
 * Converts an int array to float array on GPU with separate
 * input and output arrays.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] T Input int array
 * @param[in] ldf Leading dimension of T
 * @param[out] F Output float array
 * @param[in] ldi Leading dimension of F
 * @param[in] stream CUDA stream
 */
void int2float_array(int nrows, int ncols,
                    int *T, int ldf,
                   float *F, int ldi,
                    cudaStream_t stream);

/**
 * @brief Convert float array to int array on GPU
 * 
 * Converts a float array to int array on GPU with separate
 * input and output arrays.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] F Input float array
 * @param[in] ldf Leading dimension of F
 * @param[out] T Output int array
 * @param[in] ldi Leading dimension of T
 * @param[in] stream CUDA stream
 */
void float_2int_array(int nrows, int ncols,
		    float *F, int ldf,
                    int *T, int ldi,
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

/**
 * @brief Convert int array to float array on GPU
 * 
 * Converts an int array to float array on GPU with separate
 * input and output arrays.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] T Input int array
 * @param[in] ldi Leading dimension of T
 * @param[out] F Output float array
 * @param[in] ldf Leading dimension of F
 * @param[in] stream CUDA stream
 */
void int_2float_array(int nrows, int ncols,
                    int *T, int ldi,
                    float *F, int ldf,
                    cudaStream_t stream);

/**
 * @brief Convert double array to int array on GPU (in-place)
 * 
 * Converts a double array to int array on GPU in-place without
 * allocating additional memory.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] F Input double array
 * @param[in] ldf Leading dimension of F
 * @param[in] stream CUDA stream
 */
void double_2int_array_unary(int nrows, int ncols,
                    double *F, int ldf,
                    cudaStream_t stream);

/**
 * @brief Convert int array to double array on GPU
 * 
 * Converts an int array to double array on GPU with separate
 * input and output arrays.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] T Input int array
 * @param[in] ldf Leading dimension of T
 * @param[out] F Output double array
 * @param[in] ldi Leading dimension of F
 * @param[in] stream CUDA stream
 */
void int2double_array(int nrows, int ncols,
                    int *T, int ldf,
                   double *F, int ldi,
                    cudaStream_t stream);

/**
 * @brief Convert double array to int array on GPU
 * 
 * Converts a double array to int array on GPU with separate
 * input and output arrays.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] F Input double array
 * @param[in] ldf Leading dimension of F
 * @param[out] T Output int array
 * @param[in] ldi Leading dimension of T
 * @param[in] stream CUDA stream
 */
void double_2int_array(int nrows, int ncols,
		    double *F, int ldf,
                    int *T, int ldi,
                    cudaStream_t stream);

/**
 * @brief Convert int array to double array on GPU (in-place)
 * 
 * Converts an int array to double array on GPU in-place without
 * allocating additional memory.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] T Input int array
 * @param[in] ldi Leading dimension of T
 * @param[in] stream CUDA stream
 */
void int_2double_array_unary(int nrows, int ncols,
                    int *T, int ldi,
                    cudaStream_t stream);

/**
 * @brief Convert int array to double array on GPU
 * 
 * Converts an int array to double array on GPU with separate
 * input and output arrays.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] T Input int array
 * @param[in] ldi Leading dimension of T
 * @param[out] F Output double array
 * @param[in] ldf Leading dimension of F
 * @param[in] stream CUDA stream
 */
void int_2double_array(int nrows, int ncols,
                    int *T, int ldi,
                    double *F, int ldf,
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
 * @brief Print double precision matrix on GPU
 * 
 * Prints the contents of a double precision matrix stored on GPU
 * for debugging and verification purposes.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] A Input matrix data
 * @param[in] stream CUDA stream
 */
void matrix_print_double_GPU( int nrows, int ncols, double *A, cudaStream_t stream );

/**
 * @brief Print single precision matrix on GPU
 * 
 * Prints the contents of a single precision matrix stored on GPU
 * for debugging and verification purposes.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] A Input matrix data
 * @param[in] stream CUDA stream
 */
void matrix_print_float_GPU( int nrows, int ncols, float *A, cudaStream_t stream );

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

/**
 * @brief Generate dense matrix on GPU using DCMG (Double-Cholesky Matrix Generation)
 * 
 * Generates a dense matrix on GPU using the DCMG algorithm for
 * matrix generation and testing purposes.
 * 
 * @param[out] A Output matrix data
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] m0 Starting row index
 * @param[in] n0 Starting column index
 * @param[in] l1_x_cuda Pointer to x-coordinates of local theta
 * @param[in] l1_y_cuda Pointer to y-coordinates of local theta
 * @param[in] l2_x_cuda Pointer to x-coordinates of local theta
 * @param[in] l2_y_cuda Pointer to y-coordinates of local theta
 * @param[in] localtheta Pointer to local theta values
 * @param[in] distance_metric Distance metric type
 * @param[in] lda Leading dimension of A
 * @param[in] stream CUDA stream
 */
void dcmg_array_GPU( double *A, int m, int n, int m0,
                int n0, double* l1_x_cuda, double* l1_y_cuda, double* l2_x_cuda, double* l2_y_cuda,
                double *localtheta, int distance_metric, int lda, cudaStream_t stream);

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

/**
 * @brief Merge matrices on GPU
 * 
 * Merges matrices on GPU for Hamming distance calculations
 * and matrix combination operations.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[out] aout Output matrix
 * @param[in] ain Input matrix
 * @param[in] lda Leading dimension of ain
 * @param[in] stream CUDA stream
 */
void hicma_parsec_hamming_merge_GPU(int nrows, int ncols, int *aout, int *ain, int lda, cudaStream_t stream);

/**
 * @brief Copy int32 data from source to destination on GPU
 * 
 * Performs memory copy operation for int32 data on GPU between
 * source and destination locations.
 * 
 * @param[in] nrows Number of rows
 * @param[in] ncols Number of columns
 * @param[in] _src Source data pointer
 * @param[out] _dest Destination data pointer
 * @param[in] stream CUDA stream
 */
void hicma_parsec_memcpy_int32_GPU( int nrows, int ncols, void *_src, void *_dest, cudaStream_t stream );

// d_dst: device pointer to 1 double (cudaMalloc)
// x: host scalar value
// stream: stream to enqueue the kernel (0 = default)
void copy_float_to_device_double(double* d_dst, float x, cudaStream_t stream);
void set_device_double(double* d_dst, double x, cudaStream_t stream);
void copy_device_float_to_device_double(double* d_dst, const float* d_src, cudaStream_t stream);

/* ============================================================================
 * GPU core operations
 * ============================================================================ */

/**
 * @brief Perform denseC denseA denseB gemm on GPU
 * 
 * Performs GEMM operation on GPU with dense matrices A, B, and C
 * using optimized cuBLAS kernels and mixed precision support.
 * 
 * @param[in] descA Descriptor for matrix A
 * @param[in] descB Descriptor for matrix B
 * @param[in] descC Descriptor for matrix C
 * @param[in] params_tlr TLR parameters
 * @param[in] ws_gpu GPU workspace
 * @param[in] cuda_device CUDA device module
 * @param[in] gpu_task GPU task
 * @param[in] cuda_stream CUDA execution stream
 * @param[out] C Output matrix C
 * @param[out] A Input matrix A
 * @param[out] B Input matrix B
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Number of columns
 * @param[in] Crank Rank of matrix C
 * @param[in] Arank Rank of matrix A
 * @param[in] Brank Rank of matrix B
 * @param[in] alpha Alpha parameter
 * @param[in] beta Beta parameter
 */
void hicma_parsec_core_gemmmp_TN_denseC_denseA_denseB_gpu( 
        parsec_tiled_matrix_t* descA, 
        parsec_tiled_matrix_t* descB, 
        parsec_tiled_matrix_t* descC,
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_workspace_t *ws_gpu,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        void *C, void *A, void *B, int m, int n, int k,
        int Crank, int Arank, int Brank,
        double alpha, double beta );

/* ============================================================================
 * cuBLAS and CUDA utility functions
 * ============================================================================ */

/**
 * @brief Perform cuBLAS GemmEx operation
 * 
 * Wrapper for cuBLAS GemmEx operation with support for different
 * data types and optimized algorithms.
 * 
 * @param[in] handle cuBLAS handle
 * @param[in] transa Transpose flag for A
 * @param[in] transb Transpose flag for B
 * @param[in] m Number of rows
 * @param[in] n Number of columns
 * @param[in] k Number of columns
 * @param[in] alpha Alpha parameter
 * @param[in] A Input matrix A
 * @param[in] Atype Data type of A
 * @param[in] lda Leading dimension of A
 * @param[in] B Input matrix B
 * @param[in] Btype Data type of B
 * @param[in] ldb Leading dimension of B
 * @param[in] beta Beta parameter
 * @param[out] C Output matrix C
 * @param[in] Ctype Data type of C
 * @param[in] ldc Leading dimension of C
 * @param[in] computeType Compute type
 * @param[in] algo Algorithm
 * @return cuBLAS status
 */
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
                            cublasGemmAlgo_t algo);

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
