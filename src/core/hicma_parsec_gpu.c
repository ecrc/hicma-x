/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2025     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"
#include "hicma_parsec_gpu.h"
#include "parsec/utils/zone_malloc.h"

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/**
 * @file hicma_parsec_gpu.c
 * @brief Implementation of GPU acceleration functions for HICMA PaRSEC
 *
 * This file implements GPU-accelerated operations for HICMA PaRSEC including
 * GPU architecture detection, load balancing, workspace management, and
 * various GPU kernels for matrix operations. The implementation supports
 * both CUDA and HIP backends for cross-platform GPU acceleration.
 */

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
int hicma_parsec_check_gpu_arch( hicma_parsec_params_t *params ) {
    struct cudaDeviceProp deviceProperties;
    int deviceIndex = 0;
    cudaError_t cudaResult = cudaGetDeviceProperties(&deviceProperties, deviceIndex); 
    if (cudaResult == cudaSuccess) {
        if( 0 == params->rank ) {
            printf("\nDevice %d: %s\n", deviceIndex, deviceProperties.name);
            printf("Compute capability: %d.%d\n", deviceProperties.major, deviceProperties.minor);
        }
        const char *v100 = "V100";
        const char *a100 = "A100";
        const char *h100 = "H100";
        char *substring_v100 = strstr(deviceProperties.name, v100);
        char *substring_a100 = strstr(deviceProperties.name, a100);
        char *substring_h100 = strstr(deviceProperties.name, h100);
        if (substring_v100 != NULL) {
            params->gpu_type = GPU_ARCH_NVIDIA_V100;
        } else if (substring_a100 != NULL) { 
            params->gpu_type = GPU_ARCH_NVIDIA_A100;
        } else if (substring_h100 != NULL) { 
            params->gpu_type = GPU_ARCH_NVIDIA_H100;
        } else {
            if( 0 == params->rank ) {
                printf("Not Nvidia GPUs.\n");
            }
        }
    } else {
        printf("Failed to get properties for device %d: %s\n", deviceIndex, cudaGetErrorString(cudaResult));
    }

    return 0;
}

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
void hicma_parsec_disable_GPU( parsec_taskpool_t * tp ) {
    for (int i = 0; i < parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( device->type == PARSEC_DEV_CUDA || device->type == PARSEC_DEV_HIP )
            tp->devices_index_mask &= ~(1<<i);
    }
}

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
int gpu_load_balance( int m, int n, hicma_parsec_params_t *params )
{
    if( params->band_size_dense < params->NT ) return n / params->P % params->gpus;
    return m / params->P % params->gpus;
}

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
int gpu_load_balance_2d( int m, int n, hicma_parsec_params_t *params )
{
    return (m / params->P % params->gpu_rows) * params->gpu_cols + n / params->Q % params->gpu_cols; 
}

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
int gpu_load_dense( int m, int n, int NT, int nb_cuda_devices)
{
    return (n * NT + m) % nb_cuda_devices;
}

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
int gpu_load_tlr( int m, int n, int nb_cuda_devices)
{
    return n % nb_cuda_devices;
}

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
int gpu_load_mixed_precision( int m, int n, int NT, int P, int nb_cuda_devices, int band_size_double, int band_size_single)
{
    if( band_size_double < NT && m - n < band_size_single )
        return (m / P) % nb_cuda_devices; 
    else
        return (n * NT + m) % nb_cuda_devices;
}

/**
 * @brief Disable CPU to just run on GPU
 *
 * Disables CPU execution for the specified task pool, forcing all
 * computation to run on GPU devices only.
 *
 * @param[in] tp Task pool to disable CPU for
 */
void disable_CPU( parsec_taskpool_t * tp ) {
    for (int i = 0; i < parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( device->type == PARSEC_DEV_CPU )
            tp->devices_index_mask &= ~(1<<i);
    }
}

/**
 * @brief Disable GPU to just run on CPU
 *
 * Disables GPU execution for the specified task pool, forcing all
 * computation to run on CPU only.
 *
 * @param[in] tp Task pool to disable GPU for
 */
void disable_GPU( parsec_taskpool_t * tp ) {
    for (int i = 0; i < parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( device->type == PARSEC_DEV_CUDA || device->type == PARSEC_DEV_HIP )
            tp->devices_index_mask &= ~(1<<i);
    }
}

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
int lookup_gpu_workspace_i( parsec_device_cuda_module_t *cuda_device ) {
    int i;

    /* Look for device */
    for(i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( NULL == device ) continue;
        if( device->type != PARSEC_DEV_CUDA && device->type != PARSEC_DEV_HIP ) continue;
        parsec_device_cuda_module_t *cuda_device_compare = (parsec_device_cuda_module_t*)device;

        if(cuda_device->cuda_index == cuda_device_compare->cuda_index)
            break;
    }

    return i;
}

int lookup_gpu_workspace_j( parsec_device_cuda_module_t *cuda_device,
        parsec_cuda_exec_stream_t *cuda_stream ) {
    int j;

    /* Look for stream; 0, h2d; 1 d2h*/
    for(j = 2; j < cuda_device->super.max_exec_streams; j++) {
        if( cuda_stream == (parsec_cuda_exec_stream_t *)cuda_device->super.exec_stream[j] )
            break;
    }

    return j;
}

parsec_potrf_stream_workspace_t *lookup_gpu_workspace( parsec_device_cuda_module_t *cuda_device,
        parsec_cuda_exec_stream_t *cuda_stream,
        parsec_potrf_workspace_t *ws ) {
    int i, j;

    /* Look for device */
    for(i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( NULL == device ) continue;
        if( device->type != PARSEC_DEV_CUDA && device->type != PARSEC_DEV_HIP ) continue;
        parsec_device_cuda_module_t *cuda_device_compare = (parsec_device_cuda_module_t*)device;

        if(cuda_device->cuda_index == cuda_device_compare->cuda_index)
            break;
    }

    /* Look for stream; 0, h2d; 1 d2h*/
    for(j = 2; j < cuda_device->super.max_exec_streams; j++) {
        if( cuda_stream == (parsec_cuda_exec_stream_t *)cuda_device->super.exec_stream[j] )
            break;
    }

    return &ws->gpu_workspace[i].stream_workspace[j];
}

/**
 * @brief Allocate memory for workspace
 *
 * Allocates memory for GPU workspace structures including device-specific
 * workspaces and stream workspaces.
 *
 * @param[out] ws Pointer to workspace pointer
 */
void workspace_memory_allocate( parsec_potrf_workspace_t **ws ) {
    *ws = (parsec_potrf_workspace_t *)malloc( sizeof(parsec_potrf_workspace_t) );
    (*ws)->gpu_workspace = (parsec_potrf_gpu_workspace_t *)calloc( parsec_nb_devices, sizeof(parsec_potrf_gpu_workspace_t) );

    for( int i = 0; i < parsec_nb_devices; i++ ) {
        (*ws)->gpu_workspace[i].stream_workspace = (parsec_potrf_stream_workspace_t *)calloc( PARSEC_GPU_MAX_STREAMS, sizeof(parsec_potrf_stream_workspace_t) );
        (*ws)->gpu_workspace[i].cuda_device = (parsec_device_cuda_module_t *)malloc( sizeof(parsec_device_cuda_module_t) );
    }
}

/**
 * @brief Free workspace memory
 *
 * Frees all allocated GPU workspace memory including cuBLAS handles,
 * cuSOLVER handles, and GPU buffers.
 *
 * @param[in] ws Workspace to free
 */
void workspace_memory_free( parsec_potrf_workspace_t *ws)
{
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( NULL == device ) continue;
        if( device->type != PARSEC_DEV_CUDA && device->type != PARSEC_DEV_HIP ) continue;

        for(int j = 0; j < ws->gpu_workspace[i].cuda_device->super.max_exec_streams; j++) {
            /* j 0, h2d; j 1, d2h */
            if( j <= 1 ) continue;

            /* FP8 */
            /*******************************/
#if HAVE_FP8
            cublasLtDestroy(ws->gpu_workspace[i].stream_workspace[j].lightHandle);
            cublasLtMatmulDescDestroy(ws->gpu_workspace[i].stream_workspace[j].matmulDesc);
            cublasLtMatrixLayoutDestroy(ws->gpu_workspace[i].stream_workspace[j].Adesc);
            cublasLtMatrixLayoutDestroy(ws->gpu_workspace[i].stream_workspace[j].Bdesc);
            cublasLtMatrixLayoutDestroy(ws->gpu_workspace[i].stream_workspace[j].Cdesc);
            cudaFree(ws->gpu_workspace[i].stream_workspace[j].workspace);
            //cudaFree(ws->gpu_workspace[i].stream_workspace[j].heuristicResultsArray);
            cublasLtMatmulPreferenceDestroy(ws->gpu_workspace[i].stream_workspace[j].pref);
#endif
            /*******************************/

            /* Free GPU handle_cusolver */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].handle_cusolver ) {
                cusolverDnHandle_t handle_cusolver = ws->gpu_workspace[i].stream_workspace[j].handle_cusolver;
                cusolverStatus_t status = cusolverDnDestroy(handle_cusolver);
                assert(status == CUSOLVER_STATUS_SUCCESS);
            }

            /* Free GPU handle_cublas */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].handle_cublas ) {
                cublasHandle_t handle_cublas = ws->gpu_workspace[i].stream_workspace[j].handle_cublas;
                cublasStatus_t status = cublasDestroy(handle_cublas);
                assert(status == CUBLAS_STATUS_SUCCESS);
            }

            /* Free GPU handle_cublas_deviceptr */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].handle_cublas_deviceptr ) {
                cublasHandle_t handle_cublas_deviceptr = ws->gpu_workspace[i].stream_workspace[j].handle_cublas_deviceptr;
                cublasStatus_t status = cublasDestroy(handle_cublas_deviceptr);
                assert(status == CUBLAS_STATUS_SUCCESS);
            }

            /* Free GPU handle_cublas_tensor */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].handle_cublas_tensor ) {
                cublasHandle_t handle_cublas_tensor = ws->gpu_workspace[i].stream_workspace[j].handle_cublas_tensor;
                cublasStatus_t status = cublasDestroy(handle_cublas_tensor);
                assert(status == CUBLAS_STATUS_SUCCESS);
            }

            /* Free GPU buffer */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].gpu_buffer ) {
                zone_free( ws->gpu_workspace[i].cuda_device->super.memory, ws->gpu_workspace[i].stream_workspace[j].gpu_buffer );
            }

            /* Free GPU buffer A */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].gpu_buffer_A ) {
                zone_free( ws->gpu_workspace[i].cuda_device->super.memory, ws->gpu_workspace[i].stream_workspace[j].gpu_buffer_A );
            }

            /* Free GPU buffer B */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].gpu_buffer_B ) {
                zone_free( ws->gpu_workspace[i].cuda_device->super.memory, ws->gpu_workspace[i].stream_workspace[j].gpu_buffer_B );
            }

            /* Free GPU buffer C */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].gpu_buffer_C ) {
                zone_free( ws->gpu_workspace[i].cuda_device->super.memory, ws->gpu_workspace[i].stream_workspace[j].gpu_buffer_C );
            }

            /* Free GPU buffer mbr */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].gpu_buffer_mbr ) {
                zone_free( ws->gpu_workspace[i].cuda_device->super.memory, ws->gpu_workspace[i].stream_workspace[j].gpu_buffer_mbr );
            }

            /* Free GPU buffer rr */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].gpu_buffer_rr ) {
                zone_free( ws->gpu_workspace[i].cuda_device->super.memory, ws->gpu_workspace[i].stream_workspace[j].gpu_buffer_rr );
            }

        }
    }

    for( int i = 0; i < parsec_nb_devices; i++ ) {
        free( ws->gpu_workspace[i].stream_workspace );
    }

    free( ws->gpu_workspace );
    free( ws );
}

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
void hicma_parsec_gpu_cache_reset( parsec_context_t *parsec ) {
    parsec_devices_release_memory();
    parsec_devices_reset_load(parsec);
}

/**
 * @brief Initialize GPU handle
 *
 * Initializes GPU handles including cuBLAS handles for all
 * available GPU devices and execution streams.
 *
 * @param[in] data HICMA PaRSEC data structure
 */
void gpu_handle_init( hicma_parsec_data_t *data ) {
    /* Allocate memory */
    workspace_memory_allocate( &data->ws_gpu );

    /* Traverse all gpu device */
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( NULL == device ) continue;
        if( device->type != PARSEC_DEV_CUDA && device->type != PARSEC_DEV_HIP ) continue;

        /* Set cuda_device */
        parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)device;
        cudaSetDevice(cuda_device->cuda_index);
        data->ws_gpu->gpu_workspace[i].cuda_device = cuda_device;

        /* Traverse all streams */
        for(int j = 0; j < cuda_device->super.max_exec_streams; j++) {
            /* j 0, h2d; j 1, d2h */
            if( j <= 1 ) continue;

            parsec_cuda_exec_stream_t* cuda_stream = (parsec_cuda_exec_stream_t*)cuda_device->super.exec_stream[j];

            cublasStatus_t status;
            cublasHandle_t handle_cublas_gpu;
            cublasHandle_t handle_cublas_gpu_deviceptr;

            /* Init to NULL */
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas = NULL;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas_deviceptr = NULL;

            /* Create handle_cublas_gpu */
            status = cublasCreate(&handle_cublas_gpu);
            cublasSetStream( handle_cublas_gpu, cuda_stream->cuda_stream );
            assert(CUBLAS_STATUS_SUCCESS == status);
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas = handle_cublas_gpu;

            /* Create handle_cublas_gpu_deviceptr */
            status = cublasCreate(&handle_cublas_gpu_deviceptr);
            cublasSetStream( handle_cublas_gpu_deviceptr, cuda_stream->cuda_stream );
            status = cublasSetPointerMode(handle_cublas_gpu_deviceptr, CUBLAS_POINTER_MODE_DEVICE);
            assert(CUBLAS_STATUS_SUCCESS == status);
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas_deviceptr = handle_cublas_gpu_deviceptr;
        }
    }
}

/**
 * @brief Finalize GPU handle
 *
 * Cleans up GPU handles and frees associated resources
 * for all GPU devices and execution streams.
 *
 * @param[in] data HICMA PaRSEC data structure
 */
void gpu_handle_fini( hicma_parsec_data_t *data ) {
    parsec_potrf_workspace_t *ws = data->ws_gpu;
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( NULL == device ) continue;
        if( device->type != PARSEC_DEV_CUDA && device->type != PARSEC_DEV_HIP ) continue;

        for(int j = 0; j < ws->gpu_workspace[i].cuda_device->super.max_exec_streams; j++) {
            /* j 0, h2d; j 1, d2h */
            if( j <= 1 ) continue;

            /* Free GPU handle_cublas */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].handle_cublas ) {
                cublasHandle_t handle_cublas = ws->gpu_workspace[i].stream_workspace[j].handle_cublas;
                cublasStatus_t status = cublasDestroy(handle_cublas);
                assert(status == CUBLAS_STATUS_SUCCESS);
            }

            /* Free GPU handle_cublas_deviceptr */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].handle_cublas_deviceptr ) {
                cublasHandle_t handle_cublas_deviceptr = ws->gpu_workspace[i].stream_workspace[j].handle_cublas_deviceptr;
                cublasStatus_t status = cublasDestroy(handle_cublas_deviceptr);
                assert(status == CUBLAS_STATUS_SUCCESS);
            }
        }
    }

    for( int i = 0; i < parsec_nb_devices; i++ ) {
        free( ws->gpu_workspace[i].stream_workspace );
    }

    free( ws->gpu_workspace );
    free( ws );
}

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
        int mb, int nb, int maxrank, int kind_of_cholesky ) {

    /* Only allocated memory for cases including GPU */
    /*   if( !(DENSE_TLR_DP == kind_of_cholesky
         || DENSE_MP_BAND == kind_of_cholesky
         || DENSE_SP_HP_BAND == kind_of_cholesky  
         || DENSE_TLR_MP == kind_of_cholesky
         || DENSE_MP_GPU == kind_of_cholesky
         || DENSE_MP_GPU_FP8 == kind_of_cholesky) ) 
         return; */

    /* Allocate memory */
    workspace_memory_allocate( &data->ws_gpu );

    /* Traverse all gpu device */
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( NULL == device ) continue;
        if( device->type != PARSEC_DEV_CUDA && device->type != PARSEC_DEV_HIP ) continue;

        /* Set cuda_device */
        parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)device;
        cudaSetDevice(cuda_device->cuda_index);
        data->ws_gpu->gpu_workspace[i].cuda_device = cuda_device;

        /* Traverse all streams */
        for(int j = 0; j < cuda_device->super.max_exec_streams; j++) {
            /* j 0, h2d; j 1, d2h */
            if( j <= 1 ) continue;

            parsec_cuda_exec_stream_t* cuda_stream = (parsec_cuda_exec_stream_t*)cuda_device->super.exec_stream[j];

            cublasStatus_t status;
            cusolverStatus_t status_cusolver;
            cudaError_t cudaStatus;
            cusolverDnHandle_t handle_cusolver;
            cublasHandle_t handle_cublas_tensor;
            cublasHandle_t handle_cublas_gpu;
            cublasHandle_t handle_cublas_gpu_deviceptr;

            /* Init to NULL */
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cusolver = NULL;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas = NULL;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas_deviceptr = NULL;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas_tensor = NULL;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer = NULL;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_size = 0;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_A = NULL;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_B = NULL;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_C = NULL;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_mbr = NULL;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_rr = NULL;

#if HAVE_FP8
            /****************************************************************************/
            /* FP8 */
            // lightHandle
            cublasLtHandle_t lightHandle;
            cublasLtCreate(&lightHandle);
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].lightHandle = lightHandle;

            // matmulDesc
            cublasLtMatmulDesc_t matmulDesc;
            cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            const cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(
                    matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                    sizeof(transa));
            cublasLtMatmulDescSetAttribute(
                    matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                    sizeof(transb));
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].matmulDesc= matmulDesc;

            // Adesc Bdesc Cdesc
            cublasLtMatrixLayout_t Adesc;
            cublasLtMatrixLayout_t Bdesc;
            cublasLtMatrixLayout_t Cdesc;
            int fp8_ld = ((mb + 15) / 16) * 16;
            cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, fp8_ld, fp8_ld, fp8_ld);
            cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, fp8_ld, fp8_ld, fp8_ld);
            //cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, mb, mb, mb);
            cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, fp8_ld, fp8_ld, fp8_ld);
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].Adesc = Adesc;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].Bdesc = Bdesc;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].Cdesc = Cdesc;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].fp8_ld = fp8_ld;

            // workspace
            size_t workspaceSize = 32 * 1024 * 1024;
            void *workspace;
            cudaMalloc(&workspace, workspaceSize);
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].workspaceSize = workspaceSize;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].workspace = workspace;

            // heuristicResultsArray 
            cublasLtMatmulPreference_t pref;
            cublasLtMatmulPreferenceCreate(&pref);
            cublasLtMatmulPreferenceSetAttribute(
                    pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &workspaceSize, sizeof(workspaceSize));

            int requestedAlgoCount = 1;
            int returnAlgoCount;
            //cudaMalloc(&returnAlgoCount, sizeof(int));
            cublasLtMatmulHeuristicResult_t heuristicResultsArray;

            cublasLtMatmulAlgoGetHeuristic(
                    lightHandle, matmulDesc, Adesc, Bdesc,
                    Cdesc, Cdesc, pref, requestedAlgoCount,
                    &heuristicResultsArray, &returnAlgoCount);

            data->ws_gpu->gpu_workspace[i].stream_workspace[j].heuristicResultsArray = heuristicResultsArray;
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].pref = pref;
            /****************************************************************************/
#endif

            /* Create handle_cusolver */
            status_cusolver = cusolverDnCreate(&handle_cusolver);
            assert(CUSOLVER_STATUS_SUCCESS == status_cusolver);
            status_cusolver = cusolverDnSetStream(handle_cusolver, cuda_stream->cuda_stream);
            assert(CUSOLVER_STATUS_SUCCESS == status_cusolver);
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cusolver = handle_cusolver;

            /* Allocate workspace for potrf handle_cusolver */
            int workspace_size;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
            status_cusolver = cusolverDnDpotrf_bufferSize(handle_cusolver, CUBLAS_FILL_MODE_LOWER, nb, NULL, mb, &workspace_size);
            assert(CUSOLVER_STATUS_SUCCESS == status_cusolver);
#endif

#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
            workspace_size = 0;
#endif
            if( DENSE_SP_HP_BAND == kind_of_cholesky || DENSE_MP_GPU_FP8_SP == kind_of_cholesky ) {
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer = zone_malloc( cuda_device->super.memory, workspace_size * sizeof(float) + sizeof(int) );
            } else {

                data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer = zone_malloc( cuda_device->super.memory, workspace_size * sizeof(double) + sizeof(int) );
            }
            //printf("data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer %d\n", ((int *)data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer)[0]);
            assert(NULL != data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer);
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_size = workspace_size;

            /* Create handle_cublas_gpu */
            status = cublasCreate(&handle_cublas_gpu);
            cublasSetStream( handle_cublas_gpu, cuda_stream->cuda_stream );
            assert(CUBLAS_STATUS_SUCCESS == status);
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas = handle_cublas_gpu;

            /* Create handle_cublas_gpu_deviceptr */
            status = cublasCreate(&handle_cublas_gpu_deviceptr);
            cublasSetStream( handle_cublas_gpu_deviceptr, cuda_stream->cuda_stream );
            status = cublasSetPointerMode(handle_cublas_gpu_deviceptr, CUBLAS_POINTER_MODE_DEVICE);
            assert(CUBLAS_STATUS_SUCCESS == status);
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas_deviceptr = handle_cublas_gpu_deviceptr;

            /* Create handle_cublas_tensor */
            status = cublasCreate(&handle_cublas_tensor);
            assert(CUBLAS_STATUS_SUCCESS == status);
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
            status = cublasSetMathMode(handle_cublas_tensor, CUBLAS_DEFAULT_MATH); /* Set the math mode to allow cuBLAS to use Tensor Cores */
            assert(CUBLAS_STATUS_SUCCESS == status);
#endif
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas_tensor = handle_cublas_tensor;
            //printf("i %d j %d : %p\n", i, j, data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas_tensor);

            /* GPU buffer for A, B, C, mbr, and rr */
            if( 1 || DENSE_MP_BAND == kind_of_cholesky
                    || DENSE_SP_HP_BAND == kind_of_cholesky
                    || DENSE_TLR_MP == kind_of_cholesky
                    || DENSE_MP_GPU == kind_of_cholesky
                    || DENSE_MP_GPU_FP8 == kind_of_cholesky
                    || DENSE_MP_GPU_FP8_ADAPTIVE == kind_of_cholesky
                    || DENSE_MP_GPU_FP8_SP == kind_of_cholesky ) {
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_A = zone_malloc( cuda_device->super.memory, mb * nb * sizeof(double) );
                assert(NULL != data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_A);

                data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_B = zone_malloc( cuda_device->super.memory, mb * nb * sizeof(double) );
                assert(NULL != data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_B);

                data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_C = zone_malloc( cuda_device->super.memory, mb * nb * sizeof(double) );
                assert(NULL != data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_C);
            }

            if( 1 || DENSE_TLR_DP == kind_of_cholesky
                    || DENSE_TLR_MP == kind_of_cholesky
                    || DENSE_MP_GPU == kind_of_cholesky
                    || DENSE_MP_GPU_FP8 == kind_of_cholesky
                    || DENSE_MP_GPU_FP8_ADAPTIVE == kind_of_cholesky
                    || DENSE_MP_GPU_FP8_SP == kind_of_cholesky ) {
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_mbr = zone_malloc( cuda_device->super.memory, mb * maxrank * sizeof(double) );
                assert(NULL != data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_mbr);

                data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_rr = zone_malloc( cuda_device->super.memory, maxrank * maxrank * sizeof(double) );
                assert(NULL != data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_rr);
            }

        }
    }

}

/**
 * @brief Finalize GPU temporary buffer
 *
 * Frees GPU temporary buffers and cleans up associated resources
 * for the specified Cholesky factorization type.
 *
 * @param[in] data HICMA PaRSEC data structure
 * @param[in] kind_of_cholesky Type of Cholesky factorization
 */
void gpu_temporay_buffer_fini( hicma_parsec_data_t *data, int kind_of_cholesky ) {

    /* Only free memory for cases including GPU */
#if 0
    if( !(DENSE_TLR_DP == kind_of_cholesky
                || DENSE_MP_BAND == kind_of_cholesky
                || DENSE_SP_HP_BAND == kind_of_cholesky
                || DENSE_TLR_MP == kind_of_cholesky
                || DENSE_MP_GPU == kind_of_cholesky
                || DENSE_MP_GPU_FP8 == kind_of_cholesky) )
        return;
#endif

    /** Find all CUDA devices */
    int nb = 0;
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( PARSEC_DEV_CUDA == device->type || PARSEC_DEV_HIP == device->type ) {
            nb++;
        }
    }

    if( nb > 0 ) {
        workspace_memory_free( data->ws_gpu );
    }

}

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
void hicma_parsec_find_cuda_devices( int **dev_index, int *nb) {
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        //printf("parsec_nb_devices %d i %d type %d PARSEC_DEV_HIP %d\n", parsec_nb_devices, i, device->type, PARSEC_DEV_HIP);
        if( PARSEC_DEV_CUDA == device->type || PARSEC_DEV_HIP == device->type ) {
            (*nb)++;
        }
    }
    if((*nb) == 0) {
        char hostname[256];
        gethostname(hostname, 256);
#if DEBUG_INFO
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "No CUDA device found on rank %d on %s\n", rank, hostname);
#endif
    }
    *dev_index = (int *)malloc((*nb) * sizeof(int));
    *nb = 0;
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( PARSEC_DEV_CUDA == device->type || PARSEC_DEV_HIP == device->type ) {
            (*dev_index)[(*nb)++] = device->device_index;
        }
    }
}

#endif
