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

/* Release builds compile assert() out, so a failed cuBLAS/cuSOLVER handle
 * creation (for example when several ranks share one GPU and the first one
 * reserved most of its memory) used to crash later in cublasSetStream with a
 * segfault. Check explicitly. Used inside the (i, j) device/stream loops. */
#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
#define HICMA_GPU_MEMORY_MCA "device_hip_memory_use"
#else
#define HICMA_GPU_MEMORY_MCA "device_cuda_memory_use"
#endif
#define HICMA_GPU_HANDLE_CHECK(ok, what) do {                                        \
        if( !(ok) ) {                                                                 \
            fprintf(stderr, "Fatal: %s failed on device %d stream %d; when several "  \
                    "processes share a GPU, lower PaRSEC's " HICMA_GPU_MEMORY_MCA "\n", \
                    (what), i, j);                                                    \
            abort();                                                                  \
        }                                                                             \
    } while(0)

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

/* GPU of tile (m, n) for the Cholesky kernels and their warmup (hicma_parsec_gpu.h).
 * m / P and n / Q are the tile's row and column among the tiles of its rank. */
int hicma_parsec_gpu_of_tile( int m, int n, hicma_parsec_params_t *params )
{
    if( !GENOMICS && params->band_size_dense < params->NT ) {
        return n / params->Q % params->gpus;
    }
    return gpu_load_balance_2d( m, n, params );
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
 * @brief Create an operand cache of a GPU (see hicma_operand_cache_t)
 *
 * Up to --operand_cache slots of slot_bytes, or by default (-1) two panels
 * (2 NT tiles) and at most 5% of the device memory zone; taken from the zone
 * only when first needed. NULL when off.
 */
static hicma_operand_cache_t *hicma_operand_cache_create(parsec_device_cuda_module_t *cuda_device,
        hicma_parsec_params_t *params, int mb, size_t slot_bytes)
{
    if( 0 == params->operand_cache ) return NULL;

    zone_malloc_t *zone = cuda_device->super.memory;
    slot_bytes = (slot_bytes + 255) / 256 * 256;
    long nslots = params->operand_cache;
    if( nslots < 0 ) {
        size_t zone_bytes = (size_t)zone->max_segment * zone->unit_size;
        long cap = (long)(zone_bytes / 20 / slot_bytes);
        nslots = 2L * params->NT;
        if( nslots > cap ) nslots = cap;
        if( nslots < 4 ) return NULL;
    }
    int nstreams = cuda_device->super.max_exec_streams;
    if( nstreams > 64 ) return NULL;

    hicma_operand_cache_t *c = (hicma_operand_cache_t *)calloc(1, sizeof(hicma_operand_cache_t));
    if( NULL == c || NULL == (c->slots = (hicma_operand_cache_slot_t *)calloc((size_t)nslots, sizeof(hicma_operand_cache_slot_t))) ) {
        fprintf(stderr, "GPU operand cache: out of host memory\n");
        abort();
    }
    pthread_mutex_init(&c->lock, NULL);
    c->zone = zone;
    c->max_slots = (int)nslots;
    c->nslots = 0;
    c->grow = 1;
    c->nstreams = nstreams;
    c->mb = mb;
    c->slot_bytes = slot_bytes;
    return c;
}

static void hicma_operand_cache_destroy(hicma_operand_cache_t *c, parsec_device_cuda_module_t *cuda_device)
{
    if( NULL == c ) return;
    cudaSetDevice(cuda_device->cuda_index);
    for( int i = 0; i < c->nslots; i++ ) {
        cudaEventDestroy(c->slots[i].ready);
        for( int j = 0; j < c->nstreams; j++ ) {
            cudaEventDestroy(c->slots[i].done[j]);
        }
        free(c->slots[i].done);
        zone_free(cuda_device->super.memory, c->slots[i].buf);
    }
    pthread_mutex_destroy(&c->lock);
    free(c->slots);
    free(c);
}

/**
 * @brief Operand cache counters, summed over this process's GPUs and caches
 *
 * @param[out] stats {conversions, reuses, fallbacks, slots taken, bytes taken}
 * @param[in] reset Zero the counters (not the slots) afterwards
 */
void hicma_operand_cache_stats( hicma_parsec_data_t *data, uint64_t stats[5], int reset )
{
    for( int q = 0; q < 5; q++ ) stats[q] = 0;
    if( NULL == data->ws_gpu ) return;
    for( int i = 0; i < (int)parsec_nb_devices; i++ ) {
        for( int t = 0; t < HICMA_OPC_NUM; t++ ) {
            hicma_operand_cache_t *c = data->ws_gpu->gpu_workspace[i].operand_cache[t];
            if( NULL == c ) continue;
            pthread_mutex_lock(&c->lock);
            stats[0] += c->misses;
            stats[1] += c->hits;
            stats[2] += c->fallbacks;
            stats[3] += (uint64_t)c->nslots;
            stats[4] += (uint64_t)c->nslots * c->slot_bytes;
            if( reset ) c->misses = c->hits = c->fallbacks = 0;
            pthread_mutex_unlock(&c->lock);
        }
    }
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
        (*ws)->gpu_workspace[i].cuda_device = NULL;  /* set to the PaRSEC device module for GPU devices */
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

        for( int t = 0; t < HICMA_OPC_NUM; t++ ) {
            hicma_operand_cache_destroy(ws->gpu_workspace[i].operand_cache[t], ws->gpu_workspace[i].cuda_device);
            ws->gpu_workspace[i].operand_cache[t] = NULL;
        }

        for(int j = 0; j < ws->gpu_workspace[i].cuda_device->super.max_exec_streams; j++) {
            /* j 0, h2d; j 1, d2h */
            if( j <= 1 ) continue;

            /* FP8 */
            /*******************************/
#if HAVE_FP8
            /* Created only when FP8 GEMMs run (gpu_temporay_buffer_init) */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].lightHandle ) {
                cublasLtDestroy(ws->gpu_workspace[i].stream_workspace[j].lightHandle);
                cublasLtMatmulDescDestroy(ws->gpu_workspace[i].stream_workspace[j].matmulDesc);
                cublasLtMatrixLayoutDestroy(ws->gpu_workspace[i].stream_workspace[j].Adesc);
                cublasLtMatrixLayoutDestroy(ws->gpu_workspace[i].stream_workspace[j].Bdesc);
                cublasLtMatrixLayoutDestroy(ws->gpu_workspace[i].stream_workspace[j].Cdesc);
                cudaFree(ws->gpu_workspace[i].stream_workspace[j].workspace);
                //cudaFree(ws->gpu_workspace[i].stream_workspace[j].heuristicResultsArray);
                cublasLtMatmulPreferenceDestroy(ws->gpu_workspace[i].stream_workspace[j].pref);
                cublasLtMatmulDescDestroy(ws->gpu_workspace[i].stream_workspace[j].matmulDesc_scaled);
                cudaFree(ws->gpu_workspace[i].stream_workspace[j].fp8_scale);
            }
#endif
            /*******************************/

            /* Free GPU handle_cusolver */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].handle_cusolver ) {
                cusolverDnHandle_t handle_cusolver = ws->gpu_workspace[i].stream_workspace[j].handle_cusolver;
                cusolverStatus_t status = cusolverDnDestroy(handle_cusolver);
                assert(status == CUSOLVER_STATUS_SUCCESS);
            }

            /* Free the norm metadata stream and pinned buffer (NULL unless created) */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].norm_meta_stream ) {
                cudaStreamDestroy(ws->gpu_workspace[i].stream_workspace[j].norm_meta_stream);
                ws->gpu_workspace[i].stream_workspace[j].norm_meta_stream = NULL;
            }
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].norm_meta_host ) {
                cudaFreeHost(ws->gpu_workspace[i].stream_workspace[j].norm_meta_host);
                ws->gpu_workspace[i].stream_workspace[j].norm_meta_host = NULL;
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
            HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == status, "cublasCreate");
            cublasSetStream( handle_cublas_gpu, cuda_stream->cuda_stream );
            HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == status, "cuBLAS handle setup");
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas = handle_cublas_gpu;

            /* Create handle_cublas_gpu_deviceptr */
            status = cublasCreate(&handle_cublas_gpu_deviceptr);
            HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == status, "cublasCreate");
            cublasSetStream( handle_cublas_gpu_deviceptr, cuda_stream->cuda_stream );
            status = cublasSetPointerMode(handle_cublas_gpu_deviceptr, CUBLAS_POINTER_MODE_DEVICE);
            HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == status, "cuBLAS handle setup");
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

            /* Free the norm metadata stream and pinned buffer (NULL unless created) */
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].norm_meta_stream ) {
                cudaStreamDestroy(ws->gpu_workspace[i].stream_workspace[j].norm_meta_stream);
                ws->gpu_workspace[i].stream_workspace[j].norm_meta_stream = NULL;
            }
            if( NULL != ws->gpu_workspace[i].stream_workspace[j].norm_meta_host ) {
                cudaFreeHost(ws->gpu_workspace[i].stream_workspace[j].norm_meta_host);
                ws->gpu_workspace[i].stream_workspace[j].norm_meta_host = NULL;
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
        }
    }

    for( int i = 0; i < parsec_nb_devices; i++ ) {
        free( ws->gpu_workspace[i].stream_workspace );
    }

    free( ws->gpu_workspace );
    free( ws );
}

#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
/* rocBLAS and rocSOLVER set a handle up (workspace, kernel code objects) at
 * its first call, which in a GPU run happened inside the factorization: on an
 * MI210 the first dense factorization at N=8192 took 0.52 s and the next ones
 * 0.055 s (DPLASMA: 0.126 s, then 0.060 s). Call every handle of a stream once,
 * at tile size, with the routines and precisions the Cholesky uses; the
 * buffers are scratch and are zeroed again afterwards. */
static void hicma_gpu_handles_warmup( parsec_potrf_stream_workspace_t *sw, hipStream_t stream, int mb )
{
    void *A = sw->gpu_buffer_A, *B = sw->gpu_buffer_B, *C = sw->gpu_buffer_C;
    const size_t bytes = (size_t)mb * mb * sizeof(double);
    const double d1 = 1.0, d0 = 0.0;
    const float s1 = 1.0f, s0 = 0.0f;
    if( NULL == A || NULL == B || NULL == C || NULL == sw->handle_cublas ) return;
    hipMemsetAsync(A, 0, bytes, stream);
    hipMemsetAsync(B, 0, bytes, stream);
    hipMemsetAsync(C, 0, bytes, stream);
    hipblasHandle_t blas[2] = { sw->handle_cublas, sw->handle_cublas_tensor };
    for( int h = 0; h < 2; h++ ) {
        if( NULL == blas[h] ) continue;
        hipblasSetStream(blas[h], stream);
        hipblasDgemm(blas[h], HIPBLAS_OP_N, HIPBLAS_OP_T, mb, mb, mb, &d1, A, mb, B, mb, &d0, C, mb);
        hipblasSgemm(blas[h], HIPBLAS_OP_N, HIPBLAS_OP_T, mb, mb, mb, &s1, A, mb, B, mb, &s0, C, mb);
        hipblasGemmEx(blas[h], HIPBLAS_OP_N, HIPBLAS_OP_T, mb, mb, mb, &d1, A, HIPBLAS_R_64F, mb, B, HIPBLAS_R_64F, mb,
                      &d0, C, HIPBLAS_R_64F, mb, HIPBLAS_R_64F, HIPBLAS_GEMM_DEFAULT);
        hipblasGemmEx(blas[h], HIPBLAS_OP_N, HIPBLAS_OP_T, mb, mb, mb, &s1, A, HIPBLAS_R_32F, mb, B, HIPBLAS_R_32F, mb,
                      &s0, C, HIPBLAS_R_32F, mb, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);
        hipblasGemmEx(blas[h], HIPBLAS_OP_N, HIPBLAS_OP_T, mb, mb, mb, &s1, A, HIPBLAS_R_16F, mb, B, HIPBLAS_R_16F, mb,
                      &s0, C, HIPBLAS_R_32F, mb, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);
        hipblasGemmEx(blas[h], HIPBLAS_OP_N, HIPBLAS_OP_T, mb, mb, mb, &s1, A, HIPBLAS_R_16B, mb, B, HIPBLAS_R_16B, mb,
                      &s0, C, HIPBLAS_R_32F, mb, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);
    }
    hipblasDtrsm(sw->handle_cublas, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT,
                 mb, mb, &d1, A, mb, C, mb);
    hipblasStrsm(sw->handle_cublas, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT,
                 mb, mb, &s1, A, mb, C, mb);
    hipblasDsyrk(sw->handle_cublas, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_N, mb, mb, &d1, A, mb, &d1, C, mb);
    hipblasSsyrk(sw->handle_cublas, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_N, mb, mb, &s1, A, mb, &s1, C, mb);
    if( NULL != sw->handle_cublas_deviceptr ) {  /* norms of the run-time decision, result on the device */
        hipblasSetStream(sw->handle_cublas_deviceptr, stream);
        hipblasDnrm2(sw->handle_cublas_deviceptr, mb * mb, A, 1, (double *)B);
        hipblasSnrm2(sw->handle_cublas_deviceptr, mb * mb, A, 1, (float *)B);
    }
    if( NULL != sw->handle_cusolver && NULL != sw->gpu_buffer ) {
        hipblasSetStream(sw->handle_cusolver, stream);
        rocsolver_dpotrf(sw->handle_cusolver, rocblas_fill_lower, mb, C, mb, (int *)sw->gpu_buffer);
        rocsolver_spotrf(sw->handle_cusolver, rocblas_fill_lower, mb, C, mb, (int *)sw->gpu_buffer);
    }
    hipMemsetAsync(A, 0, bytes, stream);
    hipMemsetAsync(B, 0, bytes, stream);
    hipMemsetAsync(C, 0, bytes, stream);
    hipStreamSynchronize(stream);
}
#endif

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

            /* Private non-blocking stream + pinned buffer for reading the
             * runtime-decision {norm, precision} metadata (see the struct). */
            {
                parsec_potrf_stream_workspace_t *sw = &data->ws_gpu->gpu_workspace[i].stream_workspace[j];
                int meta_ok = (cudaSuccess == cudaStreamCreateWithFlags(&sw->norm_meta_stream, cudaStreamNonBlocking));
#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
                meta_ok = meta_ok && (hipSuccess == hipHostMalloc((void **)&sw->norm_meta_host, 4 * sizeof(double), hipHostMallocDefault));
#else
                meta_ok = meta_ok && (cudaSuccess == cudaMallocHost((void **)&sw->norm_meta_host, 4 * sizeof(double)));
#endif
                if( !meta_ok ) {
                    fprintf(stderr, "Failed to create the norm metadata stream/buffer on GPU %d stream %d\n", i, j);
                    abort();
                }
            }

#if HAVE_FP8
            /****************************************************************************/
            /* FP8: only when FP8 GEMMs will run (fp8_enabled); otherwise the
             * handles stay NULL and the 32 MB cuBLASLt workspace per stream is
             * not allocated */
            if( params->fp8_enabled ) {
                // lightHandle
                cublasLtHandle_t lightHandle;
                HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == cublasLtCreate(&lightHandle), "cuBLASLt handle");
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].lightHandle = lightHandle;

                // matmulDesc
                cublasLtMatmulDesc_t matmulDesc;
                HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cuBLASLt matmul descriptor");
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
                int fp8_ld = hicma_fp8_ld(mb);
                HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, fp8_ld, fp8_ld, fp8_ld), "cuBLASLt A layout");
                HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, fp8_ld, fp8_ld, fp8_ld), "cuBLASLt B layout");
                //cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, mb, mb, mb);
                HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, fp8_ld, fp8_ld, fp8_ld), "cuBLASLt C layout");
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].Adesc = Adesc;
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].Bdesc = Bdesc;
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].Cdesc = Cdesc;
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].fp8_ld = fp8_ld;

                // workspace
                size_t workspaceSize = 32 * 1024 * 1024;
                void *workspace;
                HICMA_GPU_HANDLE_CHECK(cudaSuccess == cudaMalloc(&workspace, workspaceSize), "cuBLASLt workspace allocation");
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].workspaceSize = workspaceSize;
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].workspace = workspace;

                // heuristicResultsArray 
                cublasLtMatmulPreference_t pref;
                HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == cublasLtMatmulPreferenceCreate(&pref), "cuBLASLt preference");
                cublasLtMatmulPreferenceSetAttribute(
                        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                        &workspaceSize, sizeof(workspaceSize));

                int requestedAlgoCount = 1;
                int returnAlgoCount = 0;
                cublasLtMatmulHeuristicResult_t heuristicResultsArray;
                memset(&heuristicResultsArray, 0, sizeof(heuristicResultsArray));

                cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
                        lightHandle, matmulDesc, Adesc, Bdesc,
                        Cdesc, Cdesc, pref, requestedAlgoCount,
                        &heuristicResultsArray, &returnAlgoCount);
                /* FP8 GEMMs abort without an algorithm */
                if( CUBLAS_STATUS_SUCCESS != heuristic_status || 0 == returnAlgoCount ) {
                    fprintf(stderr, "Warning: no cuBLASLt FP8 GEMM algorithm on device %d (status %d)\n",
                            i, (int)heuristic_status);
                }

                data->ws_gpu->gpu_workspace[i].stream_workspace[j].heuristicResultsArray = heuristicResultsArray;
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].pref = pref;

                /* FP8 GEMMs: same GEMM, with device pointers to the per-tile
                 * scales 1/s_A and 1/s_B that hicma_parsec_fp8_operand_GPU writes,
                 * or the GEMM copies from a tile sent in FP8, on this stream before
                 * each GEMM (initially 1). */
                float *fp8_scale = NULL;
                const float fp8_scale_init[8] = { 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f };
                HICMA_GPU_HANDLE_CHECK(cudaSuccess == cudaMalloc((void **)&fp8_scale, sizeof(fp8_scale_init)), "FP8 scale buffer");
                HICMA_GPU_HANDLE_CHECK(cudaSuccess == cudaMemcpy(fp8_scale, fp8_scale_init, sizeof(fp8_scale_init), cudaMemcpyHostToDevice), "FP8 scale buffer");
                const float *a_scale = fp8_scale + 2, *b_scale = fp8_scale + 6;
                cublasLtMatmulDesc_t matmulDesc_scaled;
                HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == cublasLtMatmulDescCreate(&matmulDesc_scaled, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cuBLASLt scaled matmul descriptor");
                cublasLtMatmulDescSetAttribute(matmulDesc_scaled, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
                cublasLtMatmulDescSetAttribute(matmulDesc_scaled, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
                cublasLtMatmulDescSetAttribute(matmulDesc_scaled, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale));
                cublasLtMatmulDescSetAttribute(matmulDesc_scaled, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale));

                cublasLtMatmulHeuristicResult_t heuristic_scaled;
                memset(&heuristic_scaled, 0, sizeof(heuristic_scaled));
                returnAlgoCount = 0;
                heuristic_status = cublasLtMatmulAlgoGetHeuristic(
                        lightHandle, matmulDesc_scaled, Adesc, Bdesc,
                        Cdesc, Cdesc, pref, requestedAlgoCount,
                        &heuristic_scaled, &returnAlgoCount);
                if( CUBLAS_STATUS_SUCCESS != heuristic_status || 0 == returnAlgoCount ) {
                    fprintf(stderr, "Warning: no cuBLASLt scaled FP8 GEMM algorithm on device %d (status %d)\n",
                            i, (int)heuristic_status);
                }
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].matmulDesc_scaled = matmulDesc_scaled;
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].heuristic_scaled = heuristic_scaled;
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].fp8_scale = fp8_scale;
            }
            /****************************************************************************/
#endif

            /* Create handle_cusolver */
            status_cusolver = cusolverDnCreate(&handle_cusolver);
            HICMA_GPU_HANDLE_CHECK(CUSOLVER_STATUS_SUCCESS == status_cusolver, "cuSOLVER handle setup");
            status_cusolver = cusolverDnSetStream(handle_cusolver, cuda_stream->cuda_stream);
            HICMA_GPU_HANDLE_CHECK(CUSOLVER_STATUS_SUCCESS == status_cusolver, "cuSOLVER handle setup");
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cusolver = handle_cusolver;

            /* Allocate workspace for potrf handle_cusolver */
            int workspace_size;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
            status_cusolver = cusolverDnDpotrf_bufferSize(handle_cusolver, CUBLAS_FILL_MODE_LOWER, nb, NULL, mb, &workspace_size);
            HICMA_GPU_HANDLE_CHECK(CUSOLVER_STATUS_SUCCESS == status_cusolver, "cuSOLVER handle setup");
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
            HICMA_GPU_HANDLE_CHECK(NULL != data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer, "GPU workspace allocation");
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_size = workspace_size;

            /* Create handle_cublas_gpu */
            status = cublasCreate(&handle_cublas_gpu);
            HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == status, "cublasCreate");
            cublasSetStream( handle_cublas_gpu, cuda_stream->cuda_stream );
            HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == status, "cuBLAS handle setup");
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas = handle_cublas_gpu;

            /* Create handle_cublas_gpu_deviceptr */
            status = cublasCreate(&handle_cublas_gpu_deviceptr);
            HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == status, "cublasCreate");
            cublasSetStream( handle_cublas_gpu_deviceptr, cuda_stream->cuda_stream );
            status = cublasSetPointerMode(handle_cublas_gpu_deviceptr, CUBLAS_POINTER_MODE_DEVICE);
            HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == status, "cuBLAS handle setup");
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas_deviceptr = handle_cublas_gpu_deviceptr;

            /* Create handle_cublas_tensor */
            status = cublasCreate(&handle_cublas_tensor);
            HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == status, "cublasCreate");
            HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == status, "cuBLAS handle setup");
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
            status = cublasSetMathMode(handle_cublas_tensor, CUBLAS_DEFAULT_MATH); /* Set the math mode to allow cuBLAS to use Tensor Cores */
            HICMA_GPU_HANDLE_CHECK(CUBLAS_STATUS_SUCCESS == status, "cuBLAS handle setup");
#endif
            data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas_tensor = handle_cublas_tensor;
            //printf("i %d j %d : %p\n", i, j, data->ws_gpu->gpu_workspace[i].stream_workspace[j].handle_cublas_tensor);

            /* GPU buffers for A, B and C: a full tile, or the U/V pair of a
             * low-rank operand converted to another precision (mb x 2*rank
             * with rank <= maxrank), whichever is larger */
            {
                size_t cols = (2 * maxrank > nb) ? (size_t)(2 * maxrank) : (size_t)nb;
                size_t full_tile_bytes = (size_t)mb * cols * sizeof(double);
                data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_A = zone_malloc( cuda_device->super.memory, full_tile_bytes );
                HICMA_GPU_HANDLE_CHECK(NULL != data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_A, "GPU workspace allocation");

                data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_B = zone_malloc( cuda_device->super.memory, full_tile_bytes );
                HICMA_GPU_HANDLE_CHECK(NULL != data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_B, "GPU workspace allocation");

                data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_C = zone_malloc( cuda_device->super.memory, full_tile_bytes );
                HICMA_GPU_HANDLE_CHECK(NULL != data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_C, "GPU workspace allocation");
            }

            {
                if( maxrank > 0 ) {
                    data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_mbr =
                        zone_malloc( cuda_device->super.memory, mb * maxrank * sizeof(double) );
                    HICMA_GPU_HANDLE_CHECK(NULL != data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_mbr, "GPU workspace allocation");

                    data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_rr =
                        zone_malloc( cuda_device->super.memory, maxrank * maxrank * sizeof(double) );
                    HICMA_GPU_HANDLE_CHECK(NULL != data->ws_gpu->gpu_workspace[i].stream_workspace[j].gpu_buffer_rr, "GPU workspace allocation");
                }
            }

#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
            hicma_gpu_handles_warmup(&data->ws_gpu->gpu_workspace[i].stream_workspace[j], cuda_stream->cuda_stream, mb);
#endif
        }

        /* GEMM operand caches shared by the device's streams; device memory
         * is taken only for conversions that happen */
        {
            parsec_potrf_gpu_workspace_t *gw = &data->ws_gpu->gpu_workspace[i];
            if( params->fp8_enabled ) {
                gw->operand_cache[HICMA_OPC_FP8] = hicma_operand_cache_create(cuda_device, params, mb,
                        (size_t)hicma_fp8_tile_bytes(mb));
            }
            gw->operand_cache[HICMA_OPC_2B] = hicma_operand_cache_create(cuda_device, params, mb, (size_t)mb * nb * 2);
            gw->operand_cache[HICMA_OPC_4B] = hicma_operand_cache_create(cuda_device, params, mb, (size_t)mb * nb * 4);
            for(int j = 2; j < cuda_device->super.max_exec_streams; j++) {
                gw->stream_workspace[j].operand_cache = gw->operand_cache;
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
    (void)kind_of_cholesky;
    /* workspace_memory_free touches the stream workspaces of GPU devices only,
     * then releases the per-device arrays and the struct, which
     * gpu_temporay_buffer_init allocated even when no GPU was found */
    if( NULL != data->ws_gpu ) {
        workspace_memory_free( data->ws_gpu );
        data->ws_gpu = NULL;
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
