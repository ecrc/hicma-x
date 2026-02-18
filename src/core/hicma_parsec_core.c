/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"
#include "hicma_kernel_time.h"
#include "potrf_L_dense_mp_gpu_fp8_adaptive.h"

/* Count conversions per worker/stream; slot 0 is reduced after execution. */
void hicma_parsec_count_datatype_conversion(hicma_parsec_params_t *params_tlr,
                                            int counter_id)
{
    if( params_tlr->verbose > 1 ) {
        params_tlr->nb_datatype_conversions[counter_id] += 1;
    }
}

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
int hicma_parsec_gpu_counter_id(parsec_device_cuda_module_t *cuda_device,
                                parsec_cuda_exec_stream_t *cuda_stream)
{
    return cuda_device->cuda_index * PARSEC_GPU_MAX_STREAMS
         + lookup_gpu_workspace_j(cuda_device, cuda_stream);
}
#endif

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
/* Global variable from device_cuda_component.c in parsec runtime */
extern int parsec_device_cuda_enabled;
#endif
#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
/* Global variable from device_hip_component.c in parsec runtime */
extern int parsec_device_hip_enabled;
#endif

static void *hicma_parsec_pinned_memory_allocate(size_t size)
{
    void *ptr = NULL;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    if( cudaSuccess != cudaMallocHost(&ptr, size) ) {
        return NULL;
    }
#elif defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    if( hipSuccess != hipHostMalloc(&ptr, size, hipHostMallocDefault) ) {
        return NULL;
    }
#else
    ptr = malloc(size);
#endif
    return ptr;
}

static void hicma_parsec_pinned_memory_free(void *ptr)
{
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    cudaFreeHost(ptr);
#elif defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    hipHostFree(ptr);
#else
    free(ptr);
#endif
}

void hicma_parsec_arenas_use_pinned_memory(parsec_arena_datatype_t *arenas,
                                           int arena_count)
{
    for( int i = 0; i < arena_count; i++ ) {
        parsec_arena_t *arena = arenas[i].arena;
        if( NULL == arena ) {
            continue;
        }
        arena->data_malloc = hicma_parsec_pinned_memory_allocate;
        arena->data_free = hicma_parsec_pinned_memory_free;
    }
}

#if HAVE_FP8
static int hicma_parsec_check_cublaslt_status(cublasStatus_t status,
                                              const char *op,
                                              int m, int n, int k)
{
    if( CUBLAS_STATUS_SUCCESS == status ) {
        return 0;
    }

    fprintf(stderr, "%s failed in FP8 GEMM (%d, %d, %d): cublas status %d\n",
            op, m, n, k, (int)status);
    return -1;
}
#endif

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
#define HICMA_GPU_COPY_ATOMIC_SENTINEL 1024

static int hicma_gpu_copy_used_by_task(parsec_gpu_task_t *gpu_task,
                                       parsec_data_copy_t *copy,
                                       parsec_data_copy_t *keep_copy)
{
    parsec_task_t *task = NULL;
    int nflows = 0;

    if( copy == keep_copy ) {
        return 1;
    }
    if( NULL == gpu_task || NULL == gpu_task->ec || NULL == copy ) {
        return 0;
    }

    task = gpu_task->ec;
    nflows = (NULL != task->task_class) ? task->task_class->nb_flows : 0;
    for( int i = 0; i < nflows; i++ ) {
        parsec_data_copy_t *in = task->data[i].data_in;
        parsec_data_copy_t *out = task->data[i].data_out;
        if( copy == in || copy == out ) {
            return 1;
        }
        if( NULL != copy->original ) {
            if( NULL != in && in->original == copy->original ) {
                return 1;
            }
            if( NULL != out && out->original == copy->original ) {
                return 1;
            }
        }
    }
    return 0;
}

static int hicma_gpu_evict_one_lru_copy(parsec_device_gpu_module_t *gpu_device,
                                        parsec_gpu_task_t *gpu_task,
                                        parsec_data_copy_t *keep_copy,
                                        parsec_data_copy_t **cycling)
{
    parsec_gpu_data_copy_t *lru_gpu_elem = NULL;
    parsec_data_t *oldmaster = NULL;

    lru_gpu_elem = (parsec_gpu_data_copy_t *)parsec_list_pop_front(&gpu_device->gpu_mem_lru);
    if( NULL == lru_gpu_elem ) {
        return 0;
    }
    PARSEC_LIST_ITEM_SINGLETON(lru_gpu_elem);

    if( *cycling == lru_gpu_elem ) {
        parsec_list_push_front(&gpu_device->gpu_mem_lru, (parsec_list_item_t *)lru_gpu_elem);
        return 0;
    }

    if( hicma_gpu_copy_used_by_task(gpu_task, lru_gpu_elem, keep_copy) ) {
        parsec_list_push_back(&gpu_device->gpu_mem_lru, &lru_gpu_elem->super);
        if( NULL == *cycling ) {
            *cycling = lru_gpu_elem;
        }
        return -1;
    }

    /* In-use copies will be returned to the LRU by their owner. */
    if( 0 != lru_gpu_elem->readers ) {
        return -1;
    }

    if( lru_gpu_elem->super.super.obj_reference_count > 1
            || 0 == (lru_gpu_elem->flags & PARSEC_DATA_FLAG_PARSEC_OWNED)
            || NULL == lru_gpu_elem->device_private ) {
        parsec_list_push_back(&gpu_device->gpu_mem_lru, &lru_gpu_elem->super);
        if( NULL == *cycling ) {
            *cycling = lru_gpu_elem;
        }
        return -1;
    }

    oldmaster = lru_gpu_elem->original;
    if( NULL != oldmaster ) {
        if( !parsec_atomic_trylock(&oldmaster->lock) ) {
            parsec_list_push_back(&gpu_device->gpu_mem_lru, &lru_gpu_elem->super);
            if( NULL == *cycling ) {
                *cycling = lru_gpu_elem;
            }
            return -1;
        }
        if( !parsec_atomic_cas_int32(&lru_gpu_elem->readers, 0, -HICMA_GPU_COPY_ATOMIC_SENTINEL) ) {
            parsec_list_push_back(&gpu_device->gpu_mem_lru, &lru_gpu_elem->super);
            if( NULL == *cycling ) {
                *cycling = lru_gpu_elem;
            }
            parsec_atomic_unlock(&oldmaster->lock);
            return -1;
        }
        int do_unlock = oldmaster->super.obj_reference_count != 1;
        parsec_data_copy_detach(oldmaster, lru_gpu_elem, gpu_device->super.device_index);
        parsec_atomic_wmb();
        if( do_unlock ) {
            parsec_atomic_unlock(&oldmaster->lock);
        }
    }

    zone_free(gpu_device->memory, lru_gpu_elem->device_private);
    lru_gpu_elem->device_private = NULL;
    gpu_device->super.nb_evictions++;
    gpu_device->data_avail_epoch++;
    PARSEC_OBJ_RELEASE(lru_gpu_elem);
    return 1;
}

static void *hicma_gpu_zone_malloc_with_evict(parsec_device_gpu_module_t *gpu_device,
                                              parsec_gpu_task_t *gpu_task,
                                              parsec_data_copy_t *keep_copy,
                                              size_t target_bytes)
{
    parsec_data_copy_t *cycling = NULL;
    void *ptr = zone_malloc(gpu_device->memory, target_bytes);

    while( NULL == ptr ) {
        int evicted = hicma_gpu_evict_one_lru_copy(gpu_device, gpu_task, keep_copy, &cycling);
        if( 0 == evicted ) {
            break;
        }
        if( evicted < 0 ) {
            continue;
        }
        ptr = zone_malloc(gpu_device->memory, target_bytes);
    }
    return ptr;
}

static void hicma_update_tile_nb_elts(parsec_data_copy_t *active_copy,
                                      parsec_gpu_task_t *gpu_task,
                                      size_t target_bytes)
{
    parsec_task_t *task = NULL;
    int nflows = 0;

    if( NULL != active_copy && NULL != active_copy->original ) {
        active_copy->original->nb_elts = target_bytes;
    }
    if( NULL == gpu_task || NULL == gpu_task->ec || NULL == gpu_task->ec->task_class ) {
        return;
    }

    task = gpu_task->ec;
    nflows = task->task_class->nb_flows;
    for( int i = 0; i < nflows; i++ ) {
        parsec_data_copy_t *in = task->data[i].data_in;
        parsec_data_copy_t *out = task->data[i].data_out;
        int match = (out == active_copy) || (in == active_copy);
        if( !match && NULL != active_copy && NULL != active_copy->original ) {
            match = (NULL != out && out->original == active_copy->original)
                 || (NULL != in && in->original == active_copy->original);
        }
        if( match ) {
            gpu_task->flow_nb_elts[i] = target_bytes;
        }
    }
}

static int hicma_reallocate_tile_on_gpu(parsec_device_cuda_module_t *cuda_device,
                                        parsec_gpu_task_t *gpu_task,
                                        parsec_data_copy_t *active_copy,
                                        void *expected_ptr,
                                        size_t target_bytes,
                                        int m, int n, int k,
                                        void **new_ptr_out)
{
    parsec_device_gpu_module_t *gpu_device = &cuda_device->super;
    void *old_ptr = NULL;
    void *new_ptr = NULL;
    int freed_old = 0;

    if( NULL == active_copy ) {
        fprintf(stderr, "Missing device copy before FP64 promotion in GEMM runtime decision (%d, %d, %d)\n",
                m, n, k);
        return -1;
    }

    if( NULL != expected_ptr && expected_ptr != active_copy->device_private ) {
        fprintf(stderr, "Pointer mismatch before FP64 promotion in GEMM runtime decision (%d, %d, %d)\n",
                m, n, k);
        return -1;
    }

    if( NULL != active_copy->original
            && active_copy->original->nb_elts >= target_bytes
            && NULL != active_copy->device_private ) {
        hicma_update_tile_nb_elts(active_copy, gpu_task, target_bytes);
        *new_ptr_out = active_copy->device_private;
        return 0;
    }

    old_ptr = active_copy->device_private;

    /* Allocate the larger buffer first so a failed grow cannot drop C. */
    new_ptr = hicma_gpu_zone_malloc_with_evict(gpu_device, gpu_task, active_copy, target_bytes);

    /* Last resort: the DP GEMM result already lives in the C workspace, so the
     * old SP buffer can be released to create a contiguous FP64 slot. */
    if( NULL == new_ptr && NULL != old_ptr ) {
        zone_free(gpu_device->memory, old_ptr);
        active_copy->device_private = NULL;
        old_ptr = NULL;
        freed_old = 1;
        new_ptr = hicma_gpu_zone_malloc_with_evict(gpu_device, gpu_task, active_copy, target_bytes);
    }

    if( NULL == new_ptr ) {
        fprintf(stderr, "Failed to allocate FP64 dense C tile in GEMM runtime decision (%d, %d, %d)\n",
                m, n, k);
        return -1;
    }

    if( !freed_old && NULL != old_ptr && old_ptr != new_ptr ) {
        zone_free(gpu_device->memory, old_ptr);
    }

    active_copy->device_private = new_ptr;
    hicma_update_tile_nb_elts(active_copy, gpu_task, target_bytes);
    *new_ptr_out = new_ptr;
    return 0;
}
#endif /* PARSEC_HAVE_DEV_CUDA_SUPPORT || PARSEC_HAVE_DEV_HIP_SUPPORT */

static int hicma_reallocate_tile_on_cpu(parsec_data_copy_t *cpu_copy,
                                        size_t target_bytes,
                                        int m, int n, int k)
{
    void *new_cpu_ptr = NULL;

    if( NULL == cpu_copy ) {
        return 0;
    }

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    if( parsec_device_cuda_enabled > 0 ) {
        if( cudaSuccess != cudaMallocHost(&new_cpu_ptr, target_bytes) ) {
            fprintf(stderr, "Failed to reallocate host C tile to FP64 in GEMM runtime decision (%d, %d, %d)\n",
                    m, n, k);
            return -1;
        }
    } else
#endif
#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    if( parsec_device_hip_enabled > 0 ) {
        if( hipSuccess != hipHostMalloc(&new_cpu_ptr, target_bytes, hipHostMallocDefault) ) {
            fprintf(stderr, "Failed to reallocate host C tile to FP64 in GEMM runtime decision (%d, %d, %d)\n",
                    m, n, k);
            return -1;
        }
    } else
#endif
    {
        new_cpu_ptr = malloc(target_bytes);
        if( NULL == new_cpu_ptr ) {
            fprintf(stderr, "Failed to reallocate host C tile to FP64 in GEMM runtime decision (%d, %d, %d)\n",
                    m, n, k);
            return -1;
        }
    }

    if( NULL != cpu_copy->device_private ) {
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        if( parsec_device_cuda_enabled > 0 ) {
            cudaFreeHost(cpu_copy->device_private);
        } else
#endif
#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        if( parsec_device_hip_enabled > 0 ) {
            hipHostFree(cpu_copy->device_private);
        } else
#endif
        {
            free(cpu_copy->device_private);
        }
    }

    cpu_copy->device_private = new_cpu_ptr;
    return 0;
}

/**
 * @brief Allocates memory for HICMA computations with GPU acceleration support
 * 
 * This function provides unified memory allocation for both CPU and GPU computations.
 * It automatically selects the appropriate allocation method based on available hardware:
 * - GPU-accelerated: Uses pinned host memory for better GPU transfer performance
 * - CPU-only: Uses standard malloc for regular host memory
 * 
 * @param[out] A Pointer to the allocated memory buffer
 * @param[in] nb_elements Number of elements to allocate
 * @param[in] allocate_type Data type string (e.g., "double", "float", "int")
 * @param[out] data_size Total size of allocated memory in bytes
 */
void hicma_parsec_core_memory_allocation(void **A, size_t nb_elements, char *allocate_type, size_t *data_size) {
    // Calculate total memory size based on data type and element count
    *data_size = nb_elements * get_datatype_size(allocate_type);  
    size_t data_size_ = *data_size;
    
    // Check if GPU acceleration is available and enabled
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    if( parsec_device_cuda_enabled > 0 ) {
#elif defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    if( parsec_device_hip_enabled > 0 ) {
#endif
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        // Allocate pinned host memory for CUDA GPU acceleration
        // Pinned memory provides better transfer performance between host and GPU
        cudaMallocHost(A, data_size_);
#endif

#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        // Allocate pinned host memory for HIP/ROCm GPU acceleration
        // hipHostMallocDefault provides optimal memory allocation for GPU transfers
        hipHostMalloc(A, data_size_, hipHostMallocDefault);
        // Note: hipHostMallocNumaUser could be used for NUMA-aware allocation
        // References:
        // https://github.com/ROCm-Developer-Tools/HIP/issues/2475
        // https://rocm-developer-tools.github.io/HIP/group__GlobalDefs.html
#endif
    }
    else
#endif
    {
        // Standard CPU-only allocation using regular malloc
        *A = malloc(data_size_);
    }
}

/**
 * @brief Prints progress information for SYRK (Symmetric Rank-K update) operations
 * 
 * This function provides progress monitoring for SYRK operations during Cholesky factorization.
 * It calculates and displays completion percentage, elapsed time, and estimated total time.
 * Progress is printed at regular intervals to avoid excessive output.
 * 
 * @param[in] NT Total number of tiles in the matrix
 * @param[in] k Current tile index (0-based)
 * @param[in] start_time_syrk Starting time of the SYRK operation (in seconds)
 */
void hicma_parsec_print_process_syrk( int NT, int k, double start_time_syrk ) {
    // Print progress at most 100 times during the entire operation
    int total_time_print = 100;
    
    // Check if we should print progress (at start, end, or regular intervals)
    if( 0 == k || NT-1 == k || NT/total_time_print ) {
        struct timeval tstart;
        gettimeofday(&tstart, NULL);
        
        // Calculate elapsed time since the start of SYRK operation
        double start_time = tstart.tv_sec + tstart.tv_usec / 1.0e6 - start_time_syrk;
        
        // Estimate total time based on current progress
        // Formula: total_time = elapsed_time * total_tiles / completed_tiles
        double estimated_time = start_time * NT / (k+1);
        
        // Print progress information at key points
        if( 0 == k || NT-1 == k || 0 == k % (NT/total_time_print) ) {
            fprintf(stderr, "In potrf, k %d, %.2lf %% is finished; time: %lf of estimated total %lf\n",
                    k, 100.0*k/NT, start_time, estimated_time);
        }
    }
}

/**
 * @brief Prints progress information for POTRF (Cholesky factorization) operations
 * 
 * This function provides progress monitoring for the main Cholesky factorization process.
 * It uses a more sophisticated time estimation algorithm that accounts for the triangular
 * structure of the factorization, providing more accurate time estimates than simple
 * linear extrapolation.
 * 
 * @param[in] NT Total number of tiles in the matrix
 * @param[in] k Current tile index (0-based)
 * @param[in] start_time_potrf Starting time of the POTRF operation (in seconds)
 */
void hicma_parsec_print_process( int NT, int k, double start_time_potrf ) {
    // Print progress at most 100 times during the entire operation
    int total_time_print = 100;
    
    // Check if we should print progress (at start, end, or regular intervals)
    if( 0 == k || NT-1 == k || NT/total_time_print ) {
        struct timeval tstart;
        gettimeofday(&tstart, NULL);
        
        // Calculate elapsed time since the start of POTRF operation
        double start_time = tstart.tv_sec + tstart.tv_usec / 1.0e6 - start_time_potrf;
        
        // Initialize estimated time with current elapsed time
        double estimated_time = start_time;
        
        // Use triangular factorization time estimation for more accurate prediction
        // This accounts for the fact that Cholesky factorization has varying computational
        // complexity across different parts of the matrix due to its triangular structure
        if( k != 0 && 2*NT-2-k != 0 && k != NT-1 ) {
            // Triangular time estimation formula: accounts for decreasing work per column
            estimated_time = start_time * NT * (NT-1) / k / (2*NT-2-k);
        }

        // Print progress information at key points
        if( 0 == k || NT-1 == k || 0 == k % (NT/total_time_print) ) {
            fprintf(stderr, "In potrf, k %d, %.2lf %% is finished; time: %lf of estimated total %lf\n",
                    k, 100.0*k/NT, start_time, estimated_time);
        }
    }
}

/**
 * @brief Counts floating-point operations for TRSM (Triangular Solve) operations
 * 
 * TRSM performs triangular matrix solve: B = alpha * op(A)^(-1) * B
 * This function tracks both computational operations and communication costs
 * for performance analysis and load balancing in the PaRSEC runtime.
 * 
 * @param[in] descA Matrix descriptor for the triangular matrix A
 * @param[in,out] params_tlr HICMA parameters containing operation counters
 * @param[in] m Row index of the current tile
 * @param[in] k Column index of the current tile
 * @param[in] th_id Thread ID for per-thread operation counting
 * @param[in] tempmm Actual tile size in rows
 * @param[in] Arank Rank of matrix A (for low-rank tiles)
 */
void hicma_parsec_op_count_trsm( parsec_tiled_matrix_t *descA,
        hicma_parsec_params_t *params_tlr,
        int m, int k, int th_id, int tempmm, int Arank ) {

    /* Update critical path communication cost for TRSM operations
     * Critical path operations are those that cannot be parallelized
     * and directly affect the total execution time */
    if( m == k+1 ) {
        // Communication cost: 2 * rank * tile_size (send + receive)
        params_tlr->critical_path_trsm_message += Arank * tempmm * 2;
    }

    /* Count floating-point operations based on tile structure */
    if( IS_DENSE(m, k) ) {
        // Dense tile: standard TRSM operation count
        // Operation: B = A^(-1) * B where A is tempmm x tempmm triangular
        unsigned long int cnt = hicma_parsec_op_counts('t', tempmm, tempmm, 1 /*side*/, 0);
        params_tlr->op_band[th_id] += cnt;
        
        // Classify operations as critical path or off-path
        if( 1 == m-k )
            params_tlr->op_path[th_id] += cnt;      // Critical path operation
        else
            params_tlr->op_offpath[th_id] += cnt;   // Off-path operation
    } else {
        // Low-rank tile: reduced operation count due to rank structure
        // Operation: B = A^(-1) * B where A is tempmm x Arank low-rank
        unsigned long int cnt = hicma_parsec_op_counts('t', tempmm, Arank, 1 /*side*/, 0);
        params_tlr->op_offband[th_id] += cnt;
        
        // Classify operations as critical path or off-path
        if( 1 == m-k )
            params_tlr->op_path[th_id] += cnt;      // Critical path operation
        else
            params_tlr->op_offpath[th_id] += cnt;   // Off-path operation
    }
}


/**
 * @brief Counts floating-point operations for SYRK (Symmetric Rank-K update) operations
 * 
 * SYRK performs symmetric rank-k update: C = alpha * A * A^T + beta * C
 * This function tracks computational operations for both dense and low-rank tiles,
 * with special handling for the low-rank case that involves multiple matrix multiplications.
 * 
 * @param[in] descA Matrix descriptor for the input matrix A
 * @param[in,out] params_tlr HICMA parameters containing operation counters
 * @param[in] m Row index of the current tile
 * @param[in] k Column index of the current tile
 * @param[in] th_id Thread ID for per-thread operation counting
 * @param[in] tempmm Actual tile size in rows
 * @param[in] rank Rank of the low-rank matrix (for low-rank tiles)
 */
void hicma_parsec_op_count_syrk( parsec_tiled_matrix_t *descA,
        hicma_parsec_params_t *params_tlr,
        int m, int k, int th_id, int tempmm, int rank ) {

    /* Count floating-point operations based on tile structure */
    if( IS_DENSE(m, k) ) {
        // Dense tile: standard SYRK operation count
        // Operation: C = C + alpha * A * A^T where A is tempmm x tempmm
        unsigned long int cnt = hicma_parsec_op_counts('m', tempmm, tempmm, tempmm, 0);
        params_tlr->op_band[th_id] += cnt;
        
        // Classify operations as critical path or off-path
        if( 1 == m-k )
            params_tlr->op_path[th_id] += cnt;      // Critical path operation
        else
            params_tlr->op_offpath[th_id] += cnt;   // Off-path operation
    } else {
        // Low-rank tile: decomposed SYRK operation
        // Mathematical operation: C = C + alpha * A * A^T
        // Low-rank decomposition: A = A^u * A^v^T
        // Therefore: C = C + alpha * (A^u * A^v^T) * (A^u * A^v^T)^T
        //            C = C + alpha * A^u * (A^v * A^v^T) * A^u^T
        
        unsigned long int cnt = 0;
        
        // Step 1: A^v * A^v^T (rank x rank matrix multiplication)
        cnt += hicma_parsec_op_counts('m', rank, rank, tempmm, 0);
        
        // Step 2: A^u * (A^v * A^v^T) (tempmm x rank x rank matrix multiplication)
        cnt += hicma_parsec_op_counts('m', tempmm, rank, rank, 0);
        
        // Step 3: (A^u * (A^v * A^v^T)) * A^u^T (tempmm x tempmm x rank matrix multiplication)
        cnt += hicma_parsec_op_counts('m', tempmm, tempmm, rank, 0);
        
        params_tlr->op_band[th_id] += cnt;
        
        // Classify operations as critical path or off-path
        if( 1 == m-k )
            params_tlr->op_path[th_id] += cnt;      // Critical path operation
        else
            params_tlr->op_offpath[th_id] += cnt;   // Off-path operation
    }
}

/**
 * @brief Counts floating-point operations for GEMM (General Matrix Multiply) when result C is dense
 * 
 * GEMM performs general matrix multiplication: C = alpha * A * B + beta * C
 * This function handles various combinations of dense and low-rank input matrices A and B,
 * while the result matrix C is always dense. Different operation counts are computed
 * based on the structure of the input matrices.
 * 
 * @param[in] descA Matrix descriptor for the input matrix A
 * @param[in,out] params_tlr HICMA parameters containing operation counters
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile
 * @param[in] k Inner dimension index
 * @param[in] th_id Thread ID for per-thread operation counting
 * @param[in] tempmm Actual tile size in rows
 * @param[in] Crank Rank of matrix C (unused for dense C)
 * @param[in] Arank Rank of matrix A (for low-rank tiles)
 * @param[in] Brank Rank of matrix B (for low-rank tiles)
 */
void hicma_parsec_op_count_gemm_dense( parsec_tiled_matrix_t *descA,
        hicma_parsec_params_t *params_tlr,
        int m, int n, int k, int th_id, int tempmm,
        int Crank, int Arank, int Brank ) {

    /* Count operations based on the structure of input matrices A and B */
    
    // Case 1: All matrices are dense (A, B, and C)
    if ( IS_DENSE(m, n) && IS_DENSE(m, k) && IS_DENSE(n, k) ) {
        // Standard dense GEMM: C = A * B where all matrices are tempmm x tempmm
        unsigned long int cnt = hicma_parsec_op_counts('m', tempmm, tempmm, tempmm, 0);
        params_tlr->op_band[th_id] += cnt; 
        params_tlr->op_offpath[th_id] += cnt; 
    } 
    // Case 2: A is low-rank, B is dense, C is dense
    else if( IS_DENSE(m, n) && !IS_DENSE(m, k) && IS_DENSE(n, k) ) {
        // Low-rank A times dense B: C = (A^u * A^v^T) * B
        // This requires two matrix multiplications: A^u * (A^v^T * B)
        unsigned long int cnt = hicma_parsec_op_counts('m', tempmm, tempmm, Arank, 0) * 2; 
        params_tlr->op_band[th_id] += cnt; 
        params_tlr->op_offpath[th_id] += cnt; 
    } 
    // Case 3: Both A and B are low-rank, C is dense
    else if( IS_DENSE(m, n) && !IS_DENSE(m, k) && !IS_DENSE(n, k) ) {
        // Low-rank A times low-rank B: C = (A^u * A^v^T) * (B^u * B^v^T)
        // This requires three matrix multiplications:
        // 1. A^v^T * B^u (Arank x Brank)
        // 2. A^u * (A^v^T * B^u) (tempmm x Arank x Brank)  
        // 3. (A^u * (A^v^T * B^u)) * B^v^T (tempmm x tempmm x Brank)
        unsigned long int cnt = hicma_parsec_op_counts('m', tempmm, tempmm, hicma_parsec_min(Arank, Brank), 0)
                                + hicma_parsec_op_counts('m', tempmm, Arank, Brank, 0) * 2; 
        params_tlr->op_band[th_id] += cnt; 
        params_tlr->op_offpath[th_id] += cnt; 
    } 
}


/**
 * @brief Initializes rank tracking for low-rank tiles in Cholesky factorization
 * 
 * This function sets up comprehensive rank tracking for low-rank tiles during
 * the initial phase of Cholesky factorization. It allocates memory and initializes
 * tracking fields to monitor rank evolution throughout the factorization process.
 * 
 * @param[in] descA Matrix descriptor for the main matrix A
 * @param[in] descRank Matrix descriptor for rank tracking data
 * @param[in] params_tlr HICMA parameters (unused in this function)
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile (equals k for TRSM operations)
 * @param[in] k Current factorization step (0 for initialization)
 * @param[in] Crank Current rank of the tile
 */
void hicma_parsec_gather_rank_initial( 
        parsec_tiled_matrix_t* descA,
        parsec_tiled_matrix_t* descRank,
        hicma_parsec_params_t *params_tlr,
        int m, int n, int k, int Crank )
{
#if PRINT_RANK <= 1
    return;
#endif

    /* Only process low-rank tiles and only on the owning rank
     * This function is called at the beginning of factorization (k == 0) */
    if( 0 == k && !IS_DENSE(m, n) && descRank->super.myrank == descRank->super.rank_of(&descRank->super, m, n) ) {
        /* Create new data copy and allocate memory for rank tracking
         * The rank buffer stores multiple fields for comprehensive tracking */
        parsec_data_copy_t *my_data_copy = parsec_data_copy_new(descRank->super.data_of(&descRank->super, m, n), 0, parsec_datatype_int_t, PARSEC_DATA_FLAG_PARSEC_MANAGED);
        my_data_copy->device_private = calloc(RANK_MAP_BUFF, sizeof(int));

        /* Initialize all rank tracking fields with the current rank value
         * Field 0: Initial rank, Field 1: Minimum rank, Field 2: Maximum rank, Field 3: Final rank */
        ((int *)((descRank->super.data_of(&descRank->super, m, n))->device_copies[0]->device_private))[0] = Crank;
        ((int *)((descRank->super.data_of(&descRank->super, m, n))->device_copies[0]->device_private))[1] = Crank;
        ((int *)((descRank->super.data_of(&descRank->super, m, n))->device_copies[0]->device_private))[2] = Crank;
        ((int *)((descRank->super.data_of(&descRank->super, m, n))->device_copies[0]->device_private))[3] = Crank;
    }
}


/**
 * @brief Updates rank tracking statistics for low-rank tiles during Cholesky factorization
 * 
 * This function maintains comprehensive rank statistics throughout the factorization process.
 * It tracks the minimum, maximum, and final ranks for each low-rank tile to provide
 * insights into rank evolution and compression effectiveness.
 * 
 * @param[in] descA Matrix descriptor for the main matrix A
 * @param[in] descRank Matrix descriptor for rank tracking data
 * @param[in] params_tlr HICMA parameters (unused in this function)
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile
 * @param[in] k Current factorization step
 * @param[in] Crank Current rank of the tile
 */
void hicma_parsec_gather_rank_final(
        parsec_tiled_matrix_t* descA,
        parsec_tiled_matrix_t* descRank,
        hicma_parsec_params_t *params_tlr,
        int m, int n, int k, int Crank )
{
#if PRINT_RANK <= 1
    return;
#endif

    /* Only process low-rank tiles and only on the owning rank */
    if( !IS_DENSE(m, n) && descRank->super.myrank == descRank->super.rank_of(&descRank->super, m, n) ) {
        /* Update minimum rank if current rank is smaller
         * This tracks the best compression achieved during factorization */
        if( Crank < ((int *)((descRank->super.data_of(&descRank->super, m, n))->device_copies[0]->device_private))[1] )
            ((int *)((descRank->super.data_of(&descRank->super, m, n))->device_copies[0]->device_private))[1] = Crank;

        /* Update maximum rank if current rank is larger
         * This tracks the worst case rank during factorization */
        if( Crank > ((int *)((descRank->super.data_of(&descRank->super, m, n))->device_copies[0]->device_private))[2] )
            ((int *)((descRank->super.data_of(&descRank->super, m, n))->device_copies[0]->device_private))[2] = Crank;

        /* Record final rank when tile processing is complete (n-1 == k)
         * This captures the final rank after all factorization steps */
        if( n-1 == k )
            ((int *)((descRank->super.data_of(&descRank->super, m, n))->device_copies[0]->device_private))[3] = Crank;
    }
}


/**
 * @brief CPU implementation of POTRF (Cholesky factorization) for a single tile
 * 
 * This function performs Cholesky factorization on a single diagonal tile of the matrix.
 * It handles both single and double precision arithmetic based on the tile's data type.
 * The function is part of the PaRSEC task graph and executes on CPU cores.
 * 
 * @param[in] descA Matrix descriptor for the input/output matrix A
 * @param[in,out] params_tlr HICMA parameters containing factorization settings
 * @param[in] es PaRSEC execution stream for this task
 * @param[in,out] T Pointer to the tile data to be factorized
 * @param[in] k Index of the diagonal tile being factorized
 */
void hicma_parsec_core_potrf_cpu( parsec_tiled_matrix_t* descA,
        hicma_parsec_params_t *params_tlr,
        parsec_execution_stream_t *es,
        void *T, int k ) {

    // Calculate actual tile size (may be smaller for the last tile)
    int tempkm = k == descA->mt-1 ? descA->m - k*descA->mb : descA->mb;
    int iinfo = 0;  // Error code from LAPACK
    int ld_Ag_k = BLKLDD( descA, k );  // Leading dimension of the tile

    /* POTRF stores diagonal tiles in the precision used by the kernel. */
    params_tlr->decisions_storage_final[k*descA->lmt+k] =
        (DENSE_SP == params_tlr->decisions[k*descA->lmt+k]) ? DENSE_SP : DENSE_DP;

    if(DEBUG_INFO) printf("POTRF: %d\n", k);

    /* Print progress information for monitoring factorization progress */
    hicma_parsec_print_process( descA->mt, k, params_tlr->start_time_potrf );

    /* Count floating-point operations for performance analysis
     * Cholesky factorization of an n×n matrix requires O(n³/3) operations */
    unsigned long int cnt = hicma_parsec_op_counts('c', tempkm, 0, 0, 0);
    params_tlr->op_band[es->th_id] += cnt;    // Band operations
    params_tlr->op_path[es->th_id] += cnt;    // Critical path operations

    /* Perform Cholesky factorization based on data type */
    if( DENSE_SP == params_tlr->decisions[k*descA->lmt+k] ) {
        // Single precision Cholesky factorization
        CORE_spotrf( params_tlr->uplo, tempkm, T, ld_Ag_k, &iinfo );
    } else {
        // Double precision Cholesky factorization
        CORE_dpotrf( params_tlr->uplo, tempkm, T, ld_Ag_k, &iinfo );
    }

    /* Handle error reporting for debugging */
    if(params_tlr->verbose > 10 && iinfo != 0){
        printf("%s %d: dpotrf failed with a return value of %d. uplo:%d m:%d A:%p lda:%d\n", 
               __FILE__, __LINE__, iinfo, params_tlr->uplo, tempkm, T, ld_Ag_k);
        fflush(stdout);
    }

    /* Propagate error information up the call stack
     * Only set info if no previous error was recorded */
    if ( iinfo != 0 && params_tlr->info == 0 )
        params_tlr->info = k*descA->mb+iinfo; /* Should return here */
}


/**
 * @brief CPU implementation of TRSM (Triangular Solve) for a single tile
 * 
 * TRSM solves triangular matrix equations: C = alpha * op(A)^(-1) * C
 * This function handles various data types (double, single, half precision) and
 * both dense and low-rank matrix structures. It performs the triangular solve
 * operation as part of the Cholesky factorization process.
 * 
 * @param[in] descA Matrix descriptor for the input matrix A
 * @param[in] descRank Matrix descriptor for rank tracking (optional)
 * @param[in,out] params_tlr HICMA parameters containing factorization settings
 * @param[in] es PaRSEC execution stream for this task
 * @param[in] p_work_full_sp Memory pool for single precision temporary buffers
 * @param[in] T Pointer to the triangular matrix tile A(k,k)
 * @param[in,out] C Pointer to the matrix tile C(m,k) to be updated
 * @param[in] m Row index of the tile being updated
 * @param[in] k Column index (diagonal tile index)
 * @param[in] Crank Rank of the low-rank tile C (for low-rank case)
 */
void hicma_parsec_core_trsm_cpu( parsec_tiled_matrix_t* descA,
        parsec_tiled_matrix_t* descRank,
        hicma_parsec_params_t *params_tlr,
        parsec_execution_stream_t *es,
        parsec_memory_pool_t *p_work_full_sp,
        void *T, void *C, int m, int k, int Crank )
{
    int counter_id = es->th_id;
#if PRINT_RANK > 1
    /* Initialize rank tracking for low-rank tiles */
    hicma_parsec_gather_rank_initial( descA, descRank, params_tlr, m, k, k, Crank );
#endif

    // Calculate actual tile dimensions
    int tempmm = m == descA->mt-1 ? descA->m - m * descA->mb : descA->mb;
    int ldak = BLKLDD( descA, k );  // Leading dimension of triangular matrix T
    int ldam = BLKLDD( descA, m );  // Leading dimension of matrix C
    uint16_t storage_precision = params_tlr->decisions[k*descA->lmt+m];
    if( DENSE_DP != storage_precision && LOW_RANK_DP != storage_precision &&
            LOW_RANK_SP != storage_precision ) {
        /* Dense HP/FP8 decisions are materialized as SP by TRSM. */
        storage_precision = DENSE_SP;
    }
    params_tlr->decisions_storage_final[k*descA->lmt+m] = storage_precision;
    if(DEBUG_INFO) printf("TRSM (%d, %d): %d\n", m, k, Crank);

    /* Handle different data types and matrix structures */
    
    // Case 1: Dense double precision TRSM
    if( DENSE_DP == params_tlr->decisions[k*descA->lmt+m] ) {
        // Standard dense double precision triangular solve
        // Operation: C = C * A^(-T) where A is lower triangular
        CORE_dtrsm(PlasmaRight, PlasmaLower, PlasmaTrans, PlasmaNonUnit,
                   tempmm, descA->mb,
                   (double)1.0, T /*A(k, k)*/, ldak,
                                C /*A(m, k)*/, ldam);

    } 
    // Case 2: Dense single or half precision TRSM
    else if( DENSE_SP == params_tlr->decisions[k*descA->lmt+m] || DENSE_HP == params_tlr->decisions[k*descA->lmt+m] ) {

        /* Convert triangular matrix T to single precision if needed */
        void *T_s = parsec_private_memory_pop( p_work_full_sp );
        void *T_use = T;
        if( DENSE_DP == params_tlr->decisions_send[k*params_tlr->NT+k] ) {
            // Convert double precision T to single precision for computation
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, descA->nb, T, descA->mb, T_s, descA->mb );
            T_use = T_s;
        }

        // Single precision triangular solve
        CORE_strsm(PlasmaRight, PlasmaLower, PlasmaTrans, PlasmaNonUnit,
                   tempmm, descA->mb,
                   (float)1.0, T_use /*A(k, k)*/, ldak,
                               C     /*A(m, k)*/, ldam);

        /* Return temporary buffer to memory pool */
        parsec_private_memory_push( p_work_full_sp, T_s );

    } 
    // Case 3: Low-rank double precision TRSM
    else if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {

        int tempmm = m == descA->mt-1 ? descA->m - m * descA->mb : descA->mb;
        int ldak = BLKLDD( descA, k );
        int ldam = BLKLDD( descA, m );
        // Pointer to the V part of the low-rank matrix C = U * V^T
        void *Cv = (void *)C + descA->mb * Crank * sizeof(double);

        /* Validate rank constraints for memory allocation
         * The rank must not exceed half the tile size to ensure sufficient temporary buffers */
        if( Crank * 2 > descA->mb ) {
            fprintf(stderr, "in TRSM %d %d Crank %d is bigger than half of tile size %d\n", m, k, Crank, descA->mb);
            //exit(1);
        }

        /* Skip computational kernels on off-band tiles for performance optimization */
#if TLR_BOUND_WITHOUT_OFFBAND_GEMM_TRSM
        return;
#endif

        // Low-rank TRSM: solve T * Cv = Cv where T is triangular and Cv is rank x Crank
        CORE_dtrsm(PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaNonUnit,
                tempmm, Crank,
                (double)1.0, T, ldak,
                            Cv, ldam);

    } 
    // Case 4: Low-rank single precision TRSM
    else {
        int tempmm = m == descA->mt-1 ? descA->m - m * descA->mb : descA->mb;
        int ldak = BLKLDD( descA, k );
        int ldam = BLKLDD( descA, m );
        // Pointer to the V part of the low-rank matrix C = U * V^T
        void *Cv = (void *)C + descA->mb * Crank * sizeof(float);

        /* Validate rank constraints for memory allocation */
        if( Crank * 2 > descA->mb ) {
            fprintf(stderr, "in TRSM %d %d Crank %d is bigger than half of tile size %d\n", m, k, Crank, descA->mb);
            //exit(1);
        }

        /* Skip computational kernels on off-band tiles for performance optimization */
#if TLR_BOUND_WITHOUT_OFFBAND_GEMM_TRSM
        return;
#endif

        /* Convert triangular matrix T to single precision */
        void *T_s = parsec_private_memory_pop( p_work_full_sp );
        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
        LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, descA->nb, T, descA->mb, T_s, descA->mb );

        // Low-rank single precision TRSM
        CORE_strsm(PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaNonUnit,
                tempmm, Crank,
                (float)1.0, T_s, ldak,
                             Cv, ldam);

        /* Return temporary buffer to memory pool */
        parsec_private_memory_push( p_work_full_sp, T_s );
    }

    /* Count floating-point operations for performance analysis */
    hicma_parsec_op_count_trsm( descA, params_tlr, m, k, es->th_id, tempmm, Crank );
}


/**
 * @brief Core SYRK (Symmetric Rank-K update) operation on CPU
 * 
 * This function performs the core SYRK operation for Cholesky factorization on CPU.
 * It handles both dense and low-rank matrix operations, with automatic precision
 * conversion based on the decision matrix. The function supports mixed-precision
 * computations and memory pool management for efficient memory usage.
 * 
 * @param[in] descA Pointer to the main matrix descriptor
 * @param[in] descRank Pointer to the rank matrix descriptor
 * @param[in] params_tlr HICMA PaRSEC parameters including decision matrix
 * @param[in] es Execution stream for task execution
 * @param[in] p_work General work memory pool
 * @param[in] p_work_full_dp Double precision work memory pool
 * @param[in] p_work_uv_dp Double precision UV work memory pool
 * @param[in] p_work_mbr Memory pool for matrix block rows
 * @param[in] p_work_rr Memory pool for rank reduction
 * @param[in] T Pointer to the target matrix tile (A(m,m))
 * @param[in] A Pointer to the source matrix tile (A(m,k))
 * @param[in] m Row index of the current tile
 * @param[in] k Column index of the current tile
 * @param[in] rank Rank of the low-rank representation (if applicable)
 */
void hicma_parsec_core_syrk_cpu( parsec_tiled_matrix_t* descA,
        parsec_tiled_matrix_t* descRank,
        hicma_parsec_params_t *params_tlr,
        parsec_execution_stream_t *es,
        parsec_memory_pool_t *p_work,
        parsec_memory_pool_t *p_work_full_dp,
        parsec_memory_pool_t *p_work_uv_dp,
        parsec_memory_pool_t *p_work_mbr,
        parsec_memory_pool_t *p_work_rr,
        void *T, void *A, int m, int k, int rank )
{
    int counter_id = es->th_id;
    int tempmm = m == descA->mt-1 ? descA->m - m*descA->mb : descA->mb;
    int ldam = BLKLDD( descA, m );

    /* No recursive, dense SYRK or low rank LR_SYRK */
    if( IS_DENSE(m, k) ) {
        if( DENSE_DP == params_tlr->decisions[m*descA->lmt+m] ) {
            double *A_d = A;

            if( DENSE_DP != params_tlr->decisions[k*descA->lmt+m] ) {
                A_d = parsec_private_memory_pop( p_work_full_dp );
                hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                LAPACKE_slag2d( LAPACK_COL_MAJOR, descA->mb, descA->nb, A, descA->mb, A_d, descA->mb );
            }

            CORE_dsyrk(PlasmaLower, PlasmaNoTrans,
                    tempmm, descA->mb,
                    (double)-1.0, A_d /*A(m, k)*/, ldam,
                    (double) 1.0, T     /*A(m, m)*/, ldam);

            /* Push back to mempool */
            if( DENSE_DP != params_tlr->decisions[k*descA->lmt+m] ) {
                parsec_private_memory_push( p_work_full_dp, A_d );
            }
        } else {
            CORE_ssyrk(PlasmaLower, PlasmaNoTrans,
                    tempmm, descA->mb,
                    (float)-1.0, A /*A(m, k)*/, ldam,
                    (float) 1.0, T /*A(m, m)*/, ldam);
        }

    } else {
        /* If rank is 0, return */
        if( 0 == rank ) {
            return;
        }

        int ldau = BLKLDD( descA, m );
        int ldav = BLKLDD( descA, m );
        void *Au, *Av, *A_d;

        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            Au = (void *)A;
            Av = (void *)A + descA->mb * rank * sizeof(double);
        } else {
            A_d = parsec_private_memory_pop( p_work_uv_dp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_slag2d( LAPACK_COL_MAJOR, descA->mb, rank * 2, A, descA->mb, A_d, descA->mb );
            Au = (void *)A_d;
            Av = (void *)A_d + descA->mb * rank * sizeof(double);
        }

#define call_hcore_dsyrk 1

#if call_hcore_dsyrk
        void *p_elem_work = parsec_private_memory_pop( p_work );
        flop_counter flops;
        HCORE_dsyrk(PlasmaLower, PlasmaNoTrans,
                tempmm, rank,
                (double)-1.0,
                Au /*A(m, k)*/, ldau,
                Av /*A(m, k)*/, ldav,
                (double) 1.0, T /*A(m, m)*/, ldam, p_elem_work, &flops);
        parsec_private_memory_push( p_work, p_elem_work );

        /* If want to call 3 gemms instead */
#else
        void *p_elem_work_mbr = parsec_private_memory_pop( p_work_mbr );
        void *p_elem_work_rr = parsec_private_memory_pop( p_work_rr );

        /* tmp_rr = trans(Av) * Av */
        CORE_dgemm(PlasmaTrans, PlasmaNoTrans,
                rank, rank, descA->mb,
                (double) 1.0, Av             /*A(k, m)*/, descA->mb,
                              Av             /*A(k, n)*/, descA->mb,
                (double) 0.0, p_elem_work_rr /*A(m, n)*/, rank);

        /* tmp_mbr = tmp_rr * trans(Au) */
        CORE_dgemm(PlasmaNoTrans, PlasmaTrans,
                rank, descA->mb, rank,
                (double) 1.0, p_elem_work_rr  /*A(m, k)*/, rank,
                              Au              /*A(n, k)*/, descA->mb,
                (double) 0.0, p_elem_work_mbr /*A(m, n)*/, rank);

        /* T = T - Au * tmp_mbr */
        CORE_dgemm(PlasmaNoTrans, PlasmaNoTrans,
                descA->mb, descA->mb, rank,
                (double)-1.0, Au              /*A(m, k)*/, descA->mb,
                              p_elem_work_mbr /*A(k, n)*/, rank,
                (double) 1.0, T               /*A(m, n)*/, descA->mb);

        parsec_private_memory_push( p_work_mbr, p_elem_work_mbr );
        parsec_private_memory_push( p_work_rr, p_elem_work_rr );
#endif

        /* Push back to mempool */
        if( LOW_RANK_SP == params_tlr->decisions[k*descA->lmt+m] ) {
            parsec_private_memory_push( p_work_uv_dp, A_d );
        }
    }

    /* Operation count */
    hicma_parsec_op_count_syrk( descA, params_tlr, m, k, es->th_id, tempmm, IS_DENSE(m, k)? 0: rank );
}


/**
 * @brief Core SYRK (Symmetric Rank-K update) operation on CPU
 *        Adaptive decision during runtime
 * 
 * This function performs the core SYRK operation for Cholesky factorization on CPU.
 * It handles both dense and low-rank matrix operations, with automatic precision
 * conversion based on the decision matrix. The function supports mixed-precision
 * computations and memory pool management for efficient memory usage.
 * 
 * @param[in] descA Pointer to the main matrix descriptor
 * @param[in] descRank Pointer to the rank matrix descriptor
 * @param[in] params_tlr HICMA PaRSEC parameters including decision matrix
 * @param[in] es Execution stream for task execution
 * @param[in] p_work General work memory pool
 * @param[in] p_work_full_dp Double precision work memory pool
 * @param[in] p_work_uv_dp Double precision UV work memory pool
 * @param[in] p_work_mbr Memory pool for matrix block rows
 * @param[in] p_work_rr Memory pool for rank reduction
 * @param[in] T Pointer to the target matrix tile (A(m,m))
 * @param[in] A Pointer to the source matrix tile (A(m,k))
 * @param[in] m Row index of the current tile
 * @param[in] k Column index of the current tile
 * @param[in] rank Rank of the low-rank representation (if applicable)
 */
void hicma_parsec_core_syrk_runtime_decision_cpu( parsec_tiled_matrix_t* descA,
        parsec_tiled_matrix_t* descRank,
        hicma_parsec_params_t *params_tlr,
        parsec_execution_stream_t *es,
        parsec_memory_pool_t *p_work,
        parsec_memory_pool_t *p_work_full_dp,
        parsec_memory_pool_t *p_work_uv_dp,
        parsec_memory_pool_t *p_work_mbr,
        parsec_memory_pool_t *p_work_rr,
        void *T, void *A, int m, int k, int rank, void *A_norm )
{
    int counter_id = es->th_id;
    int tempmm = m == descA->mt-1 ? descA->m - m*descA->mb : descA->mb;
    int ldam = BLKLDD( descA, m );
    uint16_t Aprecision = (params_tlr->adaptive_decision_runtime)? (uint16_t)(((double *)A_norm)[1]): -1;

    /* No recursive, dense SYRK or low rank LR_SYRK */
    if( IS_DENSE(m, k) ) {
        if( DENSE_DP == params_tlr->decisions[m*descA->lmt+m] ) {
            double *A_d = A;

            //if( DENSE_DP != params_tlr->decisions[k*descA->lmt+m] ) {
            if( DENSE_DP != Aprecision ) {
                A_d = parsec_private_memory_pop( p_work_full_dp );
                hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                LAPACKE_slag2d( LAPACK_COL_MAJOR, descA->mb, descA->nb, A, descA->mb, A_d, descA->mb );
            }

            CORE_dsyrk(PlasmaLower, PlasmaNoTrans,
                    tempmm, descA->mb,
                    (double)-1.0, A_d /*A(m, k)*/, ldam,
                    (double) 1.0, T     /*A(m, m)*/, ldam);

            /* Push back to mempool */
            if( DENSE_DP != Aprecision ) {
                parsec_private_memory_push( p_work_full_dp, A_d );
            }
        } else {
            fprintf(stderr, "CORE_ssyrk should not be reached!\n");
            CORE_ssyrk(PlasmaLower, PlasmaNoTrans,
                    tempmm, descA->mb,
                    (float)-1.0, A /*A(m, k)*/, ldam,
                    (float) 1.0, T /*A(m, m)*/, ldam);
        }

    } else {
        fprintf(stderr, "You should reach here in hicma_parsec_core_syrk_runtime_decision_cpu!\n");
        /* If rank is 0, return */
        if( 0 == rank ) {
            return;
        }

        int ldau = BLKLDD( descA, m );
        int ldav = BLKLDD( descA, m );
        void *Au, *Av, *A_d;

        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            Au = (void *)A;
            Av = (void *)A + descA->mb * rank * sizeof(double);
        } else {
            A_d = parsec_private_memory_pop( p_work_uv_dp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_slag2d( LAPACK_COL_MAJOR, descA->mb, rank * 2, A, descA->mb, A_d, descA->mb );
            Au = (void *)A_d;
            Av = (void *)A_d + descA->mb * rank * sizeof(double);
        }

#define call_hcore_dsyrk 1

#if call_hcore_dsyrk
        void *p_elem_work = parsec_private_memory_pop( p_work );
        flop_counter flops;
        HCORE_dsyrk(PlasmaLower, PlasmaNoTrans,
                tempmm, rank,
                (double)-1.0,
                Au /*A(m, k)*/, ldau,
                Av /*A(m, k)*/, ldav,
                (double) 1.0, T /*A(m, m)*/, ldam, p_elem_work, &flops);
        parsec_private_memory_push( p_work, p_elem_work );

        /* If want to call 3 gemms instead */
#else
        void *p_elem_work_mbr = parsec_private_memory_pop( p_work_mbr );
        void *p_elem_work_rr = parsec_private_memory_pop( p_work_rr );

        /* tmp_rr = trans(Av) * Av */
        CORE_dgemm(PlasmaTrans, PlasmaNoTrans,
                rank, rank, descA->mb,
                (double) 1.0, Av             /*A(k, m)*/, descA->mb,
                              Av             /*A(k, n)*/, descA->mb,
                (double) 0.0, p_elem_work_rr /*A(m, n)*/, rank);

        /* tmp_mbr = tmp_rr * trans(Au) */
        CORE_dgemm(PlasmaNoTrans, PlasmaTrans,
                rank, descA->mb, rank,
                (double) 1.0, p_elem_work_rr  /*A(m, k)*/, rank,
                              Au              /*A(n, k)*/, descA->mb,
                (double) 0.0, p_elem_work_mbr /*A(m, n)*/, rank);

        /* T = T - Au * tmp_mbr */
        CORE_dgemm(PlasmaNoTrans, PlasmaNoTrans,
                descA->mb, descA->mb, rank,
                (double)-1.0, Au              /*A(m, k)*/, descA->mb,
                              p_elem_work_mbr /*A(k, n)*/, rank,
                (double) 1.0, T               /*A(m, n)*/, descA->mb);

        parsec_private_memory_push( p_work_mbr, p_elem_work_mbr );
        parsec_private_memory_push( p_work_rr, p_elem_work_rr );
#endif

        /* Push back to mempool */
        if( LOW_RANK_SP == params_tlr->decisions[k*descA->lmt+m] ) {
            parsec_private_memory_push( p_work_uv_dp, A_d );
        }
    }

    /* Operation count */
    hicma_parsec_op_count_syrk( descA, params_tlr, m, k, es->th_id, tempmm, IS_DENSE(m, k)? 0: rank );
}


/**
 * @brief Manually converts single precision (SP) floating point values to 8-bit format
 * 
 * This function performs a manual conversion from 32-bit single precision floating point
 * values to 8-bit format by casting the values. This is used for memory optimization
 * and reduced precision computations in HICMA.
 * 
 * @param[in] params_tlr HICMA parameters (currently unused but kept for interface consistency)
 * @param[in] A Input array of single precision floating point values
 * @param[out] A_use Output array containing the converted 8-bit values
 * @param[in] m Number of rows in the matrix
 * @param[in] n Number of columns in the matrix
 * @param[in] mb Block size in rows
 * @param[in] nb Block size in columns
 */
void hicma_parsec_convert_2fp8_bit( hicma_parsec_params_t *params_tlr,
        float *A, float *A_use, int m, int n, int mb, int nb ) {
    
        for( int j = 0; j < nb; j++ ) {
            for( int i = 0; i < mb; i++ ) {
                A_use[j*mb+i] =(uint8_t) (A[j*mb+i]); // __nv_fp8_e4m3(A[j*mb+i]);
            }
        }

} 

/**
 * @brief Prints a tile matrix in a formatted manner for debugging purposes
 * 
 * This function prints the contents of a tile matrix in row-major format,
 * displaying each element with its value. It's primarily used for debugging
 * and verification of matrix computations.
 * 
 * @param[in] A Pointer to the matrix data (row-major storage)
 * @param[in] m Number of rows in the tile
 * @param[in] n Number of columns in the tile
 */
void parsec_print_tilenew(float *A, int m, int n){
      printf("\n");
    for(int i=0;i<m;i++){
       for(int j=0;j<n;j++){
       //printf("[%d,%d]:(%f), ", i, j, tempaa[i*dcA->.super.llm+j]);
          printf("%f, ", A[j*m+i]);
       }
       printf("\n");
    }
}


/**
 * @brief CPU implementation of GEMM (General Matrix Multiply) for dense matrices
 * 
 * This function performs the general matrix multiplication C = C + A * B where
 * all matrices (C, A, B) are dense. It handles different data types (double,
 * single precision, half precision) and supports various memory layouts.
 * The function is part of the HICMA Cholesky factorization kernel.
 * 
 * @param[in] descA Matrix descriptor for matrix A
 * @param[in] descRank Matrix descriptor for rank information
 * @param[in,out] params_tlr HICMA parameters containing operation settings
 * @param[in] es Execution stream for the operation
 * @param[in] p_work Memory pool for general workspace
 * @param[in] p_work_full_dp Memory pool for double precision full matrices
 * @param[in] p_work_full_sp Memory pool for single precision full matrices
 * @param[in] p_work_full_hp Memory pool for half precision full matrices
 * @param[in] p_work_uv_dp Memory pool for double precision U/V matrices
 * @param[in] p_work_uv_sp Memory pool for single precision U/V matrices
 * @param[in] p_work_mbr Memory pool for matrix block rank operations
 * @param[in] p_work_rr Memory pool for rank-rank operations
 * @param[in,out] C Matrix C (result matrix)
 * @param[in] A Matrix A (first operand)
 * @param[in] B Matrix B (second operand)
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile
 * @param[in] k Diagonal index of the current tile
 * @param[in] Crank Rank of matrix C (unused for dense matrices)
 * @param[in] Arank Rank of matrix A (unused for dense matrices)
 * @param[in] Brank Rank of matrix B (unused for dense matrices)
 */
void hicma_parsec_core_gemm_denseC_denseA_denseB_cpu( parsec_tiled_matrix_t* descA,
        parsec_tiled_matrix_t* descRank,
        hicma_parsec_params_t *params_tlr,
        parsec_execution_stream_t *es,
        parsec_memory_pool_t *p_work,
        parsec_memory_pool_t *p_work_full_dp,
        parsec_memory_pool_t *p_work_full_sp,
        parsec_memory_pool_t *p_work_full_hp,
        parsec_memory_pool_t *p_work_uv_dp,
        parsec_memory_pool_t *p_work_uv_sp,
        parsec_memory_pool_t *p_work_mbr,
        parsec_memory_pool_t *p_work_rr,
        void *C, void *A, void *B,
        int m, int n, int k,
        int Crank, int Arank, int Brank,
        double Anorm, double Bnorm )
{
    int counter_id = es->th_id;
    int tempmm = m == descA->mt-1 ? descA->m - m * descA->mb : descA->mb;
    int tempnn = n == descA->nt-1 ? descA->n - n * descA->nb : descA->nb;
    int ldam = BLKLDD( descA, m );
    int ldan = BLKLDD( descA, n );
    int verbose = params_tlr->verbose;
    int counter_stride = params_tlr->counter_stride;
    int tid = es->th_id;
    void *A_use = A;
    void *B_use = B;
    void *A_d, *A_s, *B_d, *B_s, *A_h, *B_h;

    // Get new decision during runtime
    uint16_t new_decision = params_tlr->decisions[n*descA->lmt+m];
    if(params_tlr->adaptive_decision_runtime) {
        hicma_parsec_get_precision_tile(params_tlr, &new_decision, Anorm*Bnorm, m, n);
        if(new_decision != params_tlr->decisions[n*descA->lmt+m] && params_tlr->verbose > 99) {
            printf("The decision in gemm(%d, %d, %d) has been changed from %u to %u: norm_old %lf norm_new %.16lf (Anorm %.16lf Bnorm %.16lf)\n", m, n, k, params_tlr->decisions[n*descA->lmt+m], new_decision, params_tlr->norm_tile[n*params_tlr->NT+m], Anorm * Bnorm, Anorm, Bnorm);
            //printf("GEMM Norm %d %d : %.10lf\n", m, k, Anorm);
            //printf("GEMM Norm %d %d : %.10lf\n", n, k, Bnorm);
        }
    }
    
    if(verbose > 1) {
        if( DENSE_DP == params_tlr->decisions[n*descA->lmt+m] ) {
            params_tlr->nb_gemms[DENSE_DP*counter_stride+tid] += 1;
        } else if( DENSE_SP == params_tlr->decisions[n*descA->lmt+m] ) {
            params_tlr->nb_gemms[DENSE_SP*counter_stride+tid] += 1;
        } else if( DENSE_HP == params_tlr->decisions[n*descA->lmt+m] ) { 
            params_tlr->nb_gemms[DENSE_HP*counter_stride+tid] += 1;
        } else if(LOW_RANK_DP == params_tlr->decisions[n*descA->lmt+m] ) {
            params_tlr->nb_gemms[LOW_RANK_DP*counter_stride+tid] += 1;
        } else if(LOW_RANK_SP == params_tlr->decisions[n*descA->lmt+m] ) {
            params_tlr->nb_gemms[LOW_RANK_SP*counter_stride+tid] += 1;
        }
    }

    if(DEBUG_INFO) printf("GEMM_CPU (%d, %d, %d) : %d %d %d : C_DENSE, A_DENSE, B_DENSE\n",
            m, n, k, params_tlr->decisions[n*descA->lmt+m], params_tlr->decisions[k*descA->lmt+m], params_tlr->decisions[k*descA->lmt+n]);

    /* If dgemm */
    if( DENSE_DP == params_tlr->decisions[n*descA->lmt+m] ) {
        /* Convert datatype, A */
        if( DENSE_DP != params_tlr->decisions[k*descA->lmt+m] ) {
            A_d = parsec_private_memory_pop( p_work_full_dp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_slag2d( LAPACK_COL_MAJOR, descA->mb, descA->nb, A, descA->mb, A_d, descA->mb );
            A_use = A_d;
        }

        /* Convert datatype, B */
        if( DENSE_DP != params_tlr->decisions[k*descA->lmt+n] ) {
            B_d = parsec_private_memory_pop( p_work_full_dp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_slag2d( LAPACK_COL_MAJOR, descA->mb, descA->nb, B, descA->mb, B_d, descA->mb );
            B_use = B_d;
        }

        CORE_dgemm(PlasmaNoTrans, PlasmaTrans,
                tempmm, descA->mb, descA->mb,
                (double)-1.0, A_use /*A(m, k)*/, ldam,
                              B_use /*A(n, k)*/, ldan,
                (double) 1.0, C     /*A(m, n)*/, ldam);

        /* Push back to mempool */
        if( DENSE_DP != params_tlr->decisions[k*descA->lmt+m] )
            parsec_private_memory_push( p_work_full_dp, A_d );

        if( DENSE_DP != params_tlr->decisions[k*descA->lmt+n] )
            parsec_private_memory_push( p_work_full_dp, B_d );

        /* If sgemm */
    } else if( DENSE_SP == params_tlr->decisions[n*descA->lmt+m] ) {
        /* Convert datatype, A */
        if( DENSE_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            A_s = parsec_private_memory_pop( p_work_full_sp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, descA->nb, A, descA->mb, A_s, descA->mb );
            A_use = A_s;
        }

        /* Convert datatype, B */
        if( DENSE_DP == params_tlr->decisions[k*descA->lmt+n] ) {
            B_s = parsec_private_memory_pop( p_work_full_sp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, descA->nb, B, descA->mb, B_s, descA->mb );
            B_use = B_s;
        }

        CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                tempmm, descA->mb, descA->mb,
                (float)-1.0, A_use /*A(m, k)*/, ldam,
                             B_use /*A(n, k)*/, ldan,
                (float) 1.0, C     /*A(m, n)*/, ldam);

        /* Push back to mempool */
        if( DENSE_DP == params_tlr->decisions[k*descA->lmt+m] )
            parsec_private_memory_push( p_work_full_sp, A_s );

        if( DENSE_DP == params_tlr->decisions[k*descA->lmt+n] )
            parsec_private_memory_push( p_work_full_sp, B_s );
    }
#if !HAVE_HP_CPU
    /* Convert A and B in 16-bit and call sgemm */
    //else if( DENSE_HP == params_tlr->decisions[n*descA->lmt+m] ){
    else{
        /* Convert datatype, A */
        A_use = parsec_private_memory_pop( p_work_full_sp );
        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
        hicma_parsec_convert_2h_bit(A, A_use, descA->mb, descA->nb, params_tlr->decisions[k*descA->lmt+m]);

        /* Convert datatype, B */
        B_use = parsec_private_memory_pop( p_work_full_sp );
        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
        hicma_parsec_convert_2h_bit(B, B_use, descA->mb, descA->mb, params_tlr->decisions[k*descA->lmt+n] );

        CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                tempmm, descA->mb, descA->mb,
                (float)-1.0, A_use /*A(m, k)*/, ldam,
                             B_use /*A(n, k)*/, ldan,
                (float) 1.0, C     /*A(m, n)*/, ldam);

        parsec_private_memory_push(p_work_full_sp, A_use);
        parsec_private_memory_push(p_work_full_sp, B_use);
    }
#else
    /* If hgemm */
    //else if( DENSE_HP == params_tlr->decisions[n*descA->lmt+m] ){
    else{
        A_h = parsec_private_memory_pop( p_work_full_hp );
        B_h = parsec_private_memory_pop( p_work_full_hp );

        /* Convert datatype, A */
        if( DENSE_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            //A_s = parsec_private_memory_pop( p_work_full_sp );
            //A_h = parsec_private_memory_pop( p_work_full_hp );
            //hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            //LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, descA->nb, A, descA->mb, A_s, descA->mb );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_d2h_binary_CPU( A_h, A, descA->mb, descA->nb);
            A_use = A_h;
        } else {
            //A_h = parsec_private_memory_pop( p_work_full_hp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_binary_CPU( A_h, A, descA->mb, descA->nb);
            A_use = A_h;
        }

        /* Convert datatype, B */
        if( DENSE_DP == params_tlr->decisions[k*descA->lmt+n] ) {
            //B_s = parsec_private_memory_pop( p_work_full_sp );
            //B_h = parsec_private_memory_pop( p_work_full_hp );
            //hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            //LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, descA->nb, B, descA->mb, B_s, descA->mb );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_d2h_binary_CPU( B_h, B, descA->mb, descA->nb);
            B_use = B_h;
        } else {
            //B_h = parsec_private_memory_pop( p_work_full_hp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_binary_CPU( B_h, B, descA->mb, descA->nb);
            B_use = B_h;
        }

        /* First local GEMM convert C from single to half */
        if( 0 == k ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_unary_CPU( C, descA->mb, descA->nb );
        }

        /* Call hgemm */
        fjcblas_gemm_r16(CblasColMajor, PlasmaNoTrans, PlasmaTrans,
                tempmm, descA->mb, descA->mb,
                (__fp16)-1.0, A_use /*A(m, k)*/, ldam,
                              B_use /*A(n, k)*/, ldan,
                (__fp16) 1.0, C /*A(m, n)*/, ldam);

        /* After last local GEMM convert C from half to single */
        if( n-1 == k ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_h2s_unary_CPU( C, descA->mb, descA->nb );
        }

        parsec_private_memory_push(p_work_full_hp, A_h);
        parsec_private_memory_push(p_work_full_hp, B_h);
    }
#endif

    /* Operation count */
    unsigned long int cnt = hicma_parsec_op_counts('m', tempmm, tempmm, tempmm, 0);
    params_tlr->op_band[es->th_id] += cnt;
    params_tlr->op_offpath[es->th_id] += cnt;
}


/**
 * @brief CPU implementation of GEMM (General Matrix Multiply) for dense matrices
 *        Adaptive decision during runtime
 * 
 * This function performs the general matrix multiplication C = C + A * B where
 * all matrices (C, A, B) are dense. It handles different data types (double,
 * single precision, half precision) and supports various memory layouts.
 * The function is part of the HICMA Cholesky factorization kernel.
 * 
 * @param[in] descA Matrix descriptor for matrix A
 * @param[in] descRank Matrix descriptor for rank information
 * @param[in,out] params_tlr HICMA parameters containing operation settings
 * @param[in] es Execution stream for the operation
 * @param[in] p_work Memory pool for general workspace
 * @param[in] p_work_full_dp Memory pool for double precision full matrices
 * @param[in] p_work_full_sp Memory pool for single precision full matrices
 * @param[in] p_work_full_hp Memory pool for half precision full matrices
 * @param[in] p_work_uv_dp Memory pool for double precision U/V matrices
 * @param[in] p_work_uv_sp Memory pool for single precision U/V matrices
 * @param[in] p_work_mbr Memory pool for matrix block rank operations
 * @param[in] p_work_rr Memory pool for rank-rank operations
 * @param[in,out] C Matrix C (result matrix)
 * @param[in] A Matrix A (first operand)
 * @param[in] B Matrix B (second operand)
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile
 * @param[in] k Diagonal index of the current tile
 * @param[in] Crank Rank of matrix C (unused for dense matrices)
 * @param[in] Arank Rank of matrix A (unused for dense matrices)
 * @param[in] Brank Rank of matrix B (unused for dense matrices)
 */
void hicma_parsec_core_gemm_denseC_denseA_denseB_runtime_decision_cpu( parsec_tiled_matrix_t* descA,
        parsec_tiled_matrix_t* descRank,
        hicma_parsec_params_t *params_tlr,
        parsec_execution_stream_t *es,
        parsec_memory_pool_t *p_work,
        parsec_memory_pool_t *p_work_full_dp,
        parsec_memory_pool_t *p_work_full_sp,
        parsec_memory_pool_t *p_work_full_hp,
        parsec_memory_pool_t *p_work_uv_dp,
        parsec_memory_pool_t *p_work_uv_sp,
        parsec_memory_pool_t *p_work_mbr,
        parsec_memory_pool_t *p_work_rr,
        void *C, void *A, void *B,
        int m, int n, int k,
        int Crank, int Arank, int Brank,
        void*A_norm, void* B_norm )
{
    int counter_id = es->th_id;
    int tempmm = m == descA->mt-1 ? descA->m - m * descA->mb : descA->mb;
    int tempnn = n == descA->nt-1 ? descA->n - n * descA->nb : descA->nb;
    int ldam = BLKLDD( descA, m );
    int ldan = BLKLDD( descA, n );
    void *A_use = A;
    void *B_use = B;
    void *A_d, *A_s, *B_d, *B_s, *A_h, *B_h, *C_h;
    double Anorm = (params_tlr->adaptive_decision_runtime)? ((double *)A_norm)[0]: 0;
    double Bnorm = (params_tlr->adaptive_decision_runtime)? ((double *)B_norm)[0]: 0;
    uint16_t Aprecision = (params_tlr->adaptive_decision_runtime)? (uint16_t)(((double *)A_norm)[1]): -1;
    uint16_t Bprecision = (params_tlr->adaptive_decision_runtime)? (uint16_t)(((double *)B_norm)[1]): -1;
    int verbose = params_tlr->verbose;
    int counter_stride = params_tlr->counter_stride;
    int tid = es->th_id;

    // Get new decision during runtime
    uint16_t Cprecision = params_tlr->decisions[n*descA->lmt+m];
    uint16_t new_decision = params_tlr->decisions[n*descA->lmt+m];
    if(params_tlr->adaptive_decision_runtime) {
        hicma_parsec_get_precision_tile(params_tlr, &new_decision, Anorm*Bnorm, m, n);
        if(new_decision != params_tlr->decisions[n*descA->lmt+m] && verbose > 99) {
            printf("The decision in gemm(%d, %d, %d) has been changed from %u to %u: norm_old %lf norm_new %.16lf (Anorm %.16lf Bnorm %.16lf) Aprecision %u Bprecision %u\n", m, n, k, params_tlr->decisions[n*descA->lmt+m], new_decision, params_tlr->norm_tile[n*params_tlr->NT+m], Anorm * Bnorm, Anorm, Bnorm, Aprecision, Bprecision);
            //printf("GEMM Norm %d %d : %.10lf\n", m, k, Anorm);
            //printf("GEMM Norm %d %d : %.10lf\n", n, k, Bnorm);
        }
    }

    if(params_tlr->verbose > 1) {
        if( DENSE_DP == new_decision ) {
            params_tlr->nb_gemms[DENSE_DP*counter_stride+tid] += 1;
        } else if( DENSE_SP == new_decision ) {
            params_tlr->nb_gemms[DENSE_SP*counter_stride+tid] += 1;
        } else if( DENSE_HP == new_decision ) {
            params_tlr->nb_gemms[DENSE_HP*counter_stride+tid] += 1;
        } else if(LOW_RANK_DP == new_decision ) {
            params_tlr->nb_gemms[LOW_RANK_DP*counter_stride+tid] += 1;
        } else if(LOW_RANK_SP == new_decision ) {
            params_tlr->nb_gemms[LOW_RANK_SP*counter_stride+tid] += 1;
        }
    }

    if(DEBUG_INFO) printf("GEMM_CPU (%d, %d, %d) : %d %d %d : C_DENSE, A_DENSE, B_DENSE\n",
            m, n, k, params_tlr->decisions[n*descA->lmt+m], params_tlr->decisions[k*descA->lmt+m], params_tlr->decisions[k*descA->lmt+n]);

#define ACC_DP 1

    /* If dgemm */
    if( DENSE_DP == new_decision ) {
        /* Convert datatype, A */
        //if( DENSE_DP != params_tlr->decisions[k*descA->lmt+m] ) {
        if( DENSE_DP != Aprecision ) {
            A_d = parsec_private_memory_pop( p_work_full_dp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_slag2d( LAPACK_COL_MAJOR, descA->mb, descA->nb, A, descA->mb, A_d, descA->mb );
            A_use = A_d;
        }

        /* Convert datatype, B */
        //if( DENSE_DP != params_tlr->decisions[k*descA->lmt+n] ) {
        if( DENSE_DP != Bprecision ) {
            B_d = parsec_private_memory_pop( p_work_full_dp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_slag2d( LAPACK_COL_MAJOR, descA->mb, descA->nb, B, descA->mb, B_d, descA->mb );
            B_use = B_d;
        }

        /* Convert datatype, C, in place */
        if( DENSE_DP != params_tlr->decisions[n*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2d_unary_CPU(C, descA->mb, descA->nb);
        }

        CORE_dgemm(PlasmaNoTrans, PlasmaTrans,
                tempmm, descA->mb, descA->mb,
                (double)-1.0, A_use /*A(m, k)*/, ldam,
                              B_use /*A(n, k)*/, ldan,
                (double) 1.0, C     /*A(m, n)*/, ldam);

        /* Push back to mempool */
        if( DENSE_DP != Aprecision )
            parsec_private_memory_push( p_work_full_dp, A_d );

        if( DENSE_DP != Bprecision )
            parsec_private_memory_push( p_work_full_dp, B_d );

#if ACC_DP
        params_tlr->decisions[n*descA->lmt+m] = DENSE_DP;
#else
        /* Re-calculate the decision for C */
        uint16_t updated_decision;
        double Cnorm = hicma_parsec_core_matrix_norm_get(C, descA->mb, descA->nb, descA->mb, "double", 0, 0, 0, 0);
        hicma_parsec_get_precision_tile(params_tlr, &updated_decision, Cnorm, m, n);
        if(DENSE_DP == updated_decision) {
            params_tlr->decisions[n*descA->lmt+m] = DENSE_DP;
        } else {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_d2s_unary_CPU(C, descA->mb, descA->nb);
            params_tlr->decisions[n*descA->lmt+m] = DENSE_SP;
        }
#endif

        /* If sgemm */
    } else if( DENSE_SP == new_decision ) {
        /* Convert datatype, A */
        if( DENSE_DP == Aprecision ) {
            A_s = parsec_private_memory_pop( p_work_full_sp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, descA->nb, A, descA->mb, A_s, descA->mb );
            A_use = A_s;
        }

        /* Convert datatype, B */
        if( DENSE_DP == Bprecision ) {
            B_s = parsec_private_memory_pop( p_work_full_sp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, descA->nb, B, descA->mb, B_s, descA->mb );
            B_use = B_s;
        }

#if ACC_DP
        void *C_use = parsec_private_memory_pop( p_work_full_dp );
        //CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
        //        tempmm, descA->mb, descA->nb,
        //        (float)1.0, A_use /*A(m, k)*/, ldam,
        //                     B_use /*A(n, k)*/, ldan,
        //        //(float)1.0,  C     /*A(m, n)*/, ldam);
        //        (float) 0.0, C_use     /*A(m, n)*/, ldam);

        /* For the first GEMM */
        //if( DENSE_DP != params_tlr->decisions[n*descA->lmt+m] ) {
        //    hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
        //    convert_s2d_unary_CPU(C, descA->mb, descA->nb);
        //}

        if( DENSE_DP == params_tlr->decisions[n*descA->lmt+m] ) {
            CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                    tempmm, descA->mb, descA->nb,
                    (float)1.0, A_use /*A(m, k)*/, ldam,
                                B_use /*A(n, k)*/, ldan,
                    //(float)1.0,  C     /*A(m, n)*/, ldam);
                    (float) 0.0, C_use     /*A(m, n)*/, ldam);
            for(int j = 0; j < descA->nb; j++) {
                for(int i = 0; i < descA->mb; i++) {
                    ((double *)C)[j*descA->mb+i] -= ((float*)C_use)[j*descA->mb+i];
                }
            }
        } else {
            CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                    tempmm, descA->mb, descA->nb,
                    (float)-1.0, A_use /*A(m, k)*/, ldam,
                                B_use /*A(n, k)*/, ldan,
                    (float)1.0,  C     /*A(m, n)*/, ldam);
                    //(float) 0.0, C_use     /*A(m, n)*/, ldam);
            params_tlr->decisions[n*descA->lmt+m] = DENSE_SP;
        }
        parsec_private_memory_push( p_work_full_dp, C_use);
        //params_tlr->decisions[n*descA->lmt+m] = DENSE_DP;
#else
        /* Convert datatype, C, in place */
        if( DENSE_DP == params_tlr->decisions[n*descA->lmt+m] ) {
            void *C_use = parsec_private_memory_pop( p_work_full_dp );
            CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                    tempmm, descA->mb, descA->nb,
                    (float)1.0, A_use /*A(m, k)*/, ldam,
                    B_use /*A(n, k)*/, ldan,
                    //(float)1.0,  C     /*A(m, n)*/, ldam);
                (float) 0.0, C_use     /*A(m, n)*/, ldam);

            for(int j = 0; j < descA->nb; j++) {
                for(int i = 0; i < descA->mb; i++) {
                    ((double *)C)[j*descA->mb+i] -= ((float*)C_use)[j*descA->mb+i];
                }
            }

            parsec_private_memory_push( p_work_full_dp, C_use);

            /* Re-calculate the decision for C */
            uint16_t updated_decision;
            double Cnorm = hicma_parsec_core_matrix_norm_get(C, descA->mb, descA->nb, descA->mb, "double", 0, 0, 0, 0);
            hicma_parsec_get_precision_tile(params_tlr, &updated_decision, Cnorm, m, n);
            if(DENSE_DP == updated_decision) {
                params_tlr->decisions[n*descA->lmt+m] = DENSE_DP;
            } else {
                hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                convert_d2s_unary_CPU(C, descA->mb, descA->nb);
                params_tlr->decisions[n*descA->lmt+m] = DENSE_SP;
            }

        } else {
            CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                    tempmm, descA->mb, descA->nb,
                    (float)-1.0, A_use /*A(m, k)*/, ldam,
                    B_use /*A(n, k)*/, ldan,
                    (float)1.0,  C     /*A(m, n)*/, ldam);
            //(float) 0.0, C_use     /*A(m, n)*/, ldam);

            /* Re-calculate the decision for C */
            uint16_t updated_decision;
            double Cnorm = hicma_parsec_core_matrix_norm_get(C, descA->mb, descA->nb, descA->mb, "float", 0, 0, 0, 0);
            hicma_parsec_get_precision_tile(params_tlr, &updated_decision, Cnorm, m, n);
            if(DENSE_DP == updated_decision) {
                hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                convert_s2d_unary_CPU(C, descA->mb, descA->nb);
                params_tlr->decisions[n*descA->lmt+m] = DENSE_DP;
            } else {
                params_tlr->decisions[n*descA->lmt+m] = DENSE_SP;
            }
        }
#endif

        /* Push back to mempool */
        if( DENSE_DP == Aprecision )
            parsec_private_memory_push( p_work_full_sp, A_s );

        if( DENSE_DP == Bprecision )
            parsec_private_memory_push( p_work_full_sp, B_s );
    }
#if !HAVE_HP_CPU
    /* Convert A and B in 16-bit and call sgemm */
    //else if( DENSE_HP == params_tlr->decisions[n*descA->lmt+m] ){
    else{
            /* Convert datatype, A */
        A_use = parsec_private_memory_pop( p_work_full_sp );
        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
        hicma_parsec_convert_2h_bit(A, A_use, descA->mb, descA->nb, Aprecision);

        /* Convert datatype, B */
        B_use = parsec_private_memory_pop( p_work_full_sp );
        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
        hicma_parsec_convert_2h_bit(B, B_use, descA->mb, descA->mb, Bprecision);

#if ACC_DP
        void *C_use = parsec_private_memory_pop( p_work_full_dp );
        CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                tempmm, descA->mb, descA->nb,
                (float)1.0, A_use /*A(m, k)*/, ldam,
                             B_use /*A(n, k)*/, ldan,
                //(float)1.0,  C     /*A(m, n)*/, ldam);
                (float) 0.0, C_use     /*A(m, n)*/, ldam);

        /* For the first GEMM */
        if( DENSE_DP != params_tlr->decisions[n*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2d_unary_CPU(C, descA->mb, descA->nb);
        }

        for(int j = 0; j < descA->nb; j++) {
            for(int i = 0; i < descA->mb; i++) {
                ((double *)C)[j*descA->mb+i] -= ((float*)C_use)[j*descA->mb+i];
            }
        }
        parsec_private_memory_push( p_work_full_dp, C_use);
        params_tlr->decisions[n*descA->lmt+m] = DENSE_DP;
#else
        /* Convert datatype, C, in place */
        if( DENSE_DP == params_tlr->decisions[n*descA->lmt+m] ) {
            void *C_use = parsec_private_memory_pop( p_work_full_dp );
            CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                    tempmm, descA->mb, descA->nb,
                    (float)1.0, A_use /*A(m, k)*/, ldam,
                    B_use /*A(n, k)*/, ldan,
                    //(float)1.0,  C     /*A(m, n)*/, ldam);
                (float) 0.0, C_use     /*A(m, n)*/, ldam);

            for(int j = 0; j < descA->nb; j++) {
                for(int i = 0; i < descA->mb; i++) {
                    ((double *)C)[j*descA->mb+i] -= ((float*)C_use)[j*descA->mb+i];
                }
            }
            parsec_private_memory_push( p_work_full_dp, C_use);

            /* Re-calculate the decision for C */
            uint16_t updated_decision;
            double Cnorm = hicma_parsec_core_matrix_norm_get(C, descA->mb, descA->nb, descA->mb, "double", 0, 0, 0, 0);
            hicma_parsec_get_precision_tile(params_tlr, &updated_decision, Cnorm, m, n);
            if(DENSE_DP == updated_decision) {
                params_tlr->decisions[n*descA->lmt+m] = DENSE_DP;
            } else {
                hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                convert_d2s_unary_CPU(C, descA->mb, descA->nb);
                params_tlr->decisions[n*descA->lmt+m] = DENSE_SP;
            }
        } else {
            CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                    tempmm, descA->mb, descA->mb,
                    (float)-1.0, A_use /*A(m, k)*/, ldam,
                    B_use /*A(n, k)*/, ldan,
                    (float) 1.0, C     /*A(m, n)*/, ldam);

            /* Re-calculate the decision for C */
            uint16_t updated_decision;
            double Cnorm = hicma_parsec_core_matrix_norm_get(C, descA->mb, descA->nb, descA->mb, "float", 0, 0, 0, 0);
            hicma_parsec_get_precision_tile(params_tlr, &updated_decision, Cnorm, m, n);
            if(DENSE_DP == updated_decision) {
                hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                convert_s2d_unary_CPU(C, descA->mb, descA->nb);
                params_tlr->decisions[n*descA->lmt+m] = DENSE_DP;
            } else {
                params_tlr->decisions[n*descA->lmt+m] = DENSE_SP;
            }
        }
#endif

        parsec_private_memory_push(p_work_full_sp, A_use);
        parsec_private_memory_push(p_work_full_sp, B_use);
    }
#else
        /* If hgemm */
        //else if( DENSE_HP == params_tlr->decisions[n*descA->lmt+m] ){
    else {
        A_h = parsec_private_memory_pop( p_work_full_hp );
        B_h = parsec_private_memory_pop( p_work_full_hp );
        C_h = parsec_private_memory_pop( p_work_full_hp );

        /* Convert datatype, A */
        if( DENSE_DP == Aprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_d2h_binary_CPU( A_h, A, descA->mb, descA->nb);
            A_use = A_h;
        } else {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_binary_CPU( A_h, A, descA->mb, descA->nb);
            A_use = A_h;
        }

        /* Convert datatype, B */
        if( DENSE_DP == Bprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_d2h_binary_CPU( B_h, B, descA->mb, descA->nb);
            B_use = B_h;
        } else {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_binary_CPU( B_h, B, descA->mb, descA->nb);
            B_use = B_h;
        }

#if 0

        fjcblas_gemm_r16(CblasColMajor, PlasmaNoTrans, PlasmaTrans,
                tempmm, descA->mb, descA->mb,
                (__fp16) 1.0, A_use /*A(m, k)*/, ldam,
                              B_use /*A(n, k)*/, ldan,
                (__fp16) 0.0, C_h /*A(m, n)*/, ldam);

        /* HGEMM */
        if( DENSE_DP == params_tlr->decisions[n*descA->lmt+m] ) {
            for(int j = 0; j < descA->nb; j++) {
                for(int i = 0; i < descA->mb; i++) {
                    ((double *)C)[j*descA->nb+i] -= ((__fp16 *)C_h)[j*descA->mb+i];
                }
            }
        } else {
            for(int j = 0; j < descA->nb; j++) {
                for(int i = 0; i < descA->mb; i++) {
                    ((float*)C)[j*descA->nb+i] -= ((__fp16 *)C_h)[j*descA->mb+i];
                }
            }
            params_tlr->decisions[n*descA->lmt+m] = DENSE_SP;
        }
#else
	/* Convert datatype, C */
	if( 0 == k ) {
        if( DENSE_DP == Cprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_d2h_unary_CPU( C, descA->mb, descA->nb);
        } else {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_unary_CPU( C, descA->mb, descA->nb);
        }
	}

        /* Call hgemm */
        fjcblas_gemm_r16(CblasColMajor, PlasmaNoTrans, PlasmaTrans,
                tempmm, descA->mb, descA->mb,
                (__fp16)-1.0, A_use /*A(m, k)*/, ldam,
                              B_use /*A(n, k)*/, ldan,
                (__fp16) 1.0, C /*A(m, n)*/, ldam);

        /* Convert datatype, C */
        if( n-1  == k ) {
        if( DENSE_DP == Cprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_h2d_unary_CPU( C, descA->mb, descA->nb);
        } else {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_h2s_unary_CPU( C, descA->mb, descA->nb);
        }       
        }

#endif

        /* Push back to mempool */
        parsec_private_memory_push(p_work_full_hp, A_h);
        parsec_private_memory_push(p_work_full_hp, B_h);
        parsec_private_memory_push(p_work_full_hp, C_h);

    }
#endif

    if( 0 == k ) {
        params_tlr->decisions_gemm_first[n*descA->lmt+m] = new_decision;
    }
    if( k + 1 == n ) {
        params_tlr->decisions_gemm_last[n*descA->lmt+m] = new_decision;
    }

    /* Operation count */
    unsigned long int cnt = hicma_parsec_op_counts('m', tempmm, tempmm, tempmm, 0);
    params_tlr->op_band[es->th_id] += cnt;
    params_tlr->op_offpath[es->th_id] += cnt;
}


/**
 * @brief CPU implementation of GEMM for dense result matrix C with low-rank matrix A and dense matrix B
 * 
 * This function performs the general matrix multiplication C = C + A * B where
 * matrix A is low-rank (stored as U*V^T), matrix B is dense, and result matrix C is dense.
 * It handles the low-rank decomposition of A and performs efficient matrix operations
 * using the rank structure to reduce computational complexity.
 * 
 * @param[in] descA Matrix descriptor for matrix A
 * @param[in] descRank Matrix descriptor for rank information
 * @param[in,out] params_tlr HICMA parameters containing operation settings
 * @param[in] es Execution stream for the operation
 * @param[in] p_work Memory pool for general workspace
 * @param[in] p_work_full_dp Memory pool for double precision full matrices
 * @param[in] p_work_full_sp Memory pool for single precision full matrices
 * @param[in] p_work_full_hp Memory pool for half precision full matrices
 * @param[in] p_work_uv_dp Memory pool for double precision U/V matrices
 * @param[in] p_work_uv_sp Memory pool for single precision U/V matrices
 * @param[in] p_work_mbr Memory pool for matrix block rank operations
 * @param[in] p_work_rr Memory pool for rank-rank operations
 * @param[in,out] C Matrix C (result matrix, dense)
 * @param[in] A Matrix A (low-rank, stored as U*V^T)
 * @param[in] B Matrix B (dense)
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile
 * @param[in] k Diagonal index of the current tile
 * @param[in] Crank Rank of matrix C (unused for dense matrices)
 * @param[in] Arank Rank of matrix A (low-rank)
 * @param[in] Brank Rank of matrix B (unused for dense matrices)
 */
void hicma_parsec_core_gemm_denseC_lrA_denseB_cpu( parsec_tiled_matrix_t* descA,
        parsec_tiled_matrix_t* descRank,
        hicma_parsec_params_t *params_tlr,
        parsec_execution_stream_t *es,
        parsec_memory_pool_t *p_work,
        parsec_memory_pool_t *p_work_full_dp,
        parsec_memory_pool_t *p_work_full_sp,
        parsec_memory_pool_t *p_work_full_hp,
        parsec_memory_pool_t *p_work_uv_dp,
        parsec_memory_pool_t *p_work_uv_sp,
        parsec_memory_pool_t *p_work_mbr,
        parsec_memory_pool_t *p_work_rr,
        void *C, void *A, void *B,
        int m, int n, int k,
        int Crank, int Arank, int Brank )
{
    int counter_id = es->th_id;
    int tempmm = m == descA->mt-1 ? descA->m - m * descA->mb : descA->mb;
    void *A_use = A;
    void *B_use = B;
    void *A_d, *A_s, *B_d, *B_s, *A_h, *B_h;

    /* If rank is 0, return */
    if( 0 == Arank ) {
        return;
    }

    if(DEBUG_INFO) printf("GEMM_CPU (%d, %d, %d) : %d %d %d : C_DENSE, A_LOW_RANK, B_DENSE\n",
            m, n, k, params_tlr->decisions[n*descA->lmt+m], params_tlr->decisions[k*descA->lmt+m], params_tlr->decisions[k*descA->lmt+n]);

    void *p_elem_work_mbr = parsec_private_memory_pop( p_work_mbr );
    void *Au, *Av;

    if( DENSE_DP == params_tlr->decisions[n*descA->lmt+m] ) {
        /* Convert datatype, A */
        if( LOW_RANK_SP == params_tlr->decisions[k*descA->lmt+m] ) {
            A_d = parsec_private_memory_pop( p_work_uv_dp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_slag2d( LAPACK_COL_MAJOR, descA->mb, Arank * 2, A, descA->mb, A_d, descA->mb );
            A_use = A_d;
        }

        /* Convert datatype, B */
        if( DENSE_SP == params_tlr->decisions[k*descA->lmt+n] || DENSE_HP == params_tlr->decisions[k*descA->lmt+n] ) {
            B_d = parsec_private_memory_pop( p_work_full_dp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_slag2d( LAPACK_COL_MAJOR, descA->mb, descA->nb, B, descA->mb, B_d, descA->mb );
            B_use = B_d;
        }

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(double);

        /* tmp_mbr = trans(Av) * trans(B) */
        CORE_dgemm(PlasmaTrans, PlasmaTrans,
                Arank, descA->mb, descA->mb,
                (double)1.0, Av /*A(m, k)*/, descA->mb,
                             B_use /*A(n, m)*/, descA->mb,
                (double)0.0, p_elem_work_mbr /*A(k, n)*/, Arank);

        /* C = C - Au * tmp_mbr */
        CORE_dgemm(PlasmaNoTrans, PlasmaNoTrans,
                descA->mb, descA->mb, Arank,
                (double)-1.0, Au              /*A(m, k)*/, descA->mb,
                              p_elem_work_mbr /*A(k, n)*/, Arank,
                (double) 1.0, C               /*A(m, n)*/, descA->mb);

        /* Push back to mempool */
        if( LOW_RANK_SP == params_tlr->decisions[k*descA->lmt+m] )
            parsec_private_memory_push( p_work_uv_dp, A_d );

        if( DENSE_SP == params_tlr->decisions[k*descA->lmt+n] || DENSE_HP == params_tlr->decisions[k*descA->lmt+n] )
            parsec_private_memory_push( p_work_full_dp, B_d );

    } else if( DENSE_SP == params_tlr->decisions[n*descA->lmt+m] ) {
        /* Convert datatype, A */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            A_s = parsec_private_memory_pop( p_work_uv_sp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, Arank * 2, A, descA->mb, A_s, descA->mb );
            A_use = A_s;
        }

        /* Convert datatype, B */
        if( DENSE_DP == params_tlr->decisions[k*descA->lmt+n] ) {
            B_s = parsec_private_memory_pop( p_work_full_sp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, descA->nb, B, descA->mb, B_s, descA->mb );
            B_use = B_s;
        }

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(float);

        /* tmp_mbr = trans(Av) * trans(B) */
        CORE_sgemm(PlasmaTrans, PlasmaTrans,
                Arank, descA->mb, descA->mb,
                (float)1.0, Av /*A(m, k)*/, descA->mb,
                            B_use /*A(n, m)*/, descA->mb,
                (float)0.0, p_elem_work_mbr /*A(k, n)*/, Arank);

        /* C = C - Au * tmp_mbr */
        CORE_sgemm(PlasmaNoTrans, PlasmaNoTrans,
                descA->mb, descA->mb, Arank,
                (float)-1.0, Au              /*A(m, k)*/, descA->mb,
                             p_elem_work_mbr /*A(k, n)*/, Arank,
                (float) 1.0, C               /*A(m, n)*/, descA->mb);

        /* Push back to mempool */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] )
            parsec_private_memory_push( p_work_uv_sp, A_s );

        if( DENSE_DP == params_tlr->decisions[k*descA->lmt+n] )
            parsec_private_memory_push( p_work_full_sp, B_s );
    }

#if !HAVE_HP_CPU
    /* Convert A and B in 16-bit and call sgemm */
    else {
        /* Convert datatype, A */
        A_use = parsec_private_memory_pop( p_work_full_sp );
        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
        hicma_parsec_convert_2h_bit(A, A_use, descA->mb, Arank * 2, params_tlr->decisions[k*descA->lmt+m]);

        /* Convert datatype, B */
        B_use = parsec_private_memory_pop( p_work_full_sp );
        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
        hicma_parsec_convert_2h_bit(B, B_use, descA->mb, descA->mb, params_tlr->decisions[k*descA->lmt+n]);

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(float);

        /* tmp_mbr = trans(Av) * trans(B) */
        CORE_sgemm(PlasmaTrans, PlasmaTrans,
                Arank, descA->mb, descA->mb,
                (float)1.0, Av /*A(m, k)*/, descA->mb,
                            B_use /*A(n, m)*/, descA->mb,
                (float)0.0, p_elem_work_mbr /*A(k, n)*/, Arank);

        /* C = C - Au * tmp_mbr */
        CORE_sgemm(PlasmaNoTrans, PlasmaNoTrans,
                descA->mb, descA->mb, Arank,
                (float)-1.0, Au              /*A(m, k)*/, descA->mb,
                             p_elem_work_mbr /*A(k, n)*/, Arank,
                (float) 1.0, C               /*A(m, n)*/, descA->mb);

        /* Push back to mempool */
        parsec_private_memory_push( p_work_full_sp, A_use );
        parsec_private_memory_push( p_work_full_sp, B_use );
    }
#else
    else {
        /* Convert datatype, A */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            A_s = parsec_private_memory_pop( p_work_full_sp );
            A_h = parsec_private_memory_pop( p_work_full_hp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, Arank * 2, A, descA->mb, A_s, descA->mb );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_binary_CPU( A_h, A_s, descA->mb, Arank * 2);
            A_use = A_h;
        } else {
            A_h = parsec_private_memory_pop( p_work_full_hp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_binary_CPU( A_h, A, descA->mb, Arank * 2);
            A_use = A_h;
        }

        /* Convert datatype, B */
        if( DENSE_DP == params_tlr->decisions[k*descA->lmt+n] ) {
            B_s = parsec_private_memory_pop( p_work_full_sp );
            B_h = parsec_private_memory_pop( p_work_full_hp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, descA->nb, B, descA->mb, B_s, descA->mb );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_binary_CPU( B_h, B_s, descA->mb, descA->nb);
            B_use = B_h;
        } else {
            B_h = parsec_private_memory_pop( p_work_full_hp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_binary_CPU( B_h, B, descA->mb, descA->nb);
            B_use = B_h;
        }

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(float) / 2;

        /* First local GEMM convert C from single to half */
        if( 0 == k ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_unary_CPU( C, descA->mb, descA->nb );
        }

        /* tmp_mbr = trans(Av) * trans(B) */
        fjcblas_gemm_r16(CblasColMajor, PlasmaTrans, PlasmaTrans,
                Arank, descA->mb, descA->mb,
                (__fp16)1.0, Av              /*A(m, k)*/, descA->mb,
                B_use           /*A(n, m)*/, descA->mb,
                (__fp16)0.0, p_elem_work_mbr /*A(k, n)*/, Arank);

        /* C = C - Au * tmp_mbr */
        fjcblas_gemm_r16(CblasColMajor, PlasmaNoTrans, PlasmaNoTrans,
                descA->mb, descA->mb, Arank,
                (__fp16)-1.0, Au              /*A(m, k)*/, descA->mb,
                p_elem_work_mbr /*A(k, n)*/, Arank,
                (__fp16) 1.0, C               /*A(m, n)*/, descA->mb);

        /* After last local GEMM convert C from half to single */
        if( n-1 == k ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_h2s_unary_CPU( C, descA->mb, descA->nb );
        }

        /* Push back to mempool */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
                parsec_private_memory_push(p_work_full_sp, A_s);
                parsec_private_memory_push(p_work_full_hp, A_h);
            } else {
                parsec_private_memory_push(p_work_full_hp, A_h);
            }

            if( DENSE_DP == params_tlr->decisions[k*descA->lmt+n] ) {
                parsec_private_memory_push(p_work_full_sp, B_s);
                parsec_private_memory_push(p_work_full_hp, B_h);
            } else {
                parsec_private_memory_push(p_work_full_hp, B_h);
            }

        }
#endif

        parsec_private_memory_push( p_work_mbr, p_elem_work_mbr );

        /* Operation count */
        unsigned long int cnt = hicma_parsec_op_counts('m', tempmm, tempmm, Arank, 0) * 2;
        params_tlr->op_band[es->th_id] += cnt;
        params_tlr->op_offpath[es->th_id] += cnt;
}


/**
 * @brief CPU implementation of GEMM for dense result matrix C with low-rank matrices A and B
 * 
 * This function performs the general matrix multiplication C = C + A * B where
 * both matrices A and B are low-rank (stored as U*V^T), and result matrix C is dense.
 * It efficiently handles the low-rank decomposition of both operands and performs
 * matrix operations using the rank structure to minimize computational complexity.
 * 
 * @param[in] descA Matrix descriptor for matrix A
 * @param[in] descRank Matrix descriptor for rank information
 * @param[in,out] params_tlr HICMA parameters containing operation settings
 * @param[in] es Execution stream for the operation
 * @param[in] p_work Memory pool for general workspace
 * @param[in] p_work_full_dp Memory pool for double precision full matrices
 * @param[in] p_work_full_sp Memory pool for single precision full matrices
 * @param[in] p_work_full_hp Memory pool for half precision full matrices
 * @param[in] p_work_uv_dp Memory pool for double precision U/V matrices
 * @param[in] p_work_uv_sp Memory pool for single precision U/V matrices
 * @param[in] p_work_mbr Memory pool for matrix block rank operations
 * @param[in] p_work_rr Memory pool for rank-rank operations
 * @param[in,out] C Matrix C (result matrix, dense)
 * @param[in] A Matrix A (low-rank, stored as U*V^T)
 * @param[in] B Matrix B (low-rank, stored as U*V^T)
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile
 * @param[in] k Diagonal index of the current tile
 * @param[in] Crank Rank of matrix C (unused for dense matrices)
 * @param[in] Arank Rank of matrix A (low-rank)
 * @param[in] Brank Rank of matrix B (low-rank)
 */
void hicma_parsec_core_gemm_denseC_lrA_lrB_cpu( parsec_tiled_matrix_t* descA,
        parsec_tiled_matrix_t* descRank,
        hicma_parsec_params_t *params_tlr,
        parsec_execution_stream_t *es,
        parsec_memory_pool_t *p_work,
        parsec_memory_pool_t *p_work_full_dp,
        parsec_memory_pool_t *p_work_full_sp,
        parsec_memory_pool_t *p_work_full_hp,
        parsec_memory_pool_t *p_work_uv_dp,
        parsec_memory_pool_t *p_work_uv_sp,
        parsec_memory_pool_t *p_work_mbr,
        parsec_memory_pool_t *p_work_rr,
        void *C, void *A, void *B,
        int m, int n, int k,
        int Crank, int Arank, int Brank )
{
    int counter_id = es->th_id;
    int tempmm = m == descA->mt-1 ? descA->m - m * descA->mb : descA->mb;
    void *A_use = A;
    void *B_use = B;
    void *A_d, *A_s, *B_d, *B_s, *A_h, *B_h;

    /* If rank is 0, return */
    if( 0 == Arank || 0 == Brank ) {
        return;
    }

    if(DEBUG_INFO) printf("GEMM_CPU (%d, %d, %d) : %d %d %d : C_DENSE, A_LOW_RANK, B_LOW_RANK\n",
            m, n, k, params_tlr->decisions[n*descA->lmt+m], params_tlr->decisions[k*descA->lmt+m], params_tlr->decisions[k*descA->lmt+n]);

    void *p_elem_work_mbr = parsec_private_memory_pop( p_work_mbr );
    void *p_elem_work_rr = parsec_private_memory_pop( p_work_rr );
    void *Au, *Av, *Bu, *Bv;

    if( DENSE_DP == params_tlr->decisions[n*descA->lmt+m] )
    {
        /* Convert datatype, A */
        if( LOW_RANK_SP == params_tlr->decisions[k*descA->lmt+m] ) {
            A_d = parsec_private_memory_pop( p_work_uv_dp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_slag2d( LAPACK_COL_MAJOR, descA->mb, Arank * 2, A, descA->mb, A_d, descA->mb );
            A_use = A_d;
        }

        /* Convert datatype, B */
        if( LOW_RANK_SP == params_tlr->decisions[k*descA->lmt+n] ) {
            B_d = parsec_private_memory_pop( p_work_uv_dp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_slag2d( LAPACK_COL_MAJOR, descA->mb, Brank * 2, B, descA->mb, B_d, descA->mb );
            B_use = B_d;
        }

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(double);
        Bu = (void *)B_use;
        Bv = (void *)B_use + descA->mb * Brank * sizeof(double);

        /* tmp_rr = trans(Av) * Bv */
        CORE_dgemm(PlasmaTrans, PlasmaNoTrans,
                Arank, Brank, descA->mb,
                (double) 1.0, Av             /*A(k, m)*/, descA->mb,
                              Bv             /*A(k, n)*/, descA->mb,
                (double) 0.0, p_elem_work_rr /*A(m, n)*/, Arank);

        if( Arank > Brank ) {
            /* tmp_mbr = Au * tmp_rr */
            CORE_dgemm(PlasmaNoTrans, PlasmaNoTrans,
                    descA->mb, Brank, Arank,
                    (double) 1.0, Au              /*A(m, k)*/, descA->mb,
                                  p_elem_work_rr  /*A(k, n)*/, Arank,
                    (double) 0.0, p_elem_work_mbr /*A(m, n)*/, descA->mb);

            /* C = C - tmp_mbr * trans(Bu) */
            CORE_dgemm(PlasmaNoTrans, PlasmaTrans,
                    descA->mb, descA->mb, Brank,
                    (double)-1.0, p_elem_work_mbr /*A(m, k)*/, descA->mb,
                                  Bu              /*A(n, k)*/, descA->mb,
                    (double) 1.0, C               /*A(m, n)*/, descA->mb);
        } else {
            /* tmp_mbr = tmp_rr * trans(Bu) */
            CORE_dgemm(PlasmaNoTrans, PlasmaTrans,
                    Arank, descA->mb, Brank,
                    (double) 1.0, p_elem_work_rr  /*A(m, k)*/, Arank,
                                  Bu              /*A(n, k)*/, descA->mb,
                    (double) 0.0, p_elem_work_mbr /*A(m, n)*/, Arank);

            /* C = C - Au * tmp_mbr */
            CORE_dgemm(PlasmaNoTrans, PlasmaNoTrans,
                    descA->mb, descA->mb, Arank,
                    (double)-1.0, Au              /*A(m, k)*/, descA->mb,
                                  p_elem_work_mbr /*A(k, n)*/, Arank,
                    (double) 1.0, C               /*A(m, n)*/, descA->mb);
        }

        /* Push back to mempool */
        if( LOW_RANK_SP == params_tlr->decisions[k*descA->lmt+m] )
            parsec_private_memory_push( p_work_uv_dp, A_d );

        if( LOW_RANK_SP == params_tlr->decisions[k*descA->lmt+n] )
            parsec_private_memory_push( p_work_uv_dp, B_d );
    }
    else if( DENSE_SP == params_tlr->decisions[n*descA->lmt+m] )
    {
        /* Convert datatype, A */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            A_s = parsec_private_memory_pop( p_work_uv_sp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, Arank * 2, A, descA->mb, A_s, descA->mb );
            A_use = A_s;
        }

        /* Convert datatype, B */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+n] ) {
            B_s = parsec_private_memory_pop( p_work_uv_sp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, Brank * 2, B, descA->mb, B_s, descA->mb );
            B_use = B_s;
        }

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(float);
        Bu = (void *)B_use;
        Bv = (void *)B_use + descA->mb * Brank * sizeof(float);

        /* tmp_rr = trans(Av) * Bv */
        CORE_sgemm(PlasmaTrans, PlasmaNoTrans,
                Arank, Brank, descA->mb,
                (float) 1.0, Av             /*A(k, m)*/, descA->mb,
                             Bv             /*A(k, n)*/, descA->mb,
                (float) 0.0, p_elem_work_rr /*A(m, n)*/, Arank);

        if( Arank > Brank ) {
            /* tmp_mbr = Au * tmp_rr */
            CORE_sgemm(PlasmaNoTrans, PlasmaNoTrans,
                    descA->mb, Brank, Arank,
                    (float) 1.0, Au              /*A(m, k)*/, descA->mb,
                    p_elem_work_rr  /*A(k, n)*/, Arank,
                    (float) 0.0, p_elem_work_mbr /*A(m, n)*/, descA->mb);

            /* C = C - tmp_mbr * trans(Bu) */
            CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                    descA->mb, descA->mb, Brank,
                    (float)-1.0, p_elem_work_mbr /*A(m, k)*/, descA->mb,
                                 Bu              /*A(n, k)*/, descA->mb,
                    (float) 1.0, C               /*A(m, n)*/, descA->mb);
        } else {
            /* tmp_mbr = tmp_rr * trans(Bu) */
            CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                    Arank, descA->mb, Brank,
                    (float) 1.0, p_elem_work_rr  /*A(m, k)*/, Arank,
                                 Bu              /*A(n, k)*/, descA->mb,
                    (float) 0.0, p_elem_work_mbr /*A(m, n)*/, Arank);

            /* C = C - Au * tmp_mbr */
            CORE_sgemm(PlasmaNoTrans, PlasmaNoTrans,
                    descA->mb, descA->mb, Arank,
                    (float)-1.0, Au              /*A(m, k)*/, descA->mb,
                                 p_elem_work_mbr /*A(k, n)*/, Arank,
                    (float) 1.0, C               /*A(m, n)*/, descA->mb);
        }

        /* Push back to mempool */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] )
            parsec_private_memory_push( p_work_uv_sp, A_s );

        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+n] )
            parsec_private_memory_push( p_work_uv_sp, B_s );
    }
#if !HAVE_HP_CPU
    /* Convert A and B in 16-bit and call sgemm */
    else {
        /* Convert datatype, A */
        A_use = parsec_private_memory_pop( p_work_full_sp );
        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
        hicma_parsec_convert_2h_bit(A, A_use, descA->mb, Arank * 2, params_tlr->decisions[k*descA->lmt+m]);

        /* Convert datatype, B */
        B_use = parsec_private_memory_pop( p_work_full_sp );
        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
        hicma_parsec_convert_2h_bit(B, B_use, descA->mb, Brank * 2, params_tlr->decisions[k*descA->lmt+n]);

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(float);
        Bu = (void *)B_use;
        Bv = (void *)B_use + descA->mb * Brank * sizeof(float);

        /* tmp_rr = trans(Av) * Bv */
        CORE_sgemm(PlasmaTrans, PlasmaNoTrans,
                Arank, Brank, descA->mb,
                (float) 1.0, Av             /*A(k, m)*/, descA->mb,
                             Bv             /*A(k, n)*/, descA->mb,
                (float) 0.0, p_elem_work_rr /*A(m, n)*/, Arank);

        if( Arank > Brank ) {
            /* tmp_mbr = Au * tmp_rr */
            CORE_sgemm(PlasmaNoTrans, PlasmaNoTrans,
                    descA->mb, Brank, Arank,
                    (float) 1.0, Au              /*A(m, k)*/, descA->mb,
                                 p_elem_work_rr  /*A(k, n)*/, Arank,
                    (float) 0.0, p_elem_work_mbr /*A(m, n)*/, descA->mb);

            /* C = C - tmp_mbr * trans(Bu) */
            CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                    descA->mb, descA->mb, Brank,
                    (float)-1.0, p_elem_work_mbr /*A(m, k)*/, descA->mb,
                                 Bu              /*A(n, k)*/, descA->mb,
                    (float) 1.0, C               /*A(m, n)*/, descA->mb);
        } else {
            /* tmp_mbr = tmp_rr * trans(Bu) */
            CORE_sgemm(PlasmaNoTrans, PlasmaTrans,
                    Arank, descA->mb, Brank,
                    (float) 1.0, p_elem_work_rr  /*A(m, k)*/, Arank,
                                 Bu              /*A(n, k)*/, descA->mb,
                    (float) 0.0, p_elem_work_mbr /*A(m, n)*/, Arank);

            /* C = C - Au * tmp_mbr */
            CORE_sgemm(PlasmaNoTrans, PlasmaNoTrans,
                    descA->mb, descA->mb, Arank,
                    (float)-1.0, Au              /*A(m, k)*/, descA->mb,
                                 p_elem_work_mbr /*A(k, n)*/, Arank,
                    (float) 1.0, C               /*A(m, n)*/, descA->mb);
        }

        /* Push back to mempool */
        parsec_private_memory_push( p_work_full_sp, A_use );
        parsec_private_memory_push( p_work_full_sp, B_use );
    }
#else
    else {
        /* Convert datatype, A */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            A_s = parsec_private_memory_pop( p_work_full_sp );
            A_h = parsec_private_memory_pop( p_work_full_hp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, Arank * 2, A, descA->mb, A_s, descA->mb );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_binary_CPU( A_h, A_s, descA->mb, Arank * 2);
            A_use = A_h;
        } else {
            A_h = parsec_private_memory_pop( p_work_full_hp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_binary_CPU( A_h, A, descA->mb, Arank * 2);
            A_use = A_h;
        }

        /* Convert datatype, B */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+n] ) {
            B_s = parsec_private_memory_pop( p_work_full_sp );
            B_h = parsec_private_memory_pop( p_work_full_hp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            LAPACKE_dlag2s( LAPACK_COL_MAJOR, descA->mb, Brank * 2, B, descA->mb, B_s, descA->mb );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_binary_CPU( B_h, B_s, descA->mb, Brank * 2);
            B_use = B_h;
        } else {
            B_h = parsec_private_memory_pop( p_work_full_hp );
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_binary_CPU( B_h, B, descA->mb, Brank * 2);
            B_use = B_h;
        }

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(float) / 2;
        Bu = (void *)B_use;
        Bv = (void *)B_use + descA->mb * Brank * sizeof(float) / 2;

        /* First local GEMM convert C from single to half */
        if( 0 == k ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_s2h_unary_CPU( C, descA->mb, descA->nb );
        }

        /* tmp_rr = trans(Av) * Bv */
        fjcblas_gemm_r16(CblasColMajor, PlasmaTrans, PlasmaNoTrans,
                Arank, Brank, descA->mb,
                (__fp16)1.0, Av             /*A(k, m)*/, descA->mb,
                              Bv             /*A(k, n)*/, descA->mb,
                (__fp16)0.0, p_elem_work_rr /*A(m, n)*/, Arank);

        if( Arank > Brank ) {
            /* tmp_mbr = Au * tmp_rr */
            fjcblas_gemm_r16(CblasColMajor, PlasmaNoTrans, PlasmaNoTrans,
                    descA->mb, Brank, Arank,
                    (__fp16)1.0, Au              /*A(m, k)*/, descA->mb,
                                  p_elem_work_rr  /*A(k, n)*/, Arank,
                    (__fp16)0.0, p_elem_work_mbr /*A(m, n)*/, descA->mb);

            /* C = C - tmp_mbr * trans(Bu) */
            fjcblas_gemm_r16(CblasColMajor, PlasmaNoTrans, PlasmaTrans,
                    descA->mb, descA->mb, Brank,
                    (__fp16)-1.0, p_elem_work_mbr /*A(m, k)*/, descA->mb,
                    Bu              /*A(n, k)*/, descA->mb,
                    (__fp16)1.0, C               /*A(m, n)*/, descA->mb);
        } else {
            /* tmp_mbr = tmp_rr * trans(Bu) */
            fjcblas_gemm_r16(CblasColMajor, PlasmaNoTrans, PlasmaTrans,
                    Arank, descA->mb, Brank,
                    (__fp16)1.0, p_elem_work_rr  /*A(m, k)*/, Arank,
                                  Bu              /*A(n, k)*/, descA->mb,
                    (__fp16)0.0, p_elem_work_mbr /*A(m, n)*/, Arank);

            /* C = C - Au * tmp_mbr */
            fjcblas_gemm_r16(CblasColMajor, PlasmaNoTrans, PlasmaNoTrans,
                    descA->mb, descA->mb, Arank,
                    (__fp16)-1.0, Au              /*A(m, k)*/, descA->mb,
                                  p_elem_work_mbr /*A(k, n)*/, Arank,
                    (__fp16)1.0, C               /*A(m, n)*/, descA->mb);
        }

        /* After last local GEMM convert C from half to single */
        if( n-1 == k ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            convert_h2s_unary_CPU( C, descA->mb, descA->nb );
        }

        /* Push back to mempool */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            parsec_private_memory_push(p_work_full_sp, A_s);
            parsec_private_memory_push(p_work_full_hp, A_h);
        } else {
            parsec_private_memory_push(p_work_full_hp, A_h);
        }

        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+n] ) {
            parsec_private_memory_push(p_work_full_sp, B_s);
            parsec_private_memory_push(p_work_full_hp, B_h);
        } else {
            parsec_private_memory_push(p_work_full_hp, B_h);
        }

    }
#endif

    parsec_private_memory_push( p_work_mbr, p_elem_work_mbr );
    parsec_private_memory_push( p_work_rr, p_elem_work_rr );

    /* Operation count */
    unsigned long int cnt = hicma_parsec_op_counts('m', tempmm, tempmm, hicma_parsec_min(Arank, Brank), 0)
        + hicma_parsec_op_counts('m', tempmm, Arank, Brank, 0) * 2;
    params_tlr->op_band[es->th_id] += cnt;
    params_tlr->op_offpath[es->th_id] += cnt;
}



// TODO Rewrite low-rank kernels
#if 0

static void hicma_parsec_dtranpose( int mb, int nb, double *d, int lda_d, double *s, int lda_s ) {
    for( int j = 0; j < nb; j++ ) {
        for( int i = 0; i < mb; i++ ) {
            d[j*lda_d+i] = s[i*lda_s+j]; 
        }
    }
}

static void hicma_parsec_dnegtive( int mb, int nb, double* A, int lda ) {
    for( int j = 0; j < nb; j++ ) {
        for( int i = 0; i < mb; i++ ) {
            A[j*lda+i] = -A[j*lda+i]; 
        }
    }
}

/**
 * @brief Core DGEMM operation for low-rank matrix updates in JDF (Job Data Flow)
 * 
 * This function performs the core DGEMM operation for low-rank matrix updates:
 * Cu * trans(Cv) -= U * V
 * 
 * This is used in the context of low-rank matrix factorization where matrices
 * are represented in factorized form (U, V) and need to be updated efficiently.
 * The operation is performed in-place on the combined matrices.
 * 
 * @param[in] mb Number of rows in the matrix blocks
 * @param[in] Crank Rank of the C matrix (Cu/Cv dimensions)
 * @param[in] UVrank Rank of the U/V matrices
 * @param[in,out] Cu Left factor of C matrix (mb × Crank)
 * @param[in,out] Cv Right factor of C matrix (mb × Crank)
 * @param[in] U Left factor of update matrix (mb × UVrank)
 * @param[in] V Right factor of update matrix (mb × UVrank)
 * @param[in] tmp_buff Temporary buffer for intermediate computations
 * 
 * @note This function modifies Cu and Cv in-place
 * @note tmp_buff must be large enough to hold intermediate results
 * @note The operation is: Cu * Cv^T -= U * V^T
 */
void hicma_parsec_core_dgemm_lr_svd(
        int mb, int Crank, int UVrank,
        double *Cu, double *Cv,
        double *U, double *V,
        void *tmp_buff ) {

    char uplo = 'A';
    int rank_combine = Crank + rank;
    int size = 0;

    U_combine = (double *)(tmp_buff + size);
    size += mb * rank_combine * sizeof(double); 
    V_combine = (double *)(tmp_buff + size);
    size += mb * rank_combine * sizeof(double); 

    /* Copy Cu and U to U_tmp */
    dlacpy_( &uplo, &mb, &Crank, Cu, &mb, U_combine, &mb ); 
    dlacpy_( &uplo, &mb, &UVrank, U, &mb, U_combine, &mb ); 

    /* Copy Cv and V to V_tmp */ 
    dlacpy_( &uplo, &mb, &Crank, Cv, &mb, V_combine, &mb ); 
    dlacpy_( &uplo, &mb, &UVrank, V, &mb, V_combine, &mb ); 

    /* QR */


    /* SVD */

}


/**
 * @brief Core DGEMM operation for triple low-rank matrix multiplication in JDF
 * 
 * This function performs the core DGEMM operation for triple low-rank matrix multiplication:
 * Cu * trans(Cv) -= Au * trans(Av) * Bv * trans(Bu)
 * 
 * This is used in the context of hierarchical low-rank matrix operations where
 * three matrices (A, B, C) are all represented in factorized form and need to
 * be multiplied efficiently. The operation updates the C matrix factors.
 * 
 * @param[in] Arank Rank of the A matrix factors (Au/Av dimensions)
 * @param[in] Brank Rank of the B matrix factors (Bu/Bv dimensions)
 * @param[in] Crank Rank of the C matrix factors (Cu/Cv dimensions)
 * @param[in] mb Number of rows in the matrix blocks
 * @param[in] maxrank Maximum rank across all matrices for buffer allocation
 * @param[in] Au Left factor of A matrix (mb × Arank)
 * @param[in] Av Right factor of A matrix (mb × Arank)
 * @param[in] Bu Left factor of B matrix (mb × Brank)
 * @param[in] Bv Right factor of B matrix (mb × Brank)
 * @param[in,out] Cu Left factor of C matrix (mb × Crank)
 * @param[in,out] Cv Right factor of C matrix (mb × Crank)
 * @param[in] tmp_buff Temporary buffer for intermediate computations
 * @param[in] U Temporary buffer for U matrix (mb × maxrank)
 * @param[in] V Temporary buffer for V matrix (mb × maxrank)
 * 
 * @note This function modifies Cu and Cv in-place
 * @note tmp_buff size: max(maxrank * mb * 2 + maxrank * maxrank, mb * maxrank * 4)
 * @note Early return if Arank or Brank is zero
 * @note The operation is: Cu * Cv^T -= Au * Av^T * Bv * Bu^T
 */
void hicma_parsec_core_dgemm_lrA_lrB_lrC_svd(
        int Arank, int Brank, int Crank, int mb, int maxrank,
        double *Au, double *Av,
        double *Bu, double *Bv,
        double *Cu, double *Cv,
        void *tmp_buff, void *U, void *V ) {

    /* Direct return */
    if( 0 == Arank || 0 == Brank ) return;

    int size = 0, UVrank = 0;
    double *tmp_mbr = (double *)(tmp_buff + size);
    size += mb * maxrank * sizeof(double);
    double *tmp_rr = (double *)(tmp_buff + size); 
    size += maxrank * maxrank * sizeof(double);

    /* tmp_rr = trans(Av) * Bv */
    CORE_dgemm(PlasmaTrans, PlasmaNoTrans,
            Arank, Brank, mb,
            (double) 1.0, Av             /*A(k, m)*/, mb,
            Bv             /*A(k, n)*/, mb,
            (double) 0.0, tmp_rr /*A(m, n)*/, Arank);

    if( Arank > Brank ) {
        /* tmp_mbr = Au * tmp_rr */
        CORE_dgemm(PlasmaNoTrans, PlasmaNoTrans,
                mb, Brank, Arank,
                (double) 1.0, Au      /*A(m, k)*/, mb,
                tmp_rr  /*A(k, n)*/, Arank,
                (double) 0.0, tmp_mbr /*A(m, n)*/, mb);

        /* Update U and V */
        memcpy( U, tmp_mbr, mb * Brank * sizeof(double) ); 
        memcpy( V, Bu, mb * Brank * sizeof(double) ); 
        UVrank = Brank;

    } else {
        /* tmp_mbr = tmp_rr * trans(Bu) */
        CORE_dgemm(PlasmaNoTrans, PlasmaTrans,
                Arank, descA->mb, Brank,
                (double) 1.0, tmp_rr  /*A(m, k)*/, Arank,
                Bu      /*A(n, k)*/, descA->mb,
                (double) 0.0, tmp_mbr /*A(m, n)*/, Arank);

        double *tmp_mbr2 = (double *)(tmp_buff + size); 
        hicma_parsec_dtranpose( mb, Arank, tmp_mbr2, mb, tmp_mbr, Arank ); 

        /* Update U and V */
        memcpy( U, Au, mb * Arank * sizeof(double) ); 
        memcpy( V, tmp_mbr2, mb * Arank * sizeof(double) ); 
        UVrank = Arank;
    }

    /* V times -1 */
    hicma_parsec_dnegtive( mb, UVrank, V, mb );

    /* Call routines until SVD */ 
    hicma_parsec_core_dgemm_lr_svd( mb, Crank, UVrank, Cu, Cv, U, V, tmp_buff );
}



#endif


#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/**
 * @brief GPU wrapper for double precision GEMM using cuBLAS
 * 
 * This function provides a wrapper around cuBLAS double precision GEMM operations
 * for GPU-accelerated matrix multiplication. It uses cuBLAS's GemmEx function
 * with optimized compute types for double precision arithmetic.
 * 
 * @param[in] handle cuBLAS handle for GPU operations
 * @param[in] transa Operation type for matrix A (transpose, conjugate, etc.)
 * @param[in] transb Operation type for matrix B (transpose, conjugate, etc.)
 * @param[in] m Number of rows of matrix A and C
 * @param[in] n Number of columns of matrix B and C
 * @param[in] k Number of columns of matrix A and rows of matrix B
 * @param[in] alpha Scalar multiplier for matrix A
 * @param[in] A Pointer to matrix A data on GPU
 * @param[in] lda Leading dimension of matrix A
 * @param[in] B Pointer to matrix B data on GPU
 * @param[in] ldb Leading dimension of matrix B
 * @param[in] beta Scalar multiplier for matrix C
 * @param[in,out] C Pointer to matrix C data on GPU (result matrix)
 * @param[in] ldc Leading dimension of matrix C
 * 
 * @return cuBLAS status code indicating success or failure
 */
cublasStatus_t hicma_parsec_dgemm_gpu(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc) {

    return cublasGemmEx(handle, transa, transb,
            m, n, k,
            alpha, A, CUDA_R_64F, lda,
                   B, CUDA_R_64F, ldb,
            beta,  C, CUDA_R_64F, ldc,
            CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
}

/**
 * @brief GPU wrapper for single precision GEMM using cuBLAS
 * 
 * This function provides a wrapper around cuBLAS single precision GEMM operations
 * for GPU-accelerated matrix multiplication. It supports both standard single precision
 * and Tensor Float-32 (TF32) compute types for improved performance on modern GPUs.
 * 
 * @param[in] handle cuBLAS handle for GPU operations
 * @param[in] transa Operation type for matrix A (transpose, conjugate, etc.)
 * @param[in] transb Operation type for matrix B (transpose, conjugate, etc.)
 * @param[in] m Number of rows of matrix A and C
 * @param[in] n Number of columns of matrix B and C
 * @param[in] k Number of columns of matrix A and rows of matrix B
 * @param[in] alpha Scalar multiplier for matrix A
 * @param[in] A Pointer to matrix A data on GPU
 * @param[in] lda Leading dimension of matrix A
 * @param[in] B Pointer to matrix B data on GPU
 * @param[in] ldb Leading dimension of matrix B
 * @param[in] beta Scalar multiplier for matrix C
 * @param[in,out] C Pointer to matrix C data on GPU (result matrix)
 * @param[in] ldc Leading dimension of matrix C
 * @param[in] tensor_gemm Flag indicating whether to use TF32 compute type
 * 
 * @return cuBLAS status code indicating success or failure
 */
cublasStatus_t hicma_parsec_sgemm_gpu(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc,
                           int tensor_gemm) {

    return cublasGemmEx(handle, transa, transb,
            m, n, k,
            alpha, A, CUDA_R_32F, lda,
                   B, CUDA_R_32F, ldb,
            beta,  C, CUDA_R_32F, ldc,
            (tensor_gemm & MASK_TF32)? CUBLAS_COMPUTE_32F_FAST_TF32: CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
}

/**
 * @brief GPU implementation of POTRF (Cholesky factorization) for a single tile
 * 
 * This function performs Cholesky factorization on a single tile using GPU acceleration.
 * It handles both double and single precision arithmetic and uses cuSOLVER for the
 * actual factorization computation. The function manages GPU memory, workspace,
 * and error handling for the factorization process.
 * 
 * @param[in] descA Matrix descriptor for the input matrix
 * @param[in,out] params_tlr HICMA parameters containing operation settings and timing
 * @param[in] ws_gpu GPU workspace for POTRF operations
 * @param[in] cuda_device CUDA device module for GPU operations
 * @param[in] gpu_task GPU task descriptor
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] T Workspace matrix for factorization
 * @param[in] k Diagonal index of the current tile
 */
void hicma_parsec_core_potrf_gpu( parsec_tiled_matrix_t* descA,
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_workspace_t *ws_gpu,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        void *T, int k ) {

    if( DEBUG_INFO ) printf("GPU_potrf %d\n", k);
    int tempkn = k == descA->nt-1 ? descA->n - k*descA->nb : descA->nb;
    int ldak = BLKLDD( descA, k );

    /* POTRF stores diagonal tiles in the precision used by the kernel. */
    params_tlr->decisions_storage_final[k*descA->lmt+k] =
        (DENSE_SP == params_tlr->decisions[k*descA->lmt+k]) ? DENSE_SP : DENSE_DP;

    /* Print the progress */
    hicma_parsec_print_process( descA->mt, k, params_tlr->start_time_potrf );

    /* Lookup workspace */
    parsec_potrf_workspace_t *_ws_gpu = (parsec_potrf_workspace_t *)ws_gpu;
    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, _ws_gpu);
    int buffer_size = stream_found->gpu_buffer_size;
    cusolverDnHandle_t handle = stream_found->handle_cusolver;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    int *dev_info;

    /* Set stream */
    cusolverDnSetStream( handle, cuda_stream->cuda_stream );

#if PRINT_KERNEL_TIME && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    cudaEvent_t gpu_time_start = NULL;
    cudaEvent_t gpu_time_stop = NULL;
    int gpu_time_active = hicma_kernel_time_gpu_event_begin((NULL != gpu_task) ? gpu_task->ec : NULL,
                                                            cuda_stream->cuda_stream,
                                                            &gpu_time_start, &gpu_time_stop);
#endif

    /* GPU kernel */
    if( DENSE_SP == params_tlr->decisions[k*descA->lmt+k] ) { 
        dev_info = (int *)(stream_found->gpu_buffer + buffer_size * sizeof(float));
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        float *gpu_buffer = (float *)stream_found->gpu_buffer;
        assert(NULL != gpu_buffer);
        status = cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, tempkn, T, ldak, gpu_buffer, buffer_size, dev_info);
#endif

#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        rocsolver_spotrf(handle, rocblas_fill_lower, tempkn, T, ldak, dev_info); 
#endif


    } else {
        dev_info = (int *)(stream_found->gpu_buffer + buffer_size * sizeof(double));
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        double *gpu_buffer = (double *)stream_found->gpu_buffer;
        assert(NULL != gpu_buffer);
        status = cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, tempkn, T, ldak, gpu_buffer, buffer_size, dev_info);
#endif

#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        //printf("dev_info %d\n", dev_info[0]);
        rocsolver_dpotrf(handle, rocblas_fill_lower, tempkn, T, ldak, dev_info);                      
#endif

    }
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* Update info */
    cudaMemcpyAsync(&params_tlr->info_gpu[k], dev_info, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream->cuda_stream);

#if PRINT_KERNEL_TIME && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    if(gpu_time_active) {
        parsec_task_t *parsec_task = (NULL != gpu_task) ? gpu_task->ec : NULL;
        int gpu_time_id = (NULL != parsec_task && NULL != parsec_task->selected_device) ?
                          parsec_task->selected_device->device_index : cuda_device->super.super.device_index;
        hicma_kernel_time_gpu_event_record(parsec_task, gpu_time_id,
                                           cuda_stream->cuda_stream,
                                           gpu_time_start, gpu_time_stop);
    }
#endif
}


/**
 * @brief GPU implementation of TRSM (Triangular Solve) for a single tile
 * 
 * This function performs triangular solve operations on a single tile using GPU acceleration.
 * It solves the equation T * X = C where T is a triangular matrix and C is the right-hand side.
 * The function handles both double and single precision arithmetic and uses cuBLAS for
 * the actual computation.
 * 
 * @param[in] descA Matrix descriptor for the input matrix
 * @param[in,out] params_tlr HICMA parameters containing operation settings
 * @param[in] ws_gpu GPU workspace for TRSM operations
 * @param[in] cuda_device CUDA device module for GPU operations
 * @param[in] gpu_task GPU task descriptor
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] T Triangular matrix (lower triangular from Cholesky factorization)
 * @param[in,out] C Right-hand side matrix (input) and solution matrix (output)
 * @param[in] m Row index of the current tile
 * @param[in] k Column index of the current tile
 */
void hicma_parsec_core_trsm_gpu( parsec_tiled_matrix_t* descA,
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_stream_workspace_t *stream_found,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        void *T, void *C, int m, int k ) {

    int counter_id = hicma_parsec_gpu_counter_id(cuda_device, cuda_stream);
    if(DEBUG_INFO) printf("GPU_trsm %d %d\n", m, k);

    int tempmm = m == descA->mt-1 ? descA->m - m * descA->mb : descA->mb;
    int ldak = BLKLDD( descA, k );
    int ldam = BLKLDD( descA, m );
    params_tlr->decisions_storage_final[k*descA->lmt+m] =
        (DENSE_DP == params_tlr->decisions[k*descA->lmt+m]) ? DENSE_DP : DENSE_SP;
    const double alpha_double = (double)1.0;
    const float alpha_float = (float)1.0;

    /* Get handle_cublas */
    cublasHandle_t handle = stream_found->handle_cublas;

    cublasStatus_t status;
    cublasSetStream( handle, cuda_stream->cuda_stream );

#if PRINT_KERNEL_TIME && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    cudaEvent_t gpu_time_start = NULL;
    cudaEvent_t gpu_time_stop = NULL;
    int gpu_time_active = hicma_kernel_time_gpu_event_begin((NULL != gpu_task) ? gpu_task->ec : NULL,
                                                            cuda_stream->cuda_stream,
                                                            &gpu_time_start, &gpu_time_stop);
#endif

    if( DENSE_DP == params_tlr->decisions[k*descA->lmt+m] ) {
        status = cublasDtrsm( handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                tempmm, descA->mb,
                &alpha_double, (double *)T /*A(k, k)*/, ldak,
                               (double *)C /*A(m, k)*/, ldam);
    } else {
        /* Get the temporary buffer on GPU */
        float *T_s = (float *)T;
        if( DENSE_DP == params_tlr->decisions_send[k*params_tlr->NT+k] ) {
            T_s = (float *)stream_found->gpu_buffer_A;
            assert(NULL != T_s);

            /* Convert datatype */
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2float_GPU( descA->mb, descA->nb, T, descA->mb, T_s, descA->mb, cuda_stream->cuda_stream );
        }

        status = cublasStrsm( handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                tempmm, descA->mb,
                &alpha_float, (float *)T_s /*A(k, k)*/, ldak,
                              (float *)C   /*A(m, k)*/, ldam);
    }

#if PRINT_KERNEL_TIME && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    if(gpu_time_active) {
        parsec_task_t *parsec_task = (NULL != gpu_task) ? gpu_task->ec : NULL;
        int gpu_time_id = (NULL != parsec_task && NULL != parsec_task->selected_device) ?
                          parsec_task->selected_device->device_index : cuda_device->super.super.device_index;
        hicma_kernel_time_gpu_event_record(parsec_task, gpu_time_id,
                                           cuda_stream->cuda_stream,
                                           gpu_time_start, gpu_time_stop);
    }
#endif

}


/**
 * @brief GPU implementation of SYRK (Symmetric Rank-K update) for a single tile
 * 
 * This function performs symmetric rank-k update operations on a single tile using GPU acceleration.
 * It computes C = C - A * A^T where A is a matrix and C is symmetric. The function handles
 * both double and single precision arithmetic and uses cuBLAS for the actual computation.
 * This is a key operation in the Cholesky factorization process.
 * 
 * @param[in] descA Matrix descriptor for the input matrix
 * @param[in,out] params_tlr HICMA parameters containing operation settings
 * @param[in] ws_gpu GPU workspace for SYRK operations
 * @param[in] cuda_device CUDA device module for GPU operations
 * @param[in] gpu_task GPU task descriptor
 * @param[in] cuda_stream CUDA execution stream
 * @param[in] T Triangular matrix from Cholesky factorization
 * @param[in,out] C Symmetric matrix to be updated
 * @param[in] m Row index of the current tile
 * @param[in] k Column index of the current tile
 */
void hicma_parsec_core_syrk_gpu( parsec_tiled_matrix_t* descA,
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_workspace_t *ws_gpu,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        void *T, void *A, int m, int k, int Arank ) {

    int counter_id = hicma_parsec_gpu_counter_id(cuda_device, cuda_stream);
    int tempmm = m == descA->mt-1 ? descA->m - m*descA->mb : descA->mb;
    int ldam = BLKLDD( descA, m );
    if( DEBUG_INFO ) printf("GPU_syrk:  %d %d\n", m, k);

    /* Get handle_cublas */
    parsec_potrf_workspace_t *_ws_gpu = (parsec_potrf_workspace_t *)ws_gpu;
    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, _ws_gpu);
    cublasHandle_t handle = stream_found->handle_cublas;

    cublasStatus_t status;
    cublasSetStream( handle, cuda_stream->cuda_stream );

#if PRINT_KERNEL_TIME && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    cudaEvent_t gpu_time_start = NULL;
    cudaEvent_t gpu_time_stop = NULL;
    int gpu_time_active = hicma_kernel_time_gpu_event_begin((NULL != gpu_task) ? gpu_task->ec : NULL,
                                                            cuda_stream->cuda_stream,
                                                            &gpu_time_start, &gpu_time_stop);
#endif

    /* A is dense */
    if( IS_DENSE(m, k) ) {
        if( DENSE_DP == params_tlr->decisions[m*descA->lmt+m] ) {
            double alpha = (double)-1.0;
            double beta = (double)1.0;
            double *A_d = A;
            if( DENSE_SP == params_tlr->decisions[k*descA->lmt+m] || DENSE_HP == params_tlr->decisions[k*descA->lmt+m] || DENSE_FP8 == params_tlr->decisions[k*descA->lmt+m] ) {
                /* Get the temporary buffer on GPU */
                A_d = (double *)stream_found->gpu_buffer_A;
                assert(NULL != A_d);

                /* Convert datatype */
                hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                float2double_GPU( descA->mb, descA->nb, A, descA->mb, A_d, descA->mb, cuda_stream->cuda_stream );
            }

            status = cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    tempmm, descA->mb,
                    &alpha, A_d /*A(m, k)*/, ldam,
                    &beta,  T /*A(m, m)*/, ldam);
        } else {
            const float alpha = (float)-1.0;
            const float beta = (float)1.0;

            status = cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    tempmm, descA->mb,
                    &alpha, A  /*A(m, k)*/, ldam,
                    &beta,  T  /*A(m, m)*/, ldam);
        }
    }
    /* A is low-rank */
    else {
        /* Au, Av */
        void *Au, *Av, *A_d;
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            Au = (void *)A;
            Av = (void *)A + descA->mb * Arank * sizeof(double);
        } else {
            /* Get the temporary buffer on GPU */
            A_d = (double *)stream_found->gpu_buffer_A;
            assert(NULL != A_d);

            /* Convert datatype */
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2double_GPU( descA->mb, Arank * 2, A, descA->mb, A_d, descA->mb, cuda_stream->cuda_stream );

            Au = (void *)A_d;
            Av = (void *)A_d + descA->mb * Arank * sizeof(double);
        }

        /* Get the temporary buffer */
        double *buffer_mbr = (double *)stream_found->gpu_buffer_mbr;
        assert(NULL != buffer_mbr);

        double *buffer_rr = (double *)stream_found->gpu_buffer_rr;
        assert(NULL != buffer_rr);

        double alpha = (double)1.0;
        double beta = (double)0.0;

        /* tmp_rr = trans(Av) * Av */
        status = hicma_parsec_dgemm_gpu(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                Arank, Arank, descA->mb,
                &alpha, Av          /*A(k, m)*/, descA->mb,
                        Av          /*A(k, n)*/, descA->mb,
                &beta,  buffer_rr   /*A(m, n)*/, Arank);

        alpha = (double)1.0;
        beta = (double)0.0;

        /* tmp_mbr = tmp_rr * trans(Au) */
        status = hicma_parsec_dgemm_gpu(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                Arank, descA->mb, Arank,
                &alpha, buffer_rr    /*A(m, k)*/, Arank,
                        Au           /*A(n, k)*/, descA->mb,
                &beta,  buffer_mbr   /*A(m, n)*/, Arank);

        alpha = (double)-1.0;
        beta = (double)1.0;

        /* T = T - Au * tmp_mbr */
        status = hicma_parsec_dgemm_gpu(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                descA->mb, descA->mb, Arank,
                &alpha, Au              /*A(m, k)*/, descA->mb,
                        buffer_mbr   /*A(k, n)*/, Arank,
                &beta,  T               /*A(m, n)*/, descA->mb);

    }

#if PRINT_KERNEL_TIME && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    if(gpu_time_active) {
        parsec_task_t *parsec_task = (NULL != gpu_task) ? gpu_task->ec : NULL;
        int gpu_time_id = (NULL != parsec_task && NULL != parsec_task->selected_device) ?
                          parsec_task->selected_device->device_index : cuda_device->super.super.device_index;
        hicma_kernel_time_gpu_event_record(parsec_task, gpu_time_id,
                                           cuda_stream->cuda_stream,
                                           gpu_time_start, gpu_time_stop);
    }
#endif

}


void hicma_parsec_core_syrk_runtime_decision_gpu( parsec_tiled_matrix_t* descA,
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_workspace_t *ws_gpu,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        void *T, void *A, int m, int k, int Arank, void *A_norm )
{

    int counter_id = hicma_parsec_gpu_counter_id(cuda_device, cuda_stream);
    if(!params_tlr->adaptive_decision_runtime) {
        fprintf(stderr, "in SYRK, it needs to enable adaptive_decision_runtime\n");
        return;
    }

#if 1
    int tempmm = m == descA->mt-1 ? descA->m - m*descA->mb : descA->mb;
    int ldam = BLKLDD( descA, m );
    if( DEBUG_INFO ) printf("GPU_syrk:  %d %d\n", m, k);

    /* Get handle_cublas */
    parsec_potrf_workspace_t *_ws_gpu = (parsec_potrf_workspace_t *)ws_gpu;
    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, _ws_gpu);
    cublasHandle_t handle = stream_found->handle_cublas;

    cublasStatus_t status;
    cublasSetStream( handle, cuda_stream->cuda_stream );

#if PRINT_KERNEL_TIME && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    cudaEvent_t gpu_time_start = NULL;
    cudaEvent_t gpu_time_stop = NULL;
    int gpu_time_active = hicma_kernel_time_gpu_event_begin((NULL != gpu_task) ? gpu_task->ec : NULL,
                                                            cuda_stream->cuda_stream,
                                                            &gpu_time_start, &gpu_time_stop);
#endif

    double Aprecision_as_double;
    cudaMemcpy(&Aprecision_as_double, ((double *)A_norm) + 1, sizeof(double), cudaMemcpyDeviceToHost);
    uint16_t Aprecision = (uint16_t)Aprecision_as_double;

    if( IS_DENSE(m, k) ) {
        if( DENSE_DP == params_tlr->decisions[m*descA->lmt+m] ) {
            double alpha = (double)-1.0;
            double beta = (double)1.0;
            double *A_d = A;
            if( DENSE_DP != Aprecision ) { 
                /* Get the temporary buffer on GPU */
                A_d = (double *)stream_found->gpu_buffer_A;
                assert(NULL != A_d);

                /* Convert datatype */
                hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                float2double_GPU( descA->mb, descA->nb, A, descA->mb, A_d, descA->mb, cuda_stream->cuda_stream );
            }

            status = cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    tempmm, descA->mb,
                    &alpha, A_d /*A(m, k)*/, ldam,
                    &beta,  T /*A(m, m)*/, ldam);
        } else {
            fprintf(stderr, "Should not reach here: SYRK SP %d\n", m);
            const float alpha = (float)-1.0;
            const float beta = (float)1.0;

            status = cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    tempmm, descA->mb,
                    &alpha, A  /*A(m, k)*/, ldam,
                    &beta,  T  /*A(m, m)*/, ldam);
        }
    } else {
            fprintf(stderr, "Should not reach here: SYRK LOW_RANK %d\n", m);
    }

#if PRINT_KERNEL_TIME && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    if(gpu_time_active) {
        parsec_task_t *parsec_task = (NULL != gpu_task) ? gpu_task->ec : NULL;
        int gpu_time_id = (NULL != parsec_task && NULL != parsec_task->selected_device) ?
                          parsec_task->selected_device->device_index : cuda_device->super.super.device_index;
        hicma_kernel_time_gpu_event_record(parsec_task, gpu_time_id,
                                           cuda_stream->cuda_stream,
                                           gpu_time_start, gpu_time_stop);
    }
#endif
#else

    uint16_t old_decision = params_tlr->decisions_send[k*descA->lmt+m];
    uint16_t runtime_precision = old_decision;

    if( params_tlr->adaptive_decision_runtime && NULL != A_norm ) {
        double precision_as_double = (double)old_decision;
        cudaMemcpy(&precision_as_double, ((double *)A_norm) + 1, sizeof(double), cudaMemcpyDeviceToHost);
        runtime_precision = (uint16_t)precision_as_double;
    }

    params_tlr->decisions_send[k*descA->lmt+m] = runtime_precision;
    hicma_parsec_core_syrk_gpu(descA, params_tlr, ws_gpu, cuda_device, gpu_task, cuda_stream, T, A, m, k, Arank);
    params_tlr->decisions_send[k*descA->lmt+m] = old_decision;
#endif
}


/**
 * @brief GPU implementation of GEMM (General Matrix Multiply) for dense matrices
 * 
 * This function performs the general matrix multiplication C = C + A * B where
 * all matrices (C, A, B) are dense, using GPU acceleration. It handles different
 * data types (double, single precision, half precision) and uses cuBLAS for
 * the actual computation. The function is part of the HICMA Cholesky factorization kernel.
 * 
 * @param[in] descA Matrix descriptor for matrix A
 * @param[in,out] params_tlr HICMA parameters containing operation settings
 * @param[in] ws_gpu GPU workspace for GEMM operations
 * @param[in] cuda_device CUDA device module for GPU operations
 * @param[in] gpu_task GPU task descriptor
 * @param[in] cuda_stream CUDA execution stream
 * @param[in,out] C Matrix C (result matrix, dense)
 * @param[in] A Matrix A (first operand, dense)
 * @param[in] B Matrix B (second operand, dense)
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile
 * @param[in] k Diagonal index of the current tile
 * @param[in] Crank Rank of matrix C (unused for dense matrices)
 * @param[in] Arank Rank of matrix A (unused for dense matrices)
 * @param[in] Brank Rank of matrix B (unused for dense matrices)
 */
void hicma_parsec_core_gemm_denseC_denseA_denseB_gpu( parsec_tiled_matrix_t* descA,
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_workspace_t *ws_gpu,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        void *C, void *A, void *B, int m, int n, int k,
        int Crank, int Arank, int Brank )
{
    int counter_id = hicma_parsec_gpu_counter_id(cuda_device, cuda_stream);
    int tempmm = m == descA->mt-1 ? descA->m - m * descA->mb : descA->mb;
    int ldam = BLKLDD( descA, m );
    int ldan = BLKLDD( descA, n );
    void *A_use = A;
    void *B_use = B;
    void *A_d, *A_s, *A_h, *A_fp8, *B_d, *B_s, *B_h, *B_fp8, *C_s, *C_h, *buffer_mbr, *buffer_rr;
    cublasStatus_t status;

    /* Find workspace */
    parsec_potrf_workspace_t *_ws_gpu = (parsec_potrf_workspace_t *)ws_gpu;
    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, _ws_gpu);

    /* Get handle_cublas */
    //cublasHandle_t handle = stream_found->handle_cublas_tensor;
    cublasHandle_t handle = stream_found->handle_cublas;
    if( DENSE_HP == params_tlr->decisions[n*descA->lmt+m] ) {
        handle = stream_found->handle_cublas_tensor;
    }

    //if( DENSE_FP8 != params_tlr->decisions[n*descA->lmt+m] ) {
        cublasSetStream( handle, cuda_stream->cuda_stream );
    //}

#if PRINT_KERNEL_TIME && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    cudaEvent_t gpu_time_start = NULL;
    cudaEvent_t gpu_time_stop = NULL;
    int gpu_time_active = hicma_kernel_time_gpu_event_begin((NULL != gpu_task) ? gpu_task->ec : NULL,
                                                            cuda_stream->cuda_stream,
                                                            &gpu_time_start, &gpu_time_stop);
#define HICMA_GPU_TIME_RETURN do {                                      \
        if(gpu_time_active) {                                           \
            hicma_kernel_time_gpu_event_abort(gpu_time_start, gpu_time_stop); \
        }                                                               \
        return;                                                         \
    } while(0)
#else
#define HICMA_GPU_TIME_RETURN do { return; } while(0)
#endif

    /* Get the temporary buffer on GPU */
    A_d = (double *)stream_found->gpu_buffer_A;
    A_s = (float *)stream_found->gpu_buffer_A;
    A_h = (void *)stream_found->gpu_buffer_A;
    A_fp8 = (void *)stream_found->gpu_buffer_A;

    B_d = (double *)stream_found->gpu_buffer_B;
    B_s = (float *)stream_found->gpu_buffer_B;
    B_h = (void *)stream_found->gpu_buffer_B;
    B_fp8 = (void *)stream_found->gpu_buffer_B;

    C_s = (float *)stream_found->gpu_buffer_C;
    C_h = (void *)stream_found->gpu_buffer_C;

    buffer_mbr = (double *)stream_found->gpu_buffer_mbr;
    buffer_rr = (double *)stream_found->gpu_buffer_rr;

    if(DEBUG_INFO) printf("GPU GEMM (%d, %d, %d) : %d %d %d : C_DENSE, A_DENSE, B_DENSE\n",
            m, n, k, params_tlr->decisions[n*descA->lmt+m], params_tlr->decisions[k*descA->lmt+m], params_tlr->decisions[k*descA->lmt+n]);

    if(params_tlr->verbose > 1) {
        uint16_t new_decision = params_tlr->decisions[n*descA->lmt+m];
        int counter_stride = params_tlr->counter_stride;
        int tid = counter_id;
        if( DENSE_DP == new_decision ) {
            params_tlr->nb_gemms[DENSE_DP*counter_stride+tid] += 1;
        } else if( DENSE_SP == new_decision ) {
            params_tlr->nb_gemms[DENSE_SP*counter_stride+tid] += 1;
        } else if( DENSE_HP == new_decision ) {
            params_tlr->nb_gemms[DENSE_HP*counter_stride+tid] += 1;
        } else if( DENSE_FP8 == new_decision ) {
            params_tlr->nb_gemms[DENSE_FP8*counter_stride+tid] += 1;
        } else if(LOW_RANK_DP == new_decision ) {
            params_tlr->nb_gemms[LOW_RANK_DP*counter_stride+tid] += 1;
        } else if(LOW_RANK_SP == new_decision ) {
            params_tlr->nb_gemms[LOW_RANK_SP*counter_stride+tid] += 1;
        }
    }

    /* If dgemm */
    if( DENSE_DP == params_tlr->decisions[n*descA->lmt+m] ) {

        /* Convert datatype, A */
        if( DENSE_DP != params_tlr->decisions_send[k*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2double_GPU( descA->mb, descA->nb, A, descA->mb, A_d, descA->mb, cuda_stream->cuda_stream );
            A_use = A_d;
        }

        /* Convert datatype, B */
        if( DENSE_DP != params_tlr->decisions_send[k*descA->lmt+n] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2double_GPU( descA->mb, descA->nb, B, descA->mb, B_d, descA->mb, cuda_stream->cuda_stream );
            B_use = B_d;
        }

        double alpha = -1.0, beta = 1.0;

        status = hicma_parsec_dgemm_gpu( handle, CUBLAS_OP_N, CUBLAS_OP_T,
        //status = cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T,
                tempmm, descA->mb, descA->mb,
                &alpha, A_use, ldam,
                        B_use, ldan,
                &beta,  C,     ldam );

    }
    /* If sgemm */
    else if( DENSE_SP == params_tlr->decisions[n*descA->lmt+m] ) {

        /* Convert datatype, A */
        if( DENSE_DP == params_tlr->decisions_send[k*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2float_GPU( descA->mb, descA->nb, A, descA->mb, A_s, descA->mb, cuda_stream->cuda_stream );
            A_use = A_s;
        } else if( DENSE_HP == params_tlr->decisions_send[k*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            half2float_GPU( descA->mb, descA->nb, A, descA->mb, A_s, descA->mb, cuda_stream->cuda_stream );
            A_use = A_s;
        }

        /* Convert datatype, B */
        if( DENSE_DP == params_tlr->decisions_send[k*descA->lmt+n] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2float_GPU( descA->mb, descA->nb, B, descA->mb, B_s, descA->mb, cuda_stream->cuda_stream );
            B_use = B_s;
        } else if( DENSE_HP == params_tlr->decisions_send[k*descA->lmt+n] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            half2float_GPU( descA->mb, descA->nb, B, descA->mb, B_s, descA->mb, cuda_stream->cuda_stream );
            B_use = B_s;
        }

        float alpha = -1.0f, beta = 1.0f;

        status = hicma_parsec_sgemm_gpu( handle, CUBLAS_OP_N, CUBLAS_OP_T,
                tempmm, descA->mb, descA->mb,
                &alpha, A_use, ldam,
                        B_use, ldan,
                &beta,  C,     ldam,
                params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m]);
    }
    /* If hgemm */
    else if( DENSE_HP == params_tlr->decisions[n*descA->lmt+m] ) {
        int tensor_gemm_type = params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m] & MASK_ONLY_FP16;
        //printf("%d %d %d: %d\n", m, n, k, tensor_gemm_type);
        float alpha = -1.0f, beta = 1.0f;
        switch( tensor_gemm_type ) {
            case MASK_TF16_A16_B16_C32_OP32:
            case MASK_TF16_A16_B16_C16_OP16:
                {
                    /* Convert datatype, A */
                    if( DENSE_DP == params_tlr->decisions_send[k*descA->lmt+m] ) {
                        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                        double2half_GPU( descA->mb, descA->nb, A, descA->mb, A_h, descA->mb, cuda_stream->cuda_stream );
                        A_use = A_h;
                    } else if( DENSE_SP == params_tlr->decisions_send[k*descA->lmt+m] ) {
                        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                        float2half_GPU( descA->mb, descA->nb, A, descA->mb, A_h, descA->mb, cuda_stream->cuda_stream );
                        A_use = A_h;
                    }

                    /* Convert datatype, B */
                    if( DENSE_DP == params_tlr->decisions_send[k*descA->lmt+n] ) {
                        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                        double2half_GPU( descA->mb, descA->nb, B, descA->mb, B_h, descA->mb, cuda_stream->cuda_stream );
                        B_use = B_h;
                    } else if( DENSE_SP == params_tlr->decisions_send[k*descA->lmt+n] ) {
                        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                        float2half_GPU( descA->mb, descA->nb, B, descA->mb, B_h, descA->mb, cuda_stream->cuda_stream );
                        B_use = B_h;
                    }

                    /* First local GEMM convert C from single to half */
                    if( 0 == k && MASK_TF16_A16_B16_C16_OP16 == tensor_gemm_type ) {
                        /* Convert datatype */
                        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                        float2half_GPU( descA->mb, descA->nb, C, descA->mb, C_h, descA->mb, cuda_stream->cuda_stream );

                        /* Copy C_h to C */
                        memcpy_half_GPU( descA->mb, descA->nb, C_h, C, cuda_stream->cuda_stream );
                    }

                    /* Call hgemm */
                    if( MASK_TF16_A16_B16_C32_OP32 == tensor_gemm_type ) {
                        status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                tempmm, descA->mb, descA->mb,
                                &alpha, A_use, CUDA_R_16F, ldam,
                                        B_use, CUDA_R_16F, ldan,
                                &beta,  C,     CUDA_R_32F, ldam,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                    } else {
                        // convert alpha and beta
                        char alphah[16], betah[16];
                        float2half_host(alpha, &alphah[0]);
                        float2half_host(beta, &betah[0]);
                        status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                tempmm, descA->mb, descA->mb,
                                &alphah[0], A_use, CUDA_R_16F, ldam,
                                B_use, CUDA_R_16F, ldan,
                                &betah[0],  C,     CUDA_R_16F, ldam,
                                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);

                    }

                    /* After last local GEMM convert C from half to single */
                    if( n-1 == k && MASK_TF16_A16_B16_C16_OP16 == tensor_gemm_type ) {
                        /* Convert datatype */
                        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                        half2float_GPU( descA->mb, descA->nb, C, descA->mb, C_s, descA->mb, cuda_stream->cuda_stream );

                        /* Copy C_s to C */
                        memcpy_float_GPU( descA->mb, descA->nb, C_s, C, cuda_stream->cuda_stream );
                    }

                    break;
                }

            case MASK_BF16_A16_B16_C32_OP32:
            case MASK_BF16_A16_B16_C16_OP16:
                {
                    /* Convert datatype, A */
                    if( DENSE_DP == params_tlr->decisions_send[k*descA->lmt+m] ) {
                        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                        double2bf_GPU( descA->mb, descA->nb, A, descA->mb, A_h, descA->mb, cuda_stream->cuda_stream );
                        A_use = A_h;
                    } else if( DENSE_SP == params_tlr->decisions_send[k*descA->lmt+m] ) {
                        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                        float2bf_GPU( descA->mb, descA->nb, A, descA->mb, A_h, descA->mb, cuda_stream->cuda_stream );
                        A_use = A_h;
                    }

                    /* Convert datatype, B */
                    if( DENSE_DP == params_tlr->decisions_send[k*descA->lmt+n] ) {
                        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                        double2bf_GPU( descA->mb, descA->nb, B, descA->mb, B_h, descA->mb, cuda_stream->cuda_stream );
                        B_use = B_h;
                    } else if( DENSE_SP == params_tlr->decisions_send[k*descA->lmt+n] ) {
                        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                        float2bf_GPU( descA->mb, descA->nb, B, descA->mb, B_h, descA->mb, cuda_stream->cuda_stream );
                        B_use = B_h;
                    }

                    /* First local GEMM convert C from single to half */
                    if( 0 == k && MASK_BF16_A16_B16_C16_OP16 == tensor_gemm_type ) {
                        /* Convert datatype */
                        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                        float2bf_GPU( descA->mb, descA->nb, C, descA->mb, C_h, descA->mb, cuda_stream->cuda_stream );

                        /* Copy C_h to C */
                        memcpy_bf_GPU( descA->mb, descA->nb, C_h, C, cuda_stream->cuda_stream );
                    }

                    /* Call hgemm */
                    if( MASK_BF16_A16_B16_C32_OP32 == tensor_gemm_type ) {
                        status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                tempmm, descA->mb, descA->mb,
                                &alpha, A_use, CUDA_R_16BF, ldam,
                                        B_use, CUDA_R_16BF, ldan,
                                &beta,  C,     CUDA_R_32F, ldam,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                    } else {
                        status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                tempmm, descA->mb, descA->mb,
                                &alpha, A_use, CUDA_R_16BF, ldam,
                                        B_use, CUDA_R_16BF, ldan,
                                &beta,  C,     CUDA_R_16BF, ldam,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

                    }

                    /* After last local GEMM convert C from half to single */
                    if( n-1 == k && MASK_BF16_A16_B16_C16_OP16 == tensor_gemm_type ) {
                        /* Convert datatype */
                        hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
                        bf2float_GPU( descA->mb, descA->nb, C, descA->mb, C_s, descA->mb, cuda_stream->cuda_stream );

                        /* Copy C_s to C */
                        memcpy_float_GPU( descA->mb, descA->nb, C_s, C, cuda_stream->cuda_stream );
                    }

                    break;
                }
        }

    } else if( DENSE_FP8 == params_tlr->decisions[n*descA->lmt+m] ) {
#if HAVE_FP8
        int fp8_ld = stream_found->fp8_ld;
        int use_padded_fp8 = (fp8_ld != descA->mb) || (tempmm != descA->mb);

        /* Convert datatype, A */
        if( DENSE_DP == params_tlr->decisions_send[k*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            if( use_padded_fp8 ) {
                double2fp8_padded_GPU( tempmm, descA->mb, A, descA->mb, A_fp8, fp8_ld, fp8_ld, fp8_ld, cuda_stream->cuda_stream );
            } else {
                double2fp8_GPU( tempmm, descA->mb, A, descA->mb, A_fp8, descA->mb, cuda_stream->cuda_stream );
            }
            A_use = A_fp8;
        } else if( DENSE_SP == params_tlr->decisions_send[k*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            if( use_padded_fp8 ) {
                float2fp8_padded_GPU( tempmm, descA->mb, A, descA->mb, A_fp8, fp8_ld, fp8_ld, fp8_ld, cuda_stream->cuda_stream );
            } else {
                float2fp8_GPU( tempmm, descA->mb, A, descA->mb, A_fp8, descA->mb, cuda_stream->cuda_stream );
            }
            A_use = A_fp8;
        } else if( DENSE_HP == params_tlr->decisions_send[k*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            if( use_padded_fp8 ) {
                half2fp8_padded_GPU( tempmm, descA->mb, A, descA->mb, A_fp8, fp8_ld, fp8_ld, fp8_ld, cuda_stream->cuda_stream );
            } else {
                half2fp8_GPU( tempmm, descA->mb, A, descA->mb, A_fp8, descA->mb, cuda_stream->cuda_stream );
            }
            A_use = A_fp8;
        }

        /* Convert datatype, B */
        if( DENSE_DP == params_tlr->decisions_send[k*descA->lmt+n] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            if( use_padded_fp8 ) {
                double2fp8_padded_GPU( descA->mb, descA->mb, B, descA->mb, B_fp8, fp8_ld, fp8_ld, fp8_ld, cuda_stream->cuda_stream );
            } else {
                double2fp8_GPU( descA->mb, descA->mb, B, descA->mb, B_fp8, descA->mb, cuda_stream->cuda_stream );
            }
            B_use = B_fp8;
        } else if( DENSE_SP == params_tlr->decisions_send[k*descA->lmt+n] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            if( use_padded_fp8 ) {
                float2fp8_padded_GPU( descA->mb, descA->mb, B, descA->mb, B_fp8, fp8_ld, fp8_ld, fp8_ld, cuda_stream->cuda_stream );
            } else {
                float2fp8_GPU( descA->mb, descA->mb, B, descA->mb, B_fp8, descA->mb, cuda_stream->cuda_stream );
            }
            B_use = B_fp8;
        } else if( DENSE_HP == params_tlr->decisions_send[k*descA->lmt+n] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            if( use_padded_fp8 ) {
                half2fp8_padded_GPU( descA->mb, descA->mb, B, descA->mb, B_fp8, fp8_ld, fp8_ld, fp8_ld, cuda_stream->cuda_stream );
            } else {
                half2fp8_GPU( descA->mb, descA->mb, B, descA->mb, B_fp8, descA->mb, cuda_stream->cuda_stream );
            }
            B_use = B_fp8;
        }

#if 0
        /* First local GEMM convert C from single to half */
        if( 0 == k ) {
            /* Convert datatype */
            float2half_GPU( descA->mb, descA->nb, C, descA->mb, C_h, descA->mb, cuda_stream->cuda_stream );

            /* Copy C_h to C */
            memcpy_half_GPU( descA->mb, descA->nb, C_h, C, cuda_stream->cuda_stream );
        }
#endif

        cublasLtHandle_t lightHandle = stream_found->lightHandle; 
        cublasLtMatmulDesc_t matmulDesc = stream_found->matmulDesc;
        cublasLtMatrixLayout_t Adesc = stream_found->Adesc;
        cublasLtMatrixLayout_t Bdesc = stream_found->Bdesc;
        cublasLtMatrixLayout_t Cdesc = stream_found->Cdesc;
        float alpha = use_padded_fp8 ? 1.0f : -1.0f;
        float beta = use_padded_fp8 ? 0.0f : 1.0f;
        size_t workspaceSize = stream_found->workspaceSize; 
        void *workspace = stream_found->workspace;
        cublasLtMatmulHeuristicResult_t heuristicResultsArray = stream_found->heuristicResultsArray;
	
        //cublasSetStream(handle, cuda_stream->cuda_stream);

        // FP8
        status = cublasLtMatmul(lightHandle, matmulDesc,
                &alpha, A_use, Adesc, B_use, Bdesc,
                &beta,
                use_padded_fp8 ? C_s : C, Cdesc,
                use_padded_fp8 ? C_s : C, Cdesc,
                &heuristicResultsArray.algo, workspace, workspaceSize, cuda_stream->cuda_stream);
        if( 0 != hicma_parsec_check_cublaslt_status(status, "cublasLtMatmul", m, n, k) ) {
            HICMA_GPU_TIME_RETURN;
        }
        if( use_padded_fp8 ) {
            sub_float_from_float_ld_GPU(tempmm, descA->mb, C_s, fp8_ld, C, ldam, cuda_stream->cuda_stream);
        }

        //printf("FP8_GEMM %d %d %d\n", m, n, k);

#if 0
        /* After last local GEMM convert C from half to single */
        if( n-1 == k ) {
            /* Convert datatype */
            half2float_GPU( descA->mb, descA->nb, C, descA->mb, C_s, descA->mb, cuda_stream->cuda_stream );

            /* Copy C_s to C */
            memcpy_float_GPU( descA->mb, descA->nb, C_s, C, cuda_stream->cuda_stream );
        }
#endif

#endif // HAVE_FP8
    }

#if PRINT_KERNEL_TIME && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    if(gpu_time_active) {
        parsec_task_t *parsec_task = (NULL != gpu_task) ? gpu_task->ec : NULL;
        int gpu_time_id = (NULL != parsec_task && NULL != parsec_task->selected_device) ?
                          parsec_task->selected_device->device_index : cuda_device->super.super.device_index;
        hicma_kernel_time_gpu_event_record(parsec_task, gpu_time_id,
                                           cuda_stream->cuda_stream,
                                           gpu_time_start, gpu_time_stop);
    }
#endif
#undef HICMA_GPU_TIME_RETURN
}


void hicma_parsec_core_gemm_denseC_denseA_denseB_runtime_decision_gpu( void *this_task_,
        parsec_tiled_matrix_t* descA,
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_workspace_t *ws_gpu,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        void *C, void *A, void *B, int m, int n, int k,
        int Crank, int Arank, int Brank,
        void *A_norm, void *B_norm )
{
    int counter_id = hicma_parsec_gpu_counter_id(cuda_device, cuda_stream);
    __parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dgemm_task_t *this_task =
        (__parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dgemm_task_t *)this_task_;
    if(!params_tlr->adaptive_decision_runtime) {
        fprintf(stderr, "in GEMM, it needs to enable adaptive_decision_runtime\n");
        return;
    }

    int tempmm = m == descA->mt-1 ? descA->m - m * descA->mb : descA->mb;
    int tempnn = n == descA->nt-1 ? descA->n - n * descA->nb : descA->nb;
    int ldam = BLKLDD( descA, m );
    int ldan = BLKLDD( descA, n );
    void *A_use = A;
    void *B_use = B;
    void *C_use = C;

    const int idx_send_A = k * descA->lmt + m;
    const int idx_send_B = k * descA->lmt + n;
    const int idx_C = n * descA->lmt + m;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    uint16_t Aprecision = params_tlr->decisions_send[idx_send_A];
    uint16_t Bprecision = params_tlr->decisions_send[idx_send_B];
    uint16_t Cprecision = params_tlr->decisions[idx_C];
    uint16_t new_decision = (uint16_t)Cprecision;

    cublasStatus_t status;
    parsec_potrf_workspace_t *_ws_gpu = (parsec_potrf_workspace_t *)ws_gpu;
    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, _ws_gpu);
    cublasHandle_t handle = stream_found->handle_cublas;

    /* Get the temporary buffer on GPU */
    void *A_d, *A_s, *A_h, *A_fp8, *B_d, *B_s, *B_h, *B_fp8, *C_d, *C_s, *C_h;
    A_d = (double *)stream_found->gpu_buffer_A;
    A_s = (float *)stream_found->gpu_buffer_A;
    A_h = (void *)stream_found->gpu_buffer_A;
    A_fp8 = (void *)stream_found->gpu_buffer_A;

    B_d = (double *)stream_found->gpu_buffer_B;
    B_s = (float *)stream_found->gpu_buffer_B;
    B_h = (void *)stream_found->gpu_buffer_B;
    B_fp8 = (void *)stream_found->gpu_buffer_B;

    /* Keep DP temporary storage on a double-capable buffer. */
    C_d = (double *)stream_found->gpu_buffer_C;
    C_s = (float *)stream_found->gpu_buffer_C;
    C_h = (void *)stream_found->gpu_buffer_C;

    /* Fetch runtime {norm, precision} metadata from TRSM outputs. */
    if( NULL != A_norm ) {
        double A_meta[2];
        cudaMemcpy(A_meta, (double *)A_norm, 2 * sizeof(double), cudaMemcpyDeviceToHost);
        Anorm = A_meta[0];
        Aprecision = (uint16_t)A_meta[1];
    }
    if( NULL != B_norm ) {
        double B_meta[2];
        cudaMemcpy(B_meta, (double *)B_norm, 2 * sizeof(double), cudaMemcpyDeviceToHost);
        Bnorm = B_meta[0];
        Bprecision = (uint16_t)B_meta[1];
    }
    hicma_parsec_get_precision_tile(params_tlr, &new_decision, Anorm * Bnorm, m, n);
    cublasSetStream( handle, cuda_stream->cuda_stream );


#if 1
    /* The first local GEMM */
    if( 0 == k && 0 == params_tlr->adaptive_decision
            && DENSE_DP == params_tlr->decisions[idx_C] ) {
        double C_norm = 0.0;
        const size_t tile_elems = descA->mb * descA->nb;
        cublasDnrm2(stream_found->handle_cublas, tile_elems, C, 1, &C_norm);
        hicma_parsec_get_precision_tile(params_tlr, &new_decision, C_norm, m, n);
        if( params_tlr->verbose > 100 ) {
            printf("k 0 m %d n %d C_norm %lf new_decision %d old_decision %d\n",
                   m, n, C_norm, new_decision, params_tlr->decisions[idx_C]);
        }
        if( DENSE_DP != new_decision ) {
            double2float_GPU(descA->mb, descA->nb, C, descA->mb, C_s, descA->mb,
                             cuda_stream->cuda_stream);
            memcpy_float_GPU(descA->mb, descA->nb, C_s, C, cuda_stream->cuda_stream);
            this_task->data._f_C.data_out->original->nb_elts =
                (size_t)tempmm * (size_t)tempnn * sizeof(float);
            params_tlr->decisions[idx_C] = DENSE_SP;
        }
    }
#endif

#if PRINT_KERNEL_TIME && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    cudaEvent_t gpu_time_start = NULL;
    cudaEvent_t gpu_time_stop = NULL;
    int gpu_time_active = hicma_kernel_time_gpu_event_begin((NULL != gpu_task) ? gpu_task->ec : NULL,
                                                            cuda_stream->cuda_stream,
                                                            &gpu_time_start, &gpu_time_stop);
#define HICMA_GPU_TIME_RETURN do {                                      \
        if(gpu_time_active) {                                           \
            hicma_kernel_time_gpu_event_abort(gpu_time_start, gpu_time_stop); \
        }                                                               \
        return;                                                         \
    } while(0)
#else
#define HICMA_GPU_TIME_RETURN do { return; } while(0)
#endif

    if( new_decision != params_tlr->decisions[idx_C] && params_tlr->verbose > 99 ) {
        printf("The decision in gemm_gpu(%d, %d, %d) would change from %u to %u: norm_old %lf norm_new %.16lf (Anorm %.16lf Bnorm %.16lf) Aprecision %u Bprecision %u\n",
                m, n, k,
                params_tlr->decisions[idx_C], new_decision,
                params_tlr->norm_tile[n*params_tlr->NT+m], Anorm * Bnorm,
                Anorm, Bnorm, Aprecision, Bprecision);
    }

    if(params_tlr->verbose > 1) {
        int counter_stride = params_tlr->counter_stride;
        int tid = counter_id;
        if( DENSE_DP == new_decision ) {
            params_tlr->nb_gemms[DENSE_DP*counter_stride+tid] += 1;
        } else if( DENSE_SP == new_decision ) {
            params_tlr->nb_gemms[DENSE_SP*counter_stride+tid] += 1;
        } else if( DENSE_HP == new_decision ) {
            params_tlr->nb_gemms[DENSE_HP*counter_stride+tid] += 1;
        } else if( DENSE_FP8 == new_decision ) {
            params_tlr->nb_gemms[DENSE_FP8*counter_stride+tid] += 1;
        } else if(LOW_RANK_DP == new_decision ) {
            params_tlr->nb_gemms[LOW_RANK_DP*counter_stride+tid] += 1;
        } else if(LOW_RANK_SP == new_decision ) {
            params_tlr->nb_gemms[LOW_RANK_SP*counter_stride+tid] += 1;
        }
    }

    /* If dgemm */
    if( DENSE_DP == new_decision ) {

        if( DENSE_SP == Aprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2double_GPU(tempmm, tempnn, A, descA->mb, A_d, descA->mb, cuda_stream->cuda_stream);
            A_use = A_d;
        } else if( DENSE_DP != Aprecision ) {
            fprintf(stderr, "Unsupported A precision in DP GEMM: %u (%d,%d,%d)\n", (unsigned)Aprecision, m, n, k);
            HICMA_GPU_TIME_RETURN;
        }

        /* Convert datatype, B */
        //if( DENSE_DP != params_tlr->decisions[k*descA->lmt+n] ) {
        if( DENSE_SP == Bprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2double_GPU(tempmm, tempnn, B, descA->mb, B_d, descA->mb, cuda_stream->cuda_stream);
            B_use = B_d;
        } else if( DENSE_DP != Bprecision ) {
            fprintf(stderr, "Unsupported B precision in DP GEMM: %u (%d,%d,%d)\n", (unsigned)Bprecision, m, n, k);
            HICMA_GPU_TIME_RETURN;
        }

        /* Convert datatype, C */
        if( DENSE_DP != Cprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2double_GPU(tempmm, tempnn, C, descA->mb, C_d, descA->mb, cuda_stream->cuda_stream);
            C_use = C_d;
        }

        double alpha = -1.0, beta = 1.0;

        status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                tempmm, tempnn, descA->mb,
                &alpha, A_use, CUDA_R_64F, ldam,
                        B_use, CUDA_R_64F, ldan,
                &beta,  C_use, CUDA_R_64F, ldam,
                CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);

        // Reallocate memory on GPU if memory grows
        // Reallocate memory on CPU if adaptive_memory is enabled
        // Copy back data from C_use to C
        if( DENSE_DP != Cprecision ) {
            if( params_tlr->adaptive_memory || params_tlr->adaptive_decision ) {
                size_t target_bytes = (size_t)descA->mb * (size_t)descA->nb * sizeof(double);
                parsec_data_copy_t *c_copy = this_task->data._f_C.data_out;
                parsec_data_copy_t *c_dev_copy = NULL;
                parsec_data_copy_t *active_copy = NULL;
                parsec_data_copy_t *cpu_copy = NULL;
                void *new_C = NULL;

                if( NULL != c_copy && NULL != c_copy->original ) {
                    c_dev_copy = PARSEC_DATA_GET_COPY(c_copy->original, c_copy->device_index);
                    cpu_copy = PARSEC_DATA_GET_COPY(c_copy->original, 0);
                }
                active_copy = (NULL != c_dev_copy) ? c_dev_copy : c_copy;
                if( 0 != hicma_reallocate_tile_on_gpu(cuda_device, gpu_task, active_copy, C, target_bytes, m, n, k, &new_C) ) {
                    HICMA_GPU_TIME_RETURN;
                }
                C = new_C;

                /* Host tiles are a contiguous cudaHostRegister region unless
                 * adaptive_memory allocated them one-by-one with cudaMallocHost. */
                if( params_tlr->adaptive_memory ) {
                    if( NULL != cpu_copy && cpu_copy != active_copy ) {
                        if( 0 != hicma_reallocate_tile_on_cpu(cpu_copy, target_bytes, m, n, k) ) {
                            HICMA_GPU_TIME_RETURN;
                        }
                    }
                }
            }

            memcpy_double_GPU( tempmm, tempnn, C_use, C, cuda_stream->cuda_stream );
        }
        params_tlr->decisions[idx_C] = DENSE_DP;

        /* If sgemm */
    } else if( DENSE_SP == new_decision ) {
        /* Convert datatype, A */
        if( DENSE_DP == Aprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2float_GPU( tempmm, tempnn, A, descA->mb, A_s, descA->mb, cuda_stream->cuda_stream );
            A_use = A_s;
        } else if( DENSE_SP != Aprecision ) {
            fprintf(stderr, "Unsupported A precision in SP GEMM: %u (%d,%d,%d)\n", (unsigned)Aprecision, m, n, k);
            HICMA_GPU_TIME_RETURN;
        }

        /* Convert datatype, B */
        if( DENSE_DP == Bprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2float_GPU( tempmm, tempnn, B, descA->mb, B_s, descA->mb, cuda_stream->cuda_stream );
            B_use = B_s;
        } else if( DENSE_SP != Bprecision ) {
            fprintf(stderr, "Unsupported B precision in SP GEMM: %u (%d,%d,%d)\n", (unsigned)Bprecision, m, n, k);
            HICMA_GPU_TIME_RETURN;
        }

        /* SGEMM */
        if( DENSE_DP == params_tlr->decisions[idx_C] ) {
            float alpha = 1.0f, beta = 0.0f;
            status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    tempmm, tempnn, descA->mb,
                    &alpha, A_use, CUDA_R_32F, ldam,
                    B_use, CUDA_R_32F, ldan,
                    &beta,  C_s,   CUDA_R_32F, ldam,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            sub_float_from_double_GPU(tempmm, tempnn, C_s, C, cuda_stream->cuda_stream);
        } else {
            float alpha = -1.0f, beta = 1.0f;
            status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    tempmm, tempnn, descA->mb,
                    &alpha, A_use, CUDA_R_32F, ldam,
                    B_use, CUDA_R_32F, ldan,
                    &beta,  C,     CUDA_R_32F, ldam,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            params_tlr->decisions[idx_C] = DENSE_SP;
        }

    } else if( DENSE_HP == new_decision ){

        /* Convert datatype, A */
        if( DENSE_DP == Aprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2half_GPU( tempmm, tempnn, A, descA->mb, A_h, descA->mb, cuda_stream->cuda_stream );
            A_use = A_h;
        } else if( DENSE_SP == Aprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2half_GPU( tempmm, tempnn, A, descA->mb, A_h, descA->mb, cuda_stream->cuda_stream );
            A_use = A_h;
        } else {
            fprintf(stderr, "Precision A is not correct: %d %d %d (%u)!\n", m, n, k, (unsigned)Aprecision);
            HICMA_GPU_TIME_RETURN;
        }

        /* Convert datatype, B */
        if( DENSE_DP == Bprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2half_GPU( tempmm, tempnn, B, descA->mb, B_h, descA->mb, cuda_stream->cuda_stream );
            B_use = B_h;
        } else if( DENSE_SP == Bprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2half_GPU( tempmm, tempnn, B, descA->mb, B_h, descA->mb, cuda_stream->cuda_stream );
            B_use = B_h;
        } else {
            fprintf(stderr, "Precision B is not correct: %d %d %d (%u)!\n", m, n, k, (unsigned)Bprecision);
            HICMA_GPU_TIME_RETURN;
        }

        /* HGEMM */
        if( DENSE_DP == params_tlr->decisions[idx_C] ) {
            float alpha = 1.0f, beta = 0.0f;
            status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    tempmm, tempnn, descA->mb,
                    &alpha, A_use, CUDA_R_16F, ldam,
                    B_use, CUDA_R_16F, ldan,
                    &beta,  C_s,   CUDA_R_32F, ldam,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            sub_float_from_double_GPU(tempmm, tempnn, C_s, C, cuda_stream->cuda_stream);
        } else {
            float alpha = -1.0f, beta = 1.0f;
            status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    tempmm, tempnn, descA->mb,
                    &alpha, A_use, CUDA_R_16F, ldam,
                    B_use, CUDA_R_16F, ldan,
                    &beta,  C,     CUDA_R_32F, ldam,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            params_tlr->decisions[idx_C] = DENSE_SP;
        }

#if HAVE_FP8
    } else if( DENSE_FP8 == new_decision ){
        int fp8_ld = stream_found->fp8_ld;
        int use_padded_fp8 = (fp8_ld != descA->mb) || (tempmm != descA->mb) || (tempnn != descA->mb);

        /* Convert datatype, A */
        if( DENSE_DP == Aprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            if( use_padded_fp8 ) {
                double2fp8_padded_GPU( tempmm, descA->mb, A, descA->mb, A_fp8, fp8_ld, fp8_ld, fp8_ld, cuda_stream->cuda_stream );
            } else {
                double2fp8_GPU( tempmm, descA->mb, A, descA->mb, A_fp8, descA->mb, cuda_stream->cuda_stream );
            }
            A_use = A_fp8;
        } else if( DENSE_SP == Aprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            if( use_padded_fp8 ) {
                float2fp8_padded_GPU( tempmm, descA->mb, A, descA->mb, A_fp8, fp8_ld, fp8_ld, fp8_ld, cuda_stream->cuda_stream );
            } else {
                float2fp8_GPU( tempmm, descA->mb, A, descA->mb, A_fp8, descA->mb, cuda_stream->cuda_stream );
            }
            A_use = A_fp8;
        } else {
            fprintf(stderr, "Precision A is not correct PF8: %d %d %d: %u!\n", m, n, k, (unsigned)Aprecision);
            HICMA_GPU_TIME_RETURN;
        }

        /* Convert datatype, B */
        if( DENSE_DP == Bprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            if( use_padded_fp8 ) {
                double2fp8_padded_GPU( tempnn, descA->mb, B, descA->mb, B_fp8, fp8_ld, fp8_ld, fp8_ld, cuda_stream->cuda_stream );
            } else {
                double2fp8_GPU( tempnn, descA->mb, B, descA->mb, B_fp8, descA->mb, cuda_stream->cuda_stream );
            }
            B_use = B_fp8;
        } else if( DENSE_SP == Bprecision ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            if( use_padded_fp8 ) {
                float2fp8_padded_GPU( tempnn, descA->mb, B, descA->mb, B_fp8, fp8_ld, fp8_ld, fp8_ld, cuda_stream->cuda_stream );
            } else {
                float2fp8_GPU( tempnn, descA->mb, B, descA->mb, B_fp8, descA->mb, cuda_stream->cuda_stream );
            }
            B_use = B_fp8;
        } else {
            fprintf(stderr, "Precision B is not correct FP8: %d %d %d: %u!\n", m, n, k, (unsigned)Bprecision);
            HICMA_GPU_TIME_RETURN;
        }

        /* FP8 */
        cublasLtHandle_t lightHandle = stream_found->lightHandle;
        cublasLtMatmulDesc_t matmulDesc = stream_found->matmulDesc;
        cublasLtMatrixLayout_t Adesc = stream_found->Adesc;
        cublasLtMatrixLayout_t Bdesc = stream_found->Bdesc;
        cublasLtMatrixLayout_t Cdesc = stream_found->Cdesc;
        size_t workspaceSize = stream_found->workspaceSize;
        void *workspace = stream_found->workspace;
        cublasLtMatmulHeuristicResult_t heuristicResultsArray = stream_found->heuristicResultsArray;

        if( DENSE_DP == params_tlr->decisions[idx_C] || use_padded_fp8 ) {
            float alpha = 1.0f, beta = 0.0f;
            status = cublasLtMatmul(lightHandle, matmulDesc,
                    &alpha, A_use, Adesc, B_use, Bdesc,
                    &beta, C_s, Cdesc, C_s, Cdesc,
                    &heuristicResultsArray.algo, workspace, workspaceSize, cuda_stream->cuda_stream);
            if( 0 != hicma_parsec_check_cublaslt_status(status, "cublasLtMatmul", m, n, k) ) {
                HICMA_GPU_TIME_RETURN;
            }
            if( DENSE_DP == params_tlr->decisions[idx_C] ) {
                sub_float_from_double_ld_GPU(tempmm, tempnn, C_s, fp8_ld, C, ldam, cuda_stream->cuda_stream);
            } else {
                sub_float_from_float_ld_GPU(tempmm, tempnn, C_s, fp8_ld, C, ldam, cuda_stream->cuda_stream);
                params_tlr->decisions[idx_C] = DENSE_SP;
            }
        } else {
            float alpha = -1.0f, beta = 1.0f;
            status = cublasLtMatmul(lightHandle, matmulDesc,
                    &alpha, A_use, Adesc, B_use, Bdesc,
                    &beta, C, Cdesc, C, Cdesc,
                    &heuristicResultsArray.algo, workspace, workspaceSize, cuda_stream->cuda_stream);
            if( 0 != hicma_parsec_check_cublaslt_status(status, "cublasLtMatmul", m, n, k) ) {
                HICMA_GPU_TIME_RETURN;
            }
            params_tlr->decisions[idx_C] = DENSE_SP;
        }
#endif

    //} else if( DENSE_FP4 == new_decision ){
    } else {
            fprintf(stderr, "New decision is not supported: %d %d %d!\n", m, n, k);
    }

#if PRINT_KERNEL_TIME && defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    if(gpu_time_active) {
        parsec_task_t *parsec_task = (NULL != gpu_task) ? gpu_task->ec : NULL;
        int gpu_time_id = (NULL != parsec_task && NULL != parsec_task->selected_device) ?
                          parsec_task->selected_device->device_index : cuda_device->super.super.device_index;
        hicma_kernel_time_gpu_event_record(parsec_task, gpu_time_id,
                                           cuda_stream->cuda_stream,
                                           gpu_time_start, gpu_time_stop);
    }
#endif
#undef HICMA_GPU_TIME_RETURN

    if( 0 == k && (DENSE_DP == new_decision || DENSE_SP == new_decision ||
                   DENSE_HP == new_decision || DENSE_FP8 == new_decision) ) {
        params_tlr->decisions_gemm_first[idx_C] = new_decision;
    }

// trsm handles the first column
    if( k + 1 == n && (DENSE_DP == new_decision || DENSE_SP == new_decision ||
                      DENSE_HP == new_decision || DENSE_FP8 == new_decision) ) {
        params_tlr->decisions_gemm_last[idx_C] = new_decision;
    }

#if 1
    /* After the last GEMM that writes C, optionally store C in SP.
     * handle_cublas is HOST pointer mode (synchronous nrm2 result).
     * handle_cublas_deviceptr cannot write into a host stack address. */
    if( k+1 == n && 0 != params_tlr->adaptive_decision_runtime
            && DENSE_DP == params_tlr->decisions[idx_C] ) {
        double C_norm = 0.0;
        const size_t tile_elems = descA->mb * descA->nb;
        cublasDnrm2(stream_found->handle_cublas, tile_elems, C, 1, &C_norm);
        hicma_parsec_get_precision_tile(params_tlr, &new_decision, C_norm, m, n);
        if( params_tlr->verbose > 100 ) {
            printf("m %d n %d C_norm %lf new_decision %d old_decision %d\n",
                   m, n, C_norm, new_decision, params_tlr->decisions[idx_C]);
        }
        if( DENSE_DP != new_decision ) {
            double2float_GPU(descA->mb, descA->nb, C, descA->mb, C_s, descA->mb,
                             cuda_stream->cuda_stream);
            memcpy_float_GPU(descA->mb, descA->nb, C_s, C, cuda_stream->cuda_stream);
            this_task->data._f_C.data_out->original->nb_elts =
                (size_t)tempmm * (size_t)tempnn * sizeof(float);
            params_tlr->decisions[idx_C] = DENSE_SP;
        }
    }
#endif

}


/**
 * @brief GPU implementation of GEMM for dense result matrix C with low-rank matrix A and dense matrix B
 * 
 * This function performs the general matrix multiplication C = C + A * B where
 * matrix A is low-rank (stored as U*V^T), matrix B is dense, and result matrix C is dense,
 * using GPU acceleration. It handles the low-rank decomposition of A and performs
 * efficient matrix operations using the rank structure to reduce computational complexity.
 * 
 * @param[in] descA Matrix descriptor for matrix A
 * @param[in,out] params_tlr HICMA parameters containing operation settings
 * @param[in] ws_gpu GPU workspace for GEMM operations
 * @param[in] cuda_device CUDA device module for GPU operations
 * @param[in] gpu_task GPU task descriptor
 * @param[in] cuda_stream CUDA execution stream
 * @param[in,out] C Matrix C (result matrix, dense)
 * @param[in] A Matrix A (low-rank, stored as U*V^T)
 * @param[in] B Matrix B (dense)
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile
 * @param[in] k Diagonal index of the current tile
 * @param[in] Crank Rank of matrix C (unused for dense matrices)
 * @param[in] Arank Rank of matrix A (low-rank)
 * @param[in] Brank Rank of matrix B (unused for dense matrices)
 */
void hicma_parsec_core_gemm_denseC_lrA_denseB_gpu( parsec_tiled_matrix_t* descA,
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_workspace_t *ws_gpu,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        void *C, void *A, void *B, int m, int n, int k,
        int Crank, int Arank, int Brank )
{
    int counter_id = hicma_parsec_gpu_counter_id(cuda_device, cuda_stream);
    int tempmm = m == descA->mt-1 ? descA->m - m * descA->mb : descA->mb;
    int ldam = BLKLDD( descA, m );
    int ldan = BLKLDD( descA, n );
    void *A_use = A;
    void *B_use = B;
    void *A_d, *A_s, *A_h, *B_d, *B_s, *B_h, *C_s, *C_h, *buffer_mbr, *buffer_rr, *Au, *Av;
    cublasStatus_t status;

    /* If rank is 0, return */
    if( 0 == Arank ) {
        return;
    }

    /* Find workspace */
    parsec_potrf_workspace_t *_ws_gpu = (parsec_potrf_workspace_t *)ws_gpu;
    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, _ws_gpu);

    /* Get handle_cublas */
    cublasHandle_t handle = stream_found->handle_cublas;
    if( DENSE_HP == params_tlr->decisions[n*descA->lmt+m] ) {
        handle = stream_found->handle_cublas_tensor;
    }
    cublasSetStream( handle, cuda_stream->cuda_stream );

    /* Get the temporary buffer on GPU */
    A_d = (double *)stream_found->gpu_buffer_A;
    A_s = (float *)stream_found->gpu_buffer_A;
    A_h = (float *)stream_found->gpu_buffer_A;

    B_d = (double *)stream_found->gpu_buffer_B;
    B_s = (float *)stream_found->gpu_buffer_B;
    B_h = (float *)stream_found->gpu_buffer_B;

    C_s = (float *)stream_found->gpu_buffer_C;
    C_h = (float *)stream_found->gpu_buffer_C;

    buffer_mbr = (double *)stream_found->gpu_buffer_mbr;
    buffer_rr = (double *)stream_found->gpu_buffer_rr;

    if(DEBUG_INFO) printf("GPU GEMM (%d, %d, %d) : %d %d %d : C_DENSE, A_LOW_RANK, B_DENSE\n",
            m, n, k, params_tlr->decisions[n*descA->lmt+m], params_tlr->decisions[k*descA->lmt+m], params_tlr->decisions[k*descA->lmt+n]);

    if( DENSE_DP == params_tlr->decisions[n*descA->lmt+m] ) {
        /* Convert datatype, A */
        if( LOW_RANK_SP == params_tlr->decisions[k*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2double_GPU( descA->mb, Arank * 2, A, descA->mb, A_d, descA->mb, cuda_stream->cuda_stream );
            A_use = A_d;
        }

        /* Convert datatype, B */
        if( DENSE_SP == params_tlr->decisions[k*descA->lmt+n] || DENSE_HP == params_tlr->decisions[k*descA->lmt+n] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2double_GPU( descA->mb, descA->nb, B, descA->mb, B_d, descA->mb, cuda_stream->cuda_stream );
            B_use = B_d;
        }

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(double);

        /* tmp_mbr = trans(Av) * trans(B) */
        double alpha = 1.0;
        double beta = 0.0;

        status = hicma_parsec_dgemm_gpu( handle, CUBLAS_OP_T, CUBLAS_OP_T,
                Arank, descA->mb, descA->mb,
                &alpha, Av         /*A(m, k)*/, descA->mb,
                        B_use      /*A(n, m)*/, descA->mb,
                &beta,  buffer_mbr /*A(k, n)*/, Arank);

        /* C = C - Au * tmp_mbr */
        alpha = -1.0;
        beta = 1.0;

        status = hicma_parsec_dgemm_gpu( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                descA->mb, descA->mb, Arank,
                &alpha, Au         /*A(m, k)*/, descA->mb,
                        buffer_mbr /*A(k, n)*/, Arank,
                &beta,  C          /*A(m, n)*/, descA->mb);

    } else if( DENSE_SP == params_tlr->decisions[n*descA->lmt+m] ) {
        /* Convert datatype, A */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2float_GPU( descA->mb, Arank * 2, A, descA->mb, A_s, descA->mb, cuda_stream->cuda_stream );
            A_use = A_s;
        }

        /* Convert datatype, B */
        if( DENSE_DP == params_tlr->decisions[k*descA->lmt+n] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2float_GPU( descA->mb, descA->nb, B, descA->mb, B_s, descA->mb, cuda_stream->cuda_stream );
            B_use = B_s;
        }

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(float);

        /* tmp_mbr = trans(Av) * trans(B) */
        float alpha = 1.0f;
        float beta = 0.0f;
        status = hicma_parsec_sgemm_gpu( handle, CUBLAS_OP_T, CUBLAS_OP_T,
                Arank, descA->mb, descA->mb,
                &alpha, Av         /*A(m, k)*/, descA->mb,
                B_use      /*A(n, m)*/, descA->mb,
                &beta,  buffer_mbr /*A(k, n)*/, Arank,
                params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m]);

        /* C = C - Au * tmp_mbr */
        alpha = -1.0f;
        beta = 1.0f;

        status = hicma_parsec_sgemm_gpu( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                descA->mb, descA->mb, Arank,
                &alpha, Au         /*A(m, k)*/, descA->mb,
                        buffer_mbr /*A(k, n)*/, Arank,
                &beta,  C          /*A(m, n)*/, descA->mb,
                params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m]);

    } else {
        /* Convert datatype, A */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2half_GPU(   descA->mb, Arank * 2, A, descA->mb, A_h, descA->mb, cuda_stream->cuda_stream );
            A_use = A_h;
        } else {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2half_GPU(   descA->mb, Arank * 2, A, descA->mb, A_h, descA->mb, cuda_stream->cuda_stream );
            A_use = A_h;
        }

        /* Convert datatype, B */
        if( DENSE_DP == params_tlr->decisions[k*descA->lmt+n] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2half_GPU(   descA->mb, descA->nb, B, descA->mb, B_h, descA->mb, cuda_stream->cuda_stream );
            B_use = B_h;
        } else {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2half_GPU(   descA->mb, descA->nb, B, descA->mb, B_h, descA->mb, cuda_stream->cuda_stream );
            B_use = B_h;
        }

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(float) / 2;

        /* First local GEMM convert C from single to half */
        if( 0 == k && (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP32
                    || params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP16) ) {
            /* Convert datatype */
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2half_GPU( descA->mb, descA->nb, C, descA->mb, C_h, descA->mb, cuda_stream->cuda_stream );

            /* Copy C_h to C */
            memcpy_half_GPU( descA->mb, descA->nb, C_h, C, cuda_stream->cuda_stream );
        }

        /* tmp_mbr = trans(Av) * trans(B) */
        float alpha = 1.0f;
        float beta = 0.0f;

        status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                (int64_t)Arank, (int64_t)descA->mb, (int64_t)descA->mb,
                &alpha, Av,          CUDA_R_16F, (int64_t)descA->mb,
                        B_use,       CUDA_R_16F, (int64_t)descA->mb,
                &beta,  buffer_mbr,  CUDA_R_16F, (int64_t)Arank,
                (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP16)? CUBLAS_COMPUTE_16F: CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        /* C = C - Au * tmp_mbr */
        alpha = -1.0f;
        beta = 1.0f;

        status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int64_t)descA->mb, (int64_t)descA->mb, (int64_t)Arank,
                &alpha, Au,          CUDA_R_16F, (int64_t)descA->mb,
                        buffer_mbr,  CUDA_R_16F, (int64_t)Arank,
                &beta,  C,           (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C32_OP32)? CUDA_R_32F: CUDA_R_16F, (int64_t)descA->mb,
                (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP16)? CUBLAS_COMPUTE_16F: CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        /* After last local GEMM convert C from half to single */
        if( n-1 == k && (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP32
                    || params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP16) ) {
            /* Convert datatype */
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            half2float_GPU( descA->mb, descA->nb, C, descA->mb, C_s, descA->mb, cuda_stream->cuda_stream );

            /* Copy C_s to C */
            memcpy_float_GPU( descA->mb, descA->nb, C_s, C, cuda_stream->cuda_stream );
        }
    }

}


/**
 * @brief GPU implementation of GEMM for dense result matrix C with low-rank matrices A and B
 * 
 * This function performs the general matrix multiplication C = C + A * B where
 * both matrices A and B are low-rank (stored as U*V^T), and result matrix C is dense,
 * using GPU acceleration. It efficiently handles the low-rank decomposition of both
 * operands and performs matrix operations using the rank structure to minimize
 * computational complexity.
 * 
 * @param[in] descA Matrix descriptor for matrix A
 * @param[in,out] params_tlr HICMA parameters containing operation settings
 * @param[in] ws_gpu GPU workspace for GEMM operations
 * @param[in] cuda_device CUDA device module for GPU operations
 * @param[in] gpu_task GPU task descriptor
 * @param[in] cuda_stream CUDA execution stream
 * @param[in,out] C Matrix C (result matrix, dense)
 * @param[in] A Matrix A (low-rank, stored as U*V^T)
 * @param[in] B Matrix B (low-rank, stored as U*V^T)
 * @param[in] m Row index of the current tile
 * @param[in] n Column index of the current tile
 * @param[in] k Diagonal index of the current tile
 * @param[in] Crank Rank of matrix C (unused for dense matrices)
 * @param[in] Arank Rank of matrix A (low-rank)
 * @param[in] Brank Rank of matrix B (low-rank)
 */
void hicma_parsec_core_gemm_denseC_lrA_lrB_gpu( parsec_tiled_matrix_t* descA,
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_workspace_t *ws_gpu,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        void *C, void *A, void *B, int m, int n, int k,
        int Crank, int Arank, int Brank )
{
    int counter_id = hicma_parsec_gpu_counter_id(cuda_device, cuda_stream);
    int tempmm = m == descA->mt-1 ? descA->m - m * descA->mb : descA->mb;
    int ldam = BLKLDD( descA, m );
    int ldan = BLKLDD( descA, n );
    void *A_use = A;
    void *B_use = B;
    void *A_d, *A_s, *A_h, *B_d, *B_s, *B_h, *C_s, *C_h, *buffer_mbr, *buffer_rr, *Au, *Av, *Bu, *Bv;
    cublasStatus_t status;

    /* If rank is 0, return */
    if( 0 == Arank || 0 == Brank ) {
        return;
    }

    /* Find workspace */
    parsec_potrf_workspace_t *_ws_gpu = (parsec_potrf_workspace_t *)ws_gpu;
    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, _ws_gpu);

    /* Get handle_cublas */
    cublasHandle_t handle = stream_found->handle_cublas;
    if( DENSE_HP == params_tlr->decisions[n*descA->lmt+m] ) {
        handle = stream_found->handle_cublas_tensor;
    }
    cublasSetStream( handle, cuda_stream->cuda_stream );

    /* Get the temporary buffer on GPU */
    A_d = (double *)stream_found->gpu_buffer_A;
    A_s = (float *)stream_found->gpu_buffer_A;
    A_h = (float *)stream_found->gpu_buffer_A;

    B_d = (double *)stream_found->gpu_buffer_B;
    B_s = (float *)stream_found->gpu_buffer_B;
    B_h = (float *)stream_found->gpu_buffer_B;

    C_s = (float *)stream_found->gpu_buffer_C;
    C_h = (float *)stream_found->gpu_buffer_C;

    buffer_mbr = (double *)stream_found->gpu_buffer_mbr;
    buffer_rr = (double *)stream_found->gpu_buffer_rr;

    if(DEBUG_INFO) printf("GPU GEMM (%d, %d, %d) : %d %d %d : C_DENSE, A_LOW_RANK, B_LOW_RANK\n",
            m, n, k, params_tlr->decisions[n*descA->lmt+m], params_tlr->decisions[k*descA->lmt+m], params_tlr->decisions[k*descA->lmt+n]);

    if( DENSE_DP == params_tlr->decisions[n*descA->lmt+m] )
    {
        /* Convert datatype, A */
        if( LOW_RANK_SP == params_tlr->decisions[k*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2double_GPU( descA->mb, Arank * 2, A, descA->mb, A_d, descA->mb, cuda_stream->cuda_stream );
            A_use = A_d;
        }

        /* Convert datatype, B */
        if( LOW_RANK_SP == params_tlr->decisions[k*descA->lmt+n] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2double_GPU( descA->mb, Brank * 2, B, descA->mb, B_d, descA->mb, cuda_stream->cuda_stream );
            B_use = B_d;
        }

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(double);
        Bu = (void *)B_use;
        Bv = (void *)B_use + descA->mb * Brank * sizeof(double);

        /* tmp_rr = trans(Av) * Bv */
        double alpha = 1.0;
        double beta = 0.0;
        status = hicma_parsec_dgemm_gpu( handle, CUBLAS_OP_T, CUBLAS_OP_N,
                Arank, Brank, descA->mb,
                &alpha, Av        /*A(k, m)*/, descA->mb,
                        Bv        /*A(k, n)*/, descA->mb,
                &beta, buffer_rr  /*A(m, n)*/, Arank);

        if( Arank > Brank ) {
            /* tmp_mbr = Au * tmp_rr */
            alpha = 1.0;
            beta = 0.0;
            status = hicma_parsec_dgemm_gpu( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    descA->mb, Brank, Arank,
                    &alpha, Au         /*A(m, k)*/, descA->mb,
                            buffer_rr  /*A(k, n)*/, Arank,
                    &beta,  buffer_mbr /*A(m, n)*/, descA->mb);

            /* C = C - tmp_mbr * trans(Bu) */
            alpha = -1.0;
            beta = 1.0;
            status = hicma_parsec_dgemm_gpu( handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    descA->mb, descA->mb, Brank,
                    &alpha, buffer_mbr /*A(m, k)*/, descA->mb,
                            Bu         /*A(n, k)*/, descA->mb,
                    &beta,  C          /*A(m, n)*/, descA->mb);
        } else {
            /* tmp_mbr = tmp_rr * trans(Bu) */
            alpha = 1.0;
            beta = 0.0;
            status = hicma_parsec_dgemm_gpu( handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    Arank, descA->mb, Brank,
                    &alpha, buffer_rr  /*A(m, k)*/, Arank,
                            Bu         /*A(n, k)*/, descA->mb,
                    &beta,  buffer_mbr /*A(m, n)*/, Arank);

            /* C = C - Au * tmp_mbr */
            alpha = -1.0;
            beta = 1.0;
            status = hicma_parsec_dgemm_gpu( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    descA->mb, descA->mb, Arank,
                    &alpha, Au         /*A(m, k)*/, descA->mb,
                            buffer_mbr /*A(k, n)*/, Arank,
                    &beta,  C          /*A(m, n)*/, descA->mb);
        }

    }
    else if( DENSE_SP == params_tlr->decisions[n*descA->lmt+m] )
    {
        /* Convert datatype, A */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2float_GPU( descA->mb, Arank * 2, A, descA->mb, A_s, descA->mb, cuda_stream->cuda_stream );
            A_use = A_s;
        }

        /* Convert datatype, B */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+n] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2float_GPU( descA->mb, Brank * 2, B, descA->mb, B_s, descA->mb, cuda_stream->cuda_stream );
            B_use = B_s;
        }

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(float);
        Bu = (void *)B_use;
        Bv = (void *)B_use + descA->mb * Brank * sizeof(float);

        /* tmp_rr = trans(Av) * Bv */
        float alpha = 1.0f;
        float beta = 0.0f;
        status = hicma_parsec_sgemm_gpu( handle, CUBLAS_OP_T, CUBLAS_OP_N,
                Arank, Brank, descA->mb,
                &alpha, Av        /*A(k, m)*/, descA->mb,
                        Bv        /*A(k, n)*/, descA->mb,
                &beta, buffer_rr  /*A(m, n)*/, Arank,
                params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m]);

        if( Arank > Brank ) {
            /* tmp_mbr = Au * tmp_rr */
            alpha = 1.0f;
            beta = 0.0f;
            status = hicma_parsec_sgemm_gpu( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    descA->mb, Brank, Arank,
                    &alpha, Au         /*A(m, k)*/, descA->mb,
                            buffer_rr  /*A(k, n)*/, Arank,
                    &beta,  buffer_mbr /*A(m, n)*/, descA->mb,
                    params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m]);

            /* C = C - tmp_mbr * trans(Bu) */
            alpha = -1.0f;
            beta = 1.0f;
            status = hicma_parsec_sgemm_gpu( handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    descA->mb, descA->mb, Brank,
                    &alpha, buffer_mbr /*A(m, k)*/, descA->mb,
                            Bu         /*A(n, k)*/, descA->mb,
                    &beta,  C          /*A(m, n)*/, descA->mb,
                    params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m]);

        } else {
            /* tmp_mbr = tmp_rr * trans(Bu) */
            alpha = 1.0f;
            beta = 0.0f;
            status = hicma_parsec_sgemm_gpu( handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    Arank, descA->mb, Brank,
                    &alpha, buffer_rr  /*A(m, k)*/, Arank,
                            Bu         /*A(n, k)*/, descA->mb,
                    &beta,  buffer_mbr /*A(m, n)*/, Arank,
                    params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m]);

            /* C = C - Au * tmp_mbr */
            alpha = -1.0;
            beta = 1.0;
            status = hicma_parsec_sgemm_gpu( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    descA->mb, descA->mb, Arank,
                    &alpha, Au         /*A(m, k)*/, descA->mb,
                            buffer_mbr /*A(k, n)*/, Arank,
                    &beta,  C          /*A(m, n)*/, descA->mb,
                    params_tlr->decisions_gemm_gpu[n*params_tlr->NT+m]);
        }

    }
    else {
        /* Convert datatype, A */
        if( LOW_RANK_DP == params_tlr->decisions[k*descA->lmt+m] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2half_GPU(   descA->mb, Arank * 2, A, descA->mb, A_h, descA->mb, cuda_stream->cuda_stream );
            A_use = A_h;
        } else {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2half_GPU(   descA->mb, Arank * 2, A, descA->mb, A_h, descA->mb, cuda_stream->cuda_stream );
            A_use = A_h;
        }

        /* Convert datatype, B */
        if( DENSE_DP == params_tlr->decisions[k*descA->lmt+n] ) {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            double2half_GPU(   descA->mb, descA->nb, B, descA->mb, B_h, descA->mb, cuda_stream->cuda_stream );
            B_use = B_h;
        } else {
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2half_GPU(   descA->mb, descA->nb, B, descA->mb, B_h, descA->mb, cuda_stream->cuda_stream );
            B_use = B_h;
        }

        /* U and V pointer */
        Au = (void *)A_use;
        Av = (void *)A_use + descA->mb * Arank * sizeof(float) / 2;
        Bu = (void *)B_use;
        Bv = (void *)B_use + descA->mb * Brank * sizeof(float) / 2;

        /* First local GEMM convert C from single to half */
        if( 0 == k && (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP32
                    || params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP16) ) {
            /* Convert datatype */
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            float2half_GPU( descA->mb, descA->nb, C, descA->mb, C_h, descA->mb, cuda_stream->cuda_stream );

            /* Copy C_h to C */
            memcpy_half_GPU( descA->mb, descA->nb, C_h, C, cuda_stream->cuda_stream );
        }

        /* tmp_rr = trans(Av) * Bv */
        float alpha = 1.0f;
        float beta = 0.0f;
        status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                (int64_t)Arank, (int64_t)Brank, (int64_t)descA->mb,
                &alpha, Av,          CUDA_R_16F, (int64_t)descA->mb,
                        Bv,          CUDA_R_16F, (int64_t)descA->mb,
                &beta,  buffer_rr,   CUDA_R_16F, (int64_t)Arank,
                (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP16)? CUBLAS_COMPUTE_16F: CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        if( Arank > Brank ) {
            /* tmp_mbr = Au * tmp_rr */
            alpha = 1.0f;
            beta = 0.0f;
            status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int64_t)descA->mb, (int64_t)Brank, (int64_t)Arank,
                    &alpha, Au,          CUDA_R_16F, (int64_t)descA->mb,
                            buffer_rr,   CUDA_R_16F, (int64_t)Arank,
                    &beta,  buffer_mbr,  CUDA_R_16F, (int64_t)descA->mb,
                    (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP16)? CUBLAS_COMPUTE_16F: CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            /* C = C - tmp_mbr * trans(Bu) */
            alpha = -1.0f;
            beta = 1.0f;
            status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    (int64_t)descA->mb, (int64_t)descA->mb, (int64_t)Brank,
                    &alpha, buffer_mbr,  CUDA_R_16F, (int64_t)descA->mb,
                            Bu,          CUDA_R_16F, (int64_t)descA->mb,
                    &beta,  C,           (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C32_OP32)? CUDA_R_32F: CUDA_R_16F, (int64_t)descA->mb,
                    (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP16)? CUBLAS_COMPUTE_16F: CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        } else {
            /* tmp_mbr = tmp_rr * trans(Bu) */
            alpha = 1.0f;
            beta = 0.0f;
            status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    (int64_t)Arank, (int64_t)descA->mb, (int64_t)Brank,
                    &alpha, buffer_rr,   CUDA_R_16F, (int64_t)Arank,
                            Bu,          CUDA_R_16F, (int64_t)descA->mb,
                    &beta,  buffer_mbr,  CUDA_R_16F, (int64_t)Arank,
                    (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP16)? CUBLAS_COMPUTE_16F: CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            /* C = C - Au * tmp_mbr */
            alpha = -1.0f;
            beta = 1.0f;
            status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int64_t)descA->mb, (int64_t)descA->mb, (int64_t)Arank,
                    &alpha, Au,          CUDA_R_16F, (int64_t)descA->mb,
                            buffer_mbr,  CUDA_R_16F, (int64_t)Arank,
                    &beta,  C,           (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C32_OP32)? CUDA_R_32F: CUDA_R_16F, (int64_t)descA->mb,
                    (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP16)? CUBLAS_COMPUTE_16F: CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }

        /* After last local GEMM convert C from half to single */
        if( n-1 == k && (params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP32
                    || params_tlr->tensor_gemm & MASK_TF16_A16_B16_C16_OP16) ) {
            /* Convert datatype */
            hicma_parsec_count_datatype_conversion(params_tlr, counter_id);
            half2float_GPU( descA->mb, descA->nb, C, descA->mb, C_s, descA->mb, cuda_stream->cuda_stream );

            /* Copy C_s to C */
            memcpy_float_GPU( descA->mb, descA->nb, C_s, C, cuda_stream->cuda_stream );
        }
    }

}


#endif

/**
 * @brief Calculate matrix norm for a tile based on datatype
 * 
 * This function calculates the norm of a matrix tile based on the specified datatype.
 * It supports both row-major and column-major memory layouts.
 * 
 * @param[in] data Pointer to the matrix tile data
 * @param[in] tempmm Number of rows in the tile
 * @param[in] tempnn Number of columns in the tile
 * @param[in] lda Leading dimension
 * @param[in] datatype_str String representation of the datatype
 * @param[in] is_row_major true for row-major layout, false for column-major layout
 * @param[in] is_lower true if lower triangular, false otherwise
 * @param[in] is_upper true if upper triangular, false otherwise
 * @param[in] is_diagonal true if diagonal tile (m == n), false otherwise
 * 
 * @return The calculated norm value
 */
double hicma_parsec_core_matrix_norm_get(const void *data, int tempmm, int tempnn, int lda, 
                                        const char *datatype_str, bool is_row_major,
                                        bool is_lower, bool is_upper, bool is_diagonal) {
    if (data == NULL || datatype_str == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to hicma_parsec_core_matrix_norm_get\n");
        return 0.0;
    }
    
    if (tempmm <= 0 || tempnn <= 0 || lda <= 0) {
        fprintf(stderr, "Error: Invalid dimensions in hicma_parsec_core_matrix_norm_get: tempmm=%d, tempnn=%d, lda=%d\n", tempmm, tempnn, lda);
        return 0.0;
    }
    
    double norm_squared = 0.0;
    
    // Floating point types
    if (strcmp(datatype_str, "double") == 0 || strcmp(datatype_str, "d") == 0) {
        const double *matrix_data = (const double *)data;
        if (is_diagonal && is_lower) {
            // Lower triangular diagonal tile
            for (int j = 0; j < tempnn; j++) {
                for (int i = j; i < tempmm; i++) {
                    double current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += current_value * current_value;
                }
            }
        } else if (is_diagonal && is_upper) {
            // Upper triangular diagonal tile
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i <= j; i++) {
                    double current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += current_value * current_value;
                }
            }
        } else {
            // Full tile
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i < tempmm; i++) {
                    double current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += current_value * current_value;
                }
            }
        }
    }
    else if (strcmp(datatype_str, "float") == 0 || strcmp(datatype_str, "single") == 0 || strcmp(datatype_str, "s") == 0) {
        const float *matrix_data = (const float *)data;
        if (is_diagonal && is_lower) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = j; i < tempmm; i++) {
                    float current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else if (is_diagonal && is_upper) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i <= j; i++) {
                    float current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i < tempmm; i++) {
                    float current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        }
    }
    // Signed integer types
    else if (strcmp(datatype_str, "int8") == 0 || strcmp(datatype_str, "i8") == 0) {
        const int8_t *matrix_data = (const int8_t *)data;
        if (is_diagonal && is_lower) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = j; i < tempmm; i++) {
                    int8_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else if (is_diagonal && is_upper) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i <= j; i++) {
                    int8_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i < tempmm; i++) {
                    int8_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        }
    }
    else if (strcmp(datatype_str, "int16") == 0 || strcmp(datatype_str, "i16") == 0) {
        const int16_t *matrix_data = (const int16_t *)data;
        if (is_diagonal && is_lower) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = j; i < tempmm; i++) {
                    int16_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else if (is_diagonal && is_upper) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i <= j; i++) {
                    int16_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i < tempmm; i++) {
                    int16_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        }
    }
    else if (strcmp(datatype_str, "int32") == 0 || strcmp(datatype_str, "int") == 0 || 
             strcmp(datatype_str, "i32") == 0 || strcmp(datatype_str, "i") == 0) {
        const int *matrix_data = (const int *)data;
        if (is_diagonal && is_lower) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = j; i < tempmm; i++) {
                    int current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else if (is_diagonal && is_upper) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i <= j; i++) {
                    int current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i < tempmm; i++) {
                    int current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        }
    }
    else if (strcmp(datatype_str, "int64") == 0 || strcmp(datatype_str, "i64") == 0) {
        const int64_t *matrix_data = (const int64_t *)data;
        if (is_diagonal && is_lower) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = j; i < tempmm; i++) {
                    int64_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else if (is_diagonal && is_upper) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i <= j; i++) {
                    int64_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i < tempmm; i++) {
                    int64_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        }
    }
    // Unsigned integer types
    else if (strcmp(datatype_str, "uint8") == 0 || strcmp(datatype_str, "u8") == 0) {
        const uint8_t *matrix_data = (const uint8_t *)data;
        if (is_diagonal && is_lower) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = j; i < tempmm; i++) {
                    uint8_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else if (is_diagonal && is_upper) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i <= j; i++) {
                    uint8_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i < tempmm; i++) {
                    uint8_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        }
    }
    else if (strcmp(datatype_str, "uint16") == 0 || strcmp(datatype_str, "u16") == 0) {
        const uint16_t *matrix_data = (const uint16_t *)data;
        if (is_diagonal && is_lower) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = j; i < tempmm; i++) {
                    uint16_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else if (is_diagonal && is_upper) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i <= j; i++) {
                    uint16_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i < tempmm; i++) {
                    uint16_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        }
    }
    else if (strcmp(datatype_str, "uint32") == 0 || strcmp(datatype_str, "uint") == 0 || 
             strcmp(datatype_str, "u32") == 0 || strcmp(datatype_str, "u") == 0) {
        const unsigned int *matrix_data = (const unsigned int *)data;
        if (is_diagonal && is_lower) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = j; i < tempmm; i++) {
                    unsigned int current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else if (is_diagonal && is_upper) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i <= j; i++) {
                    unsigned int current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i < tempmm; i++) {
                    unsigned int current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        }
    }
    else if (strcmp(datatype_str, "uint64") == 0 || strcmp(datatype_str, "u64") == 0) {
        const uint64_t *matrix_data = (const uint64_t *)data;
        if (is_diagonal && is_lower) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = j; i < tempmm; i++) {
                    uint64_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else if (is_diagonal && is_upper) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i <= j; i++) {
                    uint64_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i < tempmm; i++) {
                    uint64_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        }
    }
    // Half precision
    else if (strcmp(datatype_str, "half") == 0 || strcmp(datatype_str, "fp16") == 0 || strcmp(datatype_str, "h") == 0) {
        const uint16_t *matrix_data = (const uint16_t *)data;
        if (is_diagonal && is_lower) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = j; i < tempmm; i++) {
                    uint16_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else if (is_diagonal && is_upper) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i <= j; i++) {
                    uint16_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i < tempmm; i++) {
                    uint16_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        }
    }
    // FP8, FP4, INT4, 1bit representations
    else if (strcmp(datatype_str, "fp8") == 0 || strcmp(datatype_str, "fp4") == 0 || 
             strcmp(datatype_str, "int4") == 0 || strcmp(datatype_str, "1bit") == 0) {
        const uint8_t *matrix_data = (const uint8_t *)data;
        if (is_diagonal && is_lower) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = j; i < tempmm; i++) {
                    uint8_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else if (is_diagonal && is_upper) {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i <= j; i++) {
                    uint8_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        } else {
            for (int j = 0; j < tempnn; j++) {
                for (int i = 0; i < tempmm; i++) {
                    uint8_t current_value = is_row_major ? matrix_data[i * lda + j] : matrix_data[j * lda + i];
                    norm_squared += (double)(current_value * current_value);
                }
            }
        }
    }
    else {
        fprintf(stderr, "Error: Unknown datatype '%s' in hicma_parsec_core_matrix_norm_get\n", datatype_str);
        return 0.0;
    }
    
    return norm_squared;
}
