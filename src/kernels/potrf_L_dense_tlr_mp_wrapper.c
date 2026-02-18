/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file potrf_L_dense_tlr_mp_wrapper.c
 * @brief Mixed-precision Cholesky factorization wrapper with TLR (Tile Low-Rank) support
 * 
 * This file implements wrapper functions for the mixed-precision Cholesky factorization
 * algorithm using Tile Low-Rank (TLR) representation. It provides timing instrumentation
 * and GPU/CPU kernel selection for the four main computational kernels:
 * - POTRF: Cholesky factorization of diagonal blocks
 * - TRSM: Triangular solve with multiple right-hand sides
 * - SYRK: Symmetric rank-k update
 * - GEMM: General matrix-matrix multiplication
 * 
 * The wrapper functions enable performance monitoring and automatic selection between
 * GPU and CPU implementations based on problem characteristics.
 */

#include "hicma_parsec.h"
#include "hicma_kernel_time.h"
#include "potrf_L_dense_tlr_mp.h"

/* =============================================================================
 * TIMING WRAPPER FUNCTIONS
 * =============================================================================
 * These functions wrap the actual computational kernels to measure execution time
 * for performance analysis and critical path identification.
 */

/**
 * @brief Wrapper for POTRF kernel execution with timing
 * 
 * Records the start time for POTRF (Cholesky factorization) operations and
 * delegates to the actual kernel implementation.
 * 
 * @param es Execution stream context
 * @param this_task POTRF task descriptor
 * @return Status code from the wrapped kernel
 */
static int wrap_potrf(parsec_execution_stream_t * es, 
                      __parsec_potrf_L_dense_tlr_mp_potrf_dpotrf_task_t * this_task)
{
    parsec_potrf_L_dense_tlr_mp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_mp_taskpool_t*)this_task->taskpool;
    
    /* Record start time for POTRF operation timing */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->potrf_time_temp = start_time;
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;
    
    /* Delegate to the actual POTRF kernel implementation */
    return parsec_tp->_g_params_tlr->wrap_potrf(es, (parsec_task_t *)this_task);
}

/**
 * @brief Completion wrapper for POTRF kernel with timing calculation
 * 
 * Records the end time for POTRF operations, calculates execution duration,
 * and updates cumulative timing statistics. Optionally prints critical path
 * timing information for performance analysis.
 * 
 * @param es Execution stream context
 * @param this_task POTRF task descriptor
 * @return Status code from the wrapped kernel
 */
static int wrap_potrf_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_mp_potrf_dpotrf_task_t * this_task)
{
    int val;
    double end_time;
    parsec_potrf_L_dense_tlr_mp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_mp_taskpool_t*)this_task->taskpool;
    
    /* Execute the actual POTRF completion routine */
    val = parsec_tp->_g_params_tlr->wrap_potrf_complete(es, (parsec_task_t *)this_task);
    
    /* Calculate and record execution time */
    end_time = MPI_Wtime();
    parsec_tp->_g_params_tlr->potrf_time += end_time - parsec_tp->_g_params_tlr->potrf_time_temp; 
    parsec_tp->_g_params_tlr->gather_time[es->th_id] += end_time - parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id];

#if PRINT_CRITICAL_PATH_TIME
    /* Print detailed timing information for critical path analysis */
    fprintf(stderr, "OUT_critical_path_time band_size_dense %d Nodes %d Matrix %d POTRF %d end_time %lf start_time %lf task_exe_time %lf sum_time %lf\n",
		    parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes, parsec_tp->_g_descA->lm,
		    this_task->locals.k.value, end_time, parsec_tp->_g_params_tlr->potrf_time_temp,
		    end_time - parsec_tp->_g_params_tlr->potrf_time_temp, parsec_tp->_g_params_tlr->potrf_time);
#endif

#if PRINT_KERNEL_TIME
    double start_time = hicma_kernel_time_take(this_task, end_time);
    double elapsed_time = end_time - start_time;
    const char *sum_time_scope;
    int sum_time_id;
    double sum_time = hicma_kernel_time_accumulate((parsec_task_t *)this_task, parsec_tp->_g_params_tlr, es->th_id,
                                                  start_time, end_time, &sum_time_scope, &sum_time_id);
    int gpu_time_id = -1;
    double gpu_exe_time = 0.0;
    double gpu_sum_time = 0.0;
    int has_gpu_time = hicma_kernel_time_gpu_event_take((parsec_task_t *)this_task, parsec_tp->_g_params_tlr,
                                                        &gpu_time_id, &gpu_exe_time, &gpu_sum_time);
    hicma_kernel_time_print_task("POTRF", parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes,
                                 parsec_tp->_g_descA->lm, this_task->locals.k.value, -1, -1,
                                 end_time, start_time, elapsed_time, sum_time_scope, sum_time_id, sum_time,
                                 has_gpu_time, gpu_time_id, gpu_exe_time, gpu_sum_time);
#endif

    return val;
}

/**
 * @brief Wrapper for TRSM kernel execution with conditional timing
 * 
 * Records timing for TRSM (triangular solve) operations. Only records detailed
 * timing for the first TRSM operation in each column (when m == k+1) to avoid
 * double-counting in performance measurements.
 * 
 * @param es Execution stream context
 * @param this_task TRSM task descriptor
 * @return Status code from the wrapped kernel
 */
static int wrap_trsm(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_mp_potrf_dtrsm_task_t * this_task)
{
    parsec_potrf_L_dense_tlr_mp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_mp_taskpool_t*)this_task->taskpool;
    
    /* Record thread-specific timing start */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;

    /* Only record detailed timing for the first TRSM in each column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Record start time for TRSM operation timing */
        parsec_tp->_g_params_tlr->trsm_time_temp = start_time;
        return parsec_tp->_g_params_tlr->wrap_trsm(es, (parsec_task_t *)this_task);
    } else {
        /* For subsequent TRSM operations, just delegate without detailed timing */
        return parsec_tp->_g_params_tlr->wrap_trsm(es, (parsec_task_t *)this_task);
    }
}

/**
 * @brief Completion wrapper for TRSM kernel with conditional timing calculation
 * 
 * Records the end time for TRSM operations and calculates execution duration.
 * Only updates detailed timing statistics for the first TRSM in each column
 * to maintain accurate performance measurements.
 * 
 * @param es Execution stream context
 * @param this_task TRSM task descriptor
 * @return Status code from the wrapped kernel
 */
static int wrap_trsm_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_mp_potrf_dtrsm_task_t * this_task)
{
    parsec_potrf_L_dense_tlr_mp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_mp_taskpool_t*)this_task->taskpool;
    int val;
    double end_time;

    /* Execute the actual TRSM completion routine */
    val = parsec_tp->_g_params_tlr->wrap_trsm_complete(es, (parsec_task_t *)this_task);
    end_time = MPI_Wtime();

    /* Only update detailed timing for the first TRSM in each column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Calculate and record TRSM execution time */
        parsec_tp->_g_params_tlr->trsm_time += end_time - parsec_tp->_g_params_tlr->trsm_time_temp;

#if PRINT_CRITICAL_PATH_TIME
        /* Print detailed timing information for critical path analysis */
        fprintf(stderr, "OUT_critical_path_time band_size_dense %d Nodes %d Matrix %d TRSM %d end_time %lf start_time %lf task_exe_time %lf sum_time %lf\n",
                parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes, parsec_tp->_g_descA->lm,
                this_task->locals.k.value, end_time, parsec_tp->_g_params_tlr->trsm_time_temp,
                end_time - parsec_tp->_g_params_tlr->trsm_time_temp, parsec_tp->_g_params_tlr->trsm_time);
#endif
    }

    /* Always update thread-specific timing */
    parsec_tp->_g_params_tlr->gather_time[es->th_id] += end_time - parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id];
#if PRINT_KERNEL_TIME
    double start_time = hicma_kernel_time_take(this_task, end_time);
    double elapsed_time = end_time - start_time;
    const char *sum_time_scope;
    int sum_time_id;
    double sum_time = hicma_kernel_time_accumulate((parsec_task_t *)this_task, parsec_tp->_g_params_tlr, es->th_id,
                                                  start_time, end_time, &sum_time_scope, &sum_time_id);
    int gpu_time_id = -1;
    double gpu_exe_time = 0.0;
    double gpu_sum_time = 0.0;
    int has_gpu_time = hicma_kernel_time_gpu_event_take((parsec_task_t *)this_task, parsec_tp->_g_params_tlr,
                                                        &gpu_time_id, &gpu_exe_time, &gpu_sum_time);
    hicma_kernel_time_print_task("TRSM", parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes,
                                 parsec_tp->_g_descA->lm, this_task->locals.m.value, this_task->locals.k.value, -1,
                                 end_time, start_time, elapsed_time, sum_time_scope, sum_time_id, sum_time,
                                 has_gpu_time, gpu_time_id, gpu_exe_time, gpu_sum_time);
#endif

    return val;
}

/**
 * @brief Wrapper for SYRK kernel execution with conditional timing
 * 
 * Records timing for SYRK (symmetric rank-k update) operations. Only records
 * detailed timing for the first SYRK operation in each column (when m == k+1)
 * to avoid double-counting in performance measurements.
 * 
 * @param es Execution stream context
 * @param this_task SYRK task descriptor
 * @return Status code from the wrapped kernel
 */
static int wrap_syrk(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_mp_potrf_dsyrk_task_t * this_task)
{
    parsec_potrf_L_dense_tlr_mp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_mp_taskpool_t*)this_task->taskpool;
    
    /* Record thread-specific timing start */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;
    
    /* Only record detailed timing for the first SYRK in each column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Record start time for SYRK operation timing */
        parsec_tp->_g_params_tlr->syrk_time_temp = start_time;
        return parsec_tp->_g_params_tlr->wrap_syrk(es, (parsec_task_t *)this_task);
    } else {
        /* For subsequent SYRK operations, just delegate without detailed timing */
        return parsec_tp->_g_params_tlr->wrap_syrk(es, (parsec_task_t *)this_task);
    }
}

/**
 * @brief Completion wrapper for SYRK kernel with conditional timing calculation
 * 
 * Records the end time for SYRK operations and calculates execution duration.
 * Only updates detailed timing statistics for the first SYRK in each column
 * to maintain accurate performance measurements.
 * 
 * @param es Execution stream context
 * @param this_task SYRK task descriptor
 * @return Status code from the wrapped kernel
 */
static int wrap_syrk_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_mp_potrf_dsyrk_task_t * this_task)
{
    parsec_potrf_L_dense_tlr_mp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_mp_taskpool_t*)this_task->taskpool;
    int val;
    double end_time;

    /* Execute the actual SYRK completion routine */
    val = parsec_tp->_g_params_tlr->wrap_syrk_complete(es, (parsec_task_t *)this_task);
    end_time = MPI_Wtime();

    /* Only update detailed timing for the first SYRK in each column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Calculate and record SYRK execution time */
        parsec_tp->_g_params_tlr->syrk_time += end_time - parsec_tp->_g_params_tlr->syrk_time_temp;

#if PRINT_CRITICAL_PATH_TIME
        /* Print detailed timing information for critical path analysis */
        fprintf(stderr, "OUT_critical_path_time band_size_dense %d Nodes %d Matrix %d SYRK %d end_time %lf start_time %lf task_exe_time %lf sum_time %lf\n",
                parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes, parsec_tp->_g_descA->lm,
                this_task->locals.k.value, end_time, parsec_tp->_g_params_tlr->syrk_time_temp,
                end_time - parsec_tp->_g_params_tlr->syrk_time_temp, parsec_tp->_g_params_tlr->syrk_time);
#endif
    }

    /* Always update thread-specific timing */
    parsec_tp->_g_params_tlr->gather_time[es->th_id] += end_time - parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id];
#if PRINT_KERNEL_TIME
    double start_time = hicma_kernel_time_take(this_task, end_time);
    double elapsed_time = end_time - start_time;
    const char *sum_time_scope;
    int sum_time_id;
    double sum_time = hicma_kernel_time_accumulate((parsec_task_t *)this_task, parsec_tp->_g_params_tlr, es->th_id,
                                                  start_time, end_time, &sum_time_scope, &sum_time_id);
    int gpu_time_id = -1;
    double gpu_exe_time = 0.0;
    double gpu_sum_time = 0.0;
    int has_gpu_time = hicma_kernel_time_gpu_event_take((parsec_task_t *)this_task, parsec_tp->_g_params_tlr,
                                                        &gpu_time_id, &gpu_exe_time, &gpu_sum_time);
    hicma_kernel_time_print_task("SYRK", parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes,
                                 parsec_tp->_g_descA->lm, this_task->locals.m.value, this_task->locals.k.value, -1,
                                 end_time, start_time, elapsed_time, sum_time_scope, sum_time_id, sum_time,
                                 has_gpu_time, gpu_time_id, gpu_exe_time, gpu_sum_time);
#endif

    return val;
}

/**
 * @brief Wrapper for GEMM kernel execution with timing
 * 
 * Records the start time for GEMM (general matrix-matrix multiplication)
 * operations and delegates to the actual kernel implementation.
 * 
 * @param es Execution stream context
 * @param this_task GEMM task descriptor
 * @return Status code from the wrapped kernel
 */
static int wrap_gemm(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_mp_potrf_dgemm_task_t * this_task)
{
    parsec_potrf_L_dense_tlr_mp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_mp_taskpool_t*)this_task->taskpool;
    
    /* Record start time for GEMM operation timing */
    hicma_kernel_time_record(this_task, MPI_Wtime());
    
    /* Delegate to the actual GEMM kernel implementation */
    return parsec_tp->_g_params_tlr->wrap_gemm(es, (parsec_task_t *)this_task);
}

/**
 * @brief Completion wrapper for GEMM kernel with timing calculation
 * 
 * Records the end time for GEMM operations, calculates execution duration,
 * and updates cumulative timing statistics. Optionally prints debug timing
 * information for performance analysis.
 * 
 * @param es Execution stream context
 * @param this_task GEMM task descriptor
 * @return Status code from the wrapped kernel
 */
static int wrap_gemm_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_mp_potrf_dgemm_task_t * this_task)
{
    parsec_potrf_L_dense_tlr_mp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_mp_taskpool_t*)this_task->taskpool;
    int val;
    double start_time = hicma_kernel_time_take(this_task, MPI_Wtime());
    
    /* Execute the actual GEMM completion routine */
    val = parsec_tp->_g_params_tlr->wrap_gemm_complete(es, (parsec_task_t *)this_task);
    
    /* Calculate and record execution time */
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    parsec_tp->_g_params_tlr->gather_time[es->th_id] += elapsed_time;

    /* Print GEMM timing information if enabled */
#if PRINT_KERNEL_TIME
    const char *sum_time_scope;
    int sum_time_id;
    double sum_time = hicma_kernel_time_accumulate((parsec_task_t *)this_task, parsec_tp->_g_params_tlr, es->th_id,
                                                  start_time, end_time, &sum_time_scope, &sum_time_id);
    int gpu_time_id = -1;
    double gpu_exe_time = 0.0;
    double gpu_sum_time = 0.0;
    int has_gpu_time = hicma_kernel_time_gpu_event_take((parsec_task_t *)this_task, parsec_tp->_g_params_tlr,
                                                        &gpu_time_id, &gpu_exe_time, &gpu_sum_time);
    hicma_kernel_time_print_gemm(parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes,
                                 parsec_tp->_g_descA->lm, this_task->locals.m.value,
                                 this_task->locals.n.value, this_task->locals.k.value,
                                 end_time, start_time, elapsed_time, sum_time_scope, sum_time_id, sum_time,
                                 has_gpu_time, gpu_time_id, gpu_exe_time, gpu_sum_time);
#endif
    return val;
}


/* =============================================================================
 * GPU KERNEL EVALUATION FUNCTIONS
 * =============================================================================
 * These functions determine whether GPU kernels should be used for specific
 * tasks based on problem characteristics and performance considerations.
 */

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/**
 * @brief GPU evaluation function for POTRF kernel
 * 
 * Determines whether the POTRF (Cholesky factorization) task should use
 * GPU implementation. Currently always returns DONE to use GPU.
 * 
 * @param task POTRF task descriptor
 * @return PARSEC_HOOK_RETURN_DONE to use GPU, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static parsec_hook_return_t evaluate_gpu_potrf(parsec_task_t* task) {
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief GPU evaluation function for TRSM kernel
 * 
 * Determines whether the TRSM (triangular solve) task should use GPU
 * implementation based on the distance from the diagonal block.
 * Uses GPU for operations far from the diagonal (m-k >= band_size_dense).
 * 
 * @param task TRSM task descriptor
 * @return PARSEC_HOOK_RETURN_DONE to use GPU, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static parsec_hook_return_t evaluate_gpu_trsm(parsec_task_t* task) {
    int m = ((__parsec_potrf_L_dense_tlr_mp_potrf_dtrsm_task_t *)task)->locals.m.value;
    int k = ((__parsec_potrf_L_dense_tlr_mp_potrf_dtrsm_task_t *)task)->locals.k.value;
    int band_size_dense = ((__parsec_potrf_L_dense_tlr_mp_potrf_dtrsm_task_t *)task)->locals.band_size_dense_local.value;
    
    /* Use GPU for operations far from the diagonal */
    if( m-k < band_size_dense )
        return PARSEC_HOOK_RETURN_DONE;
    else
        return PARSEC_HOOK_RETURN_NEXT;
}

/**
 * @brief GPU evaluation function for SYRK kernel
 * 
 * Determines whether the SYRK (symmetric rank-k update) task should use
 * GPU implementation. Currently always returns DONE to use GPU.
 * 
 * @param task SYRK task descriptor
 * @return PARSEC_HOOK_RETURN_DONE to use GPU, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static  parsec_hook_return_t evaluate_gpu_syrk(parsec_task_t* task) {
        return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief GPU evaluation function for GEMM kernel
 * 
 * Determines whether the GEMM (general matrix-matrix multiplication) task
 * should use GPU implementation based on the distance from the diagonal block.
 * Uses GPU for operations far from the diagonal (m-n >= band_size_dense).
 * 
 * @param task GEMM task descriptor
 * @return PARSEC_HOOK_RETURN_DONE to use GPU, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static parsec_hook_return_t evaluate_gpu_gemm(parsec_task_t* task) {
    int m = ((__parsec_potrf_L_dense_tlr_mp_potrf_dgemm_task_t *)task)->locals.m.value;
    int n = ((__parsec_potrf_L_dense_tlr_mp_potrf_dgemm_task_t *)task)->locals.n.value;
    int k = ((__parsec_potrf_L_dense_tlr_mp_potrf_dgemm_task_t *)task)->locals.k.value;
    int band_size_dense = ((__parsec_potrf_L_dense_tlr_mp_potrf_dgemm_task_t *)task)->locals.band_size_dense_local.value;
    
    /* Use GPU for operations far from the diagonal */
    if( m-n < band_size_dense )
        return PARSEC_HOOK_RETURN_DONE;
    else
        return PARSEC_HOOK_RETURN_NEXT;
}

#endif /* PARSEC_HAVE_DEV_CUDA_SUPPORT */


/* =============================================================================
 * MAIN TASKPOOL CREATION AND MANAGEMENT FUNCTIONS
 * =============================================================================
 * These functions handle the creation, configuration, and destruction of the
 * mixed-precision Cholesky factorization taskpool with TLR support.
 */

/**
 * @brief Create a new mixed-precision Cholesky factorization taskpool with TLR support
 * 
 * This function creates and configures a taskpool for mixed-precision Cholesky
 * factorization using Tile Low-Rank (TLR) representation. It sets up timing
 * instrumentation, GPU/CPU kernel selection, memory pools, and data type arenas.
 * 
 * The function supports both dense and TLR representations, with automatic
 * selection based on problem characteristics and available hardware.
 * 
 * @param parsec PaRSEC context
 * @param data Data structures containing matrix descriptors and parameters
 * @param params Algorithm parameters including precision, ranks, and hardware settings
 * 
 * @return Pointer to the created taskpool, or NULL on failure
 * 
 * @param [out] info 0 on all nodes if successful.
 *                   > 0 if the leading minor of order i of A is not positive
 *                   definite, so the factorization could not be completed, and the
 *                   solution has not been computed. Info will be equal to i on the
 *                   node that owns the diagonal element (i,i), and 0 on all other nodes
 */
parsec_taskpool_t*
potrf_L_dense_tlr_mp_New( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    /* Function pointers for hook and evaluation functions */
    void** hook;
    void** eval_gpu_potrf;
    void** eval_gpu_trsm;
    void** eval_gpu_syrk;
    void** eval_gpu_gemm;

    /* Extract key parameters for readability */
    int rank = params->rank;
    int uplo = params->uplo;
    int hmb = params->HNB;              /* Hierarchical block size */
    int compmaxrank = params->compmaxrank;    /* Maximum rank for computation */
    int storagemaxrank = params->genmaxrank;  /* Maximum rank for storage */

    /* ========================================================================
     * MATRIX DESCRIPTOR SETUP
     * ======================================================================== */
    
    /* Select appropriate matrix descriptor based on prediction mode and problem size */
    #if PREDICTION
        /* Use submatrix for prediction mode */
        parsec_tiled_matrix_t *A = parsec_tiled_matrix_submatrix(  (parsec_tiled_matrix_t *)&data->dcA, 0, 0, params->NP, params->NP);
        if( params->band_size_dense >= A->nt && params->auto_band == 0 && !params->adaptive_memory ) {
            A = parsec_tiled_matrix_submatrix(  (parsec_tiled_matrix_t *)&data->dcAd, 0, 0, params->NP, params->NP); 
        }
    #else
        /* Use full matrix descriptor */
        parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
        if( params->band_size_dense >= A->nt && params->auto_band == 0 && !params->adaptive_memory ) {
            A = (parsec_tiled_matrix_t *)&data->dcAd;    
        }
    #endif

    /* Set up auxiliary matrix descriptors for TLR representation */
    parsec_tiled_matrix_t *Ar = (parsec_tiled_matrix_t *)&data->dcAr;      /* Rank matrix */
    parsec_tiled_matrix_t *Rank = (parsec_tiled_matrix_t *)&data->dcRank;  /* Rank information */
    parsec_tiled_matrix_t *Fake = (parsec_tiled_matrix_t *)&data->dcFake;  /* Fake matrix for TLR */

    /* ========================================================================
     * PARAMETER VALIDATION
     * ======================================================================== */
    
    /* Validate matrix storage format - only lower triangular supported */
    if ((uplo != PlasmaLower)) {
        dplasma_error("potrf_L_dense_tlr_dp_New", "potrf_L_dense_tlr_dp only support PlasmaLower for now");
        return NULL;
    }

    /* Validate rank parameters - storage rank should not exceed computation rank */
    if(storagemaxrank > compmaxrank) {
        dplasma_error("potrf_L_dense_tlr_dp_New", "maxrank for storage larger than maxrank for buffers \
                is not meaningful");
        return NULL;
    }

    /* Warn if storage rank is too large relative to block size */
    if(storagemaxrank > (A->mb/2) && 0 == rank) {
        fprintf(stderr, RED "Warning: maxrank= %d is larger than half of block size\n" RESET, storagemaxrank);
    }

    /* ========================================================================
     * GPU DEVICE DETECTION AND INITIALIZATION
     * ======================================================================== */
    
    int nb = 0, *dev_index;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Detect and enumerate available CUDA/HIP devices */
    hicma_parsec_find_cuda_devices( &dev_index, &nb);

#if !GPU_BUFFER_ONCE
    /* Initialize GPU workspace buffers if not already allocated */
    gpu_temporay_buffer_init( data, params, A->mb, A->nb, storagemaxrank, params->kind_of_cholesky );
#endif
#endif

    /* ========================================================================
     * TASKPOOL CREATION
     * ======================================================================== */
    
    /* Initialize info parameter and create the main taskpool */
    params->info = 0;
    parsec_potrf_L_dense_tlr_mp_taskpool_t *hicma_dpotrf =
        parsec_potrf_L_dense_tlr_mp_new( A, Ar, Rank, Fake, params ); 

    /* ========================================================================
     * TASK CLASS IDENTIFICATION
     * ======================================================================== */
    
    /* Identify task class IDs for each computational kernel */
    int potrf_id, trsm_id, syrk_id, gemm_id;
    for( int i = 0; i < hicma_dpotrf->super.nb_task_classes; i++ ) {
        if( !strcmp(hicma_dpotrf->super.task_classes_array[i]->name, "potrf_dpotrf") )
            potrf_id = hicma_dpotrf->super.task_classes_array[i]->task_class_id;
        else if( !strcmp(hicma_dpotrf->super.task_classes_array[i]->name, "potrf_dtrsm") )
            trsm_id = hicma_dpotrf->super.task_classes_array[i]->task_class_id;
        else if( !strcmp(hicma_dpotrf->super.task_classes_array[i]->name, "potrf_dsyrk") )
            syrk_id = hicma_dpotrf->super.task_classes_array[i]->task_class_id;
        else if( !strcmp(hicma_dpotrf->super.task_classes_array[i]->name, "potrf_dgemm") )
            gemm_id = hicma_dpotrf->super.task_classes_array[i]->task_class_id;
    }
    
    /* Print task class IDs for debugging if enabled */
    if( 0 == rank && DEBUG_INFO ) printf("potrf_id= %d trsm_id= %d syrk_id= %d gemm_id= %d\n",
            potrf_id, trsm_id, syrk_id, gemm_id);

    /* ========================================================================
     * GPU WORKSPACE CONFIGURATION
     * ======================================================================== */
    
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Configure GPU workspace and device information */
    hicma_dpotrf->_g_ws_gpu = (void *)data->ws_gpu;
    hicma_dpotrf->_g_nb_cuda_devices = nb;
    hicma_dpotrf->_g_cuda_device_index = dev_index;
#endif

    /* ========================================================================
     * INCARNATION ID DETERMINATION
     * ======================================================================== */
    
    /* Determine incarnation IDs based on available hardware support */
    int gpu_id = 0, recursive_id = 0, cpu_id = 0;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    recursive_id += 1;  /* GPU support shifts recursive and CPU IDs */
    cpu_id += 1;
#endif
#if defined(PARSEC_HAVE_DEV_RECURSIVE_SUPPORT)
    cpu_id += 1;        /* Recursive support shifts CPU ID */
#endif
    
    /* Print incarnation IDs for debugging if enabled */
    if( 0 == rank && DEBUG_INFO ) printf("gpu_id= %d recursive_id= %d cpu_id= %d\n",
            gpu_id, recursive_id, cpu_id);

    /* ========================================================================
     * KERNEL HOOK AND EVALUATION FUNCTION SETUP
     * ======================================================================== */
    
    /* Configure hooks and evaluation functions based on available hardware */
    if( nb > 0 ) {
        /* GPU devices available - configure GPU kernels */
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        /* Store original GPU kernel hooks for delegation */
        hicma_dpotrf->_g_params_tlr->wrap_potrf = hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[gpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[gpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[gpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[gpu_id].hook;
        
        /* Replace GPU kernel hooks with timing wrapper functions */
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[gpu_id].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[gpu_id].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[gpu_id].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[gpu_id].hook;
        *hook = &wrap_gemm;

        /* Configure GPU evaluation functions for kernel selection */
        eval_gpu_potrf = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[gpu_id].evaluate;
        eval_gpu_trsm  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[gpu_id].evaluate;
        eval_gpu_syrk  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[gpu_id].evaluate;
        eval_gpu_gemm  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[gpu_id].evaluate;
        *eval_gpu_potrf = &evaluate_gpu_potrf;
        *eval_gpu_trsm  = &evaluate_gpu_trsm;
        *eval_gpu_syrk  = &evaluate_gpu_syrk;
        *eval_gpu_gemm  = &evaluate_gpu_gemm;
#endif

    } else if( hmb < A->mb ) { 
        /* Recursive kernels - hierarchical block size smaller than matrix block size */
        hicma_dpotrf->_g_params_tlr->wrap_potrf = hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[recursive_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[recursive_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[recursive_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[recursive_id].hook;
        
        /* Replace recursive kernel hooks with timing wrapper functions */
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[recursive_id].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[recursive_id].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[recursive_id].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[recursive_id].hook;
        *hook = &wrap_gemm;

    } else {
        /* Standard CPU kernels - no GPU or recursive support needed */
        hicma_dpotrf->_g_params_tlr->wrap_potrf = hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[cpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[cpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[cpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[cpu_id].hook;
        
        /* Replace CPU kernel hooks with timing wrapper functions */
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[cpu_id].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[cpu_id].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[cpu_id].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[cpu_id].hook;
        *hook = &wrap_gemm;
    }

    /* ========================================================================
     * COMPLETION HOOK SETUP
     * ======================================================================== */
    
    /* Configure completion hooks for all kernels to enable timing measurement */
    hicma_dpotrf->_g_params_tlr->wrap_potrf_complete = hicma_dpotrf->super.task_classes_array[potrf_id]->complete_execution;
    hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->complete_execution;
    *hook = &wrap_potrf_complete;

    hicma_dpotrf->_g_params_tlr->wrap_trsm_complete = hicma_dpotrf->super.task_classes_array[trsm_id]->complete_execution;
    hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->complete_execution;
    *hook = &wrap_trsm_complete;

    hicma_dpotrf->_g_params_tlr->wrap_syrk_complete = hicma_dpotrf->super.task_classes_array[syrk_id]->complete_execution;
    hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->complete_execution;
    *hook = &wrap_syrk_complete;

    hicma_dpotrf->_g_params_tlr->wrap_gemm_complete = hicma_dpotrf->super.task_classes_array[gemm_id]->complete_execution;
    hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->complete_execution;
    *hook = &wrap_gemm_complete;

    /* ========================================================================
     * MEMORY POOL INITIALIZATION
     * ======================================================================== */
    
    /* Initialize main worker memory pool for temporary buffers */
    hicma_dpotrf->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    
    /* Calculate workspace size for temporary buffers (from hicma/compute/pzpotrf.c line 96) */
    size_t ws_worker = 0;
    ws_worker = 
        2 * A->mb * 2 * compmaxrank            /* CU and CV temporary buffers (2*maxrk for side-by-side U's) */
        + 2 * A->mb                            /* qrtauA and qrtauB arrays */
        + compmaxrank * compmaxrank            /* qrb_aubut and AcolBcolT matrices */
        + 2 * A->mb * 2 * compmaxrank          /* newU and newV matrices */
        + (2*compmaxrank) * (2*compmaxrank)    /* svd_rA and _rA matrices */
        + (2*compmaxrank)                      /* sigma vector */
        + (2*compmaxrank);                     /* superb vector */
    ;
    ws_worker *= sizeof(double); 
    parsec_private_memory_init( hicma_dpotrf->_g_p_work, ws_worker ); 

    /* Initialize rank-rank matrix memory pool */
    hicma_dpotrf->_g_p_work_rr = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_rr, compmaxrank * compmaxrank * sizeof(double) ); 

    /* Initialize matrix-block-rank memory pool */
    hicma_dpotrf->_g_p_work_mbr = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_mbr, A->mb * compmaxrank * sizeof(double) ); 

    /* Initialize full double precision matrix memory pool */
    hicma_dpotrf->_g_p_work_full_dp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_full_dp, A->mb * A->mb * sizeof(double) );

    /* Initialize full single precision matrix memory pool */
    hicma_dpotrf->_g_p_work_full_sp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_full_sp, A->mb * A->mb * sizeof(float) );

    /* Initialize full half precision matrix memory pool */
    hicma_dpotrf->_g_p_work_full_hp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_full_hp, A->mb * A->mb * sizeof(float) / 2 );

    /* Initialize UV double precision memory pool for TLR matrices */
    hicma_dpotrf->_g_p_work_uv_dp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_uv_dp, A->mb * storagemaxrank * 2 * sizeof(double) );

    /* Initialize UV single precision memory pool for TLR matrices */
    hicma_dpotrf->_g_p_work_uv_sp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_uv_sp, A->mb * storagemaxrank * 2 * sizeof(float) );

    /* ========================================================================
     * PRIORITY AND ARENA CONFIGURATION
     * ======================================================================== */
    
    /* Set priority change limit for task scheduling */
    hicma_dpotrf->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    if(0 == hicma_dpotrf->_g_PRI_CHANGE) {
        hicma_dpotrf->_g_PRI_CHANGE = A->nt;  /* Default to number of tiles */
    }

    /* Configure data type arenas for different matrix representations */
    /* Full double precision matrix arena */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_tlr_mp_FULL_DP_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Full single precision matrix arena */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_tlr_mp_FULL_SP_ADT_IDX],
            parsec_datatype_float_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* UV double precision matrix arena for TLR representation */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_tlr_mp_UV_DP_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, storagemaxrank*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* UV single precision matrix arena for TLR representation */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_tlr_mp_UV_SP_ADT_IDX],
            parsec_datatype_float_t, PARSEC_MATRIX_FULL,
            1, A->mb, storagemaxrank*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Byte arena for general data */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_tlr_mp_BYTE_ADT_IDX],
            MPI_BYTE, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Integer array arena for rank information */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_tlr_mp_AR_ADT_IDX],
            parsec_datatype_int_t, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    #if PREDICTION
        /* Clean up prediction mode matrix descriptor if needed */
        //free(A);
    #endif

    /* Return the fully configured taskpool */
    return (parsec_taskpool_t*)hicma_dpotrf;
}

/**
 * @brief Destructor for mixed-precision Cholesky factorization taskpool
 * 
 * Properly cleans up all allocated resources including GPU workspace,
 * memory pools, data type arenas, and the taskpool itself.
 * 
 * @param _tp Taskpool to destroy
 */
void potrf_L_dense_tlr_mp_Destruct(parsec_taskpool_t* _tp)
{
    parsec_potrf_L_dense_tlr_mp_taskpool_t *tp = (parsec_potrf_L_dense_tlr_mp_taskpool_t*)_tp;

    /* ========================================================================
     * GPU RESOURCE CLEANUP
     * ======================================================================== */
    
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    if( tp->_g_nb_cuda_devices > 0 ) {
#if !GPU_BUFFER_ONCE 
        /* Free GPU workspace memory if not managed elsewhere */
        workspace_memory_free( tp->_g_ws_gpu );
#endif

        /* Free CUDA device index array */
        if( NULL != tp->_g_cuda_device_index )
            free(tp->_g_cuda_device_index);
    }
#endif

    /* ========================================================================
     * ARENA CLEANUP
     * ======================================================================== */
    
    /* Clean up all data type arenas */
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_tlr_mp_FULL_DP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_tlr_mp_FULL_SP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_tlr_mp_UV_DP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_tlr_mp_UV_SP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_tlr_mp_BYTE_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_tlr_mp_AR_ADT_IDX] );
    
    /* ========================================================================
     * MEMORY POOL CLEANUP
     * ======================================================================== */
    
    /* Clean up all memory pools */
    parsec_private_memory_fini( tp->_g_p_work );
    parsec_private_memory_fini( tp->_g_p_work_mbr );
    parsec_private_memory_fini( tp->_g_p_work_rr );
    parsec_private_memory_fini( tp->_g_p_work_full_dp );
    parsec_private_memory_fini( tp->_g_p_work_full_sp );
    parsec_private_memory_fini( tp->_g_p_work_full_hp );
    parsec_private_memory_fini( tp->_g_p_work_uv_dp );
    parsec_private_memory_fini( tp->_g_p_work_uv_sp );

    /* Free the taskpool itself */
    parsec_taskpool_free(_tp);
}

/**
 * @brief Main entry point for mixed-precision Cholesky factorization with TLR support
 * 
 * This function orchestrates the complete mixed-precision Cholesky factorization
 * process using Tile Low-Rank (TLR) representation. It creates the taskpool,
 * executes the computation, and properly cleans up resources.
 * 
 * The function supports both dense and TLR representations with automatic
 * selection based on problem characteristics and available hardware.
 * 
 * @param parsec PaRSEC context for task execution
 * @param data Data structures containing matrix descriptors and parameters
 * @param params Algorithm parameters including precision, ranks, and hardware settings
 * 
 * @return 0 on success
 * 
 * @param [out] info 0 on all nodes if successful.
 *                   > 0 if the leading minor of order i of A is not positive
 *                   definite, so the factorization could not be completed, and the
 *                   solution has not been computed. Info will be equal to i on the
 *                   node that owns the diagonal element (i,i), and 0 on all other nodes
 */
int potrf_L_dense_tlr_mp( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    parsec_taskpool_t *hicma_potrf = NULL;

    /* Print start message if verbose mode is enabled */
    if( 0 == params->rank && params->verbose )
        printf(MAG "DENSE_TLR_MP start\n" RESET);

    /* Create and configure the taskpool */
    hicma_potrf = potrf_L_dense_tlr_mp_New( parsec, data, params );

    /* Execute the factorization if taskpool creation was successful */
    if( NULL != hicma_potrf ) {
        /* Add taskpool to context and execute */
        parsec_context_add_taskpool( parsec, hicma_potrf);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        
        /* Clean up resources */
        potrf_L_dense_tlr_mp_Destruct( hicma_potrf );
        
        /* Synchronize task IDs for recursive DAGs if needed */
        if( params->HNB < params->NB )
            parsec_taskpool_sync_ids(); /* Recursive DAGs are not synchronous on IDs */
    }

    return 0;
}
