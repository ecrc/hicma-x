/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file potrf_L_dense_mp_gpu_wrapper.c
 * @brief Wrapper functions for Cholesky factorization (POTRF) with mixed-precision GPU support
 * 
 * This file provides wrapper functions for the Cholesky factorization algorithm with:
 * - Mixed-precision computation support
 * - GPU acceleration via CUDA/HIP
 * - Performance timing and profiling
 * - Task-based execution using PaRSEC runtime
 * 
 * The main operations wrapped are:
 * - POTRF: Cholesky factorization of diagonal blocks
 * - TRSM: Triangular solve with multiple right-hand sides
 * - SYRK: Symmetric rank-k update
 * - GEMM: General matrix-matrix multiplication
 */

#include "hicma_parsec.h"
#include "hicma_kernel_time.h"
#include "potrf_L_dense_mp_gpu.h"

/* ============================================================================
 * Performance Timing Wrapper Functions
 * ============================================================================
 * These wrapper functions instrument the core BLAS operations to measure
 * execution time for performance analysis and optimization.
 * ============================================================================ */
/**
 * @brief Wrapper function for POTRF (Cholesky factorization) operation
 * 
 * This function instruments the POTRF operation to measure execution time.
 * It records the start time before calling the actual POTRF implementation.
 * 
 * @param es Execution stream handle
 * @param this_task POTRF task to be executed
 * @return Return value from the actual POTRF operation
 */
static int wrap_potrf(parsec_execution_stream_t * es, 
                      __parsec_potrf_L_dense_mp_gpu_potrf_dpotrf_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_taskpool_t*)this_task->taskpool;
    
    /* Record start time for POTRF operation timing */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->potrf_time_temp = start_time;
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;
    
    /* Execute the actual POTRF operation */
    return parsec_tp->_g_params_tlr->wrap_potrf(es, (parsec_task_t *)this_task);
}

/**
 * @brief Completion wrapper for POTRF operation with timing
 * 
 * This function is called when the POTRF operation completes. It calculates
 * the total execution time and updates performance counters. Optionally prints
 * timing information for critical path analysis.
 * 
 * @param es Execution stream handle
 * @param this_task Completed POTRF task
 * @return Return value from the actual POTRF completion function
 */
static int wrap_potrf_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_potrf_dpotrf_task_t * this_task)
{
    int val;
    double end_time;
    parsec_potrf_L_dense_mp_gpu_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_taskpool_t*)this_task->taskpool;
    
    /* Execute the actual POTRF completion function */
    val = parsec_tp->_g_params_tlr->wrap_potrf_complete(es, (parsec_task_t *)this_task);
    
    /* Calculate and accumulate execution time */
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
 * @brief Wrapper function for TRSM (triangular solve) operation
 * 
 * This function instruments the TRSM operation to measure execution time.
 * It only records timing for the first TRSM operation in each column (m == k+1)
 * to avoid double-counting in performance measurements.
 * 
 * @param es Execution stream handle
 * @param this_task TRSM task to be executed
 * @return Return value from the actual TRSM operation
 */
static int wrap_trsm(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_potrf_dtrsm_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_taskpool_t*)this_task->taskpool;
    
    /* Record start time for thread-level timing */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;

    /* Only time the first TRSM operation in each column to avoid double-counting */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Record start time for TRSM operation timing */
        parsec_tp->_g_params_tlr->trsm_time_temp = start_time;
        return parsec_tp->_g_params_tlr->wrap_trsm(es, (parsec_task_t *)this_task);
    } else {
        /* Execute TRSM without timing (already counted) */
        return parsec_tp->_g_params_tlr->wrap_trsm(es, (parsec_task_t *)this_task);
    }
}

/**
 * @brief Completion wrapper for TRSM operation with timing
 * 
 * This function is called when the TRSM operation completes. It calculates
 * the total execution time and updates performance counters. Only times the
 * first TRSM operation in each column to avoid double-counting.
 * 
 * @param es Execution stream handle
 * @param this_task Completed TRSM task
 * @return Return value from the actual TRSM completion function
 */
static int wrap_trsm_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_potrf_dtrsm_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_taskpool_t*)this_task->taskpool;
    int val;
    double end_time;

    /* Only accumulate timing for the first TRSM operation in each column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Execute the actual TRSM completion function */
        val = parsec_tp->_g_params_tlr->wrap_trsm_complete(es, (parsec_task_t *)this_task);
        
        /* Calculate and accumulate TRSM execution time */
        end_time = MPI_Wtime();
        parsec_tp->_g_params_tlr->trsm_time += end_time - parsec_tp->_g_params_tlr->trsm_time_temp;

#if PRINT_CRITICAL_PATH_TIME
        /* Print detailed timing information for critical path analysis */
        fprintf(stderr, "OUT_critical_path_time band_size_dense %d Nodes %d Matrix %d TRSM %d end_time %lf start_time %lf task_exe_time %lf sum_time %lf\n",
			parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes, parsec_tp->_g_descA->lm,
			this_task->locals.k.value, end_time, parsec_tp->_g_params_tlr->trsm_time_temp,
			end_time - parsec_tp->_g_params_tlr->trsm_time_temp, parsec_tp->_g_params_tlr->trsm_time);
#endif
    } else {
        /* Execute TRSM completion without timing accumulation */
        val = parsec_tp->_g_params_tlr->wrap_trsm_complete(es, (parsec_task_t *)this_task);
        end_time = MPI_Wtime();
    }

    /* Always update thread-level timing */
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
 * @brief Wrapper function for SYRK (symmetric rank-k update) operation
 * 
 * This function instruments the SYRK operation to measure execution time.
 * It only records timing for the first SYRK operation in each column (m == k+1)
 * to avoid double-counting in performance measurements.
 * 
 * @param es Execution stream handle
 * @param this_task SYRK task to be executed
 * @return Return value from the actual SYRK operation
 */
static int wrap_syrk(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_potrf_dsyrk_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_taskpool_t*)this_task->taskpool;
    
    /* Record start time for thread-level timing */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;
    
    /* Only time the first SYRK operation in each column to avoid double-counting */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Record start time for SYRK operation timing */
        parsec_tp->_g_params_tlr->syrk_time_temp = start_time;
        return parsec_tp->_g_params_tlr->wrap_syrk(es, (parsec_task_t *)this_task);
    } else {
        /* Execute SYRK without timing (already counted) */
        return parsec_tp->_g_params_tlr->wrap_syrk(es, (parsec_task_t *)this_task);
    }
}

/**
 * @brief Completion wrapper for SYRK operation with timing
 * 
 * This function is called when the SYRK operation completes. It calculates
 * the total execution time and updates performance counters. Only times the
 * first SYRK operation in each column to avoid double-counting.
 * 
 * @param es Execution stream handle
 * @param this_task Completed SYRK task
 * @return Return value from the actual SYRK completion function
 */
static int wrap_syrk_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_potrf_dsyrk_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_taskpool_t*)this_task->taskpool;
    int val;
    double end_time;

    /* Only accumulate timing for the first SYRK operation in each column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Execute the actual SYRK completion function */
        val = parsec_tp->_g_params_tlr->wrap_syrk_complete(es, (parsec_task_t *)this_task);
        
        /* Calculate and accumulate SYRK execution time */
        end_time = MPI_Wtime();
        parsec_tp->_g_params_tlr->syrk_time += end_time - parsec_tp->_g_params_tlr->syrk_time_temp;

#if PRINT_CRITICAL_PATH_TIME
        /* Print detailed timing information for critical path analysis */
        fprintf(stderr, "OUT_critical_path_time band_size_dense %d Nodes %d Matrix %d SYRK %d end_time %lf start_time %lf task_exe_time %lf sum_time %lf\n",
			parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes, parsec_tp->_g_descA->lm,
			this_task->locals.k.value, end_time, parsec_tp->_g_params_tlr->syrk_time_temp,
			end_time - parsec_tp->_g_params_tlr->syrk_time_temp, parsec_tp->_g_params_tlr->syrk_time);
#endif
    } else {
        /* Execute SYRK completion without timing accumulation */
        val = parsec_tp->_g_params_tlr->wrap_syrk_complete(es, (parsec_task_t *)this_task);
        end_time = MPI_Wtime();
    }

    /* Always update thread-level timing */
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
 * @brief Wrapper function for GEMM (general matrix-matrix multiplication) operation
 * 
 * This function instruments the GEMM operation to measure execution time.
 * All GEMM operations are timed since they represent the bulk of computation
 * in the Cholesky factorization algorithm.
 * 
 * @param es Execution stream handle
 * @param this_task GEMM task to be executed
 * @return Return value from the actual GEMM operation
 */
static int wrap_gemm(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_potrf_dgemm_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_taskpool_t*)this_task->taskpool;
    
    /* Record start time for GEMM operation timing */
    hicma_kernel_time_record(this_task, MPI_Wtime());
    
    /* Execute the actual GEMM operation */
    return parsec_tp->_g_params_tlr->wrap_gemm(es, (parsec_task_t *)this_task);
}

/**
 * @brief Completion wrapper for GEMM operation with timing
 * 
 * This function is called when the GEMM operation completes. It calculates
 * the total execution time and updates performance counters. Optionally prints
 * detailed timing information for debugging purposes.
 * 
 * @param es Execution stream handle
 * @param this_task Completed GEMM task
 * @return Return value from the actual GEMM completion function
 */
static int wrap_gemm_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_potrf_dgemm_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_taskpool_t*)this_task->taskpool;
    int val;
    double start_time = hicma_kernel_time_take(this_task, MPI_Wtime());
    
    /* Execute the actual GEMM completion function */
    val = parsec_tp->_g_params_tlr->wrap_gemm_complete(es, (parsec_task_t *)this_task);
    
    /* Calculate and accumulate GEMM execution time */
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    parsec_tp->_g_params_tlr->gather_time[es->th_id] += elapsed_time;

    /* Print detailed GEMM timing information if enabled */
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


/* ============================================================================
 * GPU Kernel Selection Functions
 * ============================================================================
 * These functions determine whether to execute operations on GPU or CPU
 * based on problem size and available hardware resources.
 * ============================================================================ */

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/**
 * @brief GPU evaluation function for POTRF operation
 * 
 * Currently always executes POTRF on GPU when available.
 * This could be extended to include size-based heuristics.
 * 
 * @param task POTRF task to evaluate
 * @return PARSEC_HOOK_RETURN_DONE to execute on GPU
 */
static parsec_hook_return_t evaluate_gpu_potrf(parsec_task_t* task) {
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief GPU evaluation function for TRSM operation
 * 
 * Determines whether to execute TRSM on GPU based on the distance
 * from the diagonal (m-k) compared to the dense band size.
 * Operations close to the diagonal are executed on GPU.
 * 
 * @param task TRSM task to evaluate
 * @return PARSEC_HOOK_RETURN_DONE for GPU execution, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static parsec_hook_return_t evaluate_gpu_trsm(parsec_task_t* task) {
    int m = ((__parsec_potrf_L_dense_mp_gpu_potrf_dtrsm_task_t *)task)->locals.m.value;
    int k = ((__parsec_potrf_L_dense_mp_gpu_potrf_dtrsm_task_t *)task)->locals.k.value;
    int band_size_dense = ((__parsec_potrf_L_dense_mp_gpu_potrf_dtrsm_task_t *)task)->locals.band_size_dense_local.value;
    
    /* Execute on GPU if within the dense band */
    if( m-k < band_size_dense )
        return PARSEC_HOOK_RETURN_DONE;
    else
        return PARSEC_HOOK_RETURN_NEXT;
}

/**
 * @brief GPU evaluation function for SYRK operation
 * 
 * Currently always executes SYRK on GPU when available.
 * This could be extended to include size-based heuristics.
 * 
 * @param task SYRK task to evaluate
 * @return PARSEC_HOOK_RETURN_DONE to execute on GPU
 */
static parsec_hook_return_t evaluate_gpu_syrk(parsec_task_t* task) {
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief GPU evaluation function for GEMM operation
 * 
 * Determines whether to execute GEMM on GPU based on the distance
 * from the diagonal (m-n) compared to the dense band size.
 * Operations close to the diagonal are executed on GPU.
 * 
 * @param task GEMM task to evaluate
 * @return PARSEC_HOOK_RETURN_DONE for GPU execution, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static parsec_hook_return_t evaluate_gpu_gemm(parsec_task_t* task) {
    int m = ((__parsec_potrf_L_dense_mp_gpu_potrf_dgemm_task_t *)task)->locals.m.value;
    int n = ((__parsec_potrf_L_dense_mp_gpu_potrf_dgemm_task_t *)task)->locals.n.value;
    int k = ((__parsec_potrf_L_dense_mp_gpu_potrf_dgemm_task_t *)task)->locals.k.value;
    int band_size_dense = ((__parsec_potrf_L_dense_mp_gpu_potrf_dgemm_task_t *)task)->locals.band_size_dense_local.value;
    
    /* Execute on GPU if within the dense band */
    if( m-n < band_size_dense )
        return PARSEC_HOOK_RETURN_DONE;
    else
        return PARSEC_HOOK_RETURN_NEXT;
}

#endif /* PARSEC_HAVE_DEV_CUDA_SUPPORT */


/* ============================================================================
 * Main Task Pool Creation and Management Functions
 * ============================================================================ */

/**
 * @brief Create a new task pool for mixed-precision Cholesky factorization with GPU support
 * 
 * This function creates and configures a PaRSEC task pool for Cholesky factorization
 * with mixed-precision support and GPU acceleration. It sets up:
 * - Task class identification and hook installation
 * - GPU device detection and workspace allocation
 * - Memory pools for different precision types
 * - Arena configuration for data types
 * - Performance timing instrumentation
 * 
 * @param parsec PaRSEC context handle
 * @param data Matrix data structures and descriptors
 * @param params Algorithm parameters including precision settings and GPU configuration
 * @return Pointer to the created task pool, or NULL on error
 * 
 * @note This is a non-blocking, 2-flow version implementation
 * @note info parameter: 0 on all nodes if successful, > 0 if the leading minor 
 *       of order i of A is not positive definite (info = i on the node owning 
 *       diagonal element (i,i), 0 on all other nodes)
 */
parsec_taskpool_t*
potrf_L_dense_mp_gpu_New( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    /* Hook pointers for task instrumentation */
    void** hook;
    void** eval_gpu_potrf;
    void** eval_gpu_trsm;
    void** eval_gpu_syrk;
    void** eval_gpu_gemm;

    /* Extract parameters for easier access */
    int rank = params->rank;
    int uplo = params->uplo;
    int hmb = params->HNB;
    int compmaxrank = params->compmaxrank;
    int storagemaxrank = params->genmaxrank;
    
    /* Select appropriate matrix descriptor based on memory configuration */
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
    if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
        A = (parsec_tiled_matrix_t *)&data->dcAd;
    } 
    
    /* Additional matrix descriptors for rank information and fake data */
    parsec_tiled_matrix_t *Ar = (parsec_tiled_matrix_t *)&data->dcAr;
    parsec_tiled_matrix_t *Rank = (parsec_tiled_matrix_t *)&data->dcRank;
    parsec_tiled_matrix_t *Fake = (parsec_tiled_matrix_t *)&data->dcFake;

    /* Validate input parameters */
    if ((uplo != PlasmaLower)) {
        dplasma_error("potrf_L_dense_tlr_dp_New", "potrf_L_dense_tlr_dp only support PlasmaLower for now");
        return NULL /*-1*/;
    }

    /* Check rank parameter consistency */
    if(storagemaxrank > compmaxrank) {
        dplasma_error("potrf_L_dense_tlr_dp_New", "maxrank for storage larger than maxrank for buffers \
                is not meaningful");
        return NULL /*-1*/;
    }

    /* Warn about potentially inefficient rank settings */
    if(storagemaxrank > (A->mb/2) && 0 == rank) {
        fprintf(stderr, RED "Warning: maxrank= %d is larger than half of block size\n" RESET, storagemaxrank);
    }

    /* GPU device detection and initialization */
    int nb = 0, *dev_index;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Find all available CUDA/HIP devices */
    hicma_parsec_find_cuda_devices( &dev_index, &nb);

#if !GPU_BUFFER_ONCE
    /* Allocate GPU workspace if not already allocated */
    gpu_temporay_buffer_init( data, params, A->mb, A->nb, storagemaxrank, params->kind_of_cholesky );
#endif
#endif

    /* Initialize info parameter for error reporting */
    params->info = 0;
    
    /* Create the main task pool for Cholesky factorization */
    parsec_potrf_L_dense_mp_gpu_taskpool_t *hicma_dpotrf =
        parsec_potrf_L_dense_mp_gpu_new( A, Ar, Rank, Fake, params ); 

    /* Identify task class IDs for hook installation */
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

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Configure GPU workspace and device information */
    hicma_dpotrf->_g_ws_gpu = (void *)data->ws_gpu;
    hicma_dpotrf->_g_nb_cuda_devices = nb;
    hicma_dpotrf->_g_cuda_device_index = dev_index;
#endif

    /* Determine incarnation IDs for different execution backends */
    int gpu_id = 0, recursive_id = 0, cpu_id = 0;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    recursive_id += 1;  /* GPU is at index 0, so recursive starts at 1 */
    cpu_id += 1;        /* CPU is after recursive */
#endif
#if defined(PARSEC_HAVE_DEV_RECURSIVE_SUPPORT)
    cpu_id += 1;        /* CPU is after recursive if recursive is supported */
#endif
    
    /* Print incarnation IDs for debugging if enabled */
    if( 0 == rank && DEBUG_INFO ) printf("gpu_id= %d recursive_id= %d cpu_id= %d\n",
            gpu_id, recursive_id, cpu_id);

    /* Configure execution hooks based on available hardware */
    if( nb > 0 ) {
        /* GPU execution path - install GPU hooks and evaluators */
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        /* Store original GPU kernel hooks for later use */
        hicma_dpotrf->_g_params_tlr->wrap_potrf = hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[gpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[gpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[gpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[gpu_id].hook;
        
        /* Install timing wrapper hooks for GPU execution */
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[gpu_id].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[gpu_id].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[gpu_id].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[gpu_id].hook;
        *hook = &wrap_gemm;

        /* Install GPU evaluation functions for kernel selection */
        eval_gpu_potrf = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[gpu_id].evaluate;
        eval_gpu_trsm  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[gpu_id].evaluate;
        eval_gpu_syrk  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[gpu_id].evaluate;
        eval_gpu_gemm  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[gpu_id].evaluate;
        *eval_gpu_potrf = &evaluate_gpu_potrf;
        *eval_gpu_trsm  = &evaluate_gpu_trsm;
        *eval_gpu_syrk  = &evaluate_gpu_syrk;
        *eval_gpu_gemm  = &evaluate_gpu_gemm;
#endif

    /* Recursive execution path - for small block sizes */
    } else if( hmb < A->mb ) { 
        /* Store original recursive kernel hooks */
        hicma_dpotrf->_g_params_tlr->wrap_potrf = hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[recursive_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[recursive_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[recursive_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[recursive_id].hook;
        
        /* Install timing wrapper hooks for recursive execution */
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[recursive_id].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[recursive_id].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[recursive_id].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[recursive_id].hook;
        *hook = &wrap_gemm;

    /* CPU execution path - for standard block sizes */
    } else {
        /* Store original CPU kernel hooks */
        hicma_dpotrf->_g_params_tlr->wrap_potrf = hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[cpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[cpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[cpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[cpu_id].hook;
        
        /* Install timing wrapper hooks for CPU execution */
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[cpu_id].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[cpu_id].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[cpu_id].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[cpu_id].hook;
        *hook = &wrap_gemm;
    }

    /* Install completion wrapper hooks for all task types */
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

    /* ============================================================================
     * Memory Pool Initialization
     * ============================================================================ */
    
    /* Main worker memory pool for temporary buffers */
    hicma_dpotrf->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    
    /* Calculate workspace size for temporary buffers (from hicma/compute/pzpotrf.c line 96) */
    size_t ws_worker = 0;
    ws_worker = 
        2 * A->mb * 2 * compmaxrank            // CU and CV temporary buffers (2*maxrk for side-by-side U's)
        + 2 * A->mb                            // qrtauA and qrtauB arrays
        + compmaxrank * compmaxrank            // qrb_aubut and AcolBcolT matrices
        + 2 * A->mb * 2 * compmaxrank          // newU and newV matrices
        + (2*compmaxrank) * (2*compmaxrank)    // svd_rA and _rA matrices
        + (2*compmaxrank)                      // sigma array
        + (2*compmaxrank);                     // superb array
    ;
    ws_worker *= sizeof(double); 
    parsec_private_memory_init( hicma_dpotrf->_g_p_work, ws_worker ); 

    /* Rank-rank matrix memory pool */
    hicma_dpotrf->_g_p_work_rr = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_rr, compmaxrank * compmaxrank * sizeof(double) ); 

    /* Matrix-block-rank memory pool */
    hicma_dpotrf->_g_p_work_mbr = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_mbr, A->mb * compmaxrank * sizeof(double) ); 

    /* Full matrix memory pools for different precisions */
    hicma_dpotrf->_g_p_work_full_dp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_full_dp, A->mb * A->mb * sizeof(double) );

    hicma_dpotrf->_g_p_work_full_sp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_full_sp, A->mb * A->mb * sizeof(float) );

    hicma_dpotrf->_g_p_work_full_hp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_full_hp, A->mb * A->mb * sizeof(float) / 2 );

    /* U-V factor memory pools for different precisions */
    hicma_dpotrf->_g_p_work_uv_dp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_uv_dp, A->mb * storagemaxrank * 2 * sizeof(double) );

    hicma_dpotrf->_g_p_work_uv_sp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_uv_sp, A->mb * storagemaxrank * 2 * sizeof(float) );

    /* Configure priority change threshold for task scheduling */
    hicma_dpotrf->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    if(0 == hicma_dpotrf->_g_PRI_CHANGE) {
        hicma_dpotrf->_g_PRI_CHANGE = A->nt;
    }

    /* ============================================================================
     * Arena Configuration for Data Types
     * ============================================================================ */
    
    /* Double precision full matrix arena */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_FULL_DP_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Single precision full matrix arena */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_FULL_SP_ADT_IDX],
            parsec_datatype_float_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Double precision U-V factor arena */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_UV_DP_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, storagemaxrank*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Single precision U-V factor arena */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_UV_SP_ADT_IDX],
            parsec_datatype_float_t, PARSEC_MATRIX_FULL,
            1, A->mb, storagemaxrank*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Byte arena for generic data */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_BYTE_ADT_IDX],
            MPI_BYTE, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Half precision matrix arena */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_FULL_HP_ADT_IDX],
            MPI_BYTE, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb*2, A->mb, 
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* FP8 precision matrix arena */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_FULL_FP8_ADT_IDX],
            MPI_BYTE, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Integer array arena */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_AR_ADT_IDX],
            parsec_datatype_int_t, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    if( params->arena_pinned_memory && nb > 0 ) {
        hicma_parsec_arenas_use_pinned_memory(
                hicma_dpotrf->arenas_datatypes,
                PARSEC_potrf_L_dense_mp_gpu_ADT_IDX_MAX);
    }

    return (parsec_taskpool_t*)hicma_dpotrf;
}

/**
 * @brief Destructor for the Cholesky factorization task pool
 * 
 * This function properly cleans up all resources allocated during task pool creation:
 * - GPU workspace and device information
 * - Memory pools for different data types and precisions
 * - Arena configurations
 * - Task pool itself
 * 
 * @param _tp Task pool to be destroyed
 */
void potrf_L_dense_mp_gpu_Destruct(parsec_taskpool_t* _tp)
{
    parsec_potrf_L_dense_mp_gpu_taskpool_t *tp = (parsec_potrf_L_dense_mp_gpu_taskpool_t*)_tp;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Clean up GPU resources if devices were available */
    if( tp->_g_nb_cuda_devices > 0 ) {
#if !GPU_BUFFER_ONCE 
        /* Free GPU workspace memory if not managed elsewhere */
        workspace_memory_free( tp->_g_ws_gpu );
#endif

        /* Free device index array */
        if( NULL != tp->_g_cuda_device_index )
            free(tp->_g_cuda_device_index);
    }
#endif

    /* Clean up all arena configurations */
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_FULL_DP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_FULL_SP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_FULL_HP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_FULL_FP8_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_UV_DP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_UV_SP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_BYTE_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_AR_ADT_IDX] );
    
    /* Clean up all memory pools */
    parsec_private_memory_fini( tp->_g_p_work );
    parsec_private_memory_fini( tp->_g_p_work_mbr );
    parsec_private_memory_fini( tp->_g_p_work_rr );
    parsec_private_memory_fini( tp->_g_p_work_full_dp );
    parsec_private_memory_fini( tp->_g_p_work_full_sp );
    parsec_private_memory_fini( tp->_g_p_work_full_hp );
    parsec_private_memory_fini( tp->_g_p_work_uv_dp );
    parsec_private_memory_fini( tp->_g_p_work_uv_sp );

    /* Free the task pool itself */
    parsec_taskpool_free(_tp);
}

/* ============================================================================
 * Main Interface Function
 * ============================================================================ */

/**
 * @brief Main interface function for mixed-precision Cholesky factorization with GPU support
 * 
 * This function provides the main entry point for Cholesky factorization with:
 * - Mixed-precision computation (TLR - Tile Low-Rank)
 * - GPU acceleration support
 * - Task-based parallel execution using PaRSEC
 * 
 * The function creates a task pool, executes the factorization, and cleans up resources.
 * 
 * @param parsec PaRSEC context handle for task scheduling
 * @param data Matrix data structures and descriptors
 * @param params Algorithm parameters including precision settings and GPU configuration
 * @return 0 on success
 * 
 * @note info parameter: 0 on all nodes if successful, > 0 if the leading minor 
 *       of order i of A is not positive definite (info = i on the node owning 
 *       diagonal element (i,i), 0 on all other nodes)
 */
int potrf_L_dense_mp_gpu( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    parsec_taskpool_t *hicma_potrf = NULL;

    /* Print start message if verbose mode is enabled */
    if( 0 == params->rank && params->verbose )
        printf(MAG "DENSE_MP_GPU Start\n" RESET);

    /* Create the task pool for Cholesky factorization */
    hicma_potrf = potrf_L_dense_mp_gpu_New( parsec, data, params );

    /* Execute the factorization if task pool was created successfully */
    if( NULL != hicma_potrf ) {
        /* Add task pool to PaRSEC context and execute */
        parsec_context_add_taskpool( parsec, hicma_potrf);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        
        /* Clean up resources */
        potrf_L_dense_mp_gpu_Destruct( hicma_potrf );
        
        /* Synchronize task IDs for recursive DAGs if needed */
        if( params->HNB < params->NB )
            parsec_taskpool_sync_ids(); /* recursive DAGs are not synchronous on ids */
    }

    return 0;
}
