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

/**
 * @file potrf_L_dense_mp_gpu_fp8_adaptive_wrapper.c
 * @brief Wrapper functions for Cholesky factorization with mixed-precision GPU support and FP8 optimization
 * 
 * This file implements wrapper functions for the Cholesky factorization (POTRF) algorithm
 * with support for mixed-precision computations, GPU acceleration, and FP8 data types.
 * The wrappers provide timing instrumentation and kernel selection logic for different
 * computational kernels: POTRF, TRSM, SYRK, and GEMM.
 * 
 * Key features:
 * - Mixed-precision Cholesky factorization (MP)
 * - GPU acceleration support (CUDA/HIP)
 * - FP8 data type support for memory efficiency
 * - Performance timing and profiling
 * - Dynamic kernel selection based on problem size
 * - Memory pool management for different precision types
 */

/* ============================================================================ */
/* Timing and Profiling Wrapper Functions                                      */
/* ============================================================================ */

/**
 * @brief Wrapper function for POTRF kernel execution with timing instrumentation
 * 
 * This function wraps the actual POTRF kernel execution to measure and record
 * the execution time for performance analysis. It records the start time before
 * calling the original kernel function.
 * 
 * @param es Execution stream context
 * @param this_task POTRF task structure containing kernel parameters
 * @return Return value from the wrapped POTRF kernel
 */
static int wrap_potrf(parsec_execution_stream_t * es, 
                      __parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dpotrf_task_t * this_task)
{
    // Get taskpool reference for accessing timing parameters
    parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t*)this_task->taskpool;
    
    // Record start time for POTRF kernel execution timing
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->potrf_time_temp = start_time;
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;
    
    // Execute the actual POTRF kernel and return its result
    return parsec_tp->_g_params_tlr->wrap_potrf(es, (parsec_task_t *)this_task);
}

/**
 * @brief Completion wrapper for POTRF kernel with timing and performance logging
 * 
 * This function is called after POTRF kernel completion to calculate execution time,
 * update cumulative timing statistics, and optionally log performance metrics.
 * It handles critical path timing analysis for performance optimization.
 * 
 * @param es Execution stream context
 * @param this_task POTRF task structure containing kernel parameters
 * @return Return value from the wrapped POTRF completion function
 */
static int wrap_potrf_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dpotrf_task_t * this_task)
{
    int val;
    double end_time;
    parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t*)this_task->taskpool;
    
    // Calculate execution time and update timing statistics
    end_time = MPI_Wtime();
    parsec_tp->_g_params_tlr->potrf_time += end_time - parsec_tp->_g_params_tlr->potrf_time_temp; 
    parsec_tp->_g_params_tlr->gather_time[es->th_id] += end_time - parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id];

#if PRINT_CRITICAL_PATH_TIME
    // Log critical path timing information for performance analysis
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

    val = parsec_tp->_g_params_tlr->wrap_potrf_complete(es, (parsec_task_t *)this_task);
    return val;
}

/**
 * @brief Wrapper function for TRSM kernel execution with conditional timing
 * 
 * This function wraps the TRSM (Triangular Solve) kernel execution. It only records
 * timing for the first TRSM operation in each column (when m == k+1) to avoid
 * double-counting in performance measurements. This selective timing helps in
 * analyzing the critical path of the Cholesky factorization.
 * 
 * @param es Execution stream context
 * @param this_task TRSM task structure containing kernel parameters
 * @return Return value from the wrapped TRSM kernel
 */
static int wrap_trsm(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dtrsm_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t*)this_task->taskpool;
    
    // Record timing for thread-local gather time tracking
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;

    // Only record TRSM timing for the first operation in each column (critical path)
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        // Record start time for TRSM kernel execution timing
        parsec_tp->_g_params_tlr->trsm_time_temp = start_time;
        return parsec_tp->_g_params_tlr->wrap_trsm(es, (parsec_task_t *)this_task);
    } else {
        // Execute TRSM without timing for non-critical path operations
        return parsec_tp->_g_params_tlr->wrap_trsm(es, (parsec_task_t *)this_task);
    }
}

/**
 * @brief Completion wrapper for TRSM kernel with conditional timing and logging
 * 
 * This function handles TRSM kernel completion with selective timing updates.
 * Only the first TRSM operation in each column (critical path) contributes to
 * the cumulative TRSM timing statistics. All operations update thread-local
 * gather time for overall performance tracking.
 * 
 * @param es Execution stream context
 * @param this_task TRSM task structure containing kernel parameters
 * @return Return value from the wrapped TRSM completion function
 */
static int wrap_trsm_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dtrsm_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t*)this_task->taskpool;
    int val;
    double end_time;

    // Handle timing for critical path TRSM operations (first in each column)
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        end_time = MPI_Wtime();
        
        // Update cumulative TRSM timing statistics
        parsec_tp->_g_params_tlr->trsm_time += end_time - parsec_tp->_g_params_tlr->trsm_time_temp;

#if PRINT_CRITICAL_PATH_TIME
        // Log critical path timing information for TRSM operations
	fprintf(stderr, "OUT_critical_path_time band_size_dense %d Nodes %d Matrix %d TRSM %d end_time %lf start_time %lf task_exe_time %lf sum_time %lf\n",
			parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes, parsec_tp->_g_descA->lm,
			this_task->locals.k.value, end_time, parsec_tp->_g_params_tlr->trsm_time_temp,
			end_time - parsec_tp->_g_params_tlr->trsm_time_temp, parsec_tp->_g_params_tlr->trsm_time);
#endif
    } else {
        end_time = MPI_Wtime();
    }

    // Update thread-local gather time for all TRSM operations
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

    val = parsec_tp->_g_params_tlr->wrap_trsm_complete(es, (parsec_task_t *)this_task);
    return val;
}

/**
 * @brief Wrapper function for SYRK kernel execution with conditional timing
 * 
 * This function wraps the SYRK (Symmetric Rank-K update) kernel execution.
 * Similar to TRSM, it only records timing for the first SYRK operation in each
 * column (when m == k+1) to focus on critical path analysis. This selective
 * timing approach helps identify performance bottlenecks in the factorization.
 * 
 * @param es Execution stream context
 * @param this_task SYRK task structure containing kernel parameters
 * @return Return value from the wrapped SYRK kernel
 */
static int wrap_syrk(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dsyrk_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t*)this_task->taskpool;
    
    // Record timing for thread-local gather time tracking
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;
    
    // Only record SYRK timing for the first operation in each column (critical path)
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        // Record start time for SYRK kernel execution timing
        parsec_tp->_g_params_tlr->syrk_time_temp = start_time;
        return parsec_tp->_g_params_tlr->wrap_syrk(es, (parsec_task_t *)this_task);
    } else {
        // Execute SYRK without timing for non-critical path operations
        return parsec_tp->_g_params_tlr->wrap_syrk(es, (parsec_task_t *)this_task);
    }
}

/**
 * @brief Completion wrapper for SYRK kernel with conditional timing and logging
 * 
 * This function handles SYRK kernel completion with selective timing updates.
 * Only the first SYRK operation in each column (critical path) contributes to
 * the cumulative SYRK timing statistics. All operations update thread-local
 * gather time for comprehensive performance tracking.
 * 
 * @param es Execution stream context
 * @param this_task SYRK task structure containing kernel parameters
 * @return Return value from the wrapped SYRK completion function
 */
static int wrap_syrk_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dsyrk_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t*)this_task->taskpool;
    int val;
    double end_time;

    // Handle timing for critical path SYRK operations (first in each column)
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        end_time = MPI_Wtime();
        
        // Update cumulative SYRK timing statistics
        parsec_tp->_g_params_tlr->syrk_time += end_time - parsec_tp->_g_params_tlr->syrk_time_temp;

#if PRINT_CRITICAL_PATH_TIME
        // Log critical path timing information for SYRK operations
	fprintf(stderr, "OUT_critical_path_time band_size_dense %d Nodes %d Matrix %d SYRK %d end_time %lf start_time %lf task_exe_time %lf sum_time %lf\n",
			parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes, parsec_tp->_g_descA->lm,
			this_task->locals.k.value, end_time, parsec_tp->_g_params_tlr->syrk_time_temp,
			end_time - parsec_tp->_g_params_tlr->syrk_time_temp, parsec_tp->_g_params_tlr->syrk_time);
#endif
    } else {
        end_time = MPI_Wtime();
    }

    // Update thread-local gather time for all SYRK operations
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

    val = parsec_tp->_g_params_tlr->wrap_syrk_complete(es, (parsec_task_t *)this_task);
    return val;
}

/**
 * @brief Wrapper function for GEMM kernel execution with timing instrumentation
 * 
 * This function wraps the GEMM (General Matrix Multiply) kernel execution.
 * Unlike POTRF, TRSM, and SYRK, GEMM operations are not part of the critical path
 * in Cholesky factorization, so they only contribute to thread-local gather time
 * tracking for overall performance analysis.
 * 
 * @param es Execution stream context
 * @param this_task GEMM task structure containing kernel parameters
 * @return Return value from the wrapped GEMM kernel
 */
static int wrap_gemm(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dgemm_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t*)this_task->taskpool;
    
    // Record start time for thread-local gather time tracking
    hicma_kernel_time_record(this_task, MPI_Wtime());
    
    // Execute the actual GEMM kernel and return its result
    return parsec_tp->_g_params_tlr->wrap_gemm(es, (parsec_task_t *)this_task);
}

/**
 * @brief Completion wrapper for GEMM kernel with timing and debug logging
 * 
 * This function handles GEMM kernel completion with timing updates and optional
 * debug information logging. GEMM operations contribute to thread-local gather
 * time tracking and can provide detailed execution information when debug mode
 * is enabled for performance analysis and optimization.
 * 
 * @param es Execution stream context
 * @param this_task GEMM task structure containing kernel parameters
 * @return Return value from the wrapped GEMM completion function
 */
static int wrap_gemm_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dgemm_task_t * this_task)
{
    parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t *parsec_tp = (parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t*)this_task->taskpool;
    int val;
    double start_time = hicma_kernel_time_take(this_task, MPI_Wtime());
    // Calculate execution time and update thread-local gather time before releasing successors
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    parsec_tp->_g_params_tlr->gather_time[es->th_id] += elapsed_time;

    // Log detailed GEMM execution information when debug mode is enabled
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
    val = parsec_tp->_g_params_tlr->wrap_gemm_complete(es, (parsec_task_t *)this_task);
    return val;
}


/* ============================================================================ */
/* GPU Kernel Selection and Evaluation Functions                               */
/* ============================================================================ */

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/**
 * @brief GPU kernel evaluation function for POTRF operations
 * 
 * This function determines whether to use GPU acceleration for POTRF operations.
 * Currently, it always returns DONE, indicating that GPU kernels should be used
 * for all POTRF operations when GPU support is available.
 * 
 * @param task POTRF task structure
 * @return PARSEC_HOOK_RETURN_DONE to use GPU kernel
 */
static parsec_hook_return_t evaluate_gpu_potrf(parsec_task_t* task) {
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief GPU kernel evaluation function for TRSM operations
 * 
 * This function determines whether to use GPU acceleration for TRSM operations
 * based on the problem size. GPU kernels are used when the matrix size difference
 * (m-k) is smaller than the dense band size, indicating that the operation
 * involves dense matrix regions that benefit from GPU acceleration.
 * 
 * @param task TRSM task structure containing matrix dimensions
 * @return PARSEC_HOOK_RETURN_DONE to use GPU kernel, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static parsec_hook_return_t evaluate_gpu_trsm(parsec_task_t* task) {
    int m = ((__parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dtrsm_task_t *)task)->locals.m.value;
    int k = ((__parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dtrsm_task_t *)task)->locals.k.value;
    int band_size_dense = ((__parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dtrsm_task_t *)task)->locals.band_size_dense_local.value;
    
    // Use GPU for operations within the dense band region
    if( m-k < band_size_dense )
        return PARSEC_HOOK_RETURN_DONE;
    else
        return PARSEC_HOOK_RETURN_NEXT;
}

/**
 * @brief GPU kernel evaluation function for SYRK operations
 * 
 * This function determines whether to use GPU acceleration for SYRK operations.
 * Currently, it always returns DONE, indicating that GPU kernels should be used
 * for all SYRK operations when GPU support is available.
 * 
 * @param task SYRK task structure
 * @return PARSEC_HOOK_RETURN_DONE to use GPU kernel
 */
static parsec_hook_return_t evaluate_gpu_syrk(parsec_task_t* task) {
        return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief GPU kernel evaluation function for GEMM operations
 * 
 * This function determines whether to use GPU acceleration for GEMM operations
 * based on the problem size. GPU kernels are used when the matrix size difference
 * (m-n) is smaller than the dense band size, indicating that the operation
 * involves dense matrix regions that benefit from GPU acceleration.
 * 
 * @param task GEMM task structure containing matrix dimensions
 * @return PARSEC_HOOK_RETURN_DONE to use GPU kernel, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static parsec_hook_return_t evaluate_gpu_gemm(parsec_task_t* task) {
    int m = ((__parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dgemm_task_t *)task)->locals.m.value;
    int n = ((__parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dgemm_task_t *)task)->locals.n.value;
    int k = ((__parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dgemm_task_t *)task)->locals.k.value;
    int band_size_dense = ((__parsec_potrf_L_dense_mp_gpu_fp8_adaptive_potrf_dgemm_task_t *)task)->locals.band_size_dense_local.value;
    
    // Use GPU for operations within the dense band region
    if( m-n < band_size_dense )
        return PARSEC_HOOK_RETURN_DONE;
    else
        return PARSEC_HOOK_RETURN_NEXT;
}

#endif /* PARSEC_HAVE_DEV_CUDA_SUPPORT */


/* ============================================================================ */
/* Main Taskpool Creation and Management Functions                             */
/* ============================================================================ */

/**
 * @brief Create and initialize a new POTRF taskpool with mixed-precision GPU support and FP8 optimization
 * 
 * This function creates a new taskpool for Cholesky factorization with support for:
 * - Mixed-precision computations (MP)
 * - GPU acceleration (CUDA/HIP)
 * - FP8 data types for memory efficiency
 * - Dynamic kernel selection based on problem characteristics
 * - Memory pool management for different precision types
 * 
 * The function performs comprehensive initialization including:
 * - Parameter validation and matrix selection
 * - GPU device detection and workspace allocation
 * - Task class identification and hook setup
 * - Memory pool initialization for various data types
 * - Arena configuration for different precision types
 * 
 * @param parsec PaRSEC context for task scheduling
 * @param data Matrix data structures and workspace
 * @param params Algorithm parameters and configuration
 * @return Initialized taskpool pointer, or NULL on failure
 * 
 * @param [out] info 0 on all nodes if successful.
 *                   > 0 if the leading minor of order i of A is not positive
 *                   definite, so the factorization could not be completed, and the
 *                   solution has not been computed. Info will be equal to i on the
 *                   node that owns the diagonal element (i,i), and 0 on all other nodes
 */
parsec_taskpool_t*
potrf_L_dense_mp_gpu_fp8_adaptive_New( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    // Hook pointers for kernel function redirection
    void** hook;
    void** eval_gpu_potrf;
    void** eval_gpu_trsm;
    void** eval_gpu_syrk;
    void** eval_gpu_gemm;

    // Extract and validate input parameters
    int rank = params->rank;
    int uplo = params->uplo;
    int hmb = params->HNB;
    int compmaxrank = params->compmaxrank;
    int storagemaxrank = params->genmaxrank;
    
    // Select appropriate matrix descriptor based on problem characteristics
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
    if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
        A = (parsec_tiled_matrix_t *)&data->dcAd;
    } 
    
    // Initialize matrix descriptors for rank and auxiliary data
    parsec_tiled_matrix_t *Ar = (parsec_tiled_matrix_t *)&data->dcAr;
    parsec_tiled_matrix_t *Rank = (parsec_tiled_matrix_t *)&data->dcRank;
    parsec_tiled_matrix_t *Fake = (parsec_tiled_matrix_t *)&data->dcFake;
    parsec_tiled_matrix_t *Norm = (parsec_tiled_matrix_t *)&data->dcNorm;

    // Validate input parameters
    if ((uplo != PlasmaLower)) {
        dplasma_error("potrf_L_dense_tlr_dp_New", "potrf_L_dense_tlr_dp only support PlasmaLower for now");
        return NULL /*-1*/;
    }

    // Validate rank parameters for storage and computation
    if(storagemaxrank > compmaxrank) {
        dplasma_error("potrf_L_dense_tlr_dp_New", "maxrank for storage larger than maxrank for buffers \
                is not meaningful");
        return NULL /*-1*/;
    }

    // Warn about potentially inefficient rank settings
    if(storagemaxrank > (A->mb/2) && 0 == rank) {
        fprintf(stderr, RED "Warning: maxrank= %d is larger than half of block size\n" RESET, storagemaxrank);
    }

    // GPU device detection and initialization
    int nb = 0, *dev_index;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    // Find and initialize available GPU devices
    hicma_parsec_find_cuda_devices( &dev_index, &nb);

#if !GPU_BUFFER_ONCE
    // Allocate GPU workspace buffers if not already allocated
    gpu_temporay_buffer_init( data, params, A->mb, A->nb, storagemaxrank, params->kind_of_cholesky );
#endif
#endif

    // Initialize error status and create the main taskpool
    params->info = 0;
    parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t *hicma_dpotrf =
        parsec_potrf_L_dense_mp_gpu_fp8_adaptive_new( A, Ar, Rank, Norm, Fake, params ); 

    // Identify task class IDs for different kernel types
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
    
    // Debug output for task class identification
    if( 0 == rank && DEBUG_INFO ) printf("potrf_id= %d trsm_id= %d syrk_id= %d gemm_id= %d\n",
            potrf_id, trsm_id, syrk_id, gemm_id);

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    // Configure GPU workspace and device information
    hicma_dpotrf->_g_ws_gpu = (void *)data->ws_gpu;
    hicma_dpotrf->_g_nb_cuda_devices = nb;
    hicma_dpotrf->_g_cuda_device_index = dev_index;
#endif

    // Determine incarnation IDs for different execution backends
    int gpu_id = 0, recursive_id = 0, cpu_id = 0;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    recursive_id += 1;
    cpu_id += 1;
#endif
#if defined(PARSEC_HAVE_DEV_RECURSIVE_SUPPORT)
    cpu_id += 1;
#endif
    
    // Debug output for incarnation identification
    if( 0 == rank && DEBUG_INFO ) printf("gpu_id= %d recursive_id= %d cpu_id= %d\n",
            gpu_id, recursive_id, cpu_id);

    // Configure kernel execution hooks based on available hardware
    if( nb > 0 ) {
        // GPU execution path - configure GPU kernel hooks and evaluators
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        // Store original GPU kernel functions for wrapper calls
        hicma_dpotrf->_g_params_tlr->wrap_potrf = hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[gpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[gpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[gpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[gpu_id].hook;
        
        // Redirect kernel hooks to timing wrapper functions
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[gpu_id].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[gpu_id].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[gpu_id].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[gpu_id].hook;
        *hook = &wrap_gemm;

        // Configure GPU kernel evaluation functions for dynamic selection
        eval_gpu_potrf = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[gpu_id].evaluate;
        eval_gpu_trsm  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[gpu_id].evaluate;
        eval_gpu_syrk  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[gpu_id].evaluate;
        eval_gpu_gemm  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[gpu_id].evaluate;
        *eval_gpu_potrf = &evaluate_gpu_potrf;
        *eval_gpu_trsm  = &evaluate_gpu_trsm;
        *eval_gpu_syrk  = &evaluate_gpu_syrk;
        *eval_gpu_gemm  = &evaluate_gpu_gemm;
#endif

    // Recursive execution path for smaller block sizes
    } else if( hmb < A->mb ) { 
        // Store original recursive kernel functions for wrapper calls
        hicma_dpotrf->_g_params_tlr->wrap_potrf = hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[recursive_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[recursive_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[recursive_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[recursive_id].hook;
        
        // Redirect recursive kernel hooks to timing wrapper functions
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[recursive_id].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[recursive_id].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[recursive_id].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[recursive_id].hook;
        *hook = &wrap_gemm;

    // Standard CPU execution path
    } else {
        // Store original CPU kernel functions for wrapper calls
        hicma_dpotrf->_g_params_tlr->wrap_potrf = hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[cpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[cpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[cpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[cpu_id].hook;
        
        // Redirect CPU kernel hooks to timing wrapper functions
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[cpu_id].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[cpu_id].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[cpu_id].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[cpu_id].hook;
        *hook = &wrap_gemm;
    }

    // Configure completion hooks for all kernel types
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

    // Initialize memory pools for different precision types and workspace buffers
    hicma_dpotrf->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    
    // Calculate workspace size for temporary buffers (based on hicma/compute/pzpotrf.c line 96)
    size_t ws_worker = 0;
    ws_worker = 
        2 * A->mb * 2 * compmaxrank            // CU and CV temporary buffers (2*maxrk for side-by-side U storage)
        + 2 * A->mb                            // qrtauA and qrtauB arrays
        + compmaxrank * compmaxrank            // qrb_aubut and AcolBcolT matrices
        + 2 * A->mb * 2 * compmaxrank          // newU and newV matrices
        + (2*compmaxrank) * (2*compmaxrank)    // svd_rA and _rA matrices
        + (2*compmaxrank)                      // sigma vector
        + (2*compmaxrank);                     // superb vector
    ;
    ws_worker *= sizeof(double); 
    parsec_private_memory_init( hicma_dpotrf->_g_p_work, ws_worker ); 

    // Initialize rank-by-rank matrix memory pool
    hicma_dpotrf->_g_p_work_rr = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_rr, compmaxrank * compmaxrank * sizeof(double) ); 

    // Initialize matrix-block-rank memory pool
    hicma_dpotrf->_g_p_work_mbr = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_mbr, A->mb * compmaxrank * sizeof(double) ); 

    // Initialize full matrix memory pools for different precision types
    hicma_dpotrf->_g_p_work_full_dp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_full_dp, A->mb * A->mb * sizeof(double) );

    hicma_dpotrf->_g_p_work_full_sp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_full_sp, A->mb * A->mb * sizeof(float) );

    hicma_dpotrf->_g_p_work_full_hp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_full_hp, A->mb * A->mb * sizeof(float) / 2 );

    hicma_dpotrf->_g_p_work_full_fp8 = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_full_fp8, A->mb * A->mb * sizeof(float) / 4 );

    // Initialize U-V matrix memory pools for different precision types
    hicma_dpotrf->_g_p_work_uv_dp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_uv_dp, A->mb * storagemaxrank * 2 * sizeof(double) );

    hicma_dpotrf->_g_p_work_uv_sp = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_uv_sp, A->mb * storagemaxrank * 2 * sizeof(float) );

    // Configure task priority limits for scheduling optimization
    hicma_dpotrf->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    if(0 == hicma_dpotrf->_g_PRI_CHANGE) {
        hicma_dpotrf->_g_PRI_CHANGE = A->nt;
    }

    // Configure data type arenas for different precision types and matrix layouts
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_DP_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_SP_ADT_IDX],
            parsec_datatype_float_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_UV_DP_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, storagemaxrank*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_UV_SP_ADT_IDX],
            parsec_datatype_float_t, PARSEC_MATRIX_FULL,
            1, A->mb, storagemaxrank*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_BYTE_ADT_IDX],
            MPI_BYTE, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_HP_ADT_IDX],
            MPI_BYTE, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb*2, A->mb, 
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_FP8_ADT_IDX],
            MPI_BYTE, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_AR_ADT_IDX],
            parsec_datatype_int_t, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_NORM_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, 2, 1, 2,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    if( params->arena_pinned_memory && nb > 0 ) {
        hicma_parsec_arenas_use_pinned_memory(
                hicma_dpotrf->arenas_datatypes,
                PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_ADT_IDX_MAX);
    }

    return (parsec_taskpool_t*)hicma_dpotrf;
}

/**
 * @brief Destructor function for POTRF taskpool with comprehensive cleanup
 * 
 * This function performs complete cleanup of the POTRF taskpool, including:
 * - GPU workspace and device memory deallocation
 * - Data type arena cleanup for all precision types
 * - Memory pool deinitialization for all workspace buffers
 * - Taskpool structure deallocation
 * 
 * @param _tp Taskpool pointer to be destroyed
 */
void potrf_L_dense_mp_gpu_fp8_adaptive_Destruct(parsec_taskpool_t* _tp)
{
    parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t *tp = (parsec_potrf_L_dense_mp_gpu_fp8_adaptive_taskpool_t*)_tp;
    parsec_memory_pool_t *p_work = tp->_g_p_work;
    parsec_memory_pool_t *p_work_mbr = tp->_g_p_work_mbr;
    parsec_memory_pool_t *p_work_rr = tp->_g_p_work_rr;
    parsec_memory_pool_t *p_work_full_dp = tp->_g_p_work_full_dp;
    parsec_memory_pool_t *p_work_full_sp = tp->_g_p_work_full_sp;
    parsec_memory_pool_t *p_work_full_hp = tp->_g_p_work_full_hp;
    parsec_memory_pool_t *p_work_full_fp8 = tp->_g_p_work_full_fp8;
    parsec_memory_pool_t *p_work_uv_dp = tp->_g_p_work_uv_dp;
    parsec_memory_pool_t *p_work_uv_sp = tp->_g_p_work_uv_sp;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    // Clean up GPU resources if devices were used
    if( tp->_g_nb_cuda_devices > 0 ) {
#if !GPU_BUFFER_ONCE 
        // Free GPU workspace memory if not managed globally
        workspace_memory_free( tp->_g_ws_gpu );
#endif

        // Free GPU device index array
        if( NULL != tp->_g_cuda_device_index )
            free(tp->_g_cuda_device_index);
    }
#endif

    // Clean up data type arenas for all precision types
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_DP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_SP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_HP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_FULL_FP8_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_UV_DP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_UV_SP_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_BYTE_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_AR_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_mp_gpu_fp8_adaptive_NORM_ADT_IDX] );
    
    // Free the taskpool structure
    parsec_taskpool_free(_tp);

    // Clean up all memory pools after taskpool repositories are released.
    parsec_private_memory_fini(p_work);
    parsec_private_memory_fini(p_work_mbr);
    parsec_private_memory_fini(p_work_rr);
    parsec_private_memory_fini(p_work_full_dp);
    parsec_private_memory_fini(p_work_full_sp);
    parsec_private_memory_fini(p_work_full_hp);
    parsec_private_memory_fini(p_work_full_fp8);
    parsec_private_memory_fini(p_work_uv_dp);
    parsec_private_memory_fini(p_work_uv_sp);
    free(p_work);
    free(p_work_mbr);
    free(p_work_rr);
    free(p_work_full_dp);
    free(p_work_full_sp);
    free(p_work_full_hp);
    free(p_work_full_fp8);
    free(p_work_uv_dp);
    free(p_work_uv_sp);
}

/* ============================================================================ */
/* Main Interface Function                                                      */
/* ============================================================================ */

/**
 * @brief Main interface function for mixed-precision Cholesky factorization with GPU support and FP8 optimization
 * 
 * This function provides the main interface for performing Cholesky factorization with:
 * - Mixed-precision computations (MP)
 * - GPU acceleration support (CUDA/HIP)
 * - FP8 data type support for memory efficiency
 * - TLR (Tile Low-Rank) compression
 * 
 * The function orchestrates the complete factorization process including:
 * - Taskpool creation and configuration
 * - Task scheduling and execution
 * - Resource cleanup and synchronization
 * 
 * @param parsec PaRSEC context for task scheduling and execution
 * @param data Matrix data structures and workspace
 * @param params Algorithm parameters and configuration
 * @return 0 on success, error code on failure
 * 
 * @param [out] info 0 on all nodes if successful.
 *                   > 0 if the leading minor of order i of A is not positive
 *                   definite, so the factorization could not be completed, and the
 *                   solution has not been computed. Info will be equal to i on the
 *                   node that owns the diagonal element (i,i), and 0 on all other nodes
 */
int potrf_L_dense_mp_gpu_fp8_adaptive( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    parsec_taskpool_t *hicma_potrf = NULL;

    // Print startup message on rank 0 if verbose mode is enabled
    if( 0 == params->rank && params->verbose )
        printf(MAG "DENSE_MP_GPU_FP8_ADAPTIVE Start\n" RESET);

    // Create and configure the POTRF taskpool
    hicma_potrf = potrf_L_dense_mp_gpu_fp8_adaptive_New( parsec, data, params );

    // Execute the factorization if taskpool creation was successful
    if( NULL != hicma_potrf ) {
        // Add taskpool to PaRSEC context and execute
        parsec_context_add_taskpool( parsec, hicma_potrf);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        
        // Clean up resources
        potrf_L_dense_mp_gpu_fp8_adaptive_Destruct( hicma_potrf );
        
        // Synchronize task IDs for recursive DAGs if needed
        if( params->HNB < params->NB )
            parsec_taskpool_sync_ids(); /*recursive DAGs are not synchronous on ids */
    }

    return 0;
}
