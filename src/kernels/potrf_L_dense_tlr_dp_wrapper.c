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
#include "potrf_L_dense_tlr_dp.h"

/**
 * @brief Wrapper function for POTRF (Cholesky factorization) task execution
 * 
 * This function wraps the actual POTRF kernel execution to measure and record
 * timing information for performance analysis. It captures the start time
 * before delegating to the actual kernel implementation.
 * 
 * The POTRF operation performs Cholesky factorization on a diagonal block,
 * which is the most critical operation in the factorization process as it
 * determines the overall performance and numerical stability.
 * 
 * @param es Execution stream handle for the current thread
 * @param this_task Pointer to the POTRF task structure containing all necessary data
 * @return Return code from the actual POTRF kernel execution
 */
static int wrap_potrf(parsec_execution_stream_t * es, 
                      __parsec_potrf_L_dense_tlr_dp_potrf_dpotrf_task_t * this_task)
{
    /* Cast taskpool to access timing parameters */
    parsec_potrf_L_dense_tlr_dp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_dp_taskpool_t*)this_task->taskpool;
    
    /* Record start time of POTRF operation for performance profiling */
    /* This timing is used for critical path analysis and performance optimization */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->potrf_time_temp = start_time;
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;
    
    /* Delegate to the actual POTRF kernel implementation */
    /* The actual implementation depends on the execution device (CPU/GPU) */
    return parsec_tp->_g_params_tlr->wrap_potrf(es, (parsec_task_t *)this_task);
}

/**
 * @brief Completion wrapper for POTRF task execution
 * 
 * This function handles the completion phase of POTRF task execution, measuring
 * the total execution time and updating performance counters. It also provides
 * optional debug output for critical path analysis.
 * 
 * The completion phase is crucial for performance monitoring as it captures
 * the actual execution time of the most computationally intensive operation
 * in the Cholesky factorization process.
 * 
 * @param es Execution stream handle for the current thread
 * @param this_task Pointer to the POTRF task structure
 * @return Return code from the actual POTRF completion function
 */
static int wrap_potrf_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_dp_potrf_dpotrf_task_t * this_task)
{
    int val;
    double end_time;
    /* Cast taskpool to access timing parameters and matrix descriptors */
    parsec_potrf_L_dense_tlr_dp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_dp_taskpool_t*)this_task->taskpool;
    
    /* Execute the actual completion function first */
    /* This handles any cleanup or post-processing required by the kernel */
    val = parsec_tp->_g_params_tlr->wrap_potrf_complete(es, (parsec_task_t *)this_task);
    
    /* Record end time and update performance counters */
    end_time = MPI_Wtime();
    /* Accumulate total POTRF execution time across all tasks */
    parsec_tp->_g_params_tlr->potrf_time += end_time - parsec_tp->_g_params_tlr->potrf_time_temp; 
    /* Update per-thread execution time for load balancing analysis */
    parsec_tp->_g_params_tlr->gather_time[es->th_id] += end_time - parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id];

#if PRINT_CRITICAL_PATH_TIME
    /* Debug output for critical path timing analysis */
    /* This helps identify bottlenecks in the factorization process */
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
 * @brief Wrapper function for TRSM (triangular solve) task execution
 * 
 * This function wraps the TRSM kernel execution with timing measurements.
 * It only records detailed timing for the first TRSM operation in each column
 * (when m == k+1) to avoid double-counting in performance analysis.
 * 
 * TRSM performs triangular matrix solve operations, which are essential for
 * updating the lower triangular part of the matrix during Cholesky factorization.
 * The timing optimization ensures accurate performance measurement without
 * inflating the total execution time.
 * 
 * @param es Execution stream handle for the current thread
 * @param this_task Pointer to the TRSM task structure
 * @return Return code from the actual TRSM kernel execution
 */
static int wrap_trsm(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_dp_potrf_dtrsm_task_t * this_task)
{
    /* Cast taskpool to access timing parameters */
    parsec_potrf_L_dense_tlr_dp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_dp_taskpool_t*)this_task->taskpool;
    
    /* Record start time for thread-level timing analysis */
    /* This is used for load balancing and thread utilization analysis */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;

    /* Only record detailed TRSM timing for the first operation in each column */
    /* This prevents double-counting when multiple TRSM operations occur in the same column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Record start time of TRSM operation for performance profiling */
        /* This represents the first TRSM operation in the current column */
        parsec_tp->_g_params_tlr->trsm_time_temp = start_time;
    }
    
    /* Delegate to the actual TRSM kernel implementation */
    /* The actual implementation depends on the execution device (CPU/GPU) */
    return parsec_tp->_g_params_tlr->wrap_trsm(es, (parsec_task_t *)this_task);
}

/**
 * @brief Completion wrapper for TRSM task execution
 * 
 * This function handles the completion phase of TRSM task execution, measuring
 * execution time and updating performance counters. Detailed timing is only
 * recorded for the first TRSM operation in each column to avoid double-counting.
 * 
 * The completion phase ensures accurate performance measurement by only
 * accumulating timing for representative TRSM operations, preventing
 * over-counting in the performance statistics.
 * 
 * @param es Execution stream handle for the current thread
 * @param this_task Pointer to the TRSM task structure
 * @return Return code from the actual TRSM completion function
 */
static int wrap_trsm_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_dp_potrf_dtrsm_task_t * this_task)
{
    /* Cast taskpool to access timing parameters and matrix descriptors */
    parsec_potrf_L_dense_tlr_dp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_dp_taskpool_t*)this_task->taskpool;
    int val;
    double end_time;

    /* Execute the actual completion function first */
    /* This handles any cleanup or post-processing required by the kernel */
    val = parsec_tp->_g_params_tlr->wrap_trsm_complete(es, (parsec_task_t *)this_task);
    end_time = MPI_Wtime();

    /* Only update detailed TRSM timing for the first operation in each column */
    /* This prevents double-counting when multiple TRSM operations occur in the same column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Update TRSM timing counters */
        /* Accumulate total TRSM execution time across all representative tasks */
        parsec_tp->_g_params_tlr->trsm_time += end_time - parsec_tp->_g_params_tlr->trsm_time_temp;

#if PRINT_CRITICAL_PATH_TIME
        /* Debug output for critical path timing analysis */
        /* This helps identify TRSM bottlenecks in the factorization process */
        fprintf(stderr, "OUT_critical_path_time band_size_dense %d Nodes %d Matrix %d TRSM %d end_time %lf start_time %lf task_exe_time %lf sum_time %lf\n",
                parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes, parsec_tp->_g_descA->lm,
                this_task->locals.k.value, end_time, parsec_tp->_g_params_tlr->trsm_time_temp,
                end_time - parsec_tp->_g_params_tlr->trsm_time_temp, parsec_tp->_g_params_tlr->trsm_time);
#endif
    }

    /* Update thread-level timing counters */
    /* This is used for load balancing and thread utilization analysis */
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
 * @brief Wrapper function for SYRK (symmetric rank-k update) task execution
 * 
 * This function wraps the SYRK kernel execution with timing measurements.
 * It only records detailed timing for the first SYRK operation in each column
 * (when m == k+1) to avoid double-counting in performance analysis.
 * 
 * SYRK performs symmetric rank-k updates, which are crucial for maintaining
 * the symmetric structure of the matrix during Cholesky factorization.
 * The timing optimization ensures accurate performance measurement without
 * inflating the total execution time.
 * 
 * @param es Execution stream handle for the current thread
 * @param this_task Pointer to the SYRK task structure
 * @return Return code from the actual SYRK kernel execution
 */
static int wrap_syrk(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_dp_potrf_dsyrk_task_t * this_task)
{
    /* Cast taskpool to access timing parameters */
    parsec_potrf_L_dense_tlr_dp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_dp_taskpool_t*)this_task->taskpool;
    
    /* Record start time for thread-level timing analysis */
    /* This is used for load balancing and thread utilization analysis */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;
    
    /* Only record detailed SYRK timing for the first operation in each column */
    /* This prevents double-counting when multiple SYRK operations occur in the same column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Record start time of SYRK operation for performance profiling */
        /* This represents the first SYRK operation in the current column */
        parsec_tp->_g_params_tlr->syrk_time_temp = start_time;
    }
    
    /* Delegate to the actual SYRK kernel implementation */
    /* The actual implementation depends on the execution device (CPU/GPU) */
    return parsec_tp->_g_params_tlr->wrap_syrk(es, (parsec_task_t *)this_task);
}

/**
 * @brief Completion wrapper for SYRK task execution
 * 
 * This function handles the completion phase of SYRK task execution, measuring
 * execution time and updating performance counters. Detailed timing is only
 * recorded for the first SYRK operation in each column to avoid double-counting.
 * 
 * The completion phase ensures accurate performance measurement by only
 * accumulating timing for representative SYRK operations, preventing
 * over-counting in the performance statistics.
 * 
 * @param es Execution stream handle for the current thread
 * @param this_task Pointer to the SYRK task structure
 * @return Return code from the actual SYRK completion function
 */
static int wrap_syrk_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_dp_potrf_dsyrk_task_t * this_task)
{
    /* Cast taskpool to access timing parameters and matrix descriptors */
    parsec_potrf_L_dense_tlr_dp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_dp_taskpool_t*)this_task->taskpool;
    int val;
    double end_time;

    /* Execute the actual completion function first */
    /* This handles any cleanup or post-processing required by the kernel */
    val = parsec_tp->_g_params_tlr->wrap_syrk_complete(es, (parsec_task_t *)this_task);
    end_time = MPI_Wtime();

    /* Only update detailed SYRK timing for the first operation in each column */
    /* This prevents double-counting when multiple SYRK operations occur in the same column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Update SYRK timing counters */
        /* Accumulate total SYRK execution time across all representative tasks */
        parsec_tp->_g_params_tlr->syrk_time += end_time - parsec_tp->_g_params_tlr->syrk_time_temp;

#if PRINT_CRITICAL_PATH_TIME
        /* Debug output for critical path timing analysis */
        /* This helps identify SYRK bottlenecks in the factorization process */
        fprintf(stderr, "OUT_critical_path_time band_size_dense %d Nodes %d Matrix %d SYRK %d end_time %lf start_time %lf task_exe_time %lf sum_time %lf\n",
                parsec_tp->_g_params_tlr->band_size_dense, parsec_tp->_g_descA->super.nodes, parsec_tp->_g_descA->lm,
                this_task->locals.k.value, end_time, parsec_tp->_g_params_tlr->syrk_time_temp,
                end_time - parsec_tp->_g_params_tlr->syrk_time_temp, parsec_tp->_g_params_tlr->syrk_time);
#endif
    }

    /* Update thread-level timing counters */
    /* This is used for load balancing and thread utilization analysis */
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
 * @brief Wrapper function for GEMM (general matrix multiply) task execution
 * 
 * This function wraps the GEMM kernel execution with timing measurements.
 * It records the start time for thread-level performance analysis.
 * 
 * GEMM performs general matrix multiplication operations, which are the most
 * computationally intensive operations in the Cholesky factorization process.
 * These operations typically dominate the overall execution time and are
 * critical for achieving high performance.
 * 
 * @param es Execution stream handle for the current thread
 * @param this_task Pointer to the GEMM task structure
 * @return Return code from the actual GEMM kernel execution
 */
static int wrap_gemm(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_dp_potrf_dgemm_task_t * this_task)
{
    /* Cast taskpool to access timing parameters */
    parsec_potrf_L_dense_tlr_dp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_dp_taskpool_t*)this_task->taskpool;
    
    /* Record start time for thread-level timing analysis */
    /* This is used for load balancing and thread utilization analysis */
    hicma_kernel_time_record(this_task, MPI_Wtime());
    
    /* Delegate to the actual GEMM kernel implementation */
    /* The actual implementation depends on the execution device (CPU/GPU) */
    return parsec_tp->_g_params_tlr->wrap_gemm(es, (parsec_task_t *)this_task);
}

/**
 * @brief Completion wrapper for GEMM task execution
 * 
 * This function handles the completion phase of GEMM task execution, measuring
 * execution time and updating performance counters. It provides optional debug
 * output for detailed performance analysis.
 * 
 * The completion phase captures the actual execution time of GEMM operations,
 * which are typically the most time-consuming part of the factorization.
 * This timing information is crucial for performance optimization and
 * load balancing analysis.
 * 
 * @param es Execution stream handle for the current thread
 * @param this_task Pointer to the GEMM task structure
 * @return Return code from the actual GEMM completion function
 */
static int wrap_gemm_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_dense_tlr_dp_potrf_dgemm_task_t * this_task)
{
    /* Cast taskpool to access timing parameters and matrix descriptors */
    parsec_potrf_L_dense_tlr_dp_taskpool_t *parsec_tp = (parsec_potrf_L_dense_tlr_dp_taskpool_t*)this_task->taskpool;
    int val;
    /* Store start time for execution time calculation */
    double start_time = hicma_kernel_time_take(this_task, MPI_Wtime());
    
    /* Execute the actual completion function first */
    /* This handles any cleanup or post-processing required by the kernel */
    val = parsec_tp->_g_params_tlr->wrap_gemm_complete(es, (parsec_task_t *)this_task);
    
    /* Record end time and update performance counters */
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    /* Update per-thread execution time for load balancing analysis */
    parsec_tp->_g_params_tlr->gather_time[es->th_id] += elapsed_time;

    /* Optional output for detailed GEMM timing analysis */
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

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

/**
 * @brief GPU evaluation hook for POTRF tasks
 * 
 * This function determines whether a POTRF task should be executed on GPU.
 * Currently, all POTRF tasks are directed to GPU execution for optimal performance.
 * 
 * POTRF operations are computationally intensive and benefit significantly from
 * GPU acceleration, especially for larger matrix blocks. The diagonal factorization
 * is a critical path operation that should utilize the fastest available hardware.
 * 
 * @param task Pointer to the POTRF task
 * @return PARSEC_HOOK_RETURN_DONE to execute on GPU
 */
static parsec_hook_return_t evaluate_gpu_potrf(parsec_task_t* task) {
    (void)task; /* Suppress unused parameter warning */
    /* Always execute POTRF on GPU for maximum performance */
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief GPU evaluation hook for TRSM tasks
 * 
 * This function determines whether a TRSM task should be executed on GPU
 * based on the distance from the diagonal (m-k) compared to the dense band size.
 * Tasks within the dense band are executed on GPU for better performance.
 * 
 * The decision is based on the matrix structure: operations close to the diagonal
 * (within the dense band) benefit from GPU acceleration due to higher computational
 * intensity, while operations far from the diagonal may be more efficient on CPU
 * due to lower computational density and potential memory transfer overhead.
 * 
 * @param task Pointer to the TRSM task
 * @return PARSEC_HOOK_RETURN_DONE for GPU execution, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static parsec_hook_return_t evaluate_gpu_trsm(parsec_task_t* task) {
    /* Extract task parameters for device selection decision */
    int m = ((__parsec_potrf_L_dense_tlr_dp_potrf_dtrsm_task_t *)task)->locals.m.value;
    int k = ((__parsec_potrf_L_dense_tlr_dp_potrf_dtrsm_task_t *)task)->locals.k.value;
    int band_size_dense = ((__parsec_potrf_L_dense_tlr_dp_potrf_dtrsm_task_t *)task)->locals.band_size_dense_local.value;
    
    /* Execute on GPU if within the dense band, otherwise use CPU */
    /* This optimizes performance by using GPU for computationally intensive operations */
    if( m-k < band_size_dense )
        return PARSEC_HOOK_RETURN_DONE;
    else
        return PARSEC_HOOK_RETURN_NEXT;
}

/**
 * @brief GPU evaluation hook for SYRK tasks
 * 
 * This function determines whether a SYRK task should be executed on GPU.
 * Currently, all SYRK tasks are directed to GPU execution for optimal performance.
 * 
 * SYRK operations perform symmetric rank-k updates which are computationally
 * intensive and benefit significantly from GPU acceleration. These operations
 * are critical for maintaining the symmetric structure of the matrix during
 * factorization and should utilize the fastest available hardware.
 * 
 * @param task Pointer to the SYRK task
 * @return PARSEC_HOOK_RETURN_DONE to execute on GPU
 */
static parsec_hook_return_t evaluate_gpu_syrk(parsec_task_t* task) {
    (void)task; /* Suppress unused parameter warning */
    /* Always execute SYRK on GPU for maximum performance */
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * @brief GPU evaluation hook for GEMM tasks
 * 
 * This function determines whether a GEMM task should be executed on GPU
 * based on the distance from the diagonal (m-n) compared to the dense band size.
 * Tasks within the dense band are executed on GPU for better performance.
 * 
 * The decision is based on the matrix structure: operations close to the diagonal
 * (within the dense band) benefit from GPU acceleration due to higher computational
 * intensity, while operations far from the diagonal may be more efficient on CPU
 * due to lower computational density and potential memory transfer overhead.
 * GEMM operations are the most computationally intensive and benefit most from GPU acceleration.
 * 
 * @param task Pointer to the GEMM task
 * @return PARSEC_HOOK_RETURN_DONE for GPU execution, PARSEC_HOOK_RETURN_NEXT for CPU
 */
static parsec_hook_return_t evaluate_gpu_gemm(parsec_task_t* task) {
    /* Extract task parameters for device selection decision */
    int m = ((__parsec_potrf_L_dense_tlr_dp_potrf_dgemm_task_t *)task)->locals.m.value;
    int n = ((__parsec_potrf_L_dense_tlr_dp_potrf_dgemm_task_t *)task)->locals.n.value;
    int k = ((__parsec_potrf_L_dense_tlr_dp_potrf_dgemm_task_t *)task)->locals.k.value;
    int band_size_dense = ((__parsec_potrf_L_dense_tlr_dp_potrf_dgemm_task_t *)task)->locals.band_size_dense_local.value;
    int send_full = ((__parsec_potrf_L_dense_tlr_dp_potrf_dgemm_task_t *)task)->locals.send_full_tile_local.value;
    
    (void)send_full; /* Suppress unused parameter warning */
    
    /* Execute on GPU if within the dense band, otherwise use CPU */
    /* This optimizes performance by using GPU for computationally intensive operations */
    if( m-n < band_size_dense )
        return PARSEC_HOOK_RETURN_DONE;
    else
        return PARSEC_HOOK_RETURN_NEXT;
}

#endif /* PARSEC_HAVE_DEV_CUDA_SUPPORT */

/**
 * @brief Create a new POTRF taskpool for dense TLR double precision Cholesky factorization
 * 
 * This function creates and initializes a new taskpool for performing Cholesky factorization
 * on dense Tile Low-Rank (TLR) matrices using double precision arithmetic. The implementation
 * supports both CPU and GPU execution with automatic device selection based on matrix properties.
 * 
 * The function performs the following key operations:
 * - Validates input parameters and matrix properties
 * - Initializes GPU devices and memory pools if available
 * - Sets up task execution hooks for performance monitoring
 * - Configures memory arenas for different data types
 * - Establishes priority scheduling for optimal task execution order
 * 
 * @param parsec PaRSEC context for task scheduling and execution
 * @param data Matrix data structures including dense and low-rank representations
 * @param params Algorithm parameters including precision, ranks, and execution options
 * 
 * @return Pointer to the created taskpool, or NULL if creation fails
 * 
 * @note This is a non-blocking implementation that supports 2-flow execution
 * @note The info parameter will be set to 0 on all nodes if successful, or > 0 if
 *       the leading minor of order i of A is not positive definite, indicating
 *       that the factorization could not be completed. Info will be equal to i
 *       on the node that owns the diagonal element (i,i), and 0 on all other nodes.
 */
parsec_taskpool_t*
potrf_L_dense_tlr_dp_New( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    /* Hook pointers for task execution and GPU evaluation */
    /* These pointers will be used to install timing wrapper functions */
    void** hook;
    int rank = params->rank;
    void** eval_gpu_potrf;
    void** eval_gpu_trsm;
    void** eval_gpu_syrk;
    void** eval_gpu_gemm;

    /* Extract algorithm parameters for matrix processing */
    int uplo = params->uplo;           /* Matrix storage format (upper/lower triangular) */
    int hmb = params->HNB;             /* Hierarchical block size for recursive algorithms */
    int compmaxrank = params->compmaxrank;   /* Maximum rank for computation buffers */
    int storagemaxrank = params->genmaxrank; /* Maximum rank for storage (should be <= compmaxrank) */
    
    /* Set up matrix descriptors - use dense matrix if band size is large enough */
    /* The choice between dense and low-rank representation depends on the band size */
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
    if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
        /* Use dense matrix representation for large band sizes */
        A = (parsec_tiled_matrix_t *)&data->dcAd;
    } 
    /* Low-rank matrix descriptors for TLR operations */
    parsec_tiled_matrix_t *Ar = (parsec_tiled_matrix_t *)&data->dcAr;     /* Rank information matrix */
    parsec_tiled_matrix_t *Rank = (parsec_tiled_matrix_t *)&data->dcRank; /* Rank storage matrix */
    parsec_tiled_matrix_t *Fake = (parsec_tiled_matrix_t *)&data->dcFake; /* Dummy matrix for compatibility */

    /* Validate input parameters */
    /* Currently only lower triangular matrices are supported */
    if ((uplo != PlasmaLower)) {
        dplasma_error("potrf_L_dense_tlr_dp_New", "potrf_L_dense_tlr_dp only support PlasmaLower for now");
        return NULL /*-1*/;
    }

    /* Check rank consistency - storage rank should not exceed computation rank */
    /* This ensures that the storage format can accommodate the computation requirements */
    if(storagemaxrank > compmaxrank) {
        dplasma_error("potrf_L_dense_tlr_dp_New", "maxrank for storage larger than maxrank for buffers \
                is not meaningful");
        return NULL /*-1*/;
    }

    /* Warn if storage rank is too large relative to block size */
    /* Large ranks relative to block size may indicate inefficient memory usage */
    if(storagemaxrank > (A->mb/2) && 0 == rank) {
        fprintf(stderr, RED "Warning: maxrank= %d is larger than half of block size\n" RESET, storagemaxrank);
    }

    /* Initialize GPU device management */
    int nb = 0, *dev_index;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Find and initialize all available CUDA/HIP devices */
    /* This discovers and prepares GPU devices for computation */
    hicma_parsec_find_cuda_devices( &dev_index, &nb);

#if !GPU_BUFFER_ONCE
    /* Allocate GPU workspace if not already allocated */
    /* This sets up temporary buffers for GPU operations */
    gpu_temporay_buffer_init( data, params, A->mb, A->nb, storagemaxrank, params->kind_of_cholesky );
#endif
#endif

    /* Initialize error tracking */
    /* This will be set to 0 on success or > 0 if factorization fails */
    params->info = 0;
    
    /* Create the main taskpool for POTRF computation */
    /* This creates the task dependency graph and initializes all necessary data structures */
    parsec_potrf_L_dense_tlr_dp_taskpool_t *hicma_dpotrf;
    hicma_dpotrf = parsec_potrf_L_dense_tlr_dp_new( A, Ar, Rank, Fake, params );

    /* Identify task class IDs for different kernel types */
    /* These IDs are used to install timing wrapper functions for each kernel type */
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
    /* Debug output for task class identification */
    if( 0 == rank && DEBUG_INFO ) printf("potrf_id= %d trsm_id= %d syrk_id= %d gemm_id= %d\n",
            potrf_id, trsm_id, syrk_id, gemm_id);

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Configure GPU workspace and device information */
    hicma_dpotrf->_g_ws_gpu = (void *)data->ws_gpu;
    hicma_dpotrf->_g_nb_cuda_devices = nb;
    hicma_dpotrf->_g_cuda_device_index = dev_index;
#endif

    /* Determine execution incarnation IDs based on available hardware */
    int gpu_id = 0, recursive_id = 0, cpu_id = 0;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    recursive_id += 1;
    cpu_id += 1;
#endif
#if defined(PARSEC_HAVE_DEV_RECURSIVE_SUPPORT)
    cpu_id += 1;
#endif
    if( 0 == rank && DEBUG_INFO ) printf("gpu_id= %d recursive_id= %d cpu_id= %d\n",
            gpu_id, recursive_id, cpu_id);

    /* Configure execution hooks based on available hardware */
    if( nb > 0 ) {
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        /* Set up GPU execution hooks for all kernel types */
        hicma_dpotrf->_g_params_tlr->wrap_potrf = hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[gpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[gpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[gpu_id].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[gpu_id].hook;
        
        /* Install timing wrapper hooks for performance monitoring */
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[gpu_id].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[gpu_id].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[gpu_id].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[gpu_id].hook;
        *hook = &wrap_gemm;

        /* Set up GPU evaluation hooks for device selection */
        eval_gpu_potrf = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[gpu_id].evaluate;
        eval_gpu_trsm  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[gpu_id].evaluate;
        eval_gpu_syrk  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[gpu_id].evaluate;
        eval_gpu_gemm  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[gpu_id].evaluate;
        *eval_gpu_potrf = &evaluate_gpu_potrf;
        *eval_gpu_trsm  = &evaluate_gpu_trsm;
        *eval_gpu_syrk  = &evaluate_gpu_syrk;
        *eval_gpu_gemm  = &evaluate_gpu_gemm;
#endif

        /* Configure recursive execution for small block sizes */
    } else if( hmb < A->mb ) { 
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

        /* Configure standard CPU execution for larger block sizes */
    } else {
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

    /* Set up completion hooks for all kernel types */
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

    /* Initialize memory pools for temporary workspace allocation */
    /* These pools provide thread-safe memory allocation for temporary buffers */
    hicma_dpotrf->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    
    /* Calculate workspace size for temporary buffers used in low-rank operations */
    /* Note: This workspace size calculation is based on hicma/compute/pzpotrf.c line 96 */
    /* TODO: Optimize workspace size - not all tasks need this much memory */
    size_t ws_worker =  
        2 * A->mb * 2 * compmaxrank   // Temporary buffers for CU and CV copying (2*maxrk for side-by-side U storage)
        + 2 * A->mb                   // QR tau arrays (qrtauA, qrtauB)
        + compmaxrank * compmaxrank   // QR intermediate results (qrb_aubut, AcolBcolT)
        + 2 * A->mb * 2 * compmaxrank // New U and V matrices
        + (2*compmaxrank) * (2*compmaxrank)  // SVD R matrix storage
        + (2*compmaxrank)             // SVD sigma values
        + (2*compmaxrank);            // SVD superb array
    ;
    /* Initialize the main workspace pool for low-rank operations */
    parsec_private_memory_init( hicma_dpotrf->_g_p_work, ws_worker * sizeof(double) ); 

    /* Initialize rank-rank workspace pool for small matrix operations */
    /* This pool is used for rank x rank matrix operations in low-rank computations */
    hicma_dpotrf->_g_p_work_rr = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_rr, compmaxrank * compmaxrank * sizeof(double) ); 

    /* Initialize matrix-block-rank workspace pool for intermediate computations */
    /* This pool is used for matrix-block x rank operations in low-rank computations */
    hicma_dpotrf->_g_p_work_mbr = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_mbr, A->mb * compmaxrank * sizeof(double) ); 

    /* Set up priority scheduling for optimal task execution order */
    /* This ensures that critical path tasks are executed with higher priority */
    hicma_dpotrf->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    if(0 == hicma_dpotrf->_g_PRI_CHANGE) {
        hicma_dpotrf->_g_PRI_CHANGE = A->nt;
    }

    /* Configure memory arenas for different data types and access patterns */
    /* These arenas optimize memory layout and access patterns for different data types */
    
    /* Full matrix arena for dense matrix operations */
    /* Used for storing complete matrix blocks in dense format */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_tlr_dp_FULL_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Default arena for scalar and small data */
    /* Used for storing scalar values and small arrays */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_tlr_dp_DEFAULT_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* UV arena for low-rank matrix storage (U and V factors) */
    /* Used for storing low-rank factors with optimized memory layout */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_tlr_dp_UV_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, storagemaxrank*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* AR arena for rank information storage */
    /* Used for storing rank information and metadata */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_dense_tlr_dp_AR_ADT_IDX],
            parsec_datatype_int_t, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    return (parsec_taskpool_t*)hicma_dpotrf;
}

/**
 * @brief Destructor for POTRF taskpool
 * 
 * This function properly cleans up all resources associated with the POTRF taskpool,
 * including GPU memory, memory pools, and arena datatypes. It ensures that all
 * allocated resources are properly freed to prevent memory leaks.
 * 
 * @param _tp Pointer to the taskpool to be destroyed
 */
void potrf_L_dense_tlr_dp_Destruct(parsec_taskpool_t* _tp)
{
    /* Cast to the specific taskpool type for cleanup */
    parsec_potrf_L_dense_tlr_dp_taskpool_t *tp = (parsec_potrf_L_dense_tlr_dp_taskpool_t*)_tp;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Clean up GPU resources if any devices were used */
    if( tp->_g_nb_cuda_devices > 0 ) {
#if !GPU_BUFFER_ONCE 
        /* Free GPU workspace memory if not managed globally */
        /* This releases GPU memory allocated for temporary buffers */
        workspace_memory_free( tp->_g_ws_gpu );
#endif

        /* Free device index array */
        /* This releases the array storing GPU device indices */
        if( NULL != tp->_g_cuda_device_index )
            free(tp->_g_cuda_device_index);
    }
#endif

    /* Clean up memory arenas for different data types */
    /* These arenas were used for optimized memory layout during computation */
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_tlr_dp_DEFAULT_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_tlr_dp_FULL_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_tlr_dp_UV_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_dense_tlr_dp_AR_ADT_IDX] );
    
    /* Clean up memory pools */
    /* These pools provided thread-safe memory allocation for temporary buffers */
    parsec_private_memory_fini( tp->_g_p_work );
    parsec_private_memory_fini( tp->_g_p_work_mbr );
    parsec_private_memory_fini( tp->_g_p_work_rr );

    /* Free the taskpool itself */
    /* This releases the main taskpool structure and all associated resources */
    parsec_taskpool_free(_tp);
}

/**
 * @brief Main entry point for dense TLR double precision Cholesky factorization
 * 
 * This function performs the complete Cholesky factorization of a dense Tile Low-Rank
 * matrix using double precision arithmetic. It creates a taskpool, executes the
 * factorization, and properly cleans up resources.
 * 
 * The function implements a 2-flow version that supports both dense and low-rank
 * matrix representations with automatic precision selection and device management.
 * This allows for efficient factorization of large matrices by using low-rank
 * approximations for off-diagonal blocks while maintaining full precision for
 * diagonal and near-diagonal blocks.
 * 
 * @param parsec PaRSEC context for task scheduling and execution
 * @param data Matrix data structures including dense and low-rank representations
 * @param params Algorithm parameters including precision, ranks, and execution options
 * 
 * @return 0 on successful completion
 * 
 * @note The info parameter will be set to 0 on all nodes if successful, or > 0 if
 *       the leading minor of order i of A is not positive definite, indicating
 *       that the factorization could not be completed. Info will be equal to i
 *       on the node that owns the diagonal element (i,i), and 0 on all other nodes.
 */
int potrf_L_dense_tlr_dp( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params )
{
    parsec_taskpool_t *hicma_potrf = NULL;
    
    /* Print start message if verbose mode is enabled */
    /* This provides user feedback when verbose output is requested */
    if( 0 == params->rank &&  params->verbose ) {
        fprintf(stderr, MAG "DENSE_TLR_DP start\n" RESET);
    }

    /* Create the POTRF taskpool */
    /* This sets up the task dependency graph and all necessary data structures */
    hicma_potrf = potrf_L_dense_tlr_dp_New( parsec, data, params );

    /* Execute the factorization if taskpool creation was successful */
    if( NULL != hicma_potrf ) {
        /* Add taskpool to execution context and run */
        /* This starts the parallel execution of the factorization */
        parsec_context_add_taskpool( parsec, hicma_potrf);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        
        /* Clean up resources */
        /* This ensures all allocated memory and resources are properly freed */
        potrf_L_dense_tlr_dp_Destruct( hicma_potrf );
        
        /* Synchronize task IDs for recursive DAGs if needed */
        /* Recursive DAGs require special synchronization for task ID consistency */
        if( params->HNB < params->NB )
            parsec_taskpool_sync_ids(); /*recursive DAGs are not synchronous on ids */
    }

    return 0;
}
