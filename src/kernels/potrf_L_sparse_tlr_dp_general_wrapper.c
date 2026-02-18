/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file potrf_L_sparse_tlr_dp_general_wrapper.c
 * @brief Wrapper functions for sparse TLR (Tile Low-Rank) Cholesky factorization
 * 
 * This file contains wrapper functions that instrument the sparse TLR Cholesky 
 * factorization kernel to measure execution time for different BLAS operations
 * (POTRF, TRSM, SYRK, GEMM). The wrappers are used for performance analysis
 * and critical path timing in the HICMA library.
 */

#include "hicma_parsec.h"
#include "hicma_kernel_time.h"
#include "potrf_L_sparse_tlr_dp_general.h"

/**
 * @brief Wrapper functions for performance timing instrumentation
 * 
 * These wrapper functions are used to measure execution time for each BLAS operation
 * in the sparse TLR Cholesky factorization:
 * - POTRF: Cholesky factorization of diagonal blocks
 * - TRSM: Triangular solve with multiple right-hand sides
 * - SYRK: Symmetric rank-k update
 * - GEMM: General matrix-matrix multiplication
 */
/**
 * @brief Wrapper function for POTRF (Cholesky factorization) task execution
 * 
 * This function instruments the POTRF task to measure its execution time.
 * It records the start time before calling the actual POTRF implementation.
 * 
 * @param es Execution stream handle
 * @param this_task POTRF task to be executed
 * @return Return value from the actual POTRF implementation
 */
static int wrap_potrf(parsec_execution_stream_t * es, 
                      __parsec_potrf_L_sparse_tlr_dp_general_potrf_dpotrf_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_general_taskpool_t *parsec_tp = 
        (parsec_potrf_L_sparse_tlr_dp_general_taskpool_t*)this_task->taskpool;
    
    /* Record start time of POTRF operation for timing analysis */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->potrf_time_temp = start_time;
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;
    
    /* Call the actual POTRF implementation */
    return parsec_tp->_g_params_tlr->wrap_potrf(es, (parsec_task_t *)this_task);
}

/**
 * @brief Wrapper function for POTRF task completion with timing
 * 
 * This function is called when a POTRF task completes. It calculates the
 * execution time and updates the timing statistics. Optionally prints
 * critical path timing information if enabled.
 * 
 * @param es Execution stream handle
 * @param this_task Completed POTRF task
 * @return Return value from the actual POTRF completion function
 */
static int wrap_potrf_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_general_potrf_dpotrf_task_t * this_task)
{
    int val;
    double end_time;
    parsec_potrf_L_sparse_tlr_dp_general_taskpool_t *parsec_tp = 
        (parsec_potrf_L_sparse_tlr_dp_general_taskpool_t*)this_task->taskpool;
    
    /* Call the actual POTRF completion function */
    val = parsec_tp->_g_params_tlr->wrap_potrf_complete(es, (parsec_task_t *)this_task);
    
    /* Record end time and calculate execution time */
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
 * @brief Wrapper function for TRSM (triangular solve) task execution
 * 
 * This function instruments the TRSM task to measure its execution time.
 * It only records timing for the first TRSM operation in each column
 * (when m == k+1) to avoid double-counting in the critical path.
 * 
 * @param es Execution stream handle
 * @param this_task TRSM task to be executed
 * @return Return value from the actual TRSM implementation
 */
static int wrap_trsm(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_general_potrf_dtrsm_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_general_taskpool_t *parsec_tp = 
        (parsec_potrf_L_sparse_tlr_dp_general_taskpool_t*)this_task->taskpool;
    
    /* Record start time for thread-level timing */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;

    /* Only time the first TRSM operation in each column (critical path) */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Record start time of TRSM for critical path timing */
        parsec_tp->_g_params_tlr->trsm_time_temp = start_time;
        return parsec_tp->_g_params_tlr->wrap_trsm(es, (parsec_task_t *)this_task);
    } else {
        /* Skip timing for non-critical path TRSM operations */
        return parsec_tp->_g_params_tlr->wrap_trsm(es, (parsec_task_t *)this_task);
    }
}

/**
 * @brief Wrapper function for TRSM task completion with timing
 * 
 * This function is called when a TRSM task completes. It calculates the
 * execution time for critical path TRSM operations and updates timing
 * statistics. Optionally prints critical path timing information if enabled.
 * 
 * @param es Execution stream handle
 * @param this_task Completed TRSM task
 * @return Return value from the actual TRSM completion function
 */
static int wrap_trsm_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_general_potrf_dtrsm_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_general_taskpool_t *parsec_tp = 
        (parsec_potrf_L_sparse_tlr_dp_general_taskpool_t*)this_task->taskpool;
    int val;
    double end_time;

    /* Handle critical path TRSM operations (first in each column) */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Call the actual TRSM completion function */
        val = parsec_tp->_g_params_tlr->wrap_trsm_complete(es, (parsec_task_t *)this_task);
        
        /* Record end time and calculate execution time for critical path */
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
        /* Handle non-critical path TRSM operations */
        val = parsec_tp->_g_params_tlr->wrap_trsm_complete(es, (parsec_task_t *)this_task);
        end_time = MPI_Wtime();
    }

    /* Update thread-level timing statistics */
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
 * This function instruments the SYRK task to measure its execution time.
 * It only records timing for the first SYRK operation in each column
 * (when m == k+1) to avoid double-counting in the critical path.
 * 
 * @param es Execution stream handle
 * @param this_task SYRK task to be executed
 * @return Return value from the actual SYRK implementation
 */
static int wrap_syrk(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_general_potrf_dsyrk_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_general_taskpool_t *parsec_tp = 
        (parsec_potrf_L_sparse_tlr_dp_general_taskpool_t*)this_task->taskpool;
    
    /* Record start time for thread-level timing */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;
    
    /* Only time the first SYRK operation in each column (critical path) */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Record start time of SYRK for critical path timing */
        parsec_tp->_g_params_tlr->syrk_time_temp = start_time;
        return parsec_tp->_g_params_tlr->wrap_syrk(es, (parsec_task_t *)this_task);
    } else {
        /* Skip timing for non-critical path SYRK operations */
        return parsec_tp->_g_params_tlr->wrap_syrk(es, (parsec_task_t *)this_task);
    }
}

/**
 * @brief Wrapper function for SYRK task completion with timing
 * 
 * This function is called when a SYRK task completes. It calculates the
 * execution time for critical path SYRK operations and updates timing
 * statistics. Optionally prints critical path timing information if enabled.
 * 
 * @param es Execution stream handle
 * @param this_task Completed SYRK task
 * @return Return value from the actual SYRK completion function
 */
static int wrap_syrk_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_general_potrf_dsyrk_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_general_taskpool_t *parsec_tp = 
        (parsec_potrf_L_sparse_tlr_dp_general_taskpool_t*)this_task->taskpool;
    int val;
    double end_time;

    /* Handle critical path SYRK operations (first in each column) */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Call the actual SYRK completion function */
        val = parsec_tp->_g_params_tlr->wrap_syrk_complete(es, (parsec_task_t *)this_task);
        
        /* Record end time and calculate execution time for critical path */
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
        /* Handle non-critical path SYRK operations */
        val = parsec_tp->_g_params_tlr->wrap_syrk_complete(es, (parsec_task_t *)this_task);
        end_time = MPI_Wtime();
    }

    /* Update thread-level timing statistics */
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
 * @brief Wrapper function for GEMM (general matrix-matrix multiplication) task execution
 * 
 * This function instruments the GEMM task to measure its execution time.
 * GEMM operations are not part of the critical path, so only thread-level
 * timing is recorded for performance analysis.
 * 
 * @param es Execution stream handle
 * @param this_task GEMM task to be executed
 * @return Return value from the actual GEMM implementation
 */
static int wrap_gemm(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_general_potrf_dgemm_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_general_taskpool_t *parsec_tp = 
        (parsec_potrf_L_sparse_tlr_dp_general_taskpool_t*)this_task->taskpool;
    
    /* Record start time for thread-level timing (GEMM not in critical path) */
    hicma_kernel_time_record(this_task, MPI_Wtime());
    
    /* Call the actual GEMM implementation */
    return parsec_tp->_g_params_tlr->wrap_gemm(es, (parsec_task_t *)this_task);
}

/**
 * @brief Wrapper function for GEMM task completion with timing
 * 
 * This function is called when a GEMM task completes. It calculates the
 * execution time and updates thread-level timing statistics. Optionally
 * prints debug timing information if enabled.
 * 
 * @param es Execution stream handle
 * @param this_task Completed GEMM task
 * @return Return value from the actual GEMM completion function
 */
static int wrap_gemm_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_general_potrf_dgemm_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_general_taskpool_t *parsec_tp = 
        (parsec_potrf_L_sparse_tlr_dp_general_taskpool_t*)this_task->taskpool;
    int val;
    double start_time = hicma_kernel_time_take(this_task, MPI_Wtime());
    
    /* Call the actual GEMM completion function */
    val = parsec_tp->_g_params_tlr->wrap_gemm_complete(es, (parsec_task_t *)this_task);
    
    /* Record end time and calculate execution time */
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


/**
 * @brief Create a new sparse TLR Cholesky factorization taskpool with timing instrumentation
 * 
 * This function creates a new PaRSEC taskpool for sparse TLR (Tile Low-Rank) Cholesky
 * factorization with performance timing instrumentation. It sets up wrapper functions
 * to measure execution time for different BLAS operations and configures memory pools
 * and data type arenas for efficient execution.
 * 
 * @param parsec PaRSEC context handle
 * @param data Matrix data descriptors (A, Ar, Rank)
 * @param params Algorithm parameters including rank limits and matrix properties
 * @param analysis Matrix analysis results for optimization
 * 
 * @return Pointer to the created taskpool, or NULL on error
 * 
 * @note This is a non-blocking version that supports sparse 2-flow execution
 * @note Only supports PlasmaLower triangular matrices
 * @note Storage max rank must be <= computation max rank
 */
parsec_taskpool_t*
potrf_L_sparse_tlr_dp_general_New( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t* analysis )
{
    void** hook;
    int rank = params->rank;
    int uplo = params->uplo;
    int hmb = params->HNB;
    int compmaxrank = params->compmaxrank;
    int storagemaxrank = params->genmaxrank;
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
    parsec_tiled_matrix_t *Ar = (parsec_tiled_matrix_t *)&data->dcAr;
    parsec_tiled_matrix_t *Rank = (parsec_tiled_matrix_t *)&data->dcRank;

    /* Validate input arguments */
    if ((uplo != PlasmaLower)) {
        dplasma_error("potrf_L_sparse_tlr_dp_general_New", 
                      "potrf_L_sparse_tlr_dp_general only support PlasmaLower for now");
        return NULL;
    }

    /* Check rank constraints */
    if(storagemaxrank > compmaxrank) {
        dplasma_error("potrf_L_sparse_tlr_dp_general_New", 
                      "maxrank for storage larger than maxrank for buffers is not meaningful");
        return NULL;
    }

    /* Warn about potentially inefficient rank settings */
    if(storagemaxrank > (A->mb/2) && 0 == rank) {
        fprintf(stderr, RED "Warning: maxrank= %d is larger than half of block size\n" RESET, 
                storagemaxrank);
    }

    /* Initialize info parameter */
    params->info = 0;
    
    /* Create the base taskpool */
    parsec_potrf_L_sparse_tlr_dp_general_taskpool_t *hicma_dpotrf;
    hicma_dpotrf = parsec_potrf_L_sparse_tlr_dp_general_new( A, Ar, Rank, params, analysis );

    /* Find the task class IDs for each BLAS operation */
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
    if( 0 == rank && DEBUG_INFO ) 
        printf("potrf_id= %d trsm_id= %d syrk_id= %d gemm_id= %d\n",
               potrf_id, trsm_id, syrk_id, gemm_id);

    /* Set up wrapper functions based on execution mode */
    if( hmb < A->mb ) { 
        /* Recursive mode: use incarnation 0 */
        hicma_dpotrf->_g_params_tlr->wrap_potrf = 
            hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[0].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = 
            hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[0].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = 
            hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[0].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = 
            hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[0].hook;
        
        /* Install timing wrapper functions */
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[0].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[0].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[0].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[0].hook;
        *hook = &wrap_gemm;

    } else {
        /* Non-recursive mode: use incarnation 1 */
        hicma_dpotrf->_g_params_tlr->wrap_potrf = 
            hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[1].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = 
            hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[1].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = 
            hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[1].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = 
            hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[1].hook;
        
        /* Install timing wrapper functions */
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[1].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[1].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[1].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[1].hook;
        *hook = &wrap_gemm;
    }

    /* Set up completion wrapper functions for timing */
    hicma_dpotrf->_g_params_tlr->wrap_potrf_complete = 
        hicma_dpotrf->super.task_classes_array[potrf_id]->complete_execution;
    hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->complete_execution;
    *hook = &wrap_potrf_complete;

    hicma_dpotrf->_g_params_tlr->wrap_trsm_complete = 
        hicma_dpotrf->super.task_classes_array[trsm_id]->complete_execution;
    hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->complete_execution;
    *hook = &wrap_trsm_complete;

    hicma_dpotrf->_g_params_tlr->wrap_syrk_complete = 
        hicma_dpotrf->super.task_classes_array[syrk_id]->complete_execution;
    hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->complete_execution;
    *hook = &wrap_syrk_complete;

    hicma_dpotrf->_g_params_tlr->wrap_gemm_complete = 
        hicma_dpotrf->super.task_classes_array[gemm_id]->complete_execution;
    hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->complete_execution;
    *hook = &wrap_gemm_complete;

    /* Set up memory pools for temporary workspace */
    hicma_dpotrf->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    
    /* Calculate workspace size for temporary buffers */
    /* Note: This is a conservative estimate based on hicma/compute/pzpotrf.c line 96 */
    /* TODO: Optimize workspace size - not all tasks need this much memory */
    size_t ws_worker =  
        2 * A->mb * 2 * compmaxrank   // CU and CV temporary buffers (2*maxrk for side-by-side U's)
        + 2 * A->mb                   // qrtauA and qrtauB arrays
        + compmaxrank * compmaxrank   // qrb_aubut and AcolBcolT matrices
        + 2 * A->mb * 2 * compmaxrank // newU and newV matrices
        + (2*compmaxrank) * (2*compmaxrank)  // svd_rA and _rA matrices
        + (2*compmaxrank)             // sigma vector
        + (2*compmaxrank);            // superb vector
    ;
    parsec_private_memory_init( hicma_dpotrf->_g_p_work, ws_worker * sizeof(double) ); 

    /* Set up rank-by-rank workspace memory pool */
    hicma_dpotrf->_g_p_work_rr = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_rr, 
                               compmaxrank * compmaxrank * sizeof(double) ); 

    /* Set up matrix-block-rank workspace memory pool */
    hicma_dpotrf->_g_p_work_mbr = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_mbr, 
                               A->mb * compmaxrank * sizeof(double) ); 

    /* Set up priority change threshold for task scheduling */
    hicma_dpotrf->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    if(0 == hicma_dpotrf->_g_PRI_CHANGE) {
        hicma_dpotrf->_g_PRI_CHANGE = A->nt;
    }

    /* Set up data type arenas for efficient memory management */
    /* Full matrix arena for dense blocks */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_FULL_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Default arena for scalar values */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_DEFAULT_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* UV arena for low-rank factor matrices */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_UV_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, storagemaxrank*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* AR arena for rank information */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_AR_ADT_IDX],
            parsec_datatype_int_t, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    return (parsec_taskpool_t*)hicma_dpotrf;
}

/**
 * @brief Destructor for sparse TLR Cholesky factorization taskpool
 * 
 * This function properly cleans up all resources associated with the taskpool,
 * including data type arenas and memory pools.
 * 
 * @param _tp Taskpool to be destroyed
 */
void potrf_L_sparse_tlr_dp_general_Destruct(parsec_taskpool_t* _tp)
{
    parsec_potrf_L_sparse_tlr_dp_general_taskpool_t *tp = 
        (parsec_potrf_L_sparse_tlr_dp_general_taskpool_t*)_tp;

    /* Clean up data type arenas */
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_DEFAULT_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_FULL_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_UV_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_general_AR_ADT_IDX] );
    
    /* Clean up memory pools */
    parsec_private_memory_fini( tp->_g_p_work );
    parsec_private_memory_fini( tp->_g_p_work_mbr );
    parsec_private_memory_fini( tp->_g_p_work_rr );

    /* Free the taskpool */
    parsec_taskpool_free(_tp);
}


/**
 * @brief Main function for sparse TLR Cholesky factorization with timing instrumentation
 * 
 * This function performs sparse TLR (Tile Low-Rank) Cholesky factorization with
 * performance timing instrumentation. It creates a taskpool, executes the factorization,
 * and properly cleans up resources.
 * 
 * @param parsec PaRSEC context handle
 * @param data Matrix data descriptors (A, Ar, Rank)
 * @param params Algorithm parameters including rank limits and matrix properties
 * @param analysis Matrix analysis results for optimization
 * 
 * @return 0 on success
 * 
 * @note This function only supports band_size_dense = 1 for sparse case
 * @note The function is blocking and waits for completion
 */
int potrf_L_sparse_tlr_dp_general( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t* analysis )
{
    /* Validate sparse case constraints */
    if( params->band_size_dense != 1 ) {
        if( 0 == params->rank )
            fprintf(stderr, RED "SPARSE CASE ONLY SUPPORT band_size_dense = 1\n" RESET);
        exit(1);
    }

    /* Print start message if verbose mode is enabled */
    if( 0 == params->rank && params->verbose )
        fprintf(stderr, MAG "SPARSE_TLR_DP_GENERAL start\n" RESET);

    parsec_taskpool_t *hicma_potrf = NULL;

    /* Create and execute sparse TLR Cholesky factorization */
    hicma_potrf = potrf_L_sparse_tlr_dp_general_New( parsec, data, params, analysis );

    if( NULL != hicma_potrf ) {
        /* Add taskpool to context and execute */
        parsec_context_add_taskpool( parsec, hicma_potrf);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        
        /* Clean up resources */
        potrf_L_sparse_tlr_dp_general_Destruct( hicma_potrf );
        
        /* Synchronize task IDs for recursive DAGs */
        if( params->HNB < params->NB )
            parsec_taskpool_sync_ids();
    }

    return 0;
}
