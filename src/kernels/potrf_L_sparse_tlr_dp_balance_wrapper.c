/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file potrf_L_sparse_tlr_dp_balance_wrapper.c
 * @brief Wrapper functions for sparse TLR (Tile Low-Rank) Cholesky factorization with balanced workload
 * 
 * This file implements wrapper functions for the sparse TLR Cholesky factorization algorithm
 * with balanced workload distribution. It provides timing instrumentation for performance
 * analysis of individual BLAS operations (POTRF, TRSM, SYRK, GEMM) within the factorization.
 * 
 * The implementation supports:
 * - Sparse matrix factorization using TLR format
 * - Performance timing for critical path analysis
 * - Memory pool management for temporary buffers
 * - Recursive and non-recursive execution paths
 */

#include "hicma_parsec.h"
#include "hicma_kernel_time.h"
#include "potrf_L_sparse_tlr_dp_balance.h"

/**
 * @brief Wrapper functions for performance timing instrumentation
 * 
 * These wrapper functions are used to measure execution time for each BLAS operation
 * (POTRF, TRSM, SYRK, GEMM) to enable critical path analysis and performance optimization.
 * Each operation has both a start wrapper and a completion wrapper for accurate timing.
 */
/**
 * @brief Wrapper function for POTRF operation start timing
 * 
 * Records the start time for POTRF (Cholesky factorization) operation to enable
 * performance measurement and critical path analysis.
 * 
 * @param es Execution stream handle
 * @param this_task POTRF task to be executed
 * @return Return value from the actual POTRF operation
 */
static int wrap_potrf(parsec_execution_stream_t * es, 
                      __parsec_potrf_L_sparse_tlr_dp_balance_potrf_dpotrf_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t *parsec_tp = (parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t*)this_task->taskpool;
    
    /* Record start time of POTRF operation for timing analysis */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->potrf_time_temp = start_time;
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;
    
    /* Execute the actual POTRF operation */
    return parsec_tp->_g_params_tlr->wrap_potrf(es, (parsec_task_t *)this_task);
}

/**
 * @brief Wrapper function for POTRF operation completion timing
 * 
 * Records the completion time for POTRF operation and calculates execution duration.
 * Updates cumulative timing statistics and optionally prints critical path timing information.
 * 
 * @param es Execution stream handle
 * @param this_task Completed POTRF task
 * @return Return value from the actual POTRF completion operation
 */
static int wrap_potrf_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_balance_potrf_dpotrf_task_t * this_task)
{
    int val;
    double end_time;
    parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t *parsec_tp = (parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t*)this_task->taskpool;
    
    /* Execute the actual POTRF completion operation */
    val = parsec_tp->_g_params_tlr->wrap_potrf_complete(es, (parsec_task_t *)this_task);
    
    /* Record completion time and update timing statistics */
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
 * @brief Wrapper function for TRSM operation start timing
 * 
 * Records the start time for TRSM (triangular solve) operation. Only records timing
 * for the first TRSM operation in each column (when m == k+1) to avoid double counting
 * in critical path analysis.
 * 
 * @param es Execution stream handle
 * @param this_task TRSM task to be executed
 * @return Return value from the actual TRSM operation
 */
static int wrap_trsm(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_balance_potrf_dtrsm_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t *parsec_tp = (parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t*)this_task->taskpool;
    
    /* Record start time for thread-level timing */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;

    /* Only record critical path timing for the first TRSM in each column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Record start time of TRSM for critical path analysis */
        parsec_tp->_g_params_tlr->trsm_time_temp = start_time;
        return parsec_tp->_g_params_tlr->wrap_trsm(es, (parsec_task_t *)this_task);
    } else {
        /* Execute TRSM without critical path timing */
        return parsec_tp->_g_params_tlr->wrap_trsm(es, (parsec_task_t *)this_task);
    }
}

/**
 * @brief Wrapper function for TRSM operation completion timing
 * 
 * Records the completion time for TRSM operation and calculates execution duration.
 * Updates cumulative timing statistics for critical path analysis and thread-level timing.
 * 
 * @param es Execution stream handle
 * @param this_task Completed TRSM task
 * @return Return value from the actual TRSM completion operation
 */
static int wrap_trsm_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_balance_potrf_dtrsm_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t *parsec_tp = (parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t*)this_task->taskpool;
    int val;
    double end_time;

    /* Handle critical path timing for first TRSM in each column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Execute TRSM completion and record critical path timing */
        val = parsec_tp->_g_params_tlr->wrap_trsm_complete(es, (parsec_task_t *)this_task);
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
        /* Execute TRSM completion without critical path timing */
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
 * @brief Wrapper function for SYRK operation start timing
 * 
 * Records the start time for SYRK (symmetric rank-k update) operation. Only records timing
 * for the first SYRK operation in each column (when m == k+1) to avoid double counting
 * in critical path analysis.
 * 
 * @param es Execution stream handle
 * @param this_task SYRK task to be executed
 * @return Return value from the actual SYRK operation
 */
static int wrap_syrk(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_balance_potrf_dsyrk_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t *parsec_tp = (parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t*)this_task->taskpool;
    
    /* Record start time for thread-level timing */
    double start_time = MPI_Wtime();
    hicma_kernel_time_record(this_task, start_time);
    parsec_tp->_g_params_tlr->gather_time_tmp[es->th_id] = start_time;
    
    /* Only record critical path timing for the first SYRK in each column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Record start time of SYRK for critical path analysis */
        parsec_tp->_g_params_tlr->syrk_time_temp = start_time;
        return parsec_tp->_g_params_tlr->wrap_syrk(es, (parsec_task_t *)this_task);
    } else {
        /* Execute SYRK without critical path timing */
        return parsec_tp->_g_params_tlr->wrap_syrk(es, (parsec_task_t *)this_task);
    }
}

/**
 * @brief Wrapper function for SYRK operation completion timing
 * 
 * Records the completion time for SYRK operation and calculates execution duration.
 * Updates cumulative timing statistics for critical path analysis and thread-level timing.
 * 
 * @param es Execution stream handle
 * @param this_task Completed SYRK task
 * @return Return value from the actual SYRK completion operation
 */
static int wrap_syrk_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_balance_potrf_dsyrk_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t *parsec_tp = (parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t*)this_task->taskpool;
    int val;
    double end_time;

    /* Handle critical path timing for first SYRK in each column */
    if(this_task->locals.m.value == this_task->locals.k.value + 1){
        /* Execute SYRK completion and record critical path timing */
        val = parsec_tp->_g_params_tlr->wrap_syrk_complete(es, (parsec_task_t *)this_task);
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
        /* Execute SYRK completion without critical path timing */
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
 * @brief Wrapper function for GEMM operation start timing
 * 
 * Records the start time for GEMM (general matrix multiply) operation for thread-level
 * timing analysis. GEMM operations are not part of the critical path timing.
 * 
 * @param es Execution stream handle
 * @param this_task GEMM task to be executed
 * @return Return value from the actual GEMM operation
 */
static int wrap_gemm(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_balance_potrf_dgemm_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t *parsec_tp = (parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t*)this_task->taskpool;
    
    /* Record start time for thread-level timing analysis */
    hicma_kernel_time_record(this_task, MPI_Wtime());
    
    /* Execute the actual GEMM operation */
    return parsec_tp->_g_params_tlr->wrap_gemm(es, (parsec_task_t *)this_task);
}

/**
 * @brief Wrapper function for GEMM operation completion timing
 * 
 * Records the completion time for GEMM operation and calculates execution duration.
 * Updates thread-level timing statistics and optionally prints debug timing information.
 * 
 * @param es Execution stream handle
 * @param this_task Completed GEMM task
 * @return Return value from the actual GEMM completion operation
 */
static int wrap_gemm_complete(parsec_execution_stream_t * es,
                      __parsec_potrf_L_sparse_tlr_dp_balance_potrf_dgemm_task_t * this_task)
{
    parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t *parsec_tp = (parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t*)this_task->taskpool;
    int val;
    double start_time = hicma_kernel_time_take(this_task, MPI_Wtime());
    
    /* Execute the actual GEMM completion operation */
    val = parsec_tp->_g_params_tlr->wrap_gemm_complete(es, (parsec_task_t *)this_task);
    
    /* Record completion time and update thread-level timing statistics */
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
 * @brief Create a new sparse TLR Cholesky factorization taskpool with balanced workload
 * 
 * This function creates and initializes a taskpool for sparse TLR (Tile Low-Rank) Cholesky
 * factorization with balanced workload distribution. It sets up timing instrumentation,
 * memory pools, and configures the execution environment for optimal performance.
 * 
 * @param parsec PaRSEC context handle
 * @param data Matrix data structures (A, Ar, Rank, Dist)
 * @param params Algorithm parameters including rank limits and matrix properties
 * @param analysis Matrix analysis results for optimization
 * 
 * @return Pointer to the created taskpool, or NULL on error
 * 
 * @note This is a non-blocking version that supports both recursive and non-recursive execution
 * @note Only supports PlasmaLower triangular matrices
 * @note Requires band_size_dense = 1 for sparse case
 */
parsec_taskpool_t*
potrf_L_sparse_tlr_dp_balance_New( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t* analysis )
{
    void** hook;
    
    /* Extract parameters for easier access */
    int rank = params->rank;
    int uplo = params->uplo;
    int hmb = params->HNB;              /* Hierarchical block size */
    int compmaxrank = params->compmaxrank;    /* Maximum rank for computation buffers */
    int storagemaxrank = params->genmaxrank;  /* Maximum rank for storage */
    
    /* Extract matrix data structures */
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;      /* Main matrix */
    parsec_tiled_matrix_t *Ar = (parsec_tiled_matrix_t *)&data->dcAr;    /* Rank matrix */
    parsec_tiled_matrix_t *Rank = (parsec_tiled_matrix_t *)&data->dcRank; /* Rank information */
    parsec_tiled_matrix_t *Dist = (parsec_tiled_matrix_t *)&data->dcDist; /* Distance matrix */

    /* Validate input arguments */
    if (uplo != PlasmaLower) {
        dplasma_error("potrf_L_sparse_tlr_dp_balance_New", 
                     "potrf_L_sparse_tlr_dp_balance only support PlasmaLower for now");
        return NULL;
    }

    /* Check rank parameter consistency */
    if(storagemaxrank > compmaxrank) {
        dplasma_error("potrf_L_sparse_tlr_dp_balance_New", 
                     "maxrank for storage larger than maxrank for buffers is not meaningful");
        return NULL;
    }

    /* Warn about potentially inefficient rank settings */
    if(storagemaxrank > (A->mb/2) && 0 == rank) {
        fprintf(stderr, RED "Warning: maxrank= %d is larger than half of block size\n" RESET, storagemaxrank);
    }

    /* Initialize info parameter */
    params->info = 0;
    
    /* Create the main taskpool for sparse TLR Cholesky factorization */
    parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t *hicma_dpotrf;
    hicma_dpotrf = parsec_potrf_L_sparse_tlr_dp_balance_new( A, Ar, Rank, Dist, params, analysis ); 

    /* Identify task class IDs for timing wrapper installation */
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

    /* Install timing wrapper functions based on execution mode */
    if( hmb < A->mb ) { 
        /* Recursive execution mode - use incarnation 0 */
        hicma_dpotrf->_g_params_tlr->wrap_potrf = hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[0].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[0].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[0].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[0].hook;
        
        /* Replace hooks with timing wrapper functions */
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[0].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[0].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[0].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[0].hook;
        *hook = &wrap_gemm;
    } else {
        /* Non-recursive execution mode - use incarnation 1 */
        hicma_dpotrf->_g_params_tlr->wrap_potrf = hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[1].hook;
        hicma_dpotrf->_g_params_tlr->wrap_trsm  = hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[1].hook;
        hicma_dpotrf->_g_params_tlr->wrap_syrk  = hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[1].hook;
        hicma_dpotrf->_g_params_tlr->wrap_gemm  = hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[1].hook;
        
        /* Replace hooks with timing wrapper functions */
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[potrf_id]->incarnations[1].hook;
        *hook = &wrap_potrf;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[trsm_id]->incarnations[1].hook;
        *hook = &wrap_trsm;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[syrk_id]->incarnations[1].hook;
        *hook = &wrap_syrk;
        hook  = (void *)&hicma_dpotrf->super.task_classes_array[gemm_id]->incarnations[1].hook;
        *hook = &wrap_gemm;
    }

    /* Install completion wrapper functions for timing analysis */
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

    /* Initialize memory pools for temporary buffers */
    hicma_dpotrf->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    
    /* Calculate workspace size for temporary buffers */
    /* Note: This is a conservative estimate based on the maximum memory requirements */
    size_t ws_worker = 
        2 * A->mb * 2 * compmaxrank   /* CU and CV temporary buffers (2*maxrk for side-by-side U matrices) */
        + 2 * A->mb                   /* QR tau vectors (qrtauA, qrtauB) */
        + compmaxrank * compmaxrank    /* QR B matrix and AcolBcolT */
        + 2 * A->mb * 2 * compmaxrank /* newU and newV matrices */
        + (2*compmaxrank) * (2*compmaxrank)    /* SVD R matrix */
        + (2*compmaxrank)              /* SVD sigma vector */
        + (2*compmaxrank);             /* SVD superb vector */
    
    parsec_private_memory_init( hicma_dpotrf->_g_p_work, ws_worker * sizeof(double) ); 

    /* Initialize rank-by-rank workspace memory pool */
    hicma_dpotrf->_g_p_work_rr = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_rr, compmaxrank * compmaxrank * sizeof(double) ); 

    /* Initialize matrix-block-rank workspace memory pool */
    hicma_dpotrf->_g_p_work_mbr = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( hicma_dpotrf->_g_p_work_mbr, A->mb * compmaxrank * sizeof(double) ); 

    /* Set priority change threshold for task scheduling optimization */
    hicma_dpotrf->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );
    if(0 == hicma_dpotrf->_g_PRI_CHANGE) {
        hicma_dpotrf->_g_PRI_CHANGE = A->nt;  /* Default to number of tiles */
    }

    /* Initialize data type arenas for efficient memory management */
    /* Full matrix arena for complete tile data */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_balance_FULL_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, A->mb, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Default arena for scalar values */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_balance_DEFAULT_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* UV arena for low-rank factor storage (U and V matrices side by side) */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_balance_UV_ADT_IDX],
            parsec_datatype_double_t, PARSEC_MATRIX_FULL,
            1, A->mb, storagemaxrank*2, A->mb,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    /* Array arena for rank and distance information */
    parsec_add2arena(&hicma_dpotrf->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_balance_AR_ADT_IDX],
            parsec_datatype_int_t, PARSEC_MATRIX_FULL,
            1, 1, 1, 1,
            PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    return (parsec_taskpool_t*)hicma_dpotrf;
}

/**
 * @brief Destructor for sparse TLR Cholesky factorization taskpool
 * 
 * Properly cleans up all allocated resources including memory pools, arenas,
 * and the taskpool itself.
 * 
 * @param _tp Taskpool to be destroyed
 */
void potrf_L_sparse_tlr_dp_balance_Destruct(parsec_taskpool_t* _tp)
{
    parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t *tp = (parsec_potrf_L_sparse_tlr_dp_balance_taskpool_t*)_tp;

    /* Clean up data type arenas */
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_balance_DEFAULT_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_balance_FULL_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_balance_UV_ADT_IDX] );
    parsec_del2arena( &tp->arenas_datatypes[PARSEC_potrf_L_sparse_tlr_dp_balance_AR_ADT_IDX] );
    
    /* Clean up memory pools */
    parsec_private_memory_fini( tp->_g_p_work );
    parsec_private_memory_fini( tp->_g_p_work_mbr );
    parsec_private_memory_fini( tp->_g_p_work_rr );

    /* Free the taskpool */
    parsec_taskpool_free(_tp);
}

/**
 * @brief Main entry point for sparse TLR Cholesky factorization with balanced workload
 * 
 * This function performs sparse TLR (Tile Low-Rank) Cholesky factorization with balanced
 * workload distribution. It creates a taskpool, executes the factorization, and cleans up
 * resources. The function supports both recursive and non-recursive execution modes.
 * 
 * @param parsec PaRSEC context handle
 * @param data Matrix data structures (A, Ar, Rank, Dist)
 * @param params Algorithm parameters including rank limits and matrix properties
 * @param analysis Matrix analysis results for optimization
 * 
 * @return 0 on success, non-zero on error
 * 
 * @note This function requires band_size_dense = 1 for sparse case
 * @note Only supports PlasmaLower triangular matrices
 * @note The factorization is performed in-place on matrix A
 */
int potrf_L_sparse_tlr_dp_balance( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t* analysis )
{
    /* Validate input parameters for sparse case */
    if( params->band_size_dense != 1 ) { 
        if( 0 == params->rank )
            fprintf(stderr, RED "SPARSE CASE ONLY SUPPORT band_size_dense = 1\n" RESET);
        exit(1);
    }   

    /* Print start message if verbose mode is enabled */
    if( 0 == params->rank && params->verbose )
        fprintf(stderr, MAG "SPARSE_TLR_DP_BALANCE start\n" RESET);

    parsec_taskpool_t *hicma_potrf = NULL;

    /* Create and initialize the sparse TLR Cholesky taskpool */
    hicma_potrf = potrf_L_sparse_tlr_dp_balance_New( parsec, data, params, analysis );

    /* Execute the factorization if taskpool creation was successful */
    if( NULL != hicma_potrf ) {
        /* Add taskpool to context and execute */
        parsec_context_add_taskpool( parsec, hicma_potrf);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        
        /* Clean up resources */
        potrf_L_sparse_tlr_dp_balance_Destruct( hicma_potrf );
        
        /* Synchronize task IDs for recursive DAGs */
        if( params->HNB < params->NB )
            parsec_taskpool_sync_ids(); /* Recursive DAGs are not synchronous on IDs */
    }

    return 0;
}
