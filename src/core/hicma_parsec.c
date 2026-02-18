/**
 * @file hicma_parsec.c
 * @brief Main HiCMA PaRSEC implementation file
 * 
 * This file contains the core HiCMA (Hierarchical Computations on Manycore 
 * Architectures) implementation using PaRSEC (Parallel Runtime Scheduling 
 * and Execution Controller). It provides high-level interfaces for matrix
 * operations, Cholesky factorization, and various optimization strategies.
 * 
 * Key Features:
 * - Matrix generation and compression using HiCMA algorithms
 * - Pre-analysis for Cholesky factorization optimization
 * - Multiple Cholesky factorization strategies (TLR, mixed precision, GPU)
 * - Memory management and workload balancing
 * - Performance monitoring and profiling
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"
#include "hicma_parsec_sparse_analysis.h"

/* =============================================================================
 * Configuration Constants and Debugging Options
 * ============================================================================= */

/* Enable/disable process execution time printing for performance analysis */
#define PRINT_PROCESS_EXE_TIME 0

/* Enable/disable thread execution time printing for fine-grained profiling */
#define PRINT_THREAD_EXE_TIME 0

/* =============================================================================
 * Core Function Implementations
 * ============================================================================= */

/**
 * @brief Initialize HiCMA PaRSEC environment and data structures
 * 
 * This function sets up the complete HiCMA PaRSEC environment including:
 * - Command line argument parsing and validation
 * - Parameter initialization with default values
 * - PaRSEC context creation and configuration
 * - Data structures initialization and memory allocation
 * - Decision engine setup for adaptive processing
 * - GPU workspace initialization if available
 * 
 * The initialization follows a specific sequence to ensure proper setup:
 * 1. Parse and validate command line arguments
 * 2. Initialize HiCMA parameters with system-appropriate defaults
 * 3. Set up STARSH kernels for matrix operations
 * 4. Create PaRSEC execution context
 * 5. Initialize data descriptors and GPU workspace
 * 6. Configure decision engine for adaptive tile processing
 * 
 * @param[in] argc Command line argument count
 * @param[in] argv Command line argument array
 * @param[in,out] params HiCMA parameters structure (initialized on output)
 * @param[in,out] params_kernel STARSH kernel parameters (initialized on output)
 * @param[in,out] data HiCMA data structures (initialized on output)
 * @return parsec_context_t* Initialized PaRSEC context ready for execution
 * 
 * @note This function must be called before any other HiCMA operations
 * @note GPU initialization only occurs if GPU support is available and enabled
 */
parsec_context_t * hicma_parsec_init( int argc, char ** argv,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_data_t *data) {

    parsec_context_t *parsec;

    /* Step 1: Parse and validate command line arguments */
    parse_arguments(&argc, &argv, params);

    /* Step 2: Start initialization timer for performance measurement */
    SYNC_TIME_START();

    /* Step 3: Initialize HiCMA parameters with system-appropriate default values */
    hicma_parsec_params_init(params, argv);

    /* Step 4: Initialize STARSH kernels for efficient matrix operations */
    if( hicma_parsec_kernel_init(params_kernel, params) != 0 ) {
        return NULL;
    }

    /* Step 5: Validate and adjust parameters based on system constraints */
    hicma_parsec_params_check(params);

    /* Step 6: Create and configure PaRSEC execution context */
    parsec = setup_parsec(argc, argv, params);

    /* Step 7: Display initial parameter configuration for verification */
    hicma_parsec_params_print_initial(params);

    /* Step 8: Initialize data descriptors and GPU workspace if available */
    hicma_parsec_data_init(data, params);

    /* Step 9: Initialize decision engine for adaptive tile processing strategies */
    hicma_parsec_decision_init(params);
  
    /* Step 10: Record and report initialization time */
    SYNC_TIME_PRINT(params->rank, ("hicma_parsec_init\n"));
    params->time_init = sync_time_elapsed;
  
    return parsec;
}

/**
 * @brief Generate and compress matrix using HiCMA algorithms
 * 
 * Performs matrix generation with compression using the specified kernel
 * and compression parameters. This is a key step in HiCMA that reduces
 * memory usage while maintaining numerical accuracy through low-rank
 * approximations.
 * 
 * The compression process:
 * - Generates the matrix using the specified kernel function
 * - Applies HiCMA compression algorithms to reduce memory footprint
 * - Maintains numerical accuracy within specified tolerance bounds
 * - Optimizes for the target architecture (CPU/GPU)
 * 
 * @param[in] parsec PaRSEC execution context for parallel execution
 * @param[in,out] data HiCMA data structures containing matrix information
 * @param[in] params HiCMA parameters controlling compression behavior
 * @param[in] params_kernel STARSH kernel parameters for matrix generation
 * @return int 0 on success, error code otherwise
 * 
 * @note Compression quality depends on the specified tolerance parameters
 * @note Memory usage is significantly reduced compared to dense storage
 */
int hicma_parsec_matrix_generation( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel )
{
    /* Start matrix generation timer for performance measurement */
    SYNC_TIME_START();
    
    /* Perform matrix compression using HiCMA algorithms for memory optimization */
    hicma_parsec_matrix_compress(parsec, data, params, params_kernel);

    /* Record and display generation time with compression statistics */
    SYNC_TIME_PRINT(params->rank, ("Matrix generation" "\tband_size_norm= %3d norm_global= %le\n",
                params->band_size_norm, params->norm_global));
    params->time_starsh = sync_time_elapsed;
    return 0;
}

/**
 * @brief Perform comprehensive pre-analysis of matrix before Cholesky factorization
 * 
 * This function performs comprehensive analysis of the compressed matrix
 * including rank statistics, band size optimization, memory calculations,
 * and sparse analysis. It prepares the matrix for efficient Cholesky
 * factorization by optimizing various parameters.
 * 
 * Analysis steps performed:
 * 1. Rank correctness verification (debug mode only)
 * 2. Rank statistics collection for optimization
 * 3. Band size auto-tuning for dense operations
 * 4. Memory requirement calculations
 * 5. Sparse matrix analysis for workload optimization
 * 6. Compression validation and correctness checking
 * 7. Decision engine updates with matrix characteristics
 * 8. Mixed precision conversion if enabled
 * 9. GPU warmup for optimal performance
 * 
 * @param[in] parsec PaRSEC execution context
 * @param[in,out] data HiCMA data structures
 * @param[in,out] params HiCMA parameters (may be modified during analysis)
 * @param[in] params_kernel STARSH kernel parameters
 * @param[out] analysis Matrix analysis results for optimization
 * @return int 0 on success, error code otherwise
 * 
 * @note This function is critical for achieving optimal Cholesky performance
 * @note Band size auto-tuning can significantly improve performance
 * @note Memory calculations help prevent out-of-memory errors
 */
int hicma_parsec_matrix_pre_analysis( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_matrix_analysis_t *analysis ) {

#if DEBUG_INFO
    /* Step 1: Verify rank correctness after compression (debug mode only) */
    SYNC_TIME_START();
    hicma_parsec_rank_check(parsec, (parsec_tiled_matrix_t *)&data->dcAr, params->band_size_dense);
    SYNC_TIME_PRINT(params->rank, ("Check Rank after compression\n"));
#endif

    /* Step 2: Collect rank statistics for optimization and tuning */
    SYNC_TIME_START();
    hicma_parsec_rank_stat(parsec, "init_rank_tile", (parsec_tiled_matrix_t*)&data->dcAr, params, &params->iminrk, &params->imaxrk, &params->iavgrk);
    SYNC_TIME_PRINT(params->rank, ("Gather rank after matrix compression"
                "\tmin= %d max= %d avg= %lf\n", params->iminrk, params->imaxrk, params->iavgrk));

    /* Step 3: Optimize band size for dense operations using auto-tuning */
    SYNC_TIME_START();
    hicma_parsec_band_size_dense_auto_tuning( parsec, data, params, params_kernel );
    SYNC_TIME_PRINT(params->rank, ("Band_size auto-tuning" "\tband_size_dense= %d band_size_dist= %d band_size_auto_tuning_termination= %lf\n",
                params->band_size_dense, params->band_size_dist, params->band_size_auto_tuning_termination));
    params->time_opt_band = sync_time_elapsed;

    /* Step 4: Re-gather rank statistics if band size was auto-tuned */
    if( params->auto_band && params->band_size_dense > 1 ) {
        SYNC_TIME_START();
        hicma_parsec_rank_stat(parsec, "init_rank_tile_auto_band", (parsec_tiled_matrix_t*)&data->dcAr, params, &params->iminrk_auto_band, &params->imaxrk_auto_band, &params->iavgrk_auto_band);
        SYNC_TIME_PRINT(params->rank, ("Gather rank after band_size auto-tuning" "\tmin= %d max= %d avg= %lf\n",
                    params->iminrk_auto_band, params->imaxrk_auto_band, params->iavgrk_auto_band));
    }

    /* Step 5: Calculate memory requirements for matrix allocation */
    SYNC_TIME_START();
    hicma_parsec_memory_calculation( parsec, params );
    SYNC_TIME_PRINT(params->rank, ("Memory for matrix allocatiion" "\tactual= %lf GB maxrank= %lf GB\n",
                params->memory_per_node, params->memory_per_node_maxrank));

    /* Step 6: Perform sparse matrix analysis for workload optimization */
    SYNC_TIME_START();
    hicma_parsec_sparse_analysis( parsec, data, params, analysis );
    SYNC_TIME_PRINT(params->rank, ("Matrix analysis" "\tsparse= %d band_size_dist= %d WORKLOAD_BALANCE= %d\n",
                params->sparse, params->band_size_dist, WORKLOAD_BALANCE));
    params->time_analysis = sync_time_elapsed;

    /* Step 7: Validation and correctness checking if enabled */
    if( params->check ) {
        SYNC_TIME_START();
        hicma_parsec_check_compression( parsec, data, params, params_kernel, analysis );
        SYNC_TIME_PRINT(params->rank, ("Check compress and run dense dpotrf\n"));
    }

#if DEBUG_INFO
    /* Step 8: Final rank correctness verification before Cholesky (debug mode) */
    SYNC_TIME_START();
    hicma_parsec_rank_check(parsec, (parsec_tiled_matrix_t*)&data->dcAr, params->band_size_dense);
    SYNC_TIME_PRINT(params->rank, ("Before HiCMA Check Rank\n"));
#endif

#if PRINT_RANK
    /* Step 9: Display process ID mapping for each tile (debug output) */
    hicma_parsec_process_id_print( data, params );
#endif

    /* Step 10: Update decision engine with current matrix characteristics */
    SYNC_TIME_START();
    hicma_parsec_decisions_update( parsec, data, params );
    SYNC_TIME_PRINT(params->rank, ("Decision update: %d\n", params->adaptive_decision));

    /* Step 11: Convert double precision to single precision for mixed precision operations */
#if !GENOMICS
    if( params->kind_of_cholesky == DENSE_MP_BAND
            || params->kind_of_cholesky == DENSE_SP_HP_BAND
            || params->kind_of_cholesky == DENSE_TLR_MP
            || params->kind_of_cholesky == DENSE_MP_GPU
            || params->kind_of_cholesky == DENSE_MP_GPU_FP8
            || params->kind_of_cholesky == DENSE_MP_GPU_FP8_ADAPTIVE
            || params->kind_of_cholesky == DENSE_MP_GPU_FP8_SP
      ) {
        SYNC_TIME_START();
        hicma_parsec_convert_d2s( parsec, data, params);
        SYNC_TIME_PRINT(params->rank, ("Convert D2S\n"));
    }

    /* Step 12: Calculate norm difference between double and mixed precision for validation */
    if( params->verbose > 2 && (DENSE_TLR_MP == params->kind_of_cholesky
                || DENSE_MP_GPU == params->kind_of_cholesky
                || DENSE_MP_GPU_FP8 == params->kind_of_cholesky
                || DENSE_MP_GPU_FP8_ADAPTIVE == params->kind_of_cholesky
                || DENSE_MP_GPU_FP8_SP == params->kind_of_cholesky)
            && params->band_size_dense >= params->NT ) {
        hicma_parsec_matrix_check_norm_diff( parsec, data, params, params_kernel );
    }
#endif

    /* Step 13: GPU warmup to ensure optimal performance if GPUs are available */
    if(params->gpus > 0) {
        parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
        if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
            A = (parsec_tiled_matrix_t *)&data->dcAd;
        }
        SYNC_TIME_START();
        //hicma_parsec_warmup_potrf(params->rank, params->uplo, 3872, parsec);
        hicma_parsec_potrf_L_warmup( parsec, A, params, 1);
        SYNC_TIME_PRINT(params->rank, ("Warmup\n"));
    }

    return 0;
}

/**
 * @brief Main routine for TLR (Tile Low-Rank) Cholesky factorization
 * 
 * Performs Cholesky factorization using HiCMA's tile low-rank approach.
 * This is the core computational kernel that decomposes a symmetric
 * positive definite matrix A into A = L*L^T where L is lower triangular.
 * 
 * The function supports multiple factorization strategies:
 * - DENSE_TLR_MP: Mixed precision tile low-rank Cholesky
 * - DENSE_MP_BAND: Mixed precision band-based Cholesky
 * - DENSE_SP_HP_BAND: Single precision high precision band Cholesky
 * - DENSE_MP_GPU: Mixed precision GPU-accelerated Cholesky
 * - DENSE_MP_GPU_FP8: Mixed precision GPU with FP8 support
 * - DENSE_MP_GPU_FP8_ADAPTIVE: Mixed precision GPU with FP8 support during runtime
 * - DENSE_MP_GPU_FP8_SP: Mixed precision GPU with FP8 and single precision
 * 
 * Performance optimizations include:
 * - Lookahead parameter auto-configuration
 * - Band size optimization for better performance
 * - CUDA profiling support for performance analysis
 * - MPI communication optimization recommendations
 * 
 * @param[in] parsec PaRSEC execution context for parallel execution
 * @param[in,out] data HiCMA data structures containing the matrix
 * @param[in] params HiCMA parameters controlling factorization behavior
 * @param[in] params_kernel STARSH kernel parameters
 * @return int 0 on success, >0 if factorization failed (info indicates which minor is not positive definite)
 * 
 * @note Currently only supports lower triangular matrices (PlasmaLower)
 * @note Requires single virtual process configuration (parsec->nb_vp == 1)
 * @note Lookahead parameter is auto-configured if set to -1
 * @note GPU acceleration requires CUDA support and proper configuration
 */
int hicma_parsec_potrf( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t *analysis )
{
    /* Validation: PaRSEC configuration - only single virtual process supported */
    assert( parsec->nb_vp == 1 );

    /* Validation: Currently only lower triangular matrices are supported */
    assert( params->uplo == PlasmaLower );

    /* Auto-configure lookahead parameter if not specified (-1 means auto) */
    assert( params->lookahead >= -1 );
    if( -1 == params->lookahead ) {
        params->lookahead = params->band_size_dense;
        if( 0 == params->rank && params->verbose )
            fprintf(stderr, YEL "Lookahead is not provided, set lookahead = band_size_dense = %d\n" RESET, params->lookahead);
    }

    /* Performance optimization tip for band-based algorithms */
    if( 0 == params->rank && params->band_size_dense > 1
            && !(params->band_size_dense > params->P || params->band_size_dist > params->P) ) { 
        fprintf(stderr, YEL "WARNING: band_size_dense= %d (> 1), so add flag '-- -mca runtime_comm_coll_bcast 0' at the end of command for better performance !!!\n" RESET, params->band_size_dense);
    }

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) && ENABLE_PROFILING
    /* Start CUDA profiling for performance analysis if available */
    cudaProfilerStart();
#endif

    /* Synchronize all processes and start timing for accurate measurement */
    MPI_Barrier( MPI_COMM_WORLD );
    SYNC_TIME_START();
    struct timeval tstart;
    gettimeofday(&tstart, NULL);
    params->start_time_potrf = tstart.tv_sec + tstart.tv_usec / 1.0e6;

    /* Seed skipped/zero-rank tiles; POTRF and TRSM overwrite executed tiles
     * with the precision of their final physical representation. */
    memcpy(params->decisions_storage_final, params->decisions,
            (size_t)params->MT * params->NT * sizeof(uint16_t));

    /* Execute the appropriate Cholesky factorization algorithm based on configuration */
    switch( params->kind_of_cholesky ) {
        case DENSE_TLR_MP:
            /* Mixed precision tile low-rank Cholesky factorization */
            potrf_L_dense_tlr_mp( parsec, data, params );
            break;

        case DENSE_TLR_DP:
            /* Double precision tile low-rank Cholesky */
            potrf_L_dense_tlr_dp( parsec, data, params );
            break;

        case DENSE_MP_BAND:
            /* Mixed precision band-based Cholesky */
            hicma_parsec_hsdpotrf( parsec, data, params );
            break;

        case DENSE_SP_HP_BAND:
            /* Single precision half precision band-based Cholesky */
            hicma_parsec_shpotrf( parsec, data, params );
            break;

        case DENSE_MP_GPU:
            /* Mixed precision GPU-accelerated Cholesky */
            potrf_L_dense_mp_gpu( parsec, data, params );
            break;

        case SPARSE_TLR_DP_GENERAL:
            /* Sparse tile low-rank double precision general Cholesky */
            potrf_L_sparse_tlr_dp_general( parsec, data, params, analysis );
            break;

        case SPARSE_TLR_DP_BALANCE:
            /* Sparse tile low-rank double precision balanced Cholesky */
            potrf_L_sparse_tlr_dp_balance( parsec, data, params, analysis );
            break;

        case DENSE_MP_GPU_FP8:
            /* Mixed precision GPU-accelerated Cholesky with FP8 */
            potrf_L_dense_mp_gpu_fp8( parsec, data, params );
            break;

        case DENSE_MP_GPU_FP8_SP:
            /* Mixed precision GPU-accelerated Cholesky with FP8 and single precision */
            potrf_L_dense_mp_gpu_fp8_sp( parsec, data, params );
            break;

        case DENSE_MP_GPU_FP8_ADAPTIVE: 
            /* Mixed precision GPU-accelerated Cholesky with FP8 */
            potrf_L_dense_mp_gpu_fp8_adaptive( parsec, data, params );
            break;

        default:
            /* Invalid Cholesky type specified */
            if( 0 == params->rank ) {
                fprintf(stderr, RED "ERROR: Cholesky is not called. Make sure the right parameters are used !!!\n\n");
            }
            return 1;
    }

    /* Synchronize and measure total Cholesky execution time */
    MPI_Barrier( MPI_COMM_WORLD );
    struct timeval tend;
    gettimeofday(&tend, NULL);
    double cholesky_time = tend.tv_sec + tend.tv_usec / 1.0e6 - params->start_time_potrf;
    
    /* Update best execution time if this run was faster */
    if( params->time_hicma == 0.0 || params->time_hicma > cholesky_time ) { 
        params->time_hicma = cholesky_time; 
    }
    
    /* Display comprehensive performance metrics and statistics */
    SYNC_TIME_PRINT(params->rank, ("hicma_parsec_cholesky" "\tband_size_dense= %d band_size_dist= %d "
                "lookahead= %d kind_of_problem= %d HNB= %d PxQ= %3d %-3d "
                "nb_gpus= %d NB= %4d N= %7d kind_of_cholesky= %d sparse= %d left_looking= %d "
                "nb_dense_dp= %.0lf nb_dense_sp= %.0lf nb_dense_hp= %.0lf nb_dense_fp8= %.0lf nb_low_rank_dp= %.0lf nb_low_rank_sp= %.0lf "
                "adaptive_memory %d: %14f Tflops\n",
                params->band_size_dense, params->band_size_dist, params->lookahead, params->kind_of_problem,
                params->HNB, params->P, params->Q, params->gpus, params->NB, params->N, params->kind_of_cholesky, params->sparse, params->left_looking,
                params->nb_dense_dp, params->nb_dense_sp, params->nb_dense_hp, params->nb_dense_fp8, params->nb_low_rank_dp, params->nb_low_rank_sp,
                params->adaptive_memory, params->gflops=(params->flops/1e12)/params->time_hicma));

#if PRINT_PROCESS_EXE_TIME
    /* Calculate per-process execution time statistics */
    double total_time = 0.0;
    double max_time = params->gather_time[0];
    double min_time = params->gather_time[0];
    
    for( int i = 0; i < params->cores; i++) {
        total_time += params->gather_time[i];
        if( params->gather_time[i] > max_time )
            max_time = params->gather_time[i];
        if( params->gather_time[i] < min_time )
            min_time = params->gather_time[i];
    }

    /* Display per-process execution time statistics */
    fprintf(stderr, MAG "Execution_time_each_process %d : avg= %lf max= %lf min= %lf\n" RESET, 
            parsec->my_rank, total_time/params->cores, max_time, min_time);
#endif

#if PRINT_THREAD_EXE_TIME
    /* Display per-thread execution time details */
    for( int i = 0; i < params->cores; i++ )
        fprintf(stderr, "Execution_time_each_thread %d %d : %lf\n",
                parsec->my_rank, i, params->gather_time[i]);
#endif

    /* Initialize global info variable for error reporting across all nodes */
    int ginfo = params->info;
    
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* Allocate memory for GPU error information across all tiles */
    int *gpotrf_info = calloc(params->NT, sizeof(int));
    if( params->gpus > 0) {
        memcpy( (void*)gpotrf_info, (void*)params->info_gpu, params->NT*sizeof(int) );
    }
#endif 

#if defined(PARSEC_HAVE_MPI)
    /* Perform MPI reduction only if multiple nodes are involved */
    if( params->nodes > 1 ) {
        /* Reduce error information across all processes */
        MPI_Allreduce( &params->info, &ginfo, 1, MPI_INT, MPI_MAX, (MPI_Comm)parsec->comm_ctx );
        
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        /* Reduce GPU error information across all processes if GPUs are used */
        if( params->gpus > 0) {
            MPI_Allreduce( params->info_gpu, gpotrf_info, params->NT, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        }
#endif
    }
#endif


    /* Statistics for number of GEMMs in different precision */
    if(params->verbose > 1) {
        int counter_stride = params->counter_stride;
        for( int i = 0; i < NB_DECISIONS+1; i++ ) {
            for( int j = 1; j < counter_stride; j++ ) {
                params->nb_gemms[i*counter_stride] += params->nb_gemms[i*counter_stride+j];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, params->nb_gemms, (NB_DECISIONS+1)*counter_stride, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        for( int j = 1; j < counter_stride; j++ ) {
            params->nb_datatype_conversions[0] += params->nb_datatype_conversions[j];
        }
        MPI_Allreduce(MPI_IN_PLACE, params->nb_datatype_conversions, counter_stride, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        if( 0 == params->rank ) {
            fprintf(stderr, GRN"Number_of_GEMMs: DENSE_DP= %lu DENSE_SP= %lu DENSE_HP= %lu DENSE_FP8= %lu LOW_RANK_DP= %lu LOW_RANK_SP= %lu Number_of_Datatype_Conversions= %lu\n"RESET,
                    params->nb_gemms[DENSE_DP*counter_stride], params->nb_gemms[DENSE_SP*counter_stride], params->nb_gemms[DENSE_HP*counter_stride],
                    params->nb_gemms[DENSE_FP8*counter_stride], params->nb_gemms[LOW_RANK_DP*counter_stride], params->nb_gemms[LOW_RANK_SP*counter_stride], params->nb_datatype_conversions[0]);
        }
    }

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    if( params->gpus > 0) {
        /* Process GPU error information and map to global error index */
        for(int i = 0; i< params->NT; i++) {
            if(gpotrf_info[i] > 0) {
                ginfo = gpotrf_info[i] + i * params->NB;
                break;
            }
        }
    }
    /* Clean up GPU error information array */
    free( gpotrf_info );
#endif

    params->info = ginfo;

    return ginfo;
}


/**
 * @brief Perform post-analysis of matrix after Cholesky factorization
 * 
 * This function performs various post-processing tasks including:
 * - Converting results back to double precision if mixed precision was used
 * - Gathering final rank statistics
 * - Validating factorization results
 * - Computing performance metrics and statistics
 * 
 * @param[in] parsec PaRSEC execution context
 * @param[in,out] data HiCMA data structures
 * @param[in] params HiCMA parameters
 * @param[in] params_kernel STARSH kernel parameters
 * @param[out] analysis Matrix analysis results
 * @return int 0 on success, error code otherwise
 */
int hicma_parsec_matrix_post_analysis( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_matrix_analysis_t *analysis ) {

    /* Convert single precision results back to double precision for validation */
    if( params->check
            && (params->kind_of_cholesky == DENSE_MP_BAND
            || params->kind_of_cholesky == DENSE_SP_HP_BAND
            || params->kind_of_cholesky == DENSE_TLR_MP
            || params->kind_of_cholesky == DENSE_MP_GPU
            || params->kind_of_cholesky == DENSE_MP_GPU_FP8
            || params->kind_of_cholesky == DENSE_MP_GPU_FP8_ADAPTIVE
            || params->kind_of_cholesky == DENSE_MP_GPU_FP8_SP)
      ) {
        SYNC_TIME_START();
        hicma_parsec_convert_s2d( parsec, data, params);
        SYNC_TIME_PRINT(params->rank, ("Convert S2D\n"));
    }

    if(params->verbose > 9) {
        parsec_tiled_matrix_t *decision_desc = (parsec_tiled_matrix_t *)&data->dcA;
        if( params->band_size_dense >= params->NT &&
                params->auto_band == 0 && !params->adaptive_memory ) {
            decision_desc = (parsec_tiled_matrix_t *)&data->dcAd;
        }
        get_decisions(params->decisions, (size_t)params->MT * params->NT,
                decision_desc);
        //print_decisions(params);
        if(params->adaptive_decision_runtime) {
            get_decisions(params->decisions_gemm_first,
                    (size_t)params->MT * params->NT, decision_desc);
            get_decisions(params->decisions_gemm_last,
                    (size_t)params->MT * params->NT, decision_desc);
            print_first_local_gemm_decisions(params);
            print_last_local_gemm_decisions(params);
        }
        get_decisions(params->decisions_storage_final,
                (size_t)params->MT * params->NT, decision_desc);
        print_final_storage_decisions(params);
    }

#if 0
    /* Debug code for matrix sum calculation (currently disabled) */
    double total = hicma_parsec_matrix_dsum( parsec, (parsec_tiled_matrix_t*)&data->dcAd,
            (parsec_tiled_matrix_t*)&data->dcAr, params->decisions );
    if( 0 == params->rank ) {
        fprintf(stderr, "sum of matrix %lf\n", total);
    }

    /* Debug code for matrix printing (currently disabled) */
    dplasma_dprint(parsec, params->uplo, (parsec_tiled_matrix_t *)&data->dcAd);
#endif

    /* Collect final rank statistics after Cholesky factorization */
    SYNC_TIME_START();
    hicma_parsec_rank_stat(parsec, "final_rank_tile", (parsec_tiled_matrix_t*)&data->dcAr, params, &params->fminrk, &params->fmaxrk, &params->favgrk);
    SYNC_TIME_PRINT(params->rank, ("Gather rank after Cholesky"
                "\tmin= %d max= %d avg= %lf\n", params->fminrk, params->fmaxrk, params->favgrk));

#if PRINT_RANK > 1
    /* Print detailed rank information for debugging */
    SYNC_TIME_START();
    if( 0 == params->sparse && DENSE_TLR_DP == params->kind_of_cholesky ) {
        hicma_parsec_rank_print(parsec, (parsec_tiled_matrix_t*)&data->dcRank, params->band_size_dense);
    }
    SYNC_TIME_PRINT(params->rank, ("Rank print\n"));
#endif

#if DEBUG_INFO
    /* Verify rank correctness after Cholesky factorization */
    SYNC_TIME_START();
    hicma_parsec_rank_check(parsec, (parsec_tiled_matrix_t*)&data->dcAr, params->band_size_dense);
    SYNC_TIME_PRINT(params->rank, ("Check Rank after Cholesky\n"));
#endif

    /* Perform result validation if requested */
    if( params->check ) {
        /* Validate sparse analysis results if applicable */
        if( params->sparse ) {
            SYNC_TIME_START();
            hicma_parsec_check_analysis( params, analysis );
            SYNC_TIME_PRINT(params->rank, ("Check correctness of analysis\n"));
        }

        /* Comprehensive result validation:
         * 1. Compare dense double precision vs tile low-rank results
         * 2. Verify factorization accuracy: ||L*L^T - A_original|| */
        SYNC_TIME_START();
        hicma_parsec_check_dpotrf( parsec, data, params, params_kernel, analysis );
        SYNC_TIME_PRINT(params->rank, ("Check dpotrf\n"));
    }

    /* Compute comprehensive performance statistics and operation counts */
    SYNC_TIME_START();
    hicma_parsec_cholesky_stat( params );
    SYNC_TIME_PRINT(params->rank, ("Operation_counts : total= %le op_band= %le op_offband= %le ratio_band= %lf "
                "op_critical_path= %le op_off_critical_path= %le ratio_critical_path= %lf "
                "Total_critical_path_time %lf, potrf %lf, trsm %lf, syrk %lf\n",
                (double)params->total_flops, (double)params->total_band, (double)params->total_offband, (double)params->total_band/params->total_offband,
                (double)params->total_path, (double)params->total_offpath, (double)params->total_path/params->total_offpath,
                params->critical_path_time, params->potrf_time, params->trsm_time, params->syrk_time));

    /* Display final accuracy and validation metrics */
    VERBOSE_PRINT(params->rank, params->verbose, 
            (GRN "N= %d NB= %d D= %d input_accuracy= %e result_accuracy= %e log_det_dp= %.16e log_det_mp= %.16e |log_det_dp-log_det_mp|/|log_det_dp|= %e ||A-A'||= %e <= tol*||A||= %e\n" RESET,
             params->N, params->NB, params->kind_of_problem, params->fixedacc, params->result_accuracy, params->log_det_dp, params->log_det_mp,
             fabs(params->log_det_dp-params->log_det_mp)/fabs(params->log_det_dp), params->norm_global_diff, params->fixedacc * params->norm_global));

    return 0;
}

/**
 * @brief Finalize HiCMA PaRSEC environment and cleanup resources
 * 
 * This function performs cleanup operations including:
 * - Printing final computation statistics
 * - Deallocating all memory resources
 * - Cleaning up PaRSEC context and MPI resources
 * 
 * @param[in] parsec PaRSEC execution context
 * @param[in] argc Command line argument count
 * @param[in] argv Command line argument array
 * @param[in] params HiCMA parameters
 * @param[in] params_kernel STARSH kernel parameters
 * @param[in] data HiCMA data structures
 * @param[in] analysis Matrix analysis results
 */
void hicma_parsec_fini( parsec_context_t* parsec,
        int argc, char ** argv,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_data_t *data,
        hicma_parsec_matrix_analysis_t *analysis) {

    /* Display final computation statistics and results */
    hicma_parsec_params_print_final( argc, argv, params, analysis );

    /* Deallocate all memory resources used by HiCMA */
    hicma_parsec_free_memory( parsec, data, params, params_kernel, analysis );

    /* Clean up PaRSEC context and MPI resources */
    cleanup_parsec(parsec, params);
}

/**
 * @brief Write matrix A to file in various formats
 * 
 * This function writes the matrix A to a file using either MPI-IO for distributed
 * output or standard file I/O for single-node output. The matrix can be written
 * in symmetric or general format depending on the symm parameter.
 * 
 * @param[in] parsec PaRSEC execution context
 * @param[in] filename Output filename
 * @param[in] AA Reference matrix (unused in current implementation)
 * @param[in] data HiCMA data structures containing the matrix to write
 * @param[in] params HiCMA parameters for matrix dimensions and distribution
 * @param[in] symm Flag indicating if matrix should be written in symmetric format
 */
void hicma_parsec_writeA(parsec_context_t *parsec,
        const char *filename,
        parsec_tiled_matrix_t *AA,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        int symm)
{
    /* Extract matrix A from data structures */
    parsec_tiled_matrix_t* A = (parsec_tiled_matrix_t *)&data->dcAd;
    parsec_tiled_matrix_t* X = (parsec_tiled_matrix_t *)&data->dcX;
    
    /* Extract matrix distribution parameters */
    int rank = params->rank;
    int nodes = params->nodes;
    int P = params->P;  /* Grid rows */
    int Q = params->Q;  /* Grid columns */
    int MB = A->mb;     /* Block size in rows */
    int NB = A->nb;     /* Block size in columns */
    int NT = params->NT; /* Number of tiles */
    int M = A->lm;      /* Local matrix rows */
    int N = A->ln;      /* Local matrix columns */
    
    /* Display write operation start message */
    if(rank == 0) fprintf(stderr, RED "\nWrite matrix A to file:\n" RESET);
  
#if MPIIO
    /* Use MPI-IO for distributed file writing */
    MPI_Datatype darrayA;

    /* Create MPI datatype for 2D cyclic distribution */
    Creating_MPI_Datatype_2D(nodes, rank,
            N, M, MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC,
            NB, NB, P, Q,
            MPI_DOUBLE, &darrayA);

    /* Allocate buffer for matrix data */
    double* DA = (double *) malloc( A->llm*A->lln * sizeof(double));

    /* Convert from tile format to LAPACK format */
    if(symm)
        hicma_parsec_Tile_to_Lapack_sym( parsec, (parsec_tiled_matrix_t *)A, DA, P, nodes/P);
    else
        hicma_parsec_Tile_to_Lapack( parsec, (parsec_tiled_matrix_t *)A, DA, P, nodes/P);

    /* Debug output for matrix dimensions */
    printf("\n %d, %d, %d, %d, %d, %d\n", A->llm, A->llm, M, N, MB, NB);
    
    /* Write matrix using MPI-IO */
    MPI_Writing_dfile(filename, rank, DA, A->llm*A->llm, MPI_DOUBLE, darrayA);
    
    /* Clean up allocated memory */
    parsec_data_free(DA);
#else
    /* Use standard file I/O for single-node output */
    double* DA = (double *)malloc((size_t)A->llm*(size_t)A->lln*
                                    (size_t)parsec_datadist_getsizeoftype(A->mtype));

    /* Convert from tile format to LAPACK format */
    if(symm)
        hicma_parsec_Tile_to_Lapack_sym( parsec, (parsec_tiled_matrix_t *)A, DA, P, nodes/P);
    else
        hicma_parsec_Tile_to_Lapack( parsec, (parsec_tiled_matrix_t *)A, DA, P, nodes/P);
    
    /* Write matrix to binary file */
    writeMatrixToBinaryFile(filename, N, N, DA);

    /* Clean up allocated memory */
    free(DA);
#endif
}
/**
 * @brief Write matrix B to file in various formats
 * 
 * This function writes matrix B to a file using either MPI-IO for distributed
 * output or standard file I/O for single-node output. Matrix B is typically
 * written in single precision (float) format.
 * 
 * @param[in] parsec PaRSEC execution context
 * @param[in] filename Output filename
 * @param[in] B Matrix to be written
 * @param[in] data HiCMA data structures (unused in current implementation)
 * @param[in] params HiCMA parameters for matrix dimensions and distribution
 * @param[in] symm Flag indicating if matrix should be written in symmetric format
 */
void hicma_parsec_writeB(parsec_context_t *parsec,
        const char *filename,
        parsec_tiled_matrix_t *B,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        int symm)
{
    /* Extract matrix distribution parameters */
    int rank = params->rank;
    int nodes = params->nodes;
    int P = ((parsec_matrix_block_cyclic_t *)B)->grid.rows;
    int Q = ((parsec_matrix_block_cyclic_t *)B)->grid.cols;
    int MB = B->mb;
    int NB = B->nb;
    int NT = params->NT;
    int M = B->lm;
    int N = B->ln;

    /* Display write operation start message */
    if(rank == 0) fprintf(stderr, RED "\nWrite matrix B to file:\n" RESET);
  
#if MPIIO
    /* Use MPI-IO for distributed file writing */
    MPI_Datatype darrayB;

    /* Create MPI datatype for 2D cyclic distribution with float precision */
    Creating_MPI_Datatype_2D(nodes, rank,
            M, N, MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC,
            MB, NB, P, nodes/P,
            MPI_FLOAT, &darrayB);

    /* Allocate buffer for matrix data in single precision */
    float* DB = (float *)malloc((size_t)B->llm*(size_t)B->lln*
                                    (size_t)parsec_datadist_getsizeoftype(B->mtype));

    /* Convert from tile format to LAPACK format in single precision */
    if(symm)
        hicma_parsec_Tile_to_Lapack_sym_single( parsec, (parsec_tiled_matrix_t *)B, DB, P, nodes/P);
    else
        hicma_parsec_Tile_to_Lapack_single( parsec, (parsec_tiled_matrix_t *)B, DB, P, nodes/P);
    
    /* Write matrix using MPI-IO with float precision */
    MPI_Writing_sfile(filename, rank, DB, B->llm*B->lln, MPI_FLOAT, darrayB);
    
    /* Clean up allocated memory */
    parsec_data_free(DB);
#else
    /* Use standard file I/O for single-node output */
    float* DB = (float *)malloc((size_t)B->llm*(size_t)B->lln*
                                    (size_t)parsec_datadist_getsizeoftype(B->mtype));

    /* Convert from tile format to LAPACK format in single precision */
    if(symm)
        hicma_parsec_Tile_to_Lapack_sym_single( parsec, (parsec_tiled_matrix_t *)B, DB, P, nodes/P);
    else
        hicma_parsec_Tile_to_Lapack_single( parsec, (parsec_tiled_matrix_t *)B, DB, P, nodes/P);

    /* Write matrix directly to binary file */
    FILE *FY;
    FY = fopen(filename,"wb");
    fwrite(DB,sizeof(float),B->llm*B->lln,FY);
    fclose(FY);

    /* Clean up allocated memory */
    free(DB);
#endif
}

/**
 * @brief Generate kernel matrix using HiCMA algorithms
 * 
 * This function generates a kernel matrix by performing matrix operations including:
 * - Matrix allocation and data generation
 * - Vector sum calculations
 * - SYRK operations for kernel matrix construction
 * - Matrix addition operations
 * 
 * @param[in] parsec PaRSEC execution context
 * @param[in,out] data HiCMA data structures
 * @param[in] params HiCMA parameters for matrix dimensions and operations
 */
void hicma_parsec_kernel_matrix(parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params){

    /* Select appropriate matrix storage based on band size and memory configuration */
    parsec_tiled_matrix_t *AA = (parsec_tiled_matrix_t *)&data->dcA;
    if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
        AA = (parsec_tiled_matrix_t *)&data->dcAd;
    }

    /* Allocate distributed symmetric matrix storage */
    SYNC_TIME_START();
    parsec_dist_allocate_sym(parsec, (parsec_tiled_matrix_t *)AA, params);

    /* Initialize matrix A for genotype data with block-cyclic distribution */
    parsec_matrix_block_cyclic_t A;
    parsec_matrix_block_cyclic_init(&A, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
            params->rank, ((params->nsnp<params->NB)? params->nsnp:params->NB), params->NB, params->nsnp, params->N, 0, 0,
            params->nsnp, params->N, params->P, (params->nodes)/(params->P), 1, 1, 0, 0);

    /* Initialize matrix C for intermediate calculations with block-cyclic distribution */
    parsec_matrix_block_cyclic_t C;
    parsec_matrix_block_cyclic_init(&C, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
            params->rank, 1, params->NB, 1, params->N, 0, 0,
            1, params->N, params->P, params->nodes/params->P, 1, 1, 0, 0);

    /* Generate genotype data matrix A */
    parsec_genotype_generator(parsec, (parsec_tiled_matrix_t *)&A, params, 1); 
    SYNC_TIME_PRINT(params->rank, ("memory allocate and data generation\n"));

    /* Calculate vector sum of matrix elements */
    SYNC_TIME_START();
    float *vec = (float *)malloc(sizeof(float)*params->N);
    
#if GENOMICS_ALLOCATE_SP
    /* Allocate memory for matrix C and perform square sum vector calculation */
    C.mat = parsec_data_allocate((size_t)C.super.nb_local_tiles *
            (size_t)C.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(C.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&C, "C");

    /* Compute square sum vector using matrix C as intermediate storage */
    hicma_parsec_sqr_sum_vec(parsec, (parsec_tiled_matrix_t *)&A, (parsec_tiled_matrix_t *)&C);
    parsec_copy_vector_bcdd(parsec, (parsec_tiled_matrix_t *)&C, vec);

    /* Clean up matrix C resources */
    parsec_data_free(C.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&C);
#else
    /* Direct matrix sum vector calculation */
    hicma_parsec_matrix_sum_vec(parsec, (parsec_tiled_matrix_t *)&A, params, vec);
#endif
    SYNC_TIME_PRINT(params->rank, ("sqr_sum_vec\n"));

#if 0
    for(int i = 0; i < params->N; i++) {
        if(vec[i] != vec1[i]) printf("ERROR %d %f %f\n", i, vec[i], vec1[i]);
    } 
#endif

    //float totalA1 = hicma_parsec_matrix_ssum( parsec, (parsec_tiled_matrix_t *)&A,
    //       (parsec_tiled_matrix_t*)&data->dcAr, params->decisions );
    //if( 0 == params->rank ) {
    //    fprintf(stderr, "sum of totalA1 matrix %e\n", totalA1);
    // }

#if GENOMICS_ALLOCATE_SP
    SYNC_TIME_START();
    parsec_convert_s2i(parsec, (parsec_tiled_matrix_t *)&A, 8);
    SYNC_TIME_PRINT(params->rank, ("s2i\n"));
#endif

    if(params->gpus > 0) {
        SYNC_TIME_START();
        hicma_parsec_potrf_L_warmup( parsec, (parsec_tiled_matrix_t *)&A, params, 0);
        hicma_parsec_potrf_L_warmup( parsec, (parsec_tiled_matrix_t *)AA, params, 1);
        SYNC_TIME_PRINT(params->rank, ("warmup\n"));
    }

    struct timeval tstart;
    struct timeval tend;
    for(int i = 0; i < params->nruns; i++) {
        /* Starting time */
        MPI_Barrier( MPI_COMM_WORLD );
        SYNC_TIME_START();
        gettimeofday(&tstart, NULL);
        params->start_time_syrk = tstart.tv_sec + tstart.tv_usec / 1.0e6;

#if 1
        hicma_parsec_syrk(parsec,
                PlasmaTrans, PlasmaNoTrans,
                -2.0, (parsec_tiled_matrix_t *)&A,
                (parsec_tiled_matrix_t *)&A,
                0.0,  (parsec_tiled_matrix_t *)AA, params, data);
#else
    dplasma_isyrk(parsec, params->uplo, PlasmaTrans,
               -2.0,
               (parsec_tiled_matrix_t *)&A,
               0.0,
               (parsec_tiled_matrix_t *)AA, 8,
               params, data);
#endif
      //parsec_print_cm_sym(parsec, AA,  params->nodes,  params->P, params->Q);

        /* Timer ends */
        MPI_Barrier( MPI_COMM_WORLD );
        gettimeofday(&tend, NULL);
        double isyrk_time = tend.tv_sec + tend.tv_usec / 1.0e6 - params->start_time_syrk;
        double flops = (FLOPS_DSYRK(params->nsnp, params->N) * 1e-9)/isyrk_time;
        SYNC_TIME_PRINT(params->rank, ("hicma_parsec_syrk nodes %d gpus %d N %d SNP %d lookahead %d\t : %lf Gflop/s P %d Q %d NB %d\n",
                    params->nodes, params->gpus, params->N, params->nsnp, params->lookahead, flops, params->P, params->Q, params->NB));
    }

    //parsec_convert_i2s(parsec, (parsec_tiled_matrix_t *)&A, 8);

    //parsec_data_free(A.mat);
    parsec_memory_free_tile(parsec, (parsec_tiled_matrix_t*)&A, params, 0); 
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&A);

    /* Starting time */
    MPI_Barrier( MPI_COMM_WORLD );
    SYNC_TIME_START();
    gettimeofday(&tstart, NULL);
    params->start_time_kernel_add = tstart.tv_sec + tstart.tv_usec / 1.0e6;

    hicma_parsec_matrix_add(parsec, (parsec_tiled_matrix_t *)AA, vec, params, data);

    //parsec_print_cm_sym(parsec, AA,  params->nodes,  params->P, params->Q);

    MPI_Barrier( MPI_COMM_WORLD );
    gettimeofday(&tend, NULL);
    double iadd_kernel_time = tend.tv_sec + tend.tv_usec / 1.0e6 - params->start_time_kernel_add;

    SYNC_TIME_PRINT(params->rank, ("hicma_parsec_kernel_add \n"));

    /*float totalA2 = hicma_parsec_matrix_ssum( parsec, (parsec_tiled_matrix_t *)&A,
      (parsec_tiled_matrix_t*)&data->dcAr, params->decisions );
      if( 0 == params->rank ) {
      fprintf(stderr, "sum of totalA2 matrix %e\n", totalA2);
      }*/
}


void hicma_parsec_kernel_matrix_file(parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params){
    
    char *gene_file = params->mesh_file; 

    parsec_tiled_matrix_t *AA = (parsec_tiled_matrix_t *)&data->dcA;
    if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
        AA = (parsec_tiled_matrix_t *)&data->dcAd;
    }

    SYNC_TIME_START();
    parsec_dist_allocate_sym(parsec, (parsec_tiled_matrix_t *)AA, params);

    parsec_matrix_block_cyclic_t A;
    parsec_matrix_block_cyclic_init(&A, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
            params->rank, ((params->nsnp<params->NB)? params->nsnp:params->NB), params->NB, params->nsnp, params->N, 0, 0,
            params->nsnp, params->N, params->P, (params->nodes)/(params->P), 1, 1, 0, 0);

    parsec_matrix_block_cyclic_t C;
    parsec_matrix_block_cyclic_init(&C, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
            params->rank, 1, params->NB, 1, params->N, 0, 0,
            1, params->N, params->P, params->nodes/params->P, 1, 1, 0, 0);



#if DISTFILE
    /* Distributed Reading */

    parsec_matrix_block_cyclic_t genotype_desc;

    int SNPS = params->nsnp;
    int CHK = params->order;

    dist_read(parsec, params->mesh_file, &genotype_desc, SNPS, CHK, params);

    parsec_dist_allocate(parsec, (parsec_tiled_matrix_t *)&A, params);

    parsec_redistribute(parsec,(parsec_tiled_matrix_t *) &genotype_desc, (parsec_tiled_matrix_t *)&A,  params->nsnp, params->N, 0, 0, 0, 0);

    #if PREDICTION &&MPIIO 

        parsec_tiled_matrix_t *B = (parsec_tiled_matrix_t *)&data->dcB;
        if(params->rank == 0) fprintf(stderr, RED "\nSTART READING B:\n" RESET); 
        MPI_Datatype darrayB;
        Creating_MPI_Datatype_2D(params->nodes, params->rank,
                    B->lm, B->ln, MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC,
                    B->mb, B->nb,  params->P, params->nodes/params->P,
                    MPI_FLOAT, &darrayB);

        float *phenoB=(float *)malloc((size_t)B->llm*(size_t)B->lln*
                        (size_t)parsec_datadist_getsizeoftype(B->mtype));
        
        if (phenoB == NULL ){
                printf("\n malloc() B failed \n");
                fflush(stdin);
                if (params && params->mpi_initialized_by_hicma) {
                    int finalized = 0;
                    MPI_Finalized(&finalized);
                    if (!finalized) MPI_Finalize();
                }
        	}
        
        MPI_Reading_sfile(params->pheno_file, params->rank, phenoB, B->llm*B->lln, MPI_FLOAT, darrayB);
        
        hicma_parsec_Lapack_to_Tile_Single( parsec, (parsec_tiled_matrix_t *)B, phenoB, params->P, params->nodes/params->P);
        
        free(phenoB);

    #elif  PREDICTION && !MPIIO
        if(params->rank == 0) fprintf(stderr, RED "\nSTART READING B:\n" RESET); 
        //params
        parsec_tiled_matrix_t *B = (parsec_tiled_matrix_t *)&data->dcB;

        float *phenoB=(float *)malloc((size_t)B->llm*(size_t)B->lln*
                        (size_t)parsec_datadist_getsizeoftype(B->mtype));

        printf("\n %d, %d\n", params->N, params->RHS); 
        if(params->rank == 0) fprintf(stderr, RED "\nSTART READING B:\n" RESET); 
        read_file(params->pheno_file, params->RHS, params->N, phenoB);
        
        hicma_parsec_Lapack_to_Tile_Single( parsec, (parsec_tiled_matrix_t *)B, phenoB, params->P, params->nodes/params->P);
            
        //parsec_print_cm(parsec, (parsec_tiled_matrix_t *)B,  nodes, P, Q);
        parsec_data_free(phenoB);

    
    #endif
#elif SINGLEFILE

    parsec_dist_allocate(parsec, (parsec_tiled_matrix_t *)&A, params);


    float* geneA = (float *) parsec_data_allocate((size_t)A.super.nb_local_tiles *
            (size_t)A.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(A.super.mtype));   
    parsec_data_collection_set_key((parsec_data_collection_t*)&A, "geneA"); 

    if (geneA == NULL){
        printf("\n geneA.mat parsec_data_allocate failed %d\n", __LINE__);
        fflush(stdin);
        if (params && params->mpi_initialized_by_hicma) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) MPI_Finalize();
        }
    }
        
    if(params->rank == 0) fprintf(stderr, RED "\nRead A from file:\n" RESET);

    read_file(gene_file, params->N, params->nsnp, geneA); //This is only for single precsion 
        
    hicma_parsec_Lapack_to_Tile_Single( parsec, (parsec_tiled_matrix_t *)&A, geneA, params->P, params->Q); //This is only for single precsion 
    parsec_data_free(geneA);

    #if PREDICTION 
            //params
        parsec_tiled_matrix_t *B = (parsec_tiled_matrix_t *)&data->dcB;

        float *phenoB=(float *)malloc((size_t)B->llm*(size_t)B->lln*
                        (size_t)parsec_datadist_getsizeoftype(B->mtype));

        printf("\n %d, %d\n", params->N, params->RHS); 
        if(params->rank == 0) fprintf(stderr, RED "\nSTART READING B:\n" RESET); 
        read_file(params->pheno_file, params->RHS, params->N, phenoB);
        
        hicma_parsec_Lapack_to_Tile_Single( parsec, (parsec_tiled_matrix_t *)B, phenoB, params->P, params->nodes/params->P);
            
        //parsec_print_cm(parsec, (parsec_tiled_matrix_t *)B,  nodes, P, Q);
        parsec_data_free(phenoB);

    #endif

#else

    parsec_genotype_generator(parsec, (parsec_tiled_matrix_t *)&A, params, 1); 
    SYNC_TIME_PRINT(params->rank, ("memory allocate and data generation\n"));

#endif

    //parsec_print_cm(parsec, &A,  params->nodes,  params->P, params->Q);
    
    SYNC_TIME_START();
    float *vec = (float *)malloc(sizeof(float)*params->N);
#if GENOMICS_ALLOCATE_SP
    C.mat = parsec_data_allocate((size_t)C.super.nb_local_tiles *
            (size_t)C.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(C.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&C, "C");

    hicma_parsec_sqr_sum_vec(parsec, (parsec_tiled_matrix_t *)&A, (parsec_tiled_matrix_t *)&C);
    parsec_copy_vector_bcdd(parsec, (parsec_tiled_matrix_t *)&C, vec);

    parsec_data_free(C.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&C);
#else
    hicma_parsec_matrix_sum_vec(parsec, (parsec_tiled_matrix_t *)&A, params, vec);
#endif
    SYNC_TIME_PRINT(params->rank, ("sqr_sum_vec\n"));

#if 0
    for(int i = 0; i < params->N; i++) {
        if(vec[i] != vec1[i]) printf("ERROR %d %f %f\n", i, vec[i], vec1[i]);
    } 
#endif

#if 0
    for(int i = 0; i < params->N; i++) {
        printf("%f ", vec[i]);
    }
#endif
    //float totalA1 = hicma_parsec_matrix_ssum( parsec, (parsec_tiled_matrix_t *)&A,
    //       (parsec_tiled_matrix_t*)&data->dcAr, params->decisions );
    //if( 0 == params->rank ) {
    //    fprintf(stderr, "sum of totalA1 matrix %e\n", totalA1);
    // }

#if GENOMICS_ALLOCATE_SP
    SYNC_TIME_START();
    parsec_convert_s2i(parsec, (parsec_tiled_matrix_t *)&A, 8);
    SYNC_TIME_PRINT(params->rank, ("s2i\n"));
#endif
    #if 0
    if(params->gpus > 0) {
        SYNC_TIME_START();
        hicma_parsec_potrf_L_warmup( parsec, (parsec_tiled_matrix_t *)&A, params, 0);
        hicma_parsec_potrf_L_warmup( parsec, (parsec_tiled_matrix_t *)AA, params, 1);
        SYNC_TIME_PRINT(params->rank, ("warmup\n"));
    }
    #endif

    struct timeval tstart;
    struct timeval tend;
    for(int i = 0; i < params->nruns; i++) {
        /* Starting time */
        MPI_Barrier( MPI_COMM_WORLD );
        SYNC_TIME_START();
        gettimeofday(&tstart, NULL);
        params->start_time_syrk = tstart.tv_sec + tstart.tv_usec / 1.0e6;

#if 1
        hicma_parsec_syrk(parsec,
                PlasmaTrans, PlasmaNoTrans,
                -2.0, (parsec_tiled_matrix_t *)&A,
                (parsec_tiled_matrix_t *)&A,
                0.0,  (parsec_tiled_matrix_t *)AA, params, data);
#else
    dplasma_isyrk(parsec, params->uplo, PlasmaTrans,
               -2.0,
               (parsec_tiled_matrix_t *)&A,
               0.0,
               (parsec_tiled_matrix_t *)AA, 8,
               params, data);
#endif

        /* Timer ends */
        MPI_Barrier( MPI_COMM_WORLD );
        gettimeofday(&tend, NULL);
        double isyrk_time = tend.tv_sec + tend.tv_usec / 1.0e6 - params->start_time_syrk;
        double flops = (FLOPS_DSYRK(params->nsnp, params->N) * 1e-9)/isyrk_time;
        SYNC_TIME_PRINT(params->rank, ("hicma_parsec_syrk nodes %d gpus %d N %d SNP %d lookahead %d\t : %lf Gflop/s P %d Q %d NB %d\n",
                    params->nodes, params->gpus, params->N, params->nsnp, params->lookahead, flops, params->P, params->Q, params->NB));
    }

    //parsec_print_cm_sym(parsec, AA,  params->nodes,  params->P, params->Q);
    

    //parsec_convert_i2s(parsec, (parsec_tiled_matrix_t *)&A, 8);

    //parsec_data_free(A.mat);
    parsec_memory_free_tile(parsec, (parsec_tiled_matrix_t*)&A, params, 0); 
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&A);

    /* Starting time */
    MPI_Barrier( MPI_COMM_WORLD );
    SYNC_TIME_START();
    gettimeofday(&tstart, NULL);
    params->start_time_kernel_add = tstart.tv_sec + tstart.tv_usec / 1.0e6;

    hicma_parsec_matrix_add(parsec, (parsec_tiled_matrix_t *)AA, vec, params, data);

   // parsec_print_cm_sym(parsec, AA,  params->nodes,  params->P, params->Q);

    MPI_Barrier( MPI_COMM_WORLD );
    gettimeofday(&tend, NULL);
    double iadd_kernel_time = tend.tv_sec + tend.tv_usec / 1.0e6 - params->start_time_kernel_add;

    SYNC_TIME_PRINT(params->rank, ("hicma_parsec_kernel_add \n"));

    /*float totalA2 = hicma_parsec_matrix_ssum( parsec, (parsec_tiled_matrix_t *)&A,
      (parsec_tiled_matrix_t*)&data->dcAr, params->decisions );
      if( 0 == params->rank ) {
      fprintf(stderr, "sum of totalA2 matrix %e\n", totalA2);
      }*/
}

void hicma_kernal_matrix(parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        parsec_tiled_matrix_t *AA,
        hicma_parsec_params_t *params)
{
    int rank = params->rank;
    int nodes = params->nodes;
    int P = params->P;
    int Q = params->Q;
    int NT = params->NT;
    int NB = params->NB;
    int N = params->N;
    int band_size_dense = params->band_size_dense;
    int uplo = params->uplo;
    int maxrank = params->maxrank;
    int verbose = params->verbose;
    double tol = params->fixedacc;
    int fixedrk = params->fixedrk;
    char *gene_file = params->mesh_file; 
    int SNP = params->nsnp;
    int kernel = params->rbf_kernel;
    double radius = params->radius;
    double add_diag = params->add_diag;


    parsec_matrix_block_cyclic_t A;
    parsec_matrix_block_cyclic_t C;
    parsec_matrix_block_cyclic_t T;
    parsec_tiled_matrix_t *B = (parsec_tiled_matrix_t *)&data->dcB;

    parsec_matrix_block_cyclic_init(&C, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
            rank, NB, NB, N, N, 0, 0,
            N, N, P, nodes/P, 1, 1, 0, 0);
    C.mat = parsec_data_allocate((size_t)C.super.nb_local_tiles *
            (size_t)C.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(C.super.mtype));   
    parsec_data_collection_set_key((parsec_data_collection_t*)&C, "C");    

    parsec_matrix_block_cyclic_init(&T, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
            rank, NB, NB, N, N, 0, 0,
            N, N, P, nodes/P, 1, 1, 0, 0);
    T.mat = parsec_data_allocate((size_t)T.super.nb_local_tiles *
            (size_t)T.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(T.super.mtype));   
    parsec_data_collection_set_key((parsec_data_collection_t*)&T, "T"); 

    parsec_matrix_block_cyclic_init(&A, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
            rank, ((SNP<NB)? SNP: NB), NB, SNP, N, 0, 0,
            SNP, N, P, nodes/P, 1, 1, 0, 0);
            
    A.mat = parsec_data_allocate((size_t)A.super.nb_local_tiles *
            (size_t)A.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(A.super.mtype));   
    parsec_data_collection_set_key((parsec_data_collection_t*)&A, "A"); 
    //printf("\nData sizes %d %d", A.super.llm, A.super.lln);
    
    
    if(params->kind_of_problem==14){
        if(rank == 0) fprintf(stderr, RED "\nSTART GENOMICSRATING A:\n" RESET);
        parsec_genotype_generator(parsec, (parsec_tiled_matrix_t *)&A, params, 1); 
    }
    else{
        if(rank == 0) fprintf(stderr, RED "\nSTART READING A:\n" RESET);
        #if MPIIO
            MPI_Datatype darrayA;
                Creating_MPI_Datatype_2D(nodes, rank,
                    A.super.lm, A.super.ln, MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC,
                    A.super.mb, A.super.nb, P, nodes/P,
                    MPI_FLOAT, &darrayA);
        
            float *sgeneA  = (float *) malloc(A.super.llm*A.super.lln*sizeof(float));
            double *dgeneA  = (double *)  malloc(A.super.llm*A.super.lln*sizeof(double));
        
            //parsec_data_allocate((size_t)A.super.llm*(size_t)A.super.lln*
                   // (size_t)parsec_datadist_getsizeoftype(A.super.mtype));
        
            if (sgeneA == NULL || dgeneA == NULL ){
                printf("\n malloc() A failed \n");
                fflush(stdin);
                if (params && params->mpi_initialized_by_hicma) {
                    int finalized = 0;
                    MPI_Finalized(&finalized);
                    if (!finalized) MPI_Finalize();
                }
        	}
        
            MPI_Reading_sfile(gene_file, rank, sgeneA, A.super.llm*A.super.lln, MPI_FLOAT, darrayA);
        
            for(int i =0;i<A.super.llm*A.super.lln;i++){
                dgeneA[i] = (double) sgeneA[i];
            }
        
            hicma_parsec_Lapack_to_Tile( parsec, (parsec_tiled_matrix_t *)&A, dgeneA, P, nodes/P);
            //parsec_print_cm( parsec, (parsec_tiled_matrix_t *)&A, rank, P, nodes/P);
        
            free(sgeneA);
            parsec_data_free(dgeneA);
        
        #else
        float *geneA = (float*) parsec_data_allocate((size_t)SNP*N*
                    (size_t)parsec_datadist_getsizeoftype(A.super.mtype));
        
        if (geneA == NULL){
                printf("\n malloc() A failed \n");
                fflush(stdin);
                if (params && params->mpi_initialized_by_hicma) {
                    int finalized = 0;
                    MPI_Finalized(&finalized);
                    if (!finalized) MPI_Finalize();
                }
        }
        
        if(rank == 0) fprintf(stderr, RED "\nRead A from file:\n" RESET);
        read_file(gene_file, N, SNP, geneA);
        
        hicma_parsec_Lapack_to_Tile_Single( parsec, (parsec_tiled_matrix_t *)&A, geneA, P, Q);
        free(geneA);
        #endif

    }

    if(rank == 0) fprintf(stderr, RED "\nSTART dplasma_isyrk:\n" RESET);
    //SYRK AA = A'*A
    double alpha = 1.0, beta = 0.0;
    
    /*dplasma_dsyrk(parsec, uplo, PlasmaTrans,
                        1.0, (parsec_tiled_matrix_t *)&A,
                       0.0,  (parsec_tiled_matrix_t *)AA);*/

    /* Convert matrices to integer precision for intermediate computations */
    parsec_convert_d2i(parsec, (parsec_tiled_matrix_t *)&A, 8);
    parsec_convert_d2i(parsec, (parsec_tiled_matrix_t *)AA, 32);

    /* Perform symmetric rank-k update: AA = A * A^T */
    dplasma_isyrk(parsec, uplo, PlasmaTrans, 
               1.0,
               (parsec_tiled_matrix_t *)&A,
               0.0,
               (parsec_tiled_matrix_t *)AA, 8, 
               params, data);

    /* Convert back to double precision for subsequent operations */
    parsec_convert_i2d(parsec, (parsec_tiled_matrix_t *)&A, 8);
    parsec_convert_i2d(parsec, (parsec_tiled_matrix_t *)AA, 32);

    /* Replicate matrix AA to rows for distributed computation */
    hicma_parsec_replicate_to_rows(parsec, (parsec_tiled_matrix_t *)AA, (parsec_tiled_matrix_t *)&C);
    hicma_parsec_replicate_to_rows(parsec, (parsec_tiled_matrix_t *)AA, (parsec_tiled_matrix_t *)&T);

    /* Compute T = -2*AA + T using matrix multiplication */
    dplasma_dgemm(parsec, PlasmaTrans, PlasmaNoTrans, -2.0, 
                    (parsec_tiled_matrix_t *)&A, (parsec_tiled_matrix_t *)&A, 1.0, (parsec_tiled_matrix_t *)&T);
    
    /* Update C = C' + T using matrix addition */
    dplasma_dgeadd(parsec, PlasmaTrans, 1.0, (parsec_tiled_matrix_t *)&T, 1.0, (parsec_tiled_matrix_t *)&C);

    /* Notify start of kernel calculations */
    if(rank == 0) fprintf(stderr, RED "\nSTART KERNEL CAL:\n" RESET);

    /* Apply kernel function to compute AA = kernel(sqrt(C)) */
    /* Note: Currently commented out - uncomment to enable kernel computation */
    //hicma_parsec_sqrt_kernel(parsec,(parsec_tiled_matrix_t *)&C, (parsec_tiled_matrix_t *)AA, radius, add_diag, kernel);
    //parsec_print_cm_sym(parsec, (parsec_tiled_matrix_t *)AA,  nodes, P, Q);

    /* Clean up temporary matrices to free memory */
    parsec_data_free(C.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&C);

    parsec_data_free(T.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&T);

    parsec_data_free(A.mat);
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&A);

    /* Handle different problem types for right-hand side vector B */
    if(params->kind_of_problem==14){
        /* Problem type 14: Generate random right-hand side vector B */
        #if PREDICTION || CHECKSOLVE
        if(rank == 0) fprintf(stderr, RED "\nSTART GENOMICSRATING B:\n" RESET);
        int Bseed = 2873;    /* Fixed seed for reproducible results */
        dplasma_dplrnt( parsec, 0, (parsec_tiled_matrix_t *)B,  Bseed);   
        #endif 
    }
    else{
        /* Problem types other than 14: Read right-hand side from file */
        #if MPIIO && PREDICTION
        /* Use MPI I/O for reading right-hand side vector B */
        if(rank == 0) fprintf(stderr, RED "\nSTART READING B:\n" RESET); 
        
        /* Create MPI datatype for 2D cyclic distribution */
        MPI_Datatype darrayB;
        Creating_MPI_Datatype_2D(nodes, rank,
            B->lm, B->ln, MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC,
            B->mb, B->nb,  P, nodes/P,
            MPI_FLOAT, &darrayB);
        
        /* Allocate memory for single and double precision B vectors */
        float *sphenoB  = (float *) malloc(B->llm*B->lln*sizeof(float));
        double *dphenoB  = (double *) parsec_data_allocate((size_t)B->llm*(size_t)B->lln*
                (size_t)parsec_datadist_getsizeoftype(B->mtype));
        
        /* Check memory allocation success */
        if (dphenoB == NULL || sphenoB == NULL ){
            printf("\n malloc() B failed \n");
            fflush(stdin);
            if (params && params->mpi_initialized_by_hicma) {
                int finalized = 0;
                MPI_Finalized(&finalized);
                if (!finalized) MPI_Finalize();
            }
        }
        
        /* Read single precision data from file using MPI I/O */
        MPI_Reading_sfile(params->pheno_file, rank, sphenoB, B->llm*B->lln, MPI_FLOAT, darrayB);
        
        /* Convert single precision to double precision for computation */
        for(int i =0;i<B->llm*B->lln;i++)
            dphenoB[i]=(double)sphenoB[i];
        
        /* Convert from LAPACK format to tiled format for PaRSEC */
        hicma_parsec_Lapack_to_Tile( parsec, (parsec_tiled_matrix_t *)B, dphenoB, P, nodes/P);
        
        /* Clean up allocated memory */
        parsec_data_free(dphenoB);
        free(sphenoB);
        
        #elif PREDICTION
        /* Use standard file I/O for reading right-hand side vector B */
        if(rank == 0) fprintf(stderr, RED "\nSTART READING RHS:\n" RESET);
        if(params->kind_of_problem==15){
            /* Problem type 15: Read specific RHS format */
            double *phenoB=(double *)malloc((size_t)B->llm*(size_t)B->lln*
                                            (size_t)parsec_datadist_getsizeoftype(B->mtype));
            printf("\n %d, %d\n", params->N, params->RHS); 
            
            /* Read RHS data from file */
            read_file(params->pheno_file, params->RHS, params->N, phenoB);
        
            /* Convert from LAPACK format to tiled format for PaRSEC */
            hicma_parsec_Lapack_to_Tile( parsec, (parsec_tiled_matrix_t *)B, phenoB, params->P, params->nodes/params->P);
            
            /* Clean up allocated memory */
            parsec_data_free(phenoB);
        }
        #endif
    }
   
}
