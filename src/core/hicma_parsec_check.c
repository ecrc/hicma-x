/**
 * @file hicma_parsec_check.c
 * @brief HICMA PaRSEC correctness checking implementation
 * 
 * This file implements the correctness checking functions for the HICMA library,
 * including matrix generation verification, Cholesky factorization validation,
 * compression quality assessment, and analysis verification.
 * 
 * The implementation provides comprehensive testing capabilities for:
 * - Matrix generation using STARSH kernels
 * - Cholesky factorization correctness via ||L*L'-A|| computation
 * - Compression quality validation
 * - Analysis result verification
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"

/* ============================================================================
 * Matrix generation functions
 * ============================================================================ */

/**
 * @brief Kernel operator for generating matrix tiles
 * 
 * This function is called by the PaRSEC runtime to generate individual matrix
 * tiles during computation. It uses the STARSH kernel function to populate
 * the tile data based on the specified parameters.
 * 
 * @param[in] es Execution stream (unused in current implementation)
 * @param[in] descA Matrix descriptor containing matrix metadata
 * @param[in,out] _A Pointer to matrix tile data to be generated
 * @param[in] uplo Upper/lower specification (unused in current implementation)
 * @param[in] m Row index of the tile to generate
 * @param[in] n Column index of the tile to generate
 * @param[in] op_data Operation data containing kernel parameters
 * @return 0 on success, non-zero on failure
 */
int starsh_generate_map_operator(parsec_execution_stream_t *es,
        const parsec_tiled_matrix_t *descA, void *_A, int uplo,
        int m, int n, void *op_data)
{
    int tempmm, tempnn, ldam;
    double *A = _A;
    starsh_params_t *params = op_data;
    
    /* Suppress unused parameter warnings */
    (void)es;
    (void)uplo;

    /* Calculate actual tile dimensions (may be smaller at matrix boundaries) */
    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    
    /* Get leading dimension for the tile */
    ldam = BLKLDD( descA, m );

    /* Call the STARSH kernel function to generate tile data
     * The kernel function fills the tile with appropriate values based on
     * the tile position (m,n) and the kernel parameters */
    params->kernel(tempmm, tempnn, params->index + m*descA->mb,
            params->index + n*descA->nb, params->data, params->data, A, ldam);

    return 0;
}

/**
 * @brief Generate complete matrix using STARSH kernel
 * 
 * Creates a task pool to generate all matrix tiles in parallel using the
 * STARSH kernel function. The function handles matrix generation for different
 * storage formats (upper, lower, or both).
 * 
 * @param[in] parsec PaRSEC context for task execution
 * @param[in] uplo Upper/lower specification (PlasmaLower, PlasmaUpper, PlasmaUpperLower)
 * @param[in,out] A Matrix to be generated (will be filled with generated data)
 * @param[in] params STARSH parameters containing kernel function and data
 * @return 0 on success, non-zero on failure
 */
int starsh_generate_map(parsec_context_t *parsec, int uplo,
        parsec_tiled_matrix_t *A, starsh_params_t *params)
{
    parsec_taskpool_t *pool = NULL;

    /* Validate input arguments - check for valid uplo values */
    if ((uplo != PlasmaLower) &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaUpperLower))
    {
        dplasma_error("starsh_generate_map", "illegal value of uplo");
        return -3;
    }

    /* Create a copy of parameters to ensure thread safety
     * This prevents race conditions when multiple tasks access the parameters */
    starsh_params_t *copy_params = malloc(sizeof(starsh_params_t));
    if (copy_params == NULL) {
        dplasma_error("starsh_generate_map", "memory allocation failed");
        return -1;
    }
    *copy_params = *params;
    
    /* Create task pool using PaRSEC's apply pattern
     * This will generate tasks for each tile in the matrix */
    pool = parsec_apply_New(uplo, A, starsh_generate_map_operator, copy_params);

    if ( pool != NULL ) {
        /* Add the task pool to the PaRSEC context and execute */
        parsec_context_add_taskpool(parsec, pool);
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        
        /* Clean up the task pool */
        parsec_apply_Destruct(pool);
    }
    
    /* Free the copied parameters */
    free(copy_params);
    
    return 0;
}

/* ============================================================================
 * Cholesky factorization checking functions
 * ============================================================================ */

/**
 * @brief Check Cholesky factorization correctness via ||L*L'-A||
 * 
 * This function verifies the correctness of Cholesky factorization by:
 * 1. Computing L*L' (or U'*U for upper triangular)
 * 2. Computing the difference ||L*L'-A||
 * 3. Comparing the relative error against a threshold
 * 
 * The function handles both upper and lower triangular matrices and provides
 * detailed output for debugging and validation purposes.
 * 
 * @param[in] parsec PaRSEC context for matrix operations
 * @param[in] verbose Verbosity level for output (0 = silent, 1 = verbose)
 * @param[in] uplo Upper/lower specification (PlasmaUpper, PlasmaLower)
 * @param[in] A Computed L matrix from Cholesky factorization
 * @param[in] A0 Original matrix A before factorization
 * @param[in] threshold Accuracy threshold for correctness validation
 * @param[out] result_accuracy Computed relative error ||L*L'-A||/||A||
 * @return 0 on success, non-zero on failure
 */
int check_dpotrf2( parsec_context_t *parsec, int verbose,
        int uplo,
        parsec_tiled_matrix_t *A,
        parsec_tiled_matrix_t *A0, double threshold, double *result_accuracy )
{
    parsec_matrix_block_cyclic_t *twodA = (parsec_matrix_block_cyclic_t *)A0;
    parsec_matrix_block_cyclic_t LLt;
    int info_factorization;
    double Rnorm = 0.0;  /* Norm of residual L*L'-A */
    double Anorm = 0.0;  /* Norm of original matrix A */
    double result = 0.0; /* Relative error result */
    int N = A->n;        /* Matrix dimension */
    double eps = LAPACKE_dlamch_work('e'); /* Machine epsilon */
    int side;

    /* Initialize a temporary matrix LLt to store L*L' computation
     * This matrix will be used to compute the factorization verification */
    parsec_matrix_block_cyclic_init(&LLt, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
            twodA->grid.rank,
            A->mb, A->nb, N, N, 0, 0, N, N,
            twodA->grid.rows, A->super.nodes/twodA->grid.rows,
            twodA->grid.krows, twodA->grid.kcols,
            twodA->grid.ip, twodA->grid.jq);

    /* Allocate memory for the temporary matrix */
    LLt.mat = parsec_data_allocate((size_t)LLt.super.nb_local_tiles *
            (size_t)LLt.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(LLt.super.mtype));

    /* Initialize LLt to zero and copy A into it */
    dplasma_dlaset( parsec, PlasmaUpperLower, 0., 0.,(parsec_tiled_matrix_t *)&LLt );
    dplasma_dlacpy( parsec, uplo, A, (parsec_tiled_matrix_t *)&LLt );

    /* Compute L*L' or U'*U depending on the uplo parameter
     * For upper triangular: compute U'*U (side = PlasmaLeft)
     * For lower triangular: compute L*L' (side = PlasmaRight) */
    side = (uplo == PlasmaUpper ) ? PlasmaLeft : PlasmaRight;
    dplasma_dtrmm( parsec, side, uplo, PlasmaTrans, PlasmaNonUnit, 1.0,
                   A, (parsec_tiled_matrix_t*)&LLt);

    /* Compute the residual: LLt = LLt - A0 (i.e., L*L' - A) */
    dplasma_dtradd( parsec, uplo, PlasmaNoTrans,
                    -1.0, A0, 1., (parsec_tiled_matrix_t*)&LLt);

    /* Compute Frobenius norms for error analysis */
    Anorm = dplasma_dlansy(parsec, PlasmaFrobeniusNorm, uplo, A0);
    Rnorm = dplasma_dlansy(parsec, PlasmaFrobeniusNorm, uplo,
                           (parsec_tiled_matrix_t*)&LLt);

    /* Compute relative error: ||L*L'-A|| / ||A||
     * Note: The original code had a commented line using eps*N factor */
    result = Rnorm / ( Anorm ) ;
    *result_accuracy = result;

    /* Print detailed results if verbose mode is enabled */
    if ( verbose ) {
        printf("============\n");
        printf("Checking the Cholesky factorization. Threshold is " BLU "%.2e" RESET " \n", threshold);
        printf( "-- ||A||_F = %e, ||L'L-A||_F = %e\n", Anorm, Rnorm );
        printf("-- ||L'L-A||_F/(||A||_F) = " RED "%e" RESET "\n", result);
    }

    /* Determine if factorization is correct based on threshold
     * Check for NaN or infinite values to catch numerical issues */
    if ( isnan(Rnorm)  || isinf(Rnorm)  ||
         isnan(result) || isinf(result) ||
         (result > threshold) )
    {
        if( verbose ) printf(RED "-- Factorization is suspicious ! \n" RESET);
        if( 0 == A0->super.myrank )
            fprintf(stderr, RED "Check FAILED: ||LL^T-A||_F/||A||_F = %e, accepted bound %e (--check_tol)\n" RESET, result, threshold);
        info_factorization = 1;
    }
    else
    {
        if( verbose ) printf(GRN "-- Factorization is CORRECT ! \n" RESET);
        info_factorization = 0;
    }

    /* Clean up temporary matrix memory */
    parsec_data_free(LLt.mat); 
    LLt.mat = NULL;
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&LLt);

    /* Every rank computes the same norms (dplasma_dlansy is collective), so
     * the verdict is identical everywhere and needs no reduction. */
    return info_factorization;
}

/**
 * @brief Check solution L difference from the dense counterpart
 * 
 * This function compares the computed L matrix with the reference L0 matrix
 * to verify factorization accuracy. It computes the difference ||L-L0||/||L||
 * and compares it against a specified threshold.
 * 
 * @param[in] parsec PaRSEC context for matrix operations
 * @param[in] verbose Verbosity level for output (0 = silent, 1 = verbose)
 * @param[in] uplo Upper/lower specification (PlasmaUpper, PlasmaLower)
 * @param[in] A Computed L matrix from factorization
 * @param[in] A0 Reference L matrix for comparison
 * @param[in] threshold Accuracy threshold for correctness validation
 * @return 0 on success, non-zero on failure
 */
int check_diff( parsec_context_t *parsec, int verbose,
                  int uplo,
                  parsec_tiled_matrix_t *A,
                  parsec_tiled_matrix_t *A0, double threshold )
{
    int info_factorization;
    double Rnorm = 0.0;  /* Norm of difference L-L0 */
    double A0norm = 0.0; /* Norm of reference matrix L0 */
    double Anorm = 0.0;  /* Norm of computed matrix L */
    double result = 0.0; /* Relative difference result */

    /* Compute Frobenius norms of both matrices for comparison */
    Anorm = dplasma_dlansy(parsec, PlasmaFrobeniusNorm, uplo, A);
    A0norm = dplasma_dlansy(parsec, PlasmaFrobeniusNorm, uplo, A0);

    /* Compute the difference: A = A - A0 (i.e., L - L0) */
    dplasma_dtradd( parsec, uplo, PlasmaNoTrans,
                    -1.0, A0, 1., A);

    /* Compute the norm of the difference matrix */
    Rnorm = dplasma_dlansy(parsec, PlasmaFrobeniusNorm, uplo, A);
    result = Rnorm / Anorm;

    /* Print detailed comparison results if verbose mode is enabled */
    if ( verbose ) {
        printf("============\n");
        printf("Checking the equality of lower/upper triangular parts of two matrices. Threshold is %.2e\n", threshold);
        printf( "-- dpotrf: ||L0||_F = %e, HiCMA: ||L||_F = %e, ||L-L0||_F = %e, ||L-L0||_F/||L||_F = %e \n", A0norm, Anorm, Rnorm, result );
    }

    /* Determine if matrices are equal based on threshold
     * Check for NaN or infinite values to catch numerical issues */
    if ( isnan(Rnorm)  || isinf(Rnorm)  ||
         isnan(result) || isinf(result) ||
         (result > threshold) )
    {
        if( verbose ) printf("-- DIFFERENT matrices ! \n");
        if( 0 == A->super.myrank )
            fprintf(stderr, RED "Check FAILED: ||L-L0||_F/||L||_F = %e, accepted bound %e (--check_tol)\n" RESET, result, threshold);
        info_factorization = 1;
    }
    else
    {
        if( verbose ) printf("-- SAME matrices ! \n");
        info_factorization = 0;
    }

    return info_factorization;
}

/* ============================================================================
 * HICMA-specific checking functions
 * ============================================================================ */

/**
 * @brief Check compression quality and perform dense Cholesky validation
 * 
 * This function performs comprehensive validation of matrix compression by:
 * 1. Generating the original problem matrix without approximation
 * 2. Uncompressing the approximate matrix
 * 3. Computing the difference between original and uncompressed matrices
 * 4. Performing dense Cholesky factorization on the uncompressed matrix
 * 5. Computing determinant for additional validation
 * 
 * @param[in] parsec PaRSEC context for matrix operations
 * @param[in] data HICMA data structure containing matrices and parameters
 * @param[in] params HICMA parameters for computation
 * @param[in] params_kernel STARSH kernel parameters for matrix generation
 * @param[in] analysis Matrix analysis structure with rank information
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_check_compression( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_matrix_analysis_t *analysis )
{
    int rank = params->rank;
    int nodes = params->nodes;
    int P = params->P;
    int NT = params->NT;
    int NB = params->NB;
    int N = params->N;
    int band_size_dense = params->band_size_dense;
    int uplo = params->uplo;
    int maxrank = params->maxrank;

    /* Early return if result checking is disabled */
    if( 0 == params->check ) return 0;

    if(rank == 0) fprintf(stderr, RED "\nCheck Compression:\n" RESET);

    /**********************************************
     *************** Check compression ************
     **********************************************/
    
    /* Initialize dcA1 matrix for storing the original problem matrix
     * This matrix will contain the exact values without any approximation */
    parsec_matrix_sym_block_cyclic_init(&data->dcA1, PARSEC_MATRIX_DOUBLE,
            rank, NB, NB, N, N, 0, 0,
            N, N, P, nodes/P, uplo);
#if BAND_MEMORY_CONTIGUOUS
    data->dcA1.mat = parsec_data_allocate((size_t)data->dcA1.super.nb_local_tiles *
            (size_t)data->dcA1.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(data->dcA1.super.mtype));
#endif
    parsec_data_collection_set_key((parsec_data_collection_t*)&data->dcA1, "dcA1");

    /* Generate the original problem matrix WITHOUT approximation in dcA1
     * This serves as the reference for compression quality assessment */
#if BAND_MEMORY_CONTIGUOUS
    starsh_generate_map(parsec, uplo, (parsec_tiled_matrix_t *)&data->dcA1, params_kernel);
#elif !GENOMICS
    parsec_band_regenerate(parsec, uplo, (parsec_tiled_matrix_t *)&data->dcA1, params, params_kernel, NT);
#else
    /* TODO: Rabab - Implement kernel matrix generation */
    hicma_kernal_matrix(parsec, data, (parsec_tiled_matrix_t *)&data->dcA1, params);
#endif

    /* Initialize dcA0 matrix for storing the uncompressed approximate matrix
     * This matrix will be used to store the result of uncompression */
    parsec_matrix_sym_block_cyclic_init(&data->dcA0, PARSEC_MATRIX_DOUBLE,
            rank, NB, NB, N, N, 0, 0,
            N, N, P, nodes/P, uplo);
    data->dcA0.mat = parsec_data_allocate((size_t)data->dcA0.super.nb_local_tiles *
            (size_t)data->dcA0.super.bsiz *
            (size_t)parsec_datadist_getsizeoftype(data->dcA0.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&data->dcA0, "dcA0");

    /* Uncompress the approximate matrix dcA into dcA0
     * This reconstructs the full matrix from its compressed representation */
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
    if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
        A = (parsec_tiled_matrix_t *)&data->dcAd;
    }
    parsec_tiled_matrix_t *Ar = (parsec_tiled_matrix_t *)&data->dcAr;
    parsec_tiled_matrix_t *A0 = (parsec_tiled_matrix_t *)&data->dcA0;
    hicma_parsec_matrix_uncompress(parsec, uplo, A0, A, Ar, analysis,
            band_size_dense, maxrank, &params->info);

    /* Check for uncompression errors */
    if( params->info != 0 ) {
        if( 0 == rank ) fprintf(stderr, "\nCheck compression: %d\n", params->info);
        exit(1);
    }

    /* Compute Frobenius norms for compression quality assessment */
    double norm1 = dplasma_dlansy(parsec, PlasmaFrobeniusNorm, uplo,
            (parsec_tiled_matrix_t *)&data->dcA1);
    double norm2 = dplasma_dlansy(parsec, PlasmaFrobeniusNorm, uplo,
            (parsec_tiled_matrix_t *)&data->dcA0);

    if( 0 == rank ) {
        fprintf(stderr, "Norms: map=%e lr_jdf=%e\n", norm1, norm2);
    }

    /* Compute the difference: dcA0 = -1 * dcA1 + 1 * dcA0
     * This gives us the error introduced by compression */
    dplasma_dtradd(parsec, uplo, PlasmaNoTrans, -1.0,
            (parsec_tiled_matrix_t *)&data->dcA1, 1.0,
            (parsec_tiled_matrix_t *)&data->dcA0);

    /* Compute the norm of the difference matrix */
    double diff = dplasma_dlansy(parsec, PlasmaFrobeniusNorm, uplo,
            (parsec_tiled_matrix_t *)&data->dcA0);

    if( 0 == rank ) {
        fprintf(stderr, "Norms: map-lr_jdf=%e\n", diff);
    }

    /* Report the relative error of approximation */
    if(rank == 0) {
        fprintf(stderr, "Relative error of approximation: %e\n\n", diff/norm1);
    }

    /**********************************************
     ***************** dense dpotrf ***************
     **********************************************/
    
    /* Uncompress the approximate matrix again for dense Cholesky validation
     * This ensures we have a clean copy for the factorization */
    hicma_parsec_matrix_uncompress(parsec, uplo, A0, A, Ar, analysis,
            band_size_dense, maxrank, &params->info);

    if( params->info != 0 ) {
        if( 0 == rank ) fprintf(stderr, "\nCheck compression: %d\n", params->info);
        exit(1);
    }

    /* Perform dense Cholesky factorization on the uncompressed matrix
     * This validates that the uncompressed matrix is numerically stable */
    {
        parsec_taskpool_t *parsec_dpotrf = NULL;
        int info = 0;

        /* Create dense Cholesky task pool */
        parsec_dpotrf = dplasma_dpotrf_New( uplo, (parsec_tiled_matrix_t*)&data->dcA0, &info );

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        /* TODO: Temporary solution - disable GPU for dense Cholesky
         * This ensures compatibility across different GPU configurations */
        hicma_parsec_disable_GPU( parsec_dpotrf );
#endif

        if ( parsec_dpotrf != NULL )
        {
            /* Execute dense Cholesky factorization */
            parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_dpotrf);
            dplasma_wait_until_completion(parsec);
            dplasma_dpotrf_Destruct( parsec_dpotrf );
        }

        /* info is set on the rank that owns the failing diagonal tile only.
         * A reference factorization that breaks down means the uncompressed
         * matrix is not positive definite (fixedacc too loose, or a wrong
         * --adddiag), and the comparisons below would be against garbage. */
        MPI_Allreduce( MPI_IN_PLACE, &info, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
        if( 0 != info ) {
            if( 0 == rank ) {
                fprintf(stderr, RED "The reference dense Cholesky of the uncompressed matrix failed: "
                        "info= %d (leading minor not positive definite)\n" RESET, info);
            }
            params->info = info;
            exit(1);
        }
    }

    /* Compute determinant of the dense Cholesky result
     * This provides additional validation of the uncompressed matrix */
    hicma_parsec_matrix_det( parsec, (parsec_tiled_matrix_t*)&data->dcA0, &params->log_det_dp );

    return 0;
}

/**
 * @brief Check Cholesky factorization results comprehensively
 * 
 * This function performs comprehensive verification of Cholesky factorization
 * results by validating:
 * 1. Difference between HiCMA and dense Cholesky results
 * 2. Factorization correctness via ||L*L'-A|| computation
 * 3. Numerical accuracy against specified thresholds
 * 
 * @param[in] parsec PaRSEC context for matrix operations
 * @param[in] data HICMA data structure containing matrices and parameters
 * @param[in] params HICMA parameters for computation
 * @param[in] params_kernel STARSH kernel parameters for matrix generation
 * @param[in] analysis Matrix analysis structure with rank information
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_check_dpotrf( parsec_context_t *parsec,
        hicma_parsec_data_t *data,
        hicma_parsec_params_t *params,
        starsh_params_t *params_kernel,
        hicma_parsec_matrix_analysis_t *analysis )
{
    /* Early return if result checking is disabled */
    if( 0 == params->check ) return 0;

    int rank = params->rank;
    int nodes = params->nodes;
    int P = params->P;
    int NT = params->NT;
    int NB = params->NB;
    int N = params->N;
    int band_size_dense = params->band_size_dense;
    int uplo = params->uplo;
    int maxrank = params->maxrank;
    int verbose = params->verbose;
    /* --check_tol when given, otherwise the compression tolerance */
    double tol = params->check_tol > 0.0 ? params->check_tol : params->fixedacc;
    int fixedrk = params->fixedrk;

    if(rank == 0) fprintf(stderr, RED "\nCheck Cholesky:\n" RESET);

    /**********************************************
     **** diff of dense dpotrf and tlr dpotrf *****
     **********************************************/

    /* Check equality of two matrices coming from dense and low rank Cholesky
     * Uncompress the approximate matrix into dcA1 for comparison */
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&data->dcA;
    if( params->band_size_dense >= params->NT && params->auto_band == 0 && !params->adaptive_memory ) {
        A = (parsec_tiled_matrix_t *)&data->dcAd;
    }
    parsec_tiled_matrix_t *Ar = (parsec_tiled_matrix_t *)&data->dcAr;
    parsec_tiled_matrix_t *A1 = (parsec_tiled_matrix_t *)&data->dcA1;
    hicma_parsec_matrix_uncompress(parsec, uplo, A1, A, Ar, analysis,
            band_size_dense, maxrank, &params->info);

    /* Print detailed information about the comparison */
    if(rank == 0) {
        printf("+++++++++++++++++++++++++++\n");
        printf("Checking HiCMA L vs dense L\n");
        printf("dcA1 is matrix_uncompressed matrix L obtained from HiCMA_dpotrf_L.jdf on approximate A.\n");
        printf("dcA0  is L factor obtained from dense cholesky (dpotrf_L.jdf in this repo) on approximate A.\n");
        printf("difference between these two matrices must be less than fixed accuracy threshold %.1e if fixed rank %d is zero.\n", tol, fixedrk);
        fflush(stdout);
    }

    /* Perform the comparison between HiCMA and dense Cholesky results */
    params->info |= check_diff( parsec, (rank == 0) ? verbose : 0, uplo,
            (parsec_tiled_matrix_t *)&data->dcA1,
            (parsec_tiled_matrix_t *)&data->dcA0, tol);

    /* Check the factorization correctness via ||L*L'-A|| computation
     * Input: initial A and computed L where A=LLT
     * Performs A-L*L^T and reports norms */
    
    /* Generate the original problem matrix WITHOUT approximation in dcA0
     * This serves as the reference for factorization correctness */
#if BAND_MEMORY_CONTIGUOUS
    starsh_generate_map(parsec, uplo, (parsec_tiled_matrix_t *)&data->dcA0,
            params_kernel);
#elif !GENOMICS
    parsec_band_regenerate(parsec, uplo, (parsec_tiled_matrix_t *)&data->dcA0, params, params_kernel, NT);
#else
    hicma_kernal_matrix(parsec, data, (parsec_tiled_matrix_t *)&data->dcA0, params);
#endif

    if(rank == 0) {
        printf("++++++++++++++++++++++++++\n");
        printf("Checking HiCMA L vs original A via computing LLT\n");
        printf("dcA0 is original problem A withOUT any approximation.\n");
        printf("dcA1 is matrix_uncompressed matrix L obtained from HiCMA_dpotrf_L.jdf on approximate A.\n");
        printf("difference between these two matrices must be less than fixed accuracy threshold %.1e if fixed rank %d is zero.\n", tol, fixedrk);
        fflush(stdout);
    }

    /* dcA1 must be filled again because it was changed in the previous comparison
     * This ensures we have the correct HiCMA result for the factorization check */
    hicma_parsec_matrix_uncompress(parsec, uplo, A1, A, Ar, analysis,
            band_size_dense, maxrank, &params->info);

    /* Perform the factorization correctness check using ||L*L'-A|| computation
     * This validates that the HiCMA factorization satisfies A = L*L' */
    params->info |= check_dpotrf2( parsec, (rank == 0) ? verbose : 0, uplo,
            (parsec_tiled_matrix_t *)&data->dcA1,
            (parsec_tiled_matrix_t *)&data->dcA0, tol, &params->result_accuracy);

    /* Check for factorization errors and report results */
    if( params->info != 0 ) {
        if( 0 == rank ) fprintf(stderr, "\nCheck dpotrf: %d\n", params->info);
        exit(1);
    }

    return 0;
}

/**
 * @brief Check correctness of matrix analysis results
 * 
 * This function verifies the correctness and consistency of matrix analysis
 * results by comparing the computed ranks with the expected ranks from the
 * rank array. It validates that the sparse analysis correctly identified
 * the matrix structure and fill-in patterns.
 * 
 * @param[in] params HICMA parameters containing rank array and analysis settings
 * @param[in] analysis Matrix analysis structure with computed ranks
 * @return 0 on success, non-zero on failure
 */
int hicma_parsec_check_analysis( hicma_parsec_params_t *params,
        hicma_parsec_matrix_analysis_t *analysis )
{
    /* Early return if sparse analysis is disabled */
    if( 0 == params->sparse ) return 0; 
    
    int band_size_dense = params->band_size_dense;
    int NT = params->NT;
    int rank = params->rank;
    int nodes = params->nodes;
    int root = 0;
    int fillin_correct = 1;      /* Local correctness flag */
    int fillin_correct_all = 0;  /* Global correctness flag */

    if(rank == 0) fprintf(stderr, RED "\nCheck analysis:\n" RESET);

    /* Validate the analysis results by comparing computed ranks with expected ranks
     * This loop checks each tile position (i,j) in the matrix */
    for(int i = band_size_dense; i < NT; i++) {
        for(int j = 0; j <= i-band_size_dense ; j++) {
            /* Check if the computed rank matches the expected rank
             * For positive expected ranks: computed rank should be 1
             * For non-positive expected ranks: computed rank should be 0 */
            if( params->rank_array[j*NT+i] > 0 && analysis->final_rank[j*NT+i] == 1
                    || params->rank_array[j*NT+i] <= 0 && analysis->final_rank[j*NT+i] == 0 )
                ; /* Rank is correct - continue */
            else {
                /* Rank is incorrect - mark as failed and report error */
                fillin_correct = 0;
                fprintf(stderr, RED "Matrix fillins is not calculated CORRECT : ( %d %d ) : %d %d!\n" RESET,
                        i, j, params->rank_array[j*NT+i], analysis->final_rank[j*NT+i]);
            }
        }
    }

    /* Perform global reduction to check if all nodes have correct analysis
     * This ensures consistency across the entire distributed computation */
    MPI_Reduce( &fillin_correct, &fillin_correct_all, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD );

    /* Report the global analysis results */
    if( root == rank ) {
        if( fillin_correct_all == nodes )
            fprintf(stderr, GRN "Matrix fillins are calculated CORRECT !\n" RESET);
        else
            fprintf(stderr, RED "Matrix fillins are NOT calculated CORRECT !\n" RESET);
    }

    return 0;
}
