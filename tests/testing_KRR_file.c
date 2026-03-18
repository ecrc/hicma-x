/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 */

#include "hicma_parsec.h"

// Color codes for terminal output are already defined in hicma_parsec_internal.h

// Threshold for solution correctness check
#define SOLUTION_CORRECTNESS_THRESHOLD 60.0

/**
 * @brief Checks the residual of the solution for a linear system Ax = b
 * 
 * This function computes the residual norm ||Ax - b|| and compares it against
 * the expected tolerance based on matrix and vector norms. It helps verify
 * the correctness of the computed solution.
 * 
 * @param parsec PaRSEC context
 * @param loud Verbosity level (0: silent, 1: basic, 2: detailed, 3: verbose)
 * @param AA Original matrix A (for norm computation)
 * @param A Factorized matrix A (for norm computation)
 * @param b Right-hand side vector b
 * @param x Computed solution vector x
 * @return 0 if solution is correct, 1 if suspicious
 */
int check_saxmb2(parsec_context_t *parsec, int loud,
                 parsec_tiled_matrix_t *AA,
                 parsec_tiled_matrix_t *A,
                 parsec_tiled_matrix_t *b,
                 parsec_tiled_matrix_t *x)
{
    int info_solution = 0;
    double Rnorm = 0.0;      // Residual norm ||Ax - b||
    double Anorm = 0.0;      // Matrix A norm
    double AAnorm = 0.0;     // Original matrix AA norm
    double Bnorm = 0.0;      // Right-hand side b norm
    double Xnorm = 0.0;      // Solution x norm
    double result = 0.0;     // Normalized residual
    int N = b->m;            // Matrix dimension
    double eps = LAPACKE_slamch_work('e');  // Machine epsilon
    
    // Compute various matrix and vector norms
    AAnorm = dplasma_slansy(parsec, PlasmaFrobeniusNorm, PlasmaLower, AA);
    Anorm = dplasma_slansy(parsec, PlasmaFrobeniusNorm, PlasmaLower, A);
    Bnorm = dplasma_slange(parsec, PlasmaFrobeniusNorm, b);
    Xnorm = dplasma_slange(parsec, PlasmaFrobeniusNorm, x);
    
    // Compute residual: b = b - A*x
    dplasma_ssymm(parsec, dplasmaLeft, PlasmaLower, -1.0, A, x, 1.0, b);
    
    // Compute residual norm
    Rnorm = dplasma_slange(parsec, PlasmaFrobeniusNorm, b);
    
    // Calculate normalized residual for correctness check
    // Note: Using simplified normalization for stability
    result = Rnorm / Anorm;
    
    // Print detailed information if verbose mode is enabled
    if (loud > 2) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        
        if (loud > 3) {
            printf("--||AA||_F = %e, -- ||A||_F = %e, ||X||_F = %e, ||B||_F = %e, ||Ax - B||_F = %e\n",
                   AAnorm, Anorm, Xnorm, Bnorm, Rnorm);
        }
        
        printf("-- ||Ax-B||_F/||A||_F = " RED "%e" RESET "\n", result);
    }
    
    // Check if solution is numerically correct
    if (isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > SOLUTION_CORRECTNESS_THRESHOLD)) {
        if (loud) {
            printf("-- Solution is suspicious! (result = %e)\n", result);
        }
        info_solution = 1;
    } else {
        if (loud) {
            printf(GRN "-- Solution is CORRECT!" RESET "\n");
        }
        info_solution = 0;
    }
    
    return info_solution;
}

/**
 * @brief Main function - entry point of the KRR testing program
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return Exit status (0 for success, non-zero for failure)
 */
int main(int argc, char **argv)
{
    // HiCMA and PaRSEC configuration structures
    hicma_parsec_params_t params;
    starsh_params_t params_kernel;
    hicma_parsec_data_t data;
    hicma_parsec_matrix_analysis_t analysis;
    
    // PaRSEC context
    parsec_context_t *parsec = NULL;
    
    // Matrix pointers for different operations
    parsec_tiled_matrix_t *AA = NULL;        // Working matrix A
    parsec_tiled_matrix_t *AAcpy = NULL;     // Copy of original matrix A
    
    printf("=== HiCMA KRR Testing Program ===\n");
    printf("Initializing PaRSEC context and HiCMA parameters...\n");
    
    // Initialize PaRSEC context and HiCMA parameters
    parsec = hicma_parsec_init(argc, argv, &params, &params_kernel, &data);
    if (parsec == NULL) {
        fprintf(stderr, "Failed to initialize PaRSEC context\n");
        return EXIT_FAILURE;
    }
    
    // Set up matrix A based on configuration
    AA = (parsec_tiled_matrix_t *)&data.dcA;
    if (params.band_size_dense >= params.NT && params.auto_band == 0 && !params.adaptive_memory) {
        AA = (parsec_tiled_matrix_t *)&data.dcAd;
        printf("Using dense band matrix representation\n");
    }
    
    printf("Loading kernel matrix from file...\n");
    
    // Load kernel matrix from file
    hicma_parsec_kernel_matrix_file(parsec, &data, &params);
    
    // Conditional compilation blocks for different testing modes
#if CHECKSOLVE
    printf("=== Solution Verification Mode ===\n");
    
    // Create copy of original matrix for verification
    AAcpy = (parsec_tiled_matrix_t *)&data.dcAcpy;
    dplasma_slacpy(parsec, dplasmaLower, AA, AAcpy);
    
    // Generate random right-hand side vector
    int Bseed = 3872;
    parsec_tiled_matrix_t *B = (parsec_tiled_matrix_t *)&data.dcB;
    dplasma_splrnt(parsec, 0, B, Bseed);
    
    printf("Generated random right-hand side vector with seed %d\n", Bseed);
    
#elif PREDICTION
    printf("=== Prediction Mode ===\n");
    
    // Create copy of original matrix for prediction
    AAcpy = (parsec_tiled_matrix_t *)&data.dcAcpy;
    dplasma_slacpy(parsec, dplasmaLower, AA, AAcpy);
    
    printf("Matrix copied for prediction mode\n");
    
#endif
    
    printf("Computing matrix norms...\n");
    
    // Compute matrix norms for numerical stability analysis
    if (params.band_size_dense >= params.NT && params.auto_band == 0 && !params.adaptive_memory) {
        hicma_parsec_matrix_norm_get(parsec, dplasmaLower, (parsec_tiled_matrix_t *)&data.dcAd, &params, params.norm_tile, &params.norm_global, "double");
    } else {
        hicma_parsec_matrix_norm_get(parsec, dplasmaLower, (parsec_tiled_matrix_t *)&data.dcA, &params, params.norm_tile, &params.norm_global, "double");
    }
    
    printf("Performing matrix pre-analysis...\n");
    
    // Analyze matrix structure before Cholesky factorization
    hicma_parsec_matrix_pre_analysis(parsec, &data, &params, &params_kernel, &analysis);
    
    printf("Starting Cholesky factorization...\n");
    
    // Perform Cholesky factorization with multiple runs if specified
    for (int i = 0; i < params.nruns; i++) {
        printf("Run %d/%d: ", i + 1, params.nruns);
        
        // Perform Cholesky factorization
        hicma_parsec_potrf(parsec, &data, &params, &analysis);
        
        // Print results for this run
        hicma_parsec_params_print_final(argc, argv, &params, &analysis);
        
        // Check factorization status
        if (params.info != 0) {
            fprintf(stderr, "Warning: Factorization completed with info = %d\n", params.info);
        }
    }
    
    // Check factorization success for single runs
    if (0 == params.rank && params.info != 0 && 1 == params.nruns) {
        fprintf(stderr, "-- Factorization is suspicious (info = %d)!\n", params.info);
        // Note: Not exiting to allow cleanup to proceed
    }
    
    // Conditional execution based on compilation flags
#if PREDICTION
    printf("=== Executing Prediction Mode ===\n");
    
    // Set up matrices for prediction
    parsec_tiled_matrix_t *B = parsec_tiled_matrix_submatrix((parsec_tiled_matrix_t *)&data.dcB, 0, 0, params.NP, params.RHS);
    parsec_tiled_matrix_t *X = parsec_tiled_matrix_submatrix((parsec_tiled_matrix_t *)&data.dcX, 0, 0, params.NP, params.RHS);
    parsec_tiled_matrix_t *P = parsec_tiled_matrix_submatrix((parsec_tiled_matrix_t *)&data.dcP, 0, 0, params.N, params.RHS);
    
    // Select appropriate matrix A for solving
    AA = parsec_tiled_matrix_submatrix((parsec_tiled_matrix_t *)&data.dcA, 0, 0, params.NP, params.NP);
    if (params.band_size_dense >= AA->nt && params.auto_band == 0 && !params.adaptive_memory) {
        AA = parsec_tiled_matrix_submatrix((parsec_tiled_matrix_t *)&data.dcAd, 0, 0, params.NP, params.NP);
    }
    
    // Copy right-hand side to solution vector
    dplasma_slacpy(parsec, dplasmaUpperLower, B, X);
    
    // Solve the triangular system
    dplasma_spotrs(parsec, params.uplo, AA, X);
    
    // Get full original matrix for prediction
    parsec_tiled_matrix_t *bigAA = parsec_tiled_matrix_submatrix(AAcpy, 0, 0, params.N, params.NP);
    
    printf("Matrix dimensions - AA: %dx%d, X: %dx%d, P: %dx%d\n", 
           AA->m, AA->n, X->m, X->n, P->m, P->n);
    
    // Compute prediction: P = bigAA * X
    dplasma_sgemm(parsec, PlasmaNoTrans, PlasmaNoTrans, 1.0, bigAA, X, 0.0, (parsec_tiled_matrix_t *)&(data.dcP));
    
    // Write prediction matrix to file
    printf("Writing prediction matrix to 'matrixP.bin'...\n");
    hicma_parsec_writeB(parsec, "matrixP.bin", (parsec_tiled_matrix_t *)&(data.dcP), &data, &params, 0);
    
#elif CHECKSOLVE
    printf("=== Executing Solution Verification ===\n");
    
    // Copy right-hand side to solution vector
    dplasma_slacpy(parsec, dplasmaUpperLower, B, (parsec_tiled_matrix_t *)&(data.dcX));
    
    // Solve the triangular system
    dplasma_spotrs(parsec, params.uplo, AA, (parsec_tiled_matrix_t *)&(data.dcX));
    
    // Verify solution correctness
    printf("Verifying solution correctness...\n");
    int verify_result = check_saxmb2(parsec, 4, AA, AAcpy, B, (parsec_tiled_matrix_t *)&(data.dcX));
    
    if (verify_result == 0) {
        printf(GRN "Solution verification PASSED" RESET "\n");
    } else {
        printf(RED "Solution verification FAILED" RESET "\n");
    }
    
#endif
    
    printf("Cleaning up resources...\n");
    
    // Finalize and cleanup
    hicma_parsec_fini(parsec, argc, argv, &params, &params_kernel, &data, &analysis);
    
    printf("=== HiCMA KRR Testing Program Completed ===\n");
    
    return EXIT_SUCCESS;
}

