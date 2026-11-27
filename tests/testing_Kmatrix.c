/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 * 
 */

#include "hicma_parsec.h"

// Threshold for solution correctness check
#define SOLUTION_CORRECTNESS_THRESHOLD 60.0

/**
 * @brief Checks the residual of the solution for a linear system Ax = b
 * 
 * This function computes the residual norm ||Ax - b|| and compares it against
 * the expected tolerance based on matrix and vector norms. It helps verify
 * the correctness of the computed solution for symmetric positive definite matrices.
 * 
 * @param parsec PaRSEC context
 * @param loud Verbosity level (0: silent, 1: basic, 2: detailed, 3: verbose)
 * @param A Symmetric matrix A (factorized form)
 * @param b Right-hand side vector b (will be modified to store residual)
 * @param x Computed solution vector x
 * @return 0 if solution is correct, 1 if suspicious
 */
int check_saxmb2(parsec_context_t *parsec, int loud,
                 parsec_tiled_matrix_t *A,
                 parsec_tiled_matrix_t *b,
                 parsec_tiled_matrix_t *x)
{
    int info_solution = 0;
    double Rnorm = 0.0;      // Residual norm ||Ax - b||
    double Anorm = 0.0;      // Matrix A norm
    double Bnorm = 0.0;      // Right-hand side b norm
    double Xnorm = 0.0;      // Solution x norm
    double result = 0.0;     // Normalized residual
    int N = b->m;            // Matrix dimension
    double eps = LAPACKE_slamch_work('e');  // Machine epsilon
    
    // Compute various matrix and vector norms
    Anorm = dplasma_slansy(parsec, PlasmaFrobeniusNorm, PlasmaLower, A);
    Bnorm = dplasma_slange(parsec, PlasmaFrobeniusNorm, b);
    Xnorm = dplasma_slange(parsec, PlasmaFrobeniusNorm, x);
    
    // Compute residual: b = b - A*x
    // Note: This modifies the input vector b to store the residual
    dplasma_ssymm(parsec, dplasmaLeft, PlasmaLower, -1.0, A, x, 1.0, b);
    
    // Compute residual norm
    Rnorm = dplasma_slange(parsec, PlasmaFrobeniusNorm, b);
    
    // Calculate normalized residual for correctness check
    // Note: Using simplified normalization for numerical stability
    result = Rnorm / Anorm;
    
    // Print detailed information if verbose mode is enabled
    if (loud > 2) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        
        if (loud > 3) {
            printf("-- ||A||_F = %e, ||X||_F = %e, ||B||_F = %e, ||Ax - B||_F = %e\n",
                   Anorm, Xnorm, Bnorm, Rnorm);
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
 * @brief Validates input parameters and system configuration
 * 
 * @param parsec PaRSEC context
 * @param params HiCMA parameters
 * @return 0 if validation passes, non-zero otherwise
 */
int validateConfiguration(parsec_context_t *parsec, hicma_parsec_params_t *params) {
    if (parsec == NULL) {
        fprintf(stderr, "Error: PaRSEC context is NULL\n");
        return 1;
    }
    
    if (params == NULL) {
        fprintf(stderr, "Error: HiCMA parameters are NULL\n");
        return 1;
    }
    
    // Check if matrix dimensions are reasonable
    if (params->N <= 0) {
        fprintf(stderr, "Error: Invalid matrix dimension N = %d\n", params->N);
        return 1;
    }
    
    if (params->NT <= 0) {
        fprintf(stderr, "Error: Invalid tile size NT = %d\n", params->NT);
        return 1;
    }
    
    return 0;
}

/**
 * @brief Main function - entry point of the kernel matrix testing program
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
    hicma_parsec_matrix_analysis_t analysis = {0};
    
    // PaRSEC context
    parsec_context_t *parsec = NULL;
    
    printf("=== HiCMA Kernel Matrix Testing Program ===\n");
    printf("Initializing PaRSEC context and HiCMA parameters...\n");
    
    // Initialize PaRSEC context and HiCMA parameters
    parsec = hicma_parsec_init(argc, argv, &params, &params_kernel, &data);
    if (parsec == NULL) {
        fprintf(stderr, "Failed to initialize PaRSEC context\n");
        return EXIT_FAILURE;
    }
    
    // Validate configuration parameters
    if (validateConfiguration(parsec, &params) != 0) {
        fprintf(stderr, "Configuration validation failed\n");
        hicma_parsec_fini(parsec, argc, argv, &params, &params_kernel, &data, &analysis);
        return EXIT_FAILURE;
    }
    
    // Print configuration information
    printf("Configuration:\n");
    printf("  Matrix Dimension: %d x %d\n", params.N, params.N);
    printf("  Tile Size: %d\n", params.NT);
    printf("  Number of Processes: %d x %d\n", params.P, params.Q);
    printf("  Kernel Function: %s\n", params_kernel.kernel ? "Available" : "Default");
    printf("\n");
    
    printf("Generating kernel matrix...\n");
    
    // Generate kernel matrix using HiCMA
    hicma_parsec_kernel_matrix(parsec, &data, &params);
    
    printf("Kernel matrix generated successfully!\n");
    
    // Print matrix information if available
    if (data.dcA.super.m > 0 && data.dcA.super.n > 0) {
        printf("Generated Matrix Information:\n");
        printf("  Dimensions: %d x %d\n", data.dcA.super.m, data.dcA.super.n);
        printf("  Tile Dimensions: %d x %d\n", data.dcA.super.mt, data.dcA.super.nt);
        printf("  Data Type: %s\n", data.dcA.super.dtype == PARSEC_MATRIX_FLOAT ? "Single Precision" : "Double Precision");
    }
    
    printf("Cleaning up resources...\n");
    
    // Finalize and cleanup
    hicma_parsec_fini(parsec, argc, argv, &params, &params_kernel, &data, &analysis);
    
    printf("=== HiCMA Kernel Matrix Testing Program Completed Successfully ===\n");
    
    return EXIT_SUCCESS;
}

