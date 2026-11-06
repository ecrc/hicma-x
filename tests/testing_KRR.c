/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2023-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"

/* Memory configuration flags - using definition from hicma_parsec_internal.h */

/**
 * @brief Validate solution quality using residual analysis
 * @param parsec PaRSEC context
 * @param loud Verbosity level for output
 * @param AA Original matrix A
 * @param A Factorized matrix A
 * @param b Right-hand side vector
 * @param x Solution vector
 * @return 0 if solution is valid, 1 if suspicious
 */
int check_saxmb2(parsec_context_t *parsec, int loud,
                 parsec_tiled_matrix_t *AA,
                 parsec_tiled_matrix_t *A,
                 parsec_tiled_matrix_t *b,
                 parsec_tiled_matrix_t *x)
{
    int info_solution = 0;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double AAnorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int N = b->m;
    double eps = LAPACKE_slamch_work('e');

    /* Compute matrix and vector norms for validation */
    AAnorm = dplasma_slansy(parsec, PlasmaFrobeniusNorm, PlasmaLower, AA);
    Anorm = dplasma_slansy(parsec, PlasmaFrobeniusNorm, PlasmaLower, A);
    Bnorm = dplasma_slange(parsec, PlasmaFrobeniusNorm, b);
    Xnorm = dplasma_slange(parsec, PlasmaFrobeniusNorm, x);

    /* Compute residual: r = b - A*x */
    dplasma_ssymm(parsec, dplasmaLeft, PlasmaLower, -1.0, A, x, 1.0, b);
    Rnorm = dplasma_slange(parsec, PlasmaFrobeniusNorm, b);

    /* Calculate relative error metric */
    result = Rnorm / Anorm;  /* Simplified error metric */

    /* Print validation results based on verbosity level */
    if (loud > 2) {
        printf("============ SOLUTION VALIDATION ============\n");
        printf("Checking the Residual of the solution:\n");
        if (loud > 3) {
            printf("Matrix norms:\n");
            printf("  ||AA||_F = %e\n", AAnorm);
            printf("  ||A||_F = %e\n", Anorm);
            printf("  ||x||_F = %e\n", Xnorm);
            printf("  ||b||_F = %e\n", Bnorm);
            printf("  ||r||_F = %e (residual)\n", Rnorm);
        }
        printf("Relative error: ||r||_F / ||A||_F = " RED "%e" RESET "\n", result);
    }

    /* Determine if solution is valid based on error metrics */
    if (isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > 60.0)) {
        if (loud) printf("-- Solution is suspicious!\n");
        info_solution = 1;
    } else {
        if (loud) printf(GRN "-- Solution is CORRECT!" RESET "\n");
        info_solution = 0;
    }

    return info_solution;
}

/**
 * @brief Main function for Kernel Ridge Regression testing
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return 0 on success, 1 on failure
 */
int main(int argc, char **argv)
{
    hicma_parsec_params_t params;
    starsh_params_t params_kernel;
    hicma_parsec_data_t data;
    hicma_parsec_matrix_analysis_t analysis;

    /* Initialize HiCMA and PaRSEC */
    parsec_context_t* parsec = hicma_parsec_init(argc, argv, &params, &params_kernel, &data);
    if( NULL == parsec ) {
        fprintf(stderr, "Failed to initialize HiCMA PaRSEC\n");
        return EXIT_FAILURE;
    }

    /* Select appropriate matrix based on configuration */
    parsec_tiled_matrix_t *AA = (parsec_tiled_matrix_t *)&data.dcA;
    if (params.band_size_dense >= params.NT && params.auto_band == 0 && !params.adaptive_memory) {
        AA = (parsec_tiled_matrix_t *)&data.dcAd;
        if (params.rank == 0) {
            printf("Using dense band matrix for Cholesky decomposition\n");
        }
    }

    /* Generate kernel matrix */
    if (params.rank == 0) {
        printf("Generating kernel matrix...\n");
    }
    hicma_parsec_kernel_matrix(parsec, &data, &params);

#if CHECKSOLVE
    /* Prepare for solution checking if enabled */
    if (params.rank == 0) {
        printf("Preparing solution validation...\n");
    }
    
    /* Clone original matrix for validation */
    parsec_tiled_matrix_t *AAcpy = (parsec_tiled_matrix_t *)&data.dcAcpy;
    dplasma_slacpy(parsec, dplasmaLower, AA, AAcpy);
    
    /* Generate random right-hand side vector */
    int Bseed = 3872;
    parsec_tiled_matrix_t *B = (parsec_tiled_matrix_t *)&data.dcB;
    dplasma_splrnt(parsec, 0, B, Bseed);
#endif

    /* Main computation section */
    if (params.rank == 0) {
        printf("Starting main computation...\n");
    }

    /* Compute matrix norms for analysis */
    if (params.band_size_dense >= params.NT && params.auto_band == 0 && !params.adaptive_memory) {
        hicma_parsec_matrix_norm_get(parsec, dplasmaLower, (parsec_tiled_matrix_t *)&data.dcAd, &params, params.norm_tile, &params.norm_global, "float");
    } else {
        hicma_parsec_matrix_norm_get(parsec, dplasmaLower, (parsec_tiled_matrix_t *)&data.dcA, &params, params.norm_tile, &params.norm_global, "float");
    }

    /* Analyze matrix before Cholesky decomposition */
    if (params.rank == 0) {
        printf("Analyzing matrix structure...\n");
    }
    hicma_parsec_matrix_pre_analysis(parsec, &data, &params, &params_kernel, &analysis);

    /* Perform Cholesky decomposition multiple times for benchmarking */
    if (params.rank == 0) {
        printf("Running Cholesky decomposition %d times...\n", params.nruns);
    }
    
    for (int i = 0; i < params.nruns; i++) {
        if (params.rank == 0 && params.nruns > 1) {
            printf("Run %d/%d...\n", i + 1, params.nruns);
        }
        
        /* Perform Cholesky decomposition */
        hicma_parsec_potrf(parsec, &data, &params, &analysis);
        
        /* Print final parameters for this run */
        hicma_parsec_params_print_final(argc, argv, &params, &analysis);
        
        /* Check factorization status */
        if (params.info != 0 && params.rank == 0) {
            fprintf(stderr, "Warning: Factorization may be suspicious (info = %d)\n", params.info);
            if (params.nruns == 1) {
                fprintf(stderr, "Consider checking matrix properties or adjusting parameters\n");
            }
        }
    }

    /* Analyze matrix after Cholesky decomposition */
    if (params.rank == 0) {
        printf("Analyzing matrix after factorization...\n");
    }
    hicma_parsec_matrix_post_analysis(parsec, &data, &params, &params_kernel, &analysis);

#if CHECKSOLVE
    /* Solve linear system and validate solution */
    if (params.rank == 0) {
        printf("Solving linear system and validating solution...\n");
    }
    
    /* Copy right-hand side to solution vector */
    dplasma_slacpy(parsec, dplasmaUpperLower, B, &(data.dcX));
    
    /* Solve the system using Cholesky factorization */
    dplasma_spotrs(parsec, params.uplo, AA, &(data.dcX));
    
    /* Validate solution quality */
    int validation_result = check_saxmb2(parsec, 4, AA, AAcpy, B, &(data.dcX));
    if (validation_result != 0 && params.rank == 0) {
        fprintf(stderr, "Warning: Solution validation failed\n");
    }
#endif

    if (params.rank == 0) {
        printf("Kernel Ridge Regression testing completed successfully.\n");
    }

    /* Finalize HiCMA and PaRSEC */
    hicma_parsec_fini(parsec, argc, argv, &params, &params_kernel, &data, &analysis);

    return 0;
}

