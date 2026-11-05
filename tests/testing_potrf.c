/**
 * @file testing_potrf.c
 * @brief Test program for Cholesky decomposition (POTRF) using HiCMA and PaRSEC
 * 
 * This program performs Cholesky decomposition using:
 * - HiCMA library for hierarchical matrix operations
 * - PaRSEC for parallel task scheduling
 * - TLR (Tile Low-Rank) matrix format support
 * - Performance benchmarking with multiple runs
 * - Matrix analysis before and after factorization
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"

/**
 * @brief Main function for Cholesky decomposition testing
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return 0 on success, 1 on failure
 */
int main(int argc, char **argv)
{
    /* HiCMA and PaRSEC parameter structures */
    hicma_parsec_params_t params;
    starsh_params_t params_kernel;
    hicma_parsec_data_t data;
    hicma_parsec_matrix_analysis_t analysis;

    /* Check GENOMICS feature flag */
#if GENOMICS
    fprintf(stderr, RED "GENOMICS is not supported. Please set GENOMICS to 0!!!!\n" RESET);
    fprintf(stderr, "This feature is not implemented in the current version.\n");
    exit(1);
#endif

    /* Initialize HiCMA and PaRSEC */
    parsec_context_t* parsec = hicma_parsec_init(argc, argv, &params, &params_kernel, &data);
    if( parsec == NULL ) {
        fprintf(stderr, "Failed to initialize HiCMA PaRSEC\n");
        return EXIT_FAILURE;
    }

    /* Generate test matrix */
    if (params.rank == 0) {
        printf("Generating test matrix...\n");
    }
    hicma_parsec_matrix_generation(parsec, &data, &params, &params_kernel);

    /* Analyze matrix structure before Cholesky decomposition */
    if (params.rank == 0) {
        printf("Analyzing matrix structure before factorization...\n");
    }
    hicma_parsec_matrix_pre_analysis(parsec, &data, &params, &params_kernel, &analysis);

    /* Perform Cholesky decomposition multiple times for benchmarking */
    if (params.rank == 0) {
        printf("Running Cholesky decomposition %d times...\n", params.nruns);
    }

    /* The factorization is in place, so run 2 would factorize run 1's L, and
     * in runtime-decision mode the precision mix (hence the time) depends on
     * the data. Regenerate A (and redo the pre-analysis, which resets the
     * precision decisions) before every later run where that is
     * allocation-free: the dense FP64 layout without --check, --auto_band,
     * --adaptive_memory or sparse mode. */
    int regenerate = params.nruns > 1
                     && params.band_size_dense >= params.NT && 0 == params.auto_band
                     && !params.adaptive_memory && !params.check && 0 == params.sparse;
    if (params.nruns > 1 && !regenerate && 0 == params.rank) {
        fprintf(stderr, "Warning: with this configuration runs 2..%d factorize the previous run's factor; "
                "time them with care (or use --nruns 1 per launch)\n", params.nruns);
    }

    for (int i = 0; i < params.nruns; i++) {
        if (params.rank == 0 && params.nruns > 1) {
            printf("Run %d/%d...\n", i + 1, params.nruns);
        }

        if (i > 0 && regenerate) {
            hicma_parsec_matrix_generation(parsec, &data, &params, &params_kernel);
            hicma_parsec_matrix_pre_analysis(parsec, &data, &params, &params_kernel, &analysis);
        }

        /* GEMM/conversion counters and decisions_storage_final are reset
         * inside hicma_parsec_potrf; only the GEMM precision traces here. */
        if(params.adaptive_decision_runtime) {
            memset(params.decisions_gemm_first, 0, (size_t)params.MT * params.NT * sizeof(uint16_t));
            memset(params.decisions_gemm_last, 0, (size_t)params.MT * params.NT * sizeof(uint16_t));
        }

        /* Perform Cholesky decomposition */
        hicma_parsec_potrf(parsec, &data, &params, &analysis);

        /* Print final parameters for this run */
        if(!params.check) hicma_parsec_params_print_final(argc, argv, &params, &analysis);

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

    if (params.rank == 0) {
        printf("Cholesky decomposition testing completed successfully.\n");
    }

    /* Finalize HiCMA and PaRSEC */
    hicma_parsec_fini(parsec, argc, argv, &params, &params_kernel, &data, &analysis);

    return 0;
}
