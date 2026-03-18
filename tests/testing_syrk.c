/**
 * @file testing_syrk.c
 * @brief Test program for symmetric rank-k update (SYRK) using HiCMA and PaRSEC
 * 
 * This program benchmarks the performance of SYRK operations using:
 * - HiCMA library for hierarchical matrix operations
 * - PaRSEC for parallel task scheduling
 * - GPU acceleration when available
 * - Performance benchmarking with multiple runs
 * - Memory management and cleanup
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"

/* Use work array while using HCORE functions */
int use_scratch = 1;

/**
 * @brief Forward declaration of SYRK performance function
 */
void hicma_parsec_syrk_performance(int argc, char **argv,
                                   hicma_parsec_params_t *params,
                                   starsh_params_t *params_kernel,
                                   hicma_parsec_data_t *data);

/**
 * @brief Main SYRK performance testing function
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @param params HiCMA parameters structure
 * @param params_kernel STARSH parameters structure
 * @param data HiCMA data structure
 */
void hicma_parsec_syrk_performance(int argc, char **argv,
                                   hicma_parsec_params_t *params,
                                   starsh_params_t *params_kernel,
                                   hicma_parsec_data_t *data) {

    parsec_context_t *parsec;
    
    /* Parse command line arguments */
    parse_arguments(&argc, &argv, params);
    
    /* Start initialization timing */
    SYNC_TIME_START();
    
    /* Initialize HiCMA parameters */
    hicma_parsec_params_init(params, argv);
    
    /* Initialize PaRSEC */
    parsec = hicma_parsec_setup_parsec(argc, argv, params);
    
    /* Print initial parameters */
    hicma_parsec_params_print_initial(params);
    
    SYNC_TIME_PRINT(params->rank, ("HiCMA and PaRSEC initialization completed in %.6f seconds\n", sync_time_elapsed));
    
    /* Initialize output symmetric matrix for SYRK */
    /* This matrix will store the result of A^T * A operation */
    parsec_matrix_sym_block_cyclic_init(&data->dcAd, PARSEC_MATRIX_FLOAT,
            params->rank, params->NB, params->NB, params->N, params->N, 0, 0,
            params->N, params->N, params->P, params->nodes/params->P, params->uplo);
    parsec_data_collection_set_key(&data->dcAd.super.super, "dcAd");

    /* Initialize input matrix A */
    /* Matrix A: size params.nsnp x params.N, tile size params.NB */
    parsec_matrix_block_cyclic_t A;
    parsec_matrix_block_cyclic_init(&A, PARSEC_MATRIX_FLOAT, PARSEC_MATRIX_TILE,
            params->rank, 
            ((params->nsnp < params->NB) ? params->nsnp : params->NB), 
            params->NB, params->nsnp, params->N, 0, 0,
            params->nsnp, params->N, params->P, (params->nodes)/(params->P), 1, 1, 0, 0);

    /* Allocate memory and generate test data */
    SYNC_TIME_START();
    
    /* Initialize GPU handles if CUDA/HIP support is available */
#if (defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT))
    gpu_handle_init(data);
#endif

    /* Allocate memory for symmetric matrix */
    parsec_dist_allocate_sym(parsec, (parsec_tiled_matrix_t *)&data->dcAd, params);
    
    /* Generate test data for matrix A */
    parsec_genotype_generator(parsec, (parsec_tiled_matrix_t *)&A, params, 1);
    
    SYNC_TIME_PRINT(params->rank, ("Memory allocation and data generation completed in %.6f seconds\n", sync_time_elapsed));

    /* Perform GPU warmup if GPUs are available */
    if(params->gpus > 0) {
        SYNC_TIME_START();
        
        /* Warmup with input matrix */
        hicma_parsec_potrf_L_warmup(parsec, (parsec_tiled_matrix_t *)&A, params, 0);
        
        /* Warmup with output matrix */
        hicma_parsec_potrf_L_warmup(parsec, (parsec_tiled_matrix_t *)&data->dcAd, params, 1);
        
        SYNC_TIME_PRINT(params->rank, ("GPU warmup completed in %.6f seconds\n", sync_time_elapsed));
    }

    /* Execute SYRK operation and measure performance */
    struct timeval tstart, tend;
    if (params->rank == 0) {
        printf("Starting SYRK benchmark with %d runs...\n", params->nruns);
    }
    
    for(int i = 0; i < params->nruns; i++) {
        if (params->rank == 0 && params->nruns > 1) {
            printf("Run %d/%d...\n", i + 1, params->nruns);
        }
        
        /* Synchronize all processes */
        MPI_Barrier(MPI_COMM_WORLD);
        SYNC_TIME_START();
        
        /* Start timing */
        gettimeofday(&tstart, NULL);
        params->start_time_syrk = tstart.tv_sec + tstart.tv_usec / 1.0e6;

        /* Execute SYRK operation */
#if 1
        /* Use HiCMA SYRK implementation */
        /* This computes C = alpha * A^T * A + beta * C where C is symmetric */
        hicma_parsec_syrk(parsec,
                PlasmaTrans,      /* Transpose matrix A */
                PlasmaNoTrans,    /* Don't transpose matrix A (already transposed above) */
                1.0,              /* Scaling factor alpha */
                (parsec_tiled_matrix_t *)&A,    /* Input matrix A */
                (parsec_tiled_matrix_t *)&A,    /* Input matrix A (same as A) */
                0.0,              /* Scaling factor beta */
                (parsec_tiled_matrix_t *)&data->dcAd, params, data); /* Output matrix C */
#else
        /* Use DPLASMA SYRK implementation */
        dplasma_isyrk(parsec, params->uplo, PlasmaTrans,
                1.0,
                (parsec_tiled_matrix_t *)&A,
                0.0,
                (parsec_tiled_matrix_t *)&data->dcAd, 8,
                params, data);
#endif

        /* End timing and synchronize */
        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&tend, NULL);
        
        /* Calculate execution time and performance */
        double run_time = tend.tv_sec + tend.tv_usec / 1.0e6 - params->start_time_syrk;
        double flops = (FLOPS_DSYRK(params->nsnp, params->N) * 1e-15) / run_time;
        
        /* Print run results */
        SYNC_TIME_PRINT(params->rank, ("SYRK run %d: nodes=%d gpus=%d N=%d SNP=%d lookahead=%d: %.3f Pflop/s (P=%d Q=%d NB=%d)\n",
                    i + 1, params->nodes, params->gpus, params->N, params->nsnp, params->lookahead, 
                    flops, params->P, params->Q, params->NB));
    }

    /* Print final performance summary */
    if (params->rank == 0) {
        printf("\n============ SYRK PERFORMANCE SUMMARY ============\n");
        printf("SYRK testing completed successfully.\n");
        printf("================================================\n\n");
    }

    /* Clean up allocated memory and resources */
    SYNC_TIME_START();
    
    /* Free matrix memory */
    parsec_memory_free_tile(parsec, (parsec_tiled_matrix_t*)&A, params, 0);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&A);
    
    parsec_memory_free_tile(parsec, (parsec_tiled_matrix_t*)&data->dcAd, params, 1);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t*)&data->dcAd);
    
    /* Free parameter arrays */
    free(params->rank_array);
    free(params->op_band);
    free(params->op_offband);
    free(params->op_path);
    free(params->op_offpath);
    free(params->gather_time);
    free(params->gather_time_tmp);
    free(params->decisions);
    free(params->decisions_send);
    free(params->decisions_gemm_gpu);
    free(params->norm_tile);
    
    /* Clean up GPU handles */
#if (defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT))
    gpu_handle_fini(data);
#endif
    
    SYNC_TIME_PRINT(params->rank, ("Resource cleanup completed in %.6f seconds\n", sync_time_elapsed));

    /* Clean up PaRSEC */
    hicma_parsec_cleanup_parsec(parsec, params);
}

/**
 * @brief Main function for SYRK testing
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return 0 on success
 */
int main(int argc, char **argv) {
    /* HiCMA and PaRSEC parameter structures */
    hicma_parsec_params_t params;
    starsh_params_t params_kernel;
    hicma_parsec_data_t data;
    hicma_parsec_matrix_analysis_t analysis;  /* Unused but required for compatibility */

    /* Execute SYRK performance testing */
    hicma_parsec_syrk_performance(argc, argv, &params, &params_kernel, &data);

    return 0;
}
