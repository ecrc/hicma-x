/**
 * @file testing_hamming.c
 * @brief Test program for Hamming distance computation using HiCMA and PaRSEC
 * 
 * This program computes Hamming distances between binary vectors using:
 * - Matrix-based approach for efficiency
 * - Parallel computation with PaRSEC
 * - GPU acceleration when available
 * - Performance benchmarking and validation
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2023-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#define CHECK 0  /* Set to 1 to enable result validation */

#include "hicma_parsec.h"

/**
 * @brief Compute Hamming distance using naive pairwise comparison for validation
 * @param parsec PaRSEC context
 * @param A Input binary matrix
 * @param AA Result matrix containing Hamming distances
 * @return 0 on success
 */
int check_hamming(parsec_context_t *parsec,
                 parsec_tiled_matrix_t *A,
                 parsec_tiled_matrix_t *AA)
{
    /* Allocate temporary storage for matrix A data in LAPACK format */
    int8_t *ZA = (int8_t *)malloc(A->llm * A->lln *
                                        (size_t)parsec_datadist_getsizeoftype(A->mtype));
    if (!ZA) {
        fprintf(stderr, "Failed to allocate memory for matrix A conversion\n");
        return -1;
    }

    /* Convert tiled matrix to LAPACK format for processing */
    hicma_parsec_Tile_to_Lapack_int8(parsec, A, ZA, 1, 1);

    /* Initialize counters for different and same elements */
    long int counterin = 0;   /* Count of different elements */
    long int counterout = 0;  /* Count of same elements */

    /* Compute Hamming distance using pairwise comparison */
    for (int j = 0; j < A->lln; j++) {
       for (int k = j + 1; k < A->lln; k++) {
            for (int i = 0; i < A->llm; i++) {
               /* Compare elements at positions (i,j) and (i,k) */
               if (ZA[k * A->lm + i] != ZA[j * A->lm + i]) {
                counterin++;  /* Elements are different */
               } else {
                counterout++; /* Elements are the same */
               }
           }
       }
    }
    
    /* Print results of naive Hamming distance computation */
    printf("The total different of humming distance counterin:%ld counterout:%ld\n",  
           counterin, counterout);
    free(ZA);
    
    /* Allocate temporary storage for matrix AA data in LAPACK format */
    int *ZAA = (int *)malloc(AA->llm * AA->lln *
                                        (size_t)parsec_datadist_getsizeoftype(AA->mtype));
    if (!ZAA) {
        fprintf(stderr, "Failed to allocate memory for matrix AA conversion\n");
        return -1;
    }

    /* Convert tiled matrix to LAPACK format */
    hicma_parsec_Tile_to_Lapack_int(parsec, AA, ZAA, 1, 1);
       
    /* Compute total Hamming distance from matrix AA */
    long int total = 0;
    for (int i = 0; i < AA->llm; i++) {
       for (int j = 0; j <= i; j++) {
          /* Sum up all elements in the lower triangular part */
          total += ZAA[j * AA->lm + i];
       }
    }
    
    /* Print comparison between naive and matrix-based approaches */
    printf("The total humming distance of naive:%ld and of matrix-based:%ld\n",  
           counterin, total);
    free(ZAA);
    
    return 0;
}

/**
 * @brief Main function for Hamming distance testing
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return 0 on success
 */
int main(int argc, char **argv)
{    
    hicma_parsec_params_t params;
    starsh_params_t params_kernel;
    hicma_parsec_data_t data;
    hicma_parsec_matrix_analysis_t analysis = {0};

    /* Initialize HiCMA and PaRSEC */
    parsec_context_t* parsec = hicma_parsec_init(argc, argv, &params, &params_kernel, &data); 
    if (!parsec) {
        fprintf(stderr, "Failed to initialize HiCMA and PaRSEC\n");
        return EXIT_FAILURE;
    }

    /* Initialize matrix A for binary data storage (nsnp x N) */
    parsec_matrix_block_cyclic_t A;
    parsec_matrix_block_cyclic_init(&A, PARSEC_MATRIX_BYTE, PARSEC_MATRIX_TILE,
            params.rank, ((params.nsnp < params.NB) ? params.nsnp : params.NB), params.NB, 
            params.nsnp, params.N, 0, 0,
            params.nsnp, params.N, params.P, (params.nodes)/(params.P), 1, 1, 0, 0);

    /* Initialize matrix AA for Hamming distance results (N x N) */
    parsec_matrix_block_cyclic_t AA;
    parsec_matrix_block_cyclic_init(&AA, PARSEC_MATRIX_INTEGER, PARSEC_MATRIX_TILE,
            params.rank, params.NB, params.NB, params.N, params.N, 0, 0,
            params.N, params.N, params.P, (params.nodes)/(params.P), 1, 1, 0, 0);
 
    /* Allocate memory for matrices with appropriate bit sizes */
    parsec_dist_allocate_hamming(parsec, (parsec_tiled_matrix_t *)&A, 8, &params);   /* 8-bit for binary data */
    parsec_dist_allocate_hamming(parsec, (parsec_tiled_matrix_t *)&AA, 32, &params); /* 32-bit for distance results */

    /* Generate binary matrix data */
    SYNC_TIME_START();
    parsec_hamming_binary_generator(parsec, (parsec_tiled_matrix_t *)&A, &params);
    double time_generator = sync_time_elapsed; 
    SYNC_TIME_PRINT(params.rank, ("Matrix generation completed in %.6f seconds\n", time_generator));

    /* Find unique elements in the matrix */
    SYNC_TIME_START();
    hicma_parsec_matrix_unique_element(parsec, (parsec_tiled_matrix_t *)&A, &params);
    double time_unique = sync_time_elapsed;
    SYNC_TIME_PRINT(params.rank, ("Unique elements computation: %d elements in %.6f seconds\n", 
                                  params.nb_unique_elem, time_unique));

    /* Start timing for main computation */
    MPI_Barrier(MPI_COMM_WORLD);
    struct timeval tstart;
    gettimeofday(&tstart, NULL);
    params.start_time_gemm = tstart.tv_sec + tstart.tv_usec / 1.0e6;

    /* Compute Hamming distances for each unique element */
    double time_iter = (double)INT_MAX;
    double perf_iter = 0.0;
    double flops = (double)params.N * params.N * params.nsnp / 1e12;
    
    printf("Starting Hamming distance computation for %d unique elements...\n", params.nb_unique_elem);
    
    for (int i = 0; i < params.nb_unique_elem; i++) {
        SYNC_TIME_START();
        params.current_hamming_id = i;
        
        /* Compute Hamming distance using HiCMA */
        hicma_parsec_hamming_binary(parsec,
                PlasmaTrans, PlasmaNoTrans,
                1.0, (parsec_tiled_matrix_t *)&A,
                (parsec_tiled_matrix_t *)&A,
                1.0, (parsec_tiled_matrix_t *)&AA, &params, &data);
        
        /* Print performance for this element */
        SYNC_TIME_PRINT(params.rank, ("Element %d/%d: %.6f seconds, %.3f Tflop/s\n",
                    i + 1, params.nb_unique_elem, sync_time_elapsed, 
                    flops/sync_time_elapsed));
        
        /* Track best and worst performance */
        time_iter = hicma_parsec_dmin(time_iter, sync_time_elapsed); 
        perf_iter = hicma_parsec_dmax(perf_iter, flops/sync_time_elapsed);
    }

    /* End timing and calculate total performance */
    MPI_Barrier(MPI_COMM_WORLD);
    struct timeval tend;
    gettimeofday(&tend, NULL);
    double total_time = tend.tv_sec + tend.tv_usec / 1.0e6 - params.start_time_gemm;
    double avg_perf = flops / total_time * params.nb_unique_elem;
    
    /* Print final performance summary */
    if (0 == params.rank) {
        fprintf(stderr, "[****] PERFORMANCE SUMMARY:\n");
        fprintf(stderr, "  Total time: %.6f seconds\n", total_time);
        fprintf(stderr, "  Average performance: %.3f Tflop/s\n", avg_perf);
        fprintf(stderr, "  Best iteration: %.6f seconds (%.3f Tflop/s)\n", time_iter, flops/time_iter);
        fprintf(stderr, "  Worst iteration: %.6f seconds (%.3f Tflop/s)\n", 
                flops/perf_iter, perf_iter);
        fprintf(stderr, "  Configuration: nodes=%d, gpus=%d, N=%d, NB=%d, P=%d, Q=%d\n",
                params.nodes, params.gpus, params.N, params.NB, params.P, params.Q);
    }
    
    /* Validate results if CHECK is enabled */
    double time_check = 0.0; 
#if CHECK
    SYNC_TIME_START();
    
    /* Compute Hamming distance using naive approach for validation */
    if (check_hamming(parsec, (parsec_tiled_matrix_t *)&A, (parsec_tiled_matrix_t *)&AA) != 0) {
        fprintf(stderr, "Result validation failed\n");
        goto cleanup;
    }
    
    SYNC_TIME_PRINT(params.rank, ("Result validation completed in %.6f seconds\n", sync_time_elapsed));
    time_check = sync_time_elapsed;
#endif

    /* Print final output summary */
    if (0 == params.rank) {
        fprintf(stderr, "\nOUTPUT SUMMARY: %d %d %d %d %d %d %d   ", 
                params.N, params.NB, params.nodes, params.P, params.Q, params.gpus, params.lookahead);
        fprintf(stderr, "%d %.6f %.3f %.6f %.3f %.6f %.6f %.6f\n\n", 
                params.nb_unique_elem, total_time, avg_perf, time_iter, perf_iter, 
                time_check, time_generator, time_unique);
    }

cleanup:
    /* Free allocated memory */
    parsec_memory_free_tile(parsec, (parsec_tiled_matrix_t*)&A, &params, 0); 
    parsec_memory_free_tile(parsec, (parsec_tiled_matrix_t*)&AA, &params, 0); 

    /* Finalize HiCMA and PaRSEC */
    hicma_parsec_fini(parsec, argc, argv, &params, &params_kernel, &data, &analysis);

    return EXIT_SUCCESS;
}

